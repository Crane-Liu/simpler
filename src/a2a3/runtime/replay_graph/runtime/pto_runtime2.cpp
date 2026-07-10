/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

/**
 * PTO Runtime2 - Main Implementation
 *
 * Implements the unified runtime API that combines orchestrator and scheduler.
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#include "pto_runtime2.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>

#include "aicpu/device_time.h"
#include "common/unified_log.h"
#if PTO2_PROFILING
#include "aicpu/scope_stats_collector_aicpu.h"
#endif

// Weak fallback for HOST .so builds (never called, but satisfies linker).
// The AICPU build links the strong symbol from platform/.../device_time.cpp.
// Hidden visibility prevents HOST .so from polluting global symbol table.
__attribute__((weak, visibility("hidden"))) uint64_t get_sys_cnt_aicpu() { return 0; }

// =============================================================================
// Orchestration Ops Table (function-pointer dispatch for orchestration .so)
// =============================================================================

static TaskOutputTensors submit_task_impl(PTO2Runtime *rt, const MixedKernels &mixed_kernels, const L0TaskArgs &args) {
    return rt->orchestrator.submit_task(mixed_kernels, args);
}

static TaskOutputTensors alloc_tensors_impl(PTO2Runtime *rt, const L0TaskArgs &args) {
    return rt->orchestrator.alloc_tensors(args);
}

static TaskOutputTensors submit_dummy_task_impl(PTO2Runtime *rt, const L0TaskArgs &args) {
    return rt->orchestrator.submit_dummy_task(args);
}

void rt_scope_begin(PTO2Runtime *rt) {
    PTO2ScopeMode mode = rt->pending_scope_mode;
    rt->pending_scope_mode = PTO2ScopeMode::AUTO;
    rt->orchestrator.begin_scope(mode);
}

void rt_scope_end(PTO2Runtime *rt) { rt->orchestrator.end_scope(); }

void rt_orchestration_done(PTO2Runtime *rt) {
    if (rt != nullptr && !rt->orchestrator.fatal) {
        rt->graph_pipeline.all_done.store(1, std::memory_order_release);
        rt_graph_boundary(rt);
    }
    rt->orchestrator.mark_done();
}

void rt_graph_boundary(PTO2Runtime *rt) {
    if (rt == nullptr || rt->orchestrator.sm_header == nullptr) {
        return;
    }

    PTO2ReplayGraphPipelineState &pipeline = rt->graph_pipeline;
    const int32_t buf_id = pipeline.active_buffer;
    if (buf_id < 0 || buf_id >= PTO2_REPLAY_GRAPH_BUFFER_COUNT) {
        rt->orchestrator.report_fatal(
            PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "invalid active graph buffer id=%d", buf_id
        );
        return;
    }

    PTO2ReplayGraphBufferControl &buf = pipeline.buffers[buf_id];
    const int32_t total_tasks = rt->orchestrator.sm_header->fc.task_count.load(std::memory_order_acquire);
    const int32_t task_begin = buf.task_begin;
    const int32_t task_count = total_tasks >= task_begin ? total_tasks - task_begin : 0;

    buf.task_count = task_count;
    int32_t already_completed = 0;
    for (int32_t task_id = task_begin; task_id < total_tasks; task_id++) {
        PTO2TaskSlotState &slot = rt->orchestrator.sm_header->get_slot_state_by_task_id(task_id);
        if (slot.task_state.load(std::memory_order_acquire) >= PTO2_TASK_COMPLETED) {
            already_completed++;
        }
    }
    buf.completed_count.store(already_completed, std::memory_order_release);
    if (already_completed >= task_count) {
        buf.exec_done.store(1, std::memory_order_release);
        buf.state.store(PTO2_GRAPH_BUFFER_DONE, std::memory_order_release);
    } else {
        buf.exec_done.store(0, std::memory_order_release);
        buf.state.store(PTO2_GRAPH_BUFFER_RUNNING, std::memory_order_release);
    }

    const int32_t ready_begin = pipeline.published_initial_ready_count;
    const int32_t ready_end = rt->orchestrator.initial_ready_count;
    if (rt->orchestrator.scheduler != nullptr) {
        for (int32_t i = ready_begin; i < ready_end; i++) {
            rt->orchestrator.scheduler->push_ready_routed(rt->orchestrator.initial_ready[i]);
        }
    }
    pipeline.published_initial_ready_count = ready_end;
    pipeline.published_task_count.store(total_tasks, std::memory_order_release);

    int32_t publish_gate_released = 0;
    if (rt->orchestrator.scheduler != nullptr) {
        for (int32_t task_id = task_begin; task_id < total_tasks; task_id++) {
            PTO2TaskSlotState &slot = rt->orchestrator.sm_header->get_slot_state_by_task_id(task_id);
            if (slot.task_state.load(std::memory_order_acquire) >= PTO2_TASK_COMPLETED) {
                continue;
            }
            rt->orchestrator.scheduler->release_fanin_and_check_ready(slot);
            publish_gate_released++;
        }
    }

    LOG_INFO_V0(
        "replay_graph boundary stage=publish epoch=%llu buffer=%d tasks=[%d,%d) count=%d ready=[%d,%d) gates=%d already_done=%d",
        static_cast<unsigned long long>(buf.graph_epoch), buf_id, task_begin, total_tasks, task_count, ready_begin,
        ready_end, publish_gate_released, already_completed
    );

    pipeline.graph_count++;
    pipeline.current_graph_epoch++;

    const int32_t next_buf_id = (buf_id + 1) % PTO2_REPLAY_GRAPH_BUFFER_COUNT;
    PTO2ReplayGraphBufferControl &next = pipeline.buffers[next_buf_id];
    next.dep_closed.store(1, std::memory_order_release);
    pto2_try_release_graph_buffer(next);

    if (pipeline.all_done.load(std::memory_order_acquire) != 0) {
        buf.dep_closed.store(1, std::memory_order_release);
        pto2_try_release_graph_buffer(buf);
        return;
    }

    while (next.state.load(std::memory_order_acquire) != PTO2_GRAPH_BUFFER_FREE) {
        pto2_try_release_graph_buffer(next);
        if (rt->orchestrator.fatal) {
            return;
        }
        SPIN_WAIT_HINT();
    }

    rt->orchestrator.task_allocator.begin_buffer(next_buf_id);
    next.state.store(PTO2_GRAPH_BUFFER_BUILDING, std::memory_order_release);
    next.exec_done.store(0, std::memory_order_relaxed);
    next.dep_closed.store(0, std::memory_order_relaxed);
    next.completed_count.store(0, std::memory_order_relaxed);
    next.graph_epoch = pipeline.current_graph_epoch;
    next.task_begin = total_tasks;
    next.task_count = 0;
    pipeline.active_buffer = next_buf_id;
}

static bool is_fatal_impl(PTO2Runtime *rt) { return rt->orchestrator.fatal; }

void rt_report_fatal(PTO2Runtime *rt, int32_t error_code, const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    if (fmt == nullptr || fmt[0] == '\0') {
        rt->orchestrator.report_fatal(error_code, func, nullptr);
    } else {
        char message[1024];
        vsnprintf(message, sizeof(message), fmt, args);
        rt->orchestrator.report_fatal(error_code, func, "%s", message);
    }
    va_end(args);
}

// Wait for all producers of this tensor to be safe for data access.
// Checks owner metadata (lifecycle anchor) and OverlapMap (modifier writers).
// Waits until each producer COMPLETED (done writing). The single-shot replay
// model dropped the CONSUMED lifecycle, so there is no consumer-drain (WAR)
// wait anymore — see set_tensor_data for the implications.
// Uses cycle-based timeout (checked every 1024 spins).
// Returns false on timeout (sets orch.fatal).
MAYBE_UNINITIALIZED_BEGIN
static bool wait_for_tensor_ready(PTO2Runtime *rt, const Tensor &tensor, const char *caller) {
    PTO2TaskId owner = tensor.owner_task_id;
    PTO2OrchestratorState &orch = rt->orchestrator;

    // Segmented wait: collect up to kSegmentCap producer slots, then flush by
    // spinning on each. When the segment fills, we wait for the accumulated
    // batch before continuing to gather more. Dedup is per-segment only; a
    // producer that appears in two segments is waited on twice, which is
    // idempotent (task_state is monotonic) and only adds one atomic load on
    // the second encounter.
    constexpr int kSegmentCap = 64;
    const PTO2TaskSlotState *seg[kSegmentCap];
    int seg_count = 0;
    bool failed = false;

    auto wait_one_producer = [&](const PTO2TaskSlotState &slot) {
        uint8_t ring_id = slot.ring_id;
        int32_t local_id = static_cast<int32_t>(slot.task->task_id.local());
        uint64_t t0 = get_sys_cnt_aicpu();
        int32_t spin_count = 0;
        while (slot.task_state.load(std::memory_order_acquire) < PTO2_TASK_COMPLETED) {
            SPIN_WAIT_HINT();
            if ((++spin_count & 1023) == 0 && get_sys_cnt_aicpu() - t0 > PTO2_TENSOR_DATA_TIMEOUT_CYCLES) {
                orch.report_fatal(
                    PTO2_ERROR_TENSOR_WAIT_TIMEOUT, caller,
                    "Timeout (%llu cycles): producer (ring=%d, local=%d) not completed",
                    (unsigned long long)PTO2_TENSOR_DATA_TIMEOUT_CYCLES, ring_id, local_id
                );
                failed = true;
                return;
            }
        }
    };

    auto flush_segment = [&]() {
        for (int i = 0; i < seg_count; i++) {
            wait_one_producer(*seg[i]);
            if (failed) return;
        }
        seg_count = 0;
    };

    auto try_push = [&](const PTO2TaskSlotState &s) {
        for (int j = 0; j < seg_count; j++) {
            if (seg[j] == &s) return;  // per-segment dedup
        }
        if (seg_count == kSegmentCap) {
            flush_segment();
            if (failed) return;
        }
        seg[seg_count++] = &s;
    };

    auto do_wait = [&]() {
        // Step A: creator retention — read owner directly from tensor metadata
        if (owner.is_valid()) {
            auto &s = orch.sm_header->get_slot_state_by_task_id(owner.local());
            try_push(s);
            if (failed) return;
        }

        // Step B: modifier writer lookup (OverlapMap), direct callback
        orch.tensor_map.lookup(tensor, [&](PTO2TensorMapEntry &entry, OverlapStatus) -> bool {
            PTO2TaskId pid = entry.producer_task_id;
            auto &s = orch.sm_header->get_slot_state_by_task_id(pid.local());
            try_push(s);
            return !failed;
        });
        if (failed) return;
        flush_segment();
    };

    do_wait();
    return !failed;
}
MAYBE_UNINITIALIZED_END

uint64_t get_tensor_data(PTO2Runtime *rt, const Tensor &tensor, uint32_t ndims, const uint32_t indices[]) {
    if (tensor.buffer.addr == 0) {
        unified_log_error(
            __FUNCTION__, "get_tensor_data: buffer not allocated (addr=0). "
                          "Use the Tensor returned by add_output(TensorCreateInfo) after submit returns."
        );
        return 0;
    }

    if (!wait_for_tensor_ready(rt, tensor, __FUNCTION__)) {
        return 0;
    }

    uint64_t flat_offset = tensor.compute_flat_offset(indices, ndims);
    uint64_t elem_size = get_element_size(tensor.dtype);
    const void *ptr = reinterpret_cast<const void *>(tensor.buffer.addr + flat_offset * elem_size);
    uint64_t result = 0;
    memcpy(&result, ptr, elem_size);
    return result;
}

void set_tensor_data(PTO2Runtime *rt, const Tensor &tensor, uint32_t ndims, const uint32_t indices[], uint64_t value) {
    if (tensor.buffer.addr == 0) {
        unified_log_error(
            __FUNCTION__, "set_tensor_data: buffer not allocated (addr=0). "
                          "Use the Tensor returned by add_output(TensorCreateInfo) after submit returns."
        );
        return;
    }

    // Wait for the producer to finish writing before overwriting (WAW safety).
    // The single-shot replay model has no CONSUMED lifecycle, so we no longer
    // drain consumers first (WAR protection is dropped). No workload uses
    // set_tensor_data, so there is nothing to regress; revisit if one ever does.
    if (!wait_for_tensor_ready(rt, tensor, __FUNCTION__)) {
        return;
    }

    uint64_t flat_offset = tensor.compute_flat_offset(indices, ndims);
    uint64_t elem_size = get_element_size(tensor.dtype);
    void *ptr = reinterpret_cast<void *>(tensor.buffer.addr + flat_offset * elem_size);
    memcpy(ptr, &value, elem_size);
}

// Ops-table entry that hands the call-site captured by PTO2ScopeGuard to the
// [ScopeStats] collector. The slot is always present in the struct to keep
// the layout stable; at PTO2_PROFILING=0 we fill nullptr so the orchestration
// .so's null-check skips it.
#if PTO2_PROFILING
static void scope_set_site_impl(const char *file, int line) { scope_stats_set_pending_site(file, line); }
#endif

static const PTO2RuntimeOps s_runtime_ops = {
    .submit_task = submit_task_impl,
    .scope_begin = rt_scope_begin,
    .scope_end = rt_scope_end,
    .orchestration_done = rt_orchestration_done,
    .is_fatal = is_fatal_impl,
    .report_fatal = rt_report_fatal,
    .log_error = unified_log_error,
    .log_warn = unified_log_warn,
    .log_debug = unified_log_debug,
    .log_info_v = unified_log_info_v,
    .get_tensor_data = get_tensor_data,
    .set_tensor_data = set_tensor_data,
    .alloc_tensors = alloc_tensors_impl,
    .submit_dummy_task = submit_dummy_task_impl,
#if PTO2_PROFILING
    .scope_set_site = scope_set_site_impl,
#else
    .scope_set_site = nullptr,
#endif
    .graph_boundary = rt_graph_boundary,
};

// =============================================================================
// Runtime Lifecycle (AICPU-only fixup)
// =============================================================================
//
// Layout / init_data / wire / destroy live in
// runtime/shared/pto_runtime2_init.cpp so the host build can pre-populate the
// prebuilt arena image. The pieces below — wiring the ops table and the
// SPMD core counts — depend on the device-side s_runtime_ops global and the
// AICPU SchedulerContext respectively, so they remain in the AICPU build.

void runtime_finalize_after_wire(PTO2Runtime *rt, int32_t aic_count, int32_t aiv_count) {
    rt->ops = &s_runtime_ops;
    rt->orchestrator.total_cluster_count = aic_count;
    rt->orchestrator.total_aiv_count = aiv_count;
}

void runtime_set_mode(PTO2Runtime *rt, PTO2RuntimeMode mode) {
    if (rt) {
        rt->mode = mode;
    }
}
