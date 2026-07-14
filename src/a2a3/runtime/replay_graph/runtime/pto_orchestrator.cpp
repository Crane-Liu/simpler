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
 * PTO Runtime2 - Orchestrator Implementation
 *
 * Implements orchestrator state management, scope handling, and task submission.
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#include "pto_orchestrator.h"

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#include "aicpu/dep_gen_collector_aicpu.h"
#include "common/dep_gen.h"
#include "common/unified_log.h"
#include "pto_dep_compute.h"
#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"
#include "pto_tensormap.h"
#include "pto_types.h"
#include "tensor.h"

#if SIMPLER_DFX
#include "aicpu/scope_stats_collector_aicpu.h"
#include "aicpu/args_dump_aicpu.h"
#endif

// Verify the captured Tensor blob size in DepGenRecord matches the runtime
// Tensor layout. The platform header defines DEP_GEN_TENSOR_SIZE without
// including runtime/tensor.h, so this check lives at the orch callsite.
static_assert(sizeof(Tensor) == DEP_GEN_TENSOR_SIZE, "DepGenRecord::tensors slot size out of sync with sizeof(Tensor)");
// DEP_GEN_MAX_EXPLICIT_DEPS is a diagnostic-side capture cap only; the runtime
// imposes no hard cap on explicit dep count. If a submit exceeds this cap,
// dep_gen_aicpu_record_submit() logs and truncates — runtime correctness is
// unaffected, only the captured replay record is truncated.

// Weak fallbacks: dep_gen_collector_aicpu.cpp provides the strong symbols in
// AICPU builds. Host builds (host_build_graph runtime, future dep_gen replay)
// link these no-op stubs so the runtime translation unit is self-contained.
// Visibility is hidden so the HOST .so doesn't export them into the global
// dynamic symbol table where they'd shadow the AICPU .so's strong symbols
// (same pattern as get_sys_cnt_aicpu / l2_swimlane_aicpu_record_orch_phase below).
extern "C" __attribute__((weak, visibility("hidden"))) bool is_dep_gen_enabled() { return false; }
__attribute__((weak, visibility("hidden"))) void dep_gen_aicpu_record_submit(
    uint64_t, bool, bool, int, const void *const *, const uint8_t *, int, const uint64_t *, int, const int32_t[3]
) {}

// Scope_stats enable gate, queried via the same predicate idiom as
// is_dep_gen_enabled above. The AICPU collector links the strong definition;
// host builds fall back to this weak `false`. Gating here still skips the
// cross-agent occupancy reads that feed the sample when scope_stats is disabled.
extern "C" __attribute__((weak, visibility("hidden"))) bool is_scope_stats_enabled() { return false; }

// =============================================================================
// Orchestrator Profiling (compile-time toggle)
// =============================================================================
#if SIMPLER_ORCH_PROFILING
#include "aicpu/device_time.h"
#include "aicpu/l2_swimlane_collector_aicpu.h"
// Weak fallback for builds that don't link device_time.cpp (e.g. host).
// The strong symbol from platform/.../device_time.cpp wins in the AICPU build.
//
// IMPORTANT: visibility("hidden") is required to prevent the HOST .so from
// exporting this weak fallback into the global dynamic symbol table via
// RTLD_GLOBAL. Without it, when the AICPU .so is loaded and its PLT entry
// for get_sys_cnt_aicpu is resolved, the dynamic linker finds the HOST .so's
// weak definition first (already in global table) and uses it — returning 0.
// With hidden visibility, the HOST .so does not export this symbol globally,
// so the AICPU .so's PLT resolves to its own strong definition from
// device_time.cpp.
__attribute__((weak, visibility("hidden"))) uint64_t get_sys_cnt_aicpu() { return 0; }
// Weak fallback for builds that don't link l2_swimlane_collector_aicpu.cpp.
// The strong symbol from the AICPU build wins when profiling is available.
// Also hidden to prevent HOST .so from polluting the global symbol table.
__attribute__((weak, visibility("hidden"))) void
l2_swimlane_aicpu_record_orch_phase(uint64_t, uint64_t, uint64_t, uint32_t) {}
// Accumulated cycles per sub-step (only needed for ORCH_PROFILING export)
static uint64_t g_orch_alloc_cycle = 0;   // unified task+heap alloc
static uint64_t g_orch_args_cycle = 0;    // param copy
static uint64_t g_orch_lookup_cycle = 0;  // tensormap lookup + dep building
static uint64_t g_orch_insert_cycle = 0;  // tensormap insert
static uint64_t g_orch_fanin_cycle = 0;   // fanin list + early-return check
static int64_t g_orch_submit_count = 0;
static uint32_t g_orch_submit_idx = 0;
// Cycle accumulation is unconditional under SIMPLER_ORCH_PROFILING (that's what
// the flag is for) and feeds the per-sub-step `g_orch_*_cycle` cumulatives
// printed in the cold-path log.
//
// Per-submit ORCH_SUBMIT record is the only swim-lane emit on the orch
// path — one record per submit_task() / alloc_tensors() call spanning
// the entire [start, end] window. Per-sub-step phase records were dropped
// in favour of the cumulatives + per-submit envelope; the dispatcher
// already inserts one record at the end of each submit path via
// CYCLE_COUNT_ORCH_SUBMIT_RECORD.
#define CYCLE_COUNT_START()                                                        \
    bool _prof_active = (orch->l2_swimlane_level >= L2SwimlaneLevel::ORCH_PHASES); \
    uint64_t _t0 = get_sys_cnt_aicpu(), _t1;                                       \
    uint64_t _submit_start_ts = _t0
#define CYCLE_COUNT_LAP(acc)       \
    do {                           \
        _t1 = get_sys_cnt_aicpu(); \
        acc += (_t1 - _t0);        \
        _t0 = _t1;                 \
    } while (0)
#define CYCLE_COUNT_ORCH_SUBMIT_RECORD(tid)                                                       \
    do {                                                                                          \
        if (_prof_active) {                                                                       \
            l2_swimlane_aicpu_record_orch_phase(_submit_start_ts, _t1, (tid), g_orch_submit_idx); \
        }                                                                                         \
    } while (0)
#elif SIMPLER_DFX
#include "aicpu/device_time.h"
#include "aicpu/l2_swimlane_collector_aicpu.h"
__attribute__((weak, visibility("hidden"))) uint64_t get_sys_cnt_aicpu() { return 0; }
__attribute__((weak, visibility("hidden"))) void
l2_swimlane_aicpu_record_orch_phase(uint64_t, uint64_t, uint64_t, uint32_t) {}
// submit_idx needed for swimlane task_id tagging (no cycle accumulation at this level)
static uint32_t g_orch_submit_idx = 0;
#define CYCLE_COUNT_START()                                                        \
    bool _prof_active = (orch->l2_swimlane_level >= L2SwimlaneLevel::ORCH_PHASES); \
    uint64_t _t0 = _prof_active ? get_sys_cnt_aicpu() : 0, _t1 = 0;                \
    uint64_t _submit_start_ts = _t0
#define CYCLE_COUNT_LAP(acc) \
    do {                     \
    } while (0)
#define CYCLE_COUNT_ORCH_SUBMIT_RECORD(tid)                                                       \
    do {                                                                                          \
        if (_prof_active) {                                                                       \
            _t1 = get_sys_cnt_aicpu();                                                            \
            l2_swimlane_aicpu_record_orch_phase(_submit_start_ts, _t1, (tid), g_orch_submit_idx); \
        }                                                                                         \
    } while (0)
#else
#define CYCLE_COUNT_START()
#define CYCLE_COUNT_LAP(acc)
#define CYCLE_COUNT_ORCH_SUBMIT_RECORD(tid)
#endif

static int32_t orch_mark_fatal(PTO2OrchestratorState *orch, int32_t error_code) {
    always_assert(orch != nullptr);
    orch->fatal = true;
    if (error_code == PTO2_ERROR_NONE || orch->sm_header == nullptr) {
        return PTO2_ERROR_NONE;
    }

    int32_t expected = PTO2_ERROR_NONE;
    std::atomic<int32_t> &orch_error_code = orch->sm_header->orch_error_code;
    if (orch_error_code.compare_exchange_strong(expected, error_code, std::memory_order_acq_rel)) {
        return error_code;
    }
    return expected;
}

static void
orch_report_fatal_v(PTO2OrchestratorState *orch, int32_t error_code, const char *func, const char *fmt, va_list args) {
    int32_t latched_code = orch_mark_fatal(orch, error_code);

#if SIMPLER_DFX
    // Flush the current scope's peaks BEFORE the FATAL log line, so the
    // diagnostic context (which pool/window filled up) appears right next to
    // the failure reason. on_fatal is latched, so duplicate fatals from
    // different layers don't print multiple stats lines.
    scope_stats_on_fatal();
#endif

    if (fmt == nullptr || fmt[0] == '\0') {
        if (latched_code != PTO2_ERROR_NONE && latched_code != error_code) {
            unified_log_error(func, "FATAL(code=%d, latched=%d)", error_code, latched_code);
        } else {
            unified_log_error(func, "FATAL(code=%d)", error_code);
        }
        return;
    }

    char message[1024];
    vsnprintf(message, sizeof(message), fmt, args);
    if (latched_code != PTO2_ERROR_NONE && latched_code != error_code) {
        unified_log_error(func, "FATAL(code=%d, latched=%d): %s", error_code, latched_code, message);
        return;
    }
    unified_log_error(func, "FATAL(code=%d): %s", error_code, message);
}

void PTO2OrchestratorState::report_fatal(int32_t error_code, const char *func, const char *fmt, ...) {
    auto *orch = this;
    va_list args;
    va_start(args, fmt);
    orch_report_fatal_v(orch, error_code, func, fmt, args);
    va_end(args);
}

static uint32_t next_fanin_seen_epoch(PTO2OrchestratorState *orch) {
    uint32_t next = orch->fanin_seen_current_epoch + 1;
    if (next == 0) {
        memset(orch->fanin_seen_epoch, 0, static_cast<size_t>(orch->sm_header->task_window_size) * sizeof(uint32_t));
        next = 1;
    }
    orch->fanin_seen_current_epoch = next;
    return next;
}

namespace {

constexpr int32_t PTO2_GRAPH_TEMPLATE_CAP = 16;
constexpr int32_t PTO2_GRAPH_MAX_TASKS = 1024;
constexpr int32_t PTO2_GRAPH_MAX_FANIN_PER_TASK = 128;

enum class PTO2GraphTensorSource : uint8_t {
    BOUNDARY_EXACT = 0,
    BOUNDARY_VIEW = 1,
    INTERNAL = 2,
    OWN_OUTPUT = 3,
};

enum class PTO2GraphFaninSource : uint8_t {
    INTERNAL = 0,
    EXTERNAL_LOCAL_DELTA = 1,
};

struct PTO2GraphTensorPatch {
    PTO2GraphTensorSource source{PTO2GraphTensorSource::BOUNDARY_EXACT};
    uint16_t source_index{0};
    uint64_t packed_offset{0};
};

struct PTO2GraphFaninRef {
    PTO2GraphFaninSource source{PTO2GraphFaninSource::INTERNAL};
    int32_t value{0};
};

struct PTO2GraphTaskTemplate {
    int32_t kernel_id[PTO2_SUBTASK_SLOT_COUNT]{INVALID_KERNEL_ID, INVALID_KERNEL_ID, INVALID_KERNEL_ID};
    ActiveMask active_mask{};
    int16_t logical_block_num{1};
    int16_t total_required_subtasks{0};
    bool completed_inline{false};
    bool allow_early_resolve{false};
    int32_t tensor_count{0};
    int32_t scalar_count{0};
    int32_t total_output_size{0};
    uint64_t record_packed_base{0};
    Tensor tensors[MAX_TENSOR_ARGS];
    PTO2GraphTensorPatch tensor_patch[MAX_TENSOR_ARGS];
    TensorArgType tensor_arg_types[MAX_TENSOR_ARGS];
    uint64_t scalars[MAX_SCALAR_ARGS];
    uint16_t scalar_binding_indices[MAX_SCALAR_ARGS];
    int32_t fanin_count{0};
    PTO2GraphFaninRef fanins[PTO2_GRAPH_MAX_FANIN_PER_TASK];
};

struct PTO2GraphTemplate {
    bool in_use{false};
    uint64_t full_key{0};
    int32_t task_count{0};
    PTO2GraphTaskTemplate tasks[PTO2_GRAPH_MAX_TASKS];
};

struct PTO2GraphRecordingState {
    bool active{false};
    bool unsupported{false};
    int32_t unsupported_reason{0};
    int32_t unsupported_task_index{-1};
    int32_t unsupported_tensor_index{-1};
    uint64_t full_key{0};
    int32_t start_local_task_id{0};
    int32_t current_task_index{-1};
    int32_t current_fanin_count{0};
    PTO2GraphFaninRef current_fanins[PTO2_GRAPH_MAX_FANIN_PER_TASK];
    PTO2GraphBindings bindings;
    PTO2GraphTemplate temp;
};

// The AICPU runtime DSO stays loaded for the DeviceRunner lifetime, so this
// cache survives repeated simpler_run calls in the same process.
PTO2GraphTemplate g_graph_templates[PTO2_GRAPH_TEMPLATE_CAP];
int32_t g_graph_next_replace = 0;
PTO2GraphRecordingState g_graph_recording;

enum PTO2GraphUnsupportedReason : int32_t {
    PTO2_GRAPH_UNSUPPORTED_NONE = 0,
    PTO2_GRAPH_UNSUPPORTED_TASK_WINDOW = 1,
    PTO2_GRAPH_UNSUPPORTED_NULL_PRODUCER = 2,
    PTO2_GRAPH_UNSUPPORTED_EXTERNAL_PRODUCER = 3,
    PTO2_GRAPH_UNSUPPORTED_FANIN_OVERFLOW = 4,
    PTO2_GRAPH_UNSUPPORTED_RECORD_TASK_ORDER = 5,
    PTO2_GRAPH_UNSUPPORTED_RECORD_TASK_NULL = 6,
    PTO2_GRAPH_UNSUPPORTED_ARG_OVERFLOW = 7,
    PTO2_GRAPH_UNSUPPORTED_TENSOR_SOURCE = 8,
    PTO2_GRAPH_UNSUPPORTED_NESTED_SCOPE = 9,
    PTO2_GRAPH_UNSUPPORTED_EXTERNAL_EXPLICIT_DEP = 10,
};

void graph_mark_unsupported(PTO2GraphUnsupportedReason reason, int32_t task_index = -1, int32_t tensor_index = -1) {
    if (!g_graph_recording.unsupported) {
        g_graph_recording.unsupported_reason = static_cast<int32_t>(reason);
        g_graph_recording.unsupported_task_index = task_index;
        g_graph_recording.unsupported_tensor_index = tensor_index;
    }
    g_graph_recording.unsupported = true;
}

void reset_graph_template_header(PTO2GraphTemplate *templ) {
    templ->in_use = false;
    templ->full_key = 0;
    templ->task_count = 0;
}

void reset_graph_recording() {
    g_graph_recording.active = false;
    g_graph_recording.unsupported = false;
    g_graph_recording.unsupported_reason = PTO2_GRAPH_UNSUPPORTED_NONE;
    g_graph_recording.unsupported_task_index = -1;
    g_graph_recording.unsupported_tensor_index = -1;
    g_graph_recording.full_key = 0;
    g_graph_recording.start_local_task_id = 0;
    g_graph_recording.current_task_index = -1;
    g_graph_recording.current_fanin_count = 0;
    g_graph_recording.bindings = PTO2GraphBindings{};
    reset_graph_template_header(&g_graph_recording.temp);
}

uint64_t graph_full_key(uint64_t callable_hash, uint64_t graph_key) {
    uint64_t h = 1469598103934665603ULL;
    h = pto2_graph_hash_bytes(h, &PTO2_GRAPH_CACHE_SCHEMA_VERSION, sizeof(PTO2_GRAPH_CACHE_SCHEMA_VERSION));
    h = pto2_graph_hash_bytes(h, &callable_hash, sizeof(callable_hash));
    return pto2_graph_hash_bytes(h, &graph_key, sizeof(graph_key));
}

PTO2GraphTemplate *find_graph_template(uint64_t full_key) {
    for (int32_t i = 0; i < PTO2_GRAPH_TEMPLATE_CAP; ++i) {
        if (g_graph_templates[i].in_use && g_graph_templates[i].full_key == full_key) {
            return &g_graph_templates[i];
        }
    }
    return nullptr;
}

void store_graph_template(const PTO2GraphTemplate &templ) {
    int32_t target = -1;
    for (int32_t i = 0; i < PTO2_GRAPH_TEMPLATE_CAP; ++i) {
        if (!g_graph_templates[i].in_use) {
            target = i;
            break;
        }
    }
    if (target < 0) {
        target = g_graph_next_replace;
        g_graph_next_replace = (g_graph_next_replace + 1) % PTO2_GRAPH_TEMPLATE_CAP;
    }
    g_graph_templates[target] = templ;
    g_graph_templates[target].in_use = true;
}

void graph_record_begin_task(PTO2TaskId task_id) {
    if (!g_graph_recording.active || g_graph_recording.unsupported) return;
    int32_t idx = static_cast<int32_t>(task_id.local()) - g_graph_recording.start_local_task_id;
    if (idx < 0 || idx >= PTO2_GRAPH_MAX_TASKS || idx != g_graph_recording.temp.task_count) {
        graph_mark_unsupported(PTO2_GRAPH_UNSUPPORTED_TASK_WINDOW, idx);
        return;
    }
    g_graph_recording.current_task_index = idx;
    g_graph_recording.current_fanin_count = 0;
}

void graph_record_note_fanin(PTO2TaskSlotState *producer) {
    if (!g_graph_recording.active || g_graph_recording.unsupported) return;
    if (producer == nullptr || producer->task == nullptr) {
        graph_mark_unsupported(PTO2_GRAPH_UNSUPPORTED_NULL_PRODUCER, g_graph_recording.current_task_index);
        return;
    }
    int32_t producer_local = static_cast<int32_t>(producer->task->task_id.local());
    int32_t producer_index = producer_local - g_graph_recording.start_local_task_id;
    if (producer_index >= g_graph_recording.current_task_index) {
        graph_mark_unsupported(PTO2_GRAPH_UNSUPPORTED_EXTERNAL_PRODUCER, g_graph_recording.current_task_index);
        return;
    }
    if (g_graph_recording.current_fanin_count >= PTO2_GRAPH_MAX_FANIN_PER_TASK) {
        graph_mark_unsupported(PTO2_GRAPH_UNSUPPORTED_FANIN_OVERFLOW, g_graph_recording.current_task_index);
        return;
    }
    PTO2GraphFaninRef &fanin = g_graph_recording.current_fanins[g_graph_recording.current_fanin_count++];
    if (producer_index >= 0) {
        fanin.source = PTO2GraphFaninSource::INTERNAL;
        fanin.value = producer_index;
    } else {
        fanin.source = PTO2GraphFaninSource::EXTERNAL_LOCAL_DELTA;
        fanin.value = producer_local - g_graph_recording.start_local_task_id;
    }
}

}  // namespace

static bool append_fanin_or_fail(
    PTO2OrchestratorState *orch, PTO2TaskSlotState *consumer, int32_t prod_slot, PTO2TaskSlotState *producer,
    uint32_t seen_epoch
) {
    if (producer == nullptr) return true;
    if (orch->fanin_seen_epoch[prod_slot] == seen_epoch) {
        return true;
    }
    orch->fanin_seen_epoch[prod_slot] = seen_epoch;

    graph_record_note_fanin(producer);

    PTO2DepListEntry *head = producer->fanout_head.load(std::memory_order_acquire);
    if (producer->task_state.load(std::memory_order_acquire) >= PTO2_TASK_COMPLETED || pto2_is_fanout_closed(head)) {
        return true;
    }

    PTO2DepListEntry *entry = orch->dep_pool.alloc();
    if (entry == nullptr) {
        orch_mark_fatal(orch, PTO2_ERROR_DEP_POOL_OVERFLOW);
        return false;
    }
    entry->slot_state = consumer;
    consumer->fanin_count++;
    while (true) {
        if (producer->task_state.load(std::memory_order_acquire) >= PTO2_TASK_COMPLETED ||
            pto2_is_fanout_closed(head)) {
            consumer->fanin_refcount.fetch_add(1, std::memory_order_acq_rel);
            return true;
        }
        entry->next = head;
        if (producer->fanout_head.compare_exchange_weak(
                head, entry, std::memory_order_acq_rel, std::memory_order_acquire
            )) {
            return true;
        }
    }
}

struct PTO2PreparedTask {
    PTO2TaskId task_id = PTO2TaskId::invalid();
    PTO2TaskAllocResult alloc_result = {-1, 0, nullptr, nullptr};
    PTO2TaskDescriptor *task = nullptr;
    PTO2TaskPayload *payload = nullptr;
    PTO2TaskSlotState *slot_state = nullptr;
};

namespace {

static bool prepare_graph_task(PTO2OrchestratorState *orch, const PTO2GraphTaskTemplate &templ, PTO2PreparedTask *out) {
    out->alloc_result = orch->task_allocator.alloc(templ.total_output_size);
    if (out->alloc_result.failed()) {
        orch_mark_fatal(orch, PTO2_ERROR_HEAP_RING_DEADLOCK);
        return false;
    }

    out->task_id = PTO2TaskId::make(0, static_cast<uint32_t>(out->alloc_result.task_id));
    out->slot_state = &orch->sm_header->get_slot_state_by_slot(out->alloc_result.slot);
    out->task = &orch->sm_header->task_descriptors[out->alloc_result.slot];
    out->payload = &orch->sm_header->task_payloads[out->alloc_result.slot];
    out->payload->prefetch(templ.tensor_count, templ.scalar_count);
    out->slot_state->bind_buffers(out->payload, out->task);
    out->slot_state->fanout_head.store(nullptr, std::memory_order_relaxed);
    out->slot_state->task_state.store(PTO2_TASK_PENDING, std::memory_order_relaxed);
    out->slot_state->fanin_refcount.store(0, std::memory_order_relaxed);
    out->slot_state->fanin_count = 1;
    out->slot_state->active_mask = templ.active_mask;
    out->slot_state->bind_ring(0);
    out->slot_state->set_allow_early_resolve(templ.allow_early_resolve);
    out->slot_state->ready_state.store(PTO2_READY_UNCLAIMED, std::memory_order_relaxed);
    out->slot_state->completed_subtasks.store(0, std::memory_order_relaxed);
    out->slot_state->total_required_subtasks = templ.total_required_subtasks;
    out->slot_state->logical_block_num = templ.logical_block_num;
    out->slot_state->next_block_idx.store(0, std::memory_order_relaxed);
    return true;
}

static bool graph_tensor_metadata_equal(const Tensor &lhs, const Tensor &rhs) {
    if (lhs.buffer.addr != rhs.buffer.addr || lhs.buffer.size != rhs.buffer.size ||
        lhs.start_offset != rhs.start_offset || lhs.version != rhs.version || lhs.ndims != rhs.ndims ||
        lhs.dtype != rhs.dtype || lhs.manual_dep != rhs.manual_dep || lhs.is_contiguous != rhs.is_contiguous ||
        lhs.child_memory != rhs.child_memory) {
        return false;
    }
    return memcmp(lhs.shapes, rhs.shapes, sizeof(uint32_t) * lhs.ndims) == 0 &&
           memcmp(lhs.strides, rhs.strides, sizeof(uint32_t) * lhs.ndims) == 0;
}

static bool
graph_tensor_from_boundary(const Tensor &tensor, const PTO2GraphBindings &bindings, PTO2GraphTensorPatch *patch) {
    for (uint32_t i = 0; i < bindings.tensor_count; ++i) {
        if (graph_tensor_metadata_equal(tensor, bindings.tensors[i])) {
            patch->source = PTO2GraphTensorSource::BOUNDARY_EXACT;
            patch->source_index = static_cast<uint16_t>(i);
            patch->packed_offset = 0;
            return true;
        }
    }
    uint16_t best_index = PTO2_GRAPH_INVALID_BINDING;
    uint64_t best_offset = UINT64_MAX;
    for (uint32_t i = 0; i < bindings.tensor_count; ++i) {
        const Tensor &boundary = bindings.tensors[i];
        if (tensor.buffer.addr == boundary.buffer.addr && tensor.buffer.size == boundary.buffer.size &&
            tensor.start_offset >= boundary.start_offset) {
            uint64_t offset = tensor.start_offset - boundary.start_offset;
            if (offset < best_offset) {
                best_index = static_cast<uint16_t>(i);
                best_offset = offset;
            }
        }
    }
    if (best_index == PTO2_GRAPH_INVALID_BINDING) return false;
    patch->source = PTO2GraphTensorSource::BOUNDARY_VIEW;
    patch->source_index = best_index;
    patch->packed_offset = best_offset;
    return true;
}

static bool
graph_classify_tensor(PTO2GraphTaskTemplate *task, int32_t tensor_index, const Tensor &tensor, int32_t task_index) {
    PTO2GraphTensorPatch &patch = task->tensor_patch[tensor_index];
    if (graph_tensor_from_boundary(tensor, g_graph_recording.bindings, &patch)) return true;

    uint64_t tensor_addr = tensor.buffer.addr;
    for (int32_t producer_index = task_index; producer_index >= 0; --producer_index) {
        const PTO2GraphTaskTemplate &producer =
            (producer_index == task_index) ? *task : g_graph_recording.temp.tasks[producer_index];
        if (producer.record_packed_base == 0 || producer.total_output_size <= 0) continue;
        uint64_t producer_begin = producer.record_packed_base;
        uint64_t producer_end = producer_begin + static_cast<uint64_t>(producer.total_output_size);
        if (tensor_addr < producer_begin || tensor_addr >= producer_end) continue;
        patch.source =
            producer_index == task_index ? PTO2GraphTensorSource::OWN_OUTPUT : PTO2GraphTensorSource::INTERNAL;
        patch.source_index = static_cast<uint16_t>(producer_index);
        patch.packed_offset = tensor_addr - producer_begin;
        return true;
    }
    return false;
}

static void graph_record_task(const PTO2PreparedTask &prepared, const L0TaskArgs &args, bool completed_inline) {
    if (!g_graph_recording.active || g_graph_recording.unsupported) return;
    int32_t task_index = static_cast<int32_t>(prepared.task_id.local()) - g_graph_recording.start_local_task_id;
    if (task_index < 0 || task_index >= PTO2_GRAPH_MAX_TASKS || task_index != g_graph_recording.temp.task_count) {
        graph_mark_unsupported(PTO2_GRAPH_UNSUPPORTED_RECORD_TASK_ORDER, task_index);
        return;
    }
    if (prepared.task == nullptr || prepared.payload == nullptr || prepared.slot_state == nullptr) {
        graph_mark_unsupported(PTO2_GRAPH_UNSUPPORTED_RECORD_TASK_NULL, task_index);
        return;
    }

    PTO2GraphTaskTemplate &task = g_graph_recording.temp.tasks[task_index];
    task = PTO2GraphTaskTemplate{};
    for (int i = 0; i < PTO2_SUBTASK_SLOT_COUNT; ++i)
        task.kernel_id[i] = prepared.task->kernel_id[i];
    task.active_mask = prepared.slot_state->active_mask;
    task.logical_block_num = prepared.slot_state->logical_block_num;
    task.total_required_subtasks = prepared.slot_state->total_required_subtasks;
    task.completed_inline = completed_inline;
    task.allow_early_resolve = prepared.slot_state->allow_early_resolve;
    task.tensor_count = prepared.payload->tensor_count;
    task.scalar_count = prepared.payload->scalar_count;
    task.total_output_size = static_cast<int32_t>(
        reinterpret_cast<uintptr_t>(prepared.task->packed_buffer_end) -
        reinterpret_cast<uintptr_t>(prepared.task->packed_buffer_base)
    );
    task.record_packed_base = reinterpret_cast<uint64_t>(prepared.task->packed_buffer_base);
    if (task.tensor_count < 0 || task.tensor_count > MAX_TENSOR_ARGS || task.scalar_count < 0 ||
        task.scalar_count > MAX_SCALAR_ARGS) {
        graph_mark_unsupported(PTO2_GRAPH_UNSUPPORTED_ARG_OVERFLOW, task_index);
        return;
    }

    for (int32_t i = 0; i < task.tensor_count; ++i) {
        task.tensors[i].copy(prepared.payload->tensors[i]);
        task.tensor_arg_types[i] = args.tag(i);
        if (!graph_classify_tensor(&task, i, task.tensors[i], task_index)) {
            graph_mark_unsupported(PTO2_GRAPH_UNSUPPORTED_TENSOR_SOURCE, task_index, i);
            return;
        }
    }
    memcpy(task.scalars, prepared.payload->scalars, sizeof(uint64_t) * static_cast<size_t>(task.scalar_count));
    for (int32_t i = 0; i < task.scalar_count; ++i)
        task.scalar_binding_indices[i] = args.graph_scalar_binding(i);
    task.fanin_count = g_graph_recording.current_fanin_count;
    for (int32_t i = 0; i < task.fanin_count; ++i)
        task.fanins[i] = g_graph_recording.current_fanins[i];

    g_graph_recording.temp.task_count++;
    g_graph_recording.current_task_index = -1;
}

static void graph_reset_payload(PTO2TaskPayload *payload) {
    payload->early_dispatch_state.store(PTO2_EARLY_DISPATCH_NONE, std::memory_order_relaxed);
    for (int w = 0; w < PTO2_EARLY_DISPATCH_CORE_MASK_WORDS; ++w) {
        payload->staged_core_mask[w].store(0, std::memory_order_relaxed);
    }
    payload->dispatch_fanin.store(0, std::memory_order_relaxed);
    payload->dispatch_propagated.store(0, std::memory_order_relaxed);
    payload->published_block_count.store(0, std::memory_order_relaxed);
    payload->early_dispatch_launch_state.store(PTO2_EARLY_DISPATCH_LAUNCH_NONE, std::memory_order_relaxed);
    payload->running_slot_count.store(0, std::memory_order_relaxed);
    payload->early_sync_drain_state.store(PTO2_EARLY_SYNC_DRAIN_NONE, std::memory_order_relaxed);
}

static bool graph_template_preflight(
    PTO2OrchestratorState *orch, const PTO2GraphTemplate &templ, const PTO2GraphBindings &bindings
) {
    if (templ.task_count <= 0 || templ.task_count > PTO2_GRAPH_MAX_TASKS) return false;
    uint64_t required_heap = 0;
    int32_t required_edges = 0;
    int32_t required_tensormap_entries = 0;
    for (int32_t i = 0; i < templ.task_count; ++i) {
        const PTO2GraphTaskTemplate &task = templ.tasks[i];
        if (task.tensor_count < 0 || task.tensor_count > MAX_TENSOR_ARGS || task.scalar_count < 0 ||
            task.scalar_count > MAX_SCALAR_ARGS || task.fanin_count < 0 ||
            task.fanin_count > PTO2_GRAPH_MAX_FANIN_PER_TASK || task.total_output_size < 0) {
            return false;
        }
        required_heap += PTO2_ALIGN_UP(static_cast<uint64_t>(task.total_output_size), PTO2_ALIGN_SIZE);
        for (int32_t e = 0; e < task.fanin_count; ++e) {
            if (task.fanins[e].source == PTO2GraphFaninSource::INTERNAL) required_edges++;
        }
        for (int32_t j = 0; j < task.tensor_count; ++j) {
            const PTO2GraphTensorPatch &patch = task.tensor_patch[j];
            bool is_boundary = patch.source == PTO2GraphTensorSource::BOUNDARY_EXACT ||
                               patch.source == PTO2GraphTensorSource::BOUNDARY_VIEW;
            if (is_boundary && patch.source_index >= bindings.tensor_count) {
                return false;
            }
            if (patch.source == PTO2GraphTensorSource::INTERNAL && patch.source_index >= i) return false;
            if (is_boundary &&
                (task.tensor_arg_types[j] == TensorArgType::INOUT ||
                 task.tensor_arg_types[j] == TensorArgType::OUTPUT_EXISTING) &&
                !bindings.tensors[patch.source_index].manual_dep) {
                required_tensormap_entries++;
            }
        }
        for (int32_t j = 0; j < task.scalar_count; ++j) {
            uint16_t binding_index = task.scalar_binding_indices[j];
            if (binding_index != PTO2_GRAPH_SCALAR_STATIC && binding_index >= bindings.scalar_count) return false;
        }
        for (int32_t e = 0; e < task.fanin_count; ++e) {
            const PTO2GraphFaninRef &fanin = task.fanins[e];
            if (fanin.source == PTO2GraphFaninSource::INTERNAL && (fanin.value < 0 || fanin.value >= i)) {
                return false;
            }
        }
    }
    if (templ.task_count > orch->task_allocator.task_available()) return false;
    if (required_heap > orch->task_allocator.heap_available()) return false;
    if (required_edges > orch->dep_pool.available()) return false;
    if (required_tensormap_entries > orch->tensor_map.free_entries()) return false;
    return true;
}

static bool graph_replay_template(
    PTO2OrchestratorState *orch, const PTO2GraphTemplate &templ, const PTO2GraphBindings &bindings,
    uint64_t *orch_record_task_id
) {
    if (orch_record_task_id != nullptr) *orch_record_task_id = PTO2TaskId::invalid().raw;
    if (!graph_template_preflight(orch, templ, bindings)) return false;

    PTO2PreparedTask prepared[PTO2_GRAPH_MAX_TASKS];
    for (int32_t i = 0; i < templ.task_count; ++i) {
        prepared[i] = PTO2PreparedTask{};
        if (!prepare_graph_task(orch, templ.tasks[i], &prepared[i])) return false;
    }

    for (int32_t i = 0; i < templ.task_count; ++i) {
        const PTO2GraphTaskTemplate &src = templ.tasks[i];
        PTO2PreparedTask &dst = prepared[i];
        PTO2TaskDescriptor &task = *dst.task;
        PTO2TaskPayload &payload = *dst.payload;
        PTO2TaskSlotState &slot = *dst.slot_state;

        if (orch_record_task_id != nullptr && !PTO2TaskId{*orch_record_task_id}.is_valid() && !src.completed_inline) {
            *orch_record_task_id = dst.task_id.raw;
        }
        task.task_id = dst.task_id;
        for (int k = 0; k < PTO2_SUBTASK_SLOT_COUNT; ++k)
            task.kernel_id[k] = src.kernel_id[k];
        task.packed_buffer_base = dst.alloc_result.packed_base;
        task.packed_buffer_end = dst.alloc_result.packed_end;

        payload.tensor_count = src.tensor_count;
        payload.scalar_count = src.scalar_count;
        for (int32_t j = 0; j < src.tensor_count; ++j) {
            Tensor tensor;
            tensor.copy(src.tensors[j]);
            const PTO2GraphTensorPatch &patch = src.tensor_patch[j];
            if (patch.source == PTO2GraphTensorSource::BOUNDARY_EXACT) {
                const Tensor &boundary = bindings.tensors[patch.source_index];
                tensor.copy(boundary);
            } else if (patch.source == PTO2GraphTensorSource::BOUNDARY_VIEW) {
                const Tensor &boundary = bindings.tensors[patch.source_index];
                tensor.buffer = boundary.buffer;
                tensor.owner_task_id = boundary.owner_task_id;
                tensor.start_offset = boundary.start_offset + patch.packed_offset;
                tensor.version = boundary.version;
                tensor.child_memory = boundary.child_memory;
            } else {
                int32_t producer_index =
                    patch.source == PTO2GraphTensorSource::OWN_OUTPUT ? i : static_cast<int32_t>(patch.source_index);
                PTO2PreparedTask &producer = prepared[producer_index];
                tensor.buffer.addr =
                    reinterpret_cast<uint64_t>(producer.task->packed_buffer_base) + patch.packed_offset;
                tensor.owner_task_id = producer.task_id;
            }
            payload.tensors[j].copy(tensor);
        }
        for (int32_t j = 0; j < src.scalar_count; ++j) {
            uint16_t binding_index = src.scalar_binding_indices[j];
            payload.scalars[j] =
                binding_index == PTO2_GRAPH_SCALAR_STATIC ? src.scalars[j] : bindings.scalars[binding_index];
        }
        graph_reset_payload(&payload);

        if (src.completed_inline) {
            slot.mark_completed();
            orch->inline_completed_tasks++;
        }
    }

    for (int32_t i = 0; i < templ.task_count; ++i) {
        const PTO2GraphTaskTemplate &consumer_template = templ.tasks[i];
        PTO2TaskSlotState *consumer = prepared[i].slot_state;
        if (consumer_template.completed_inline) continue;
        uint32_t seen_epoch = next_fanin_seen_epoch(orch);
        for (int32_t e = 0; e < consumer_template.fanin_count; ++e) {
            const PTO2GraphFaninRef &fanin = consumer_template.fanins[e];
            if (fanin.source != PTO2GraphFaninSource::INTERNAL) continue;
            int32_t producer_slot = prepared[fanin.value].alloc_result.slot;
            PTO2TaskSlotState *producer = prepared[fanin.value].slot_state;
            if (!append_fanin_or_fail(orch, consumer, producer_slot, producer, seen_epoch)) return false;
        }

        TensorRef boundary_tensors[MAX_TENSOR_ARGS];
        TensorArgType boundary_types[MAX_TENSOR_ARGS];
        int32_t boundary_count = 0;
        for (int32_t j = 0; j < consumer_template.tensor_count; ++j) {
            const PTO2GraphTensorPatch &patch = consumer_template.tensor_patch[j];
            if (patch.source != PTO2GraphTensorSource::BOUNDARY_EXACT &&
                patch.source != PTO2GraphTensorSource::BOUNDARY_VIEW) {
                continue;
            }
            boundary_tensors[boundary_count] = &prepared[i].payload->tensors[j];
            boundary_types[boundary_count] = consumer_template.tensor_arg_types[j];
            boundary_count++;
        }

        DepInputs boundary_inputs{boundary_count, boundary_tensors, boundary_types, 0, nullptr};
        auto boundary_emit = [&](PTO2TaskId producer_task_id) -> bool {
            int32_t producer_local = static_cast<int32_t>(producer_task_id.local());
            int32_t producer_slot = orch->sm_header->get_slot_by_task_id(producer_local);
            PTO2TaskSlotState *producer = orch->sm_header->find_live_slot_state(producer_task_id);
            if (producer == nullptr) return true;
            return append_fanin_or_fail(orch, consumer, producer_slot, producer, seen_epoch);
        };
        if (!compute_task_fanin(boundary_inputs, orch->tensor_map, orch->in_manual_scope(), boundary_emit)) {
            return false;
        }
        register_task_outputs(boundary_inputs, prepared[i].task_id, orch->tensor_map, orch->in_manual_scope());
    }

    if (orch_record_task_id != nullptr && !PTO2TaskId{*orch_record_task_id}.is_valid()) {
        *orch_record_task_id = prepared[0].task_id.raw;
    }
#if SIMPLER_DFX
    orch->tasks_submitted += templ.task_count;
#endif
    return true;
}

}  // namespace

static PTO2OutputLayout calculate_output_layout(const L0TaskArgs &args) {
    PTO2OutputLayout layout;
    for (int32_t i = 0; i < args.tensor_count(); i++) {
        if (args.tag(i) != TensorArgType::OUTPUT) {
            continue;
        }
        layout.offsets[i] = layout.total_output_size;
        layout.buffer_sizes[i] =
            PTO2_ALIGN_UP(args.tensor(i).create_info().buffer_size_bytes(), PTO2_PACKED_OUTPUT_ALIGN);
        layout.total_output_size += layout.buffer_sizes[i];
    }
    return layout;
}

static bool prepare_task(
    PTO2OrchestratorState *orch, const L0TaskArgs &args, int32_t total_output_size, ActiveMask active_mask,
    PTO2PreparedTask *out
) {
    always_assert(orch->scope_stack_top >= 0 && "Cannot submit task outside a scope");
    out->alloc_result = orch->task_allocator.alloc(total_output_size);
    if (out->alloc_result.failed()) {
        orch_mark_fatal(orch, PTO2_ERROR_HEAP_RING_DEADLOCK);
        return false;
    }

    out->task_id = PTO2TaskId::make(0, static_cast<uint32_t>(out->alloc_result.task_id));
    out->slot_state = &orch->sm_header->get_slot_state_by_slot(out->alloc_result.slot);
    out->task = &orch->sm_header->task_descriptors[out->alloc_result.slot];
    out->payload = &orch->sm_header->task_payloads[out->alloc_result.slot];

    out->payload->prefetch(args.tensor_count(), args.scalar_count());

    out->slot_state->bind_buffers(out->payload, out->task);
    graph_reset_payload(out->payload);
    out->slot_state->fanout_head.store(nullptr, std::memory_order_relaxed);
    out->slot_state->task_state.store(PTO2_TASK_PENDING, std::memory_order_relaxed);
    out->slot_state->fanin_refcount.store(0, std::memory_order_relaxed);
    out->slot_state->fanin_count = 1;
    out->slot_state->bind_ring(0);
    out->slot_state->ready_state.store(PTO2_READY_UNCLAIMED, std::memory_order_relaxed);
    out->slot_state->completed_subtasks.store(0, std::memory_order_relaxed);
    out->slot_state->next_block_idx.store(0, std::memory_order_relaxed);
    int16_t block_num = args.launch_spec.block_num();
    out->slot_state->total_required_subtasks =
        static_cast<int16_t>(block_num * __builtin_popcount(active_mask.core_mask()));
    out->slot_state->logical_block_num = block_num;
    out->slot_state->active_mask = active_mask;

    return true;
}

// =============================================================================
// Scope Management
// =============================================================================

void PTO2OrchestratorState::begin_scope(PTO2ScopeMode mode) {
    auto *orch = this;
    if (orch->fatal) {
        return;
    }
    assert(orch->scope_stack_top < static_cast<int32_t>(orch->scope_stack_capacity - 1) && "Scope stack overflow");
    if (mode == PTO2ScopeMode::AUTO && orch->in_manual_scope()) {
        report_fatal(PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "auto scope nested inside manual scope is not supported");
        return;
    }

    bool already_in_manual_scope = orch->in_manual_scope();
    ++orch->scope_stack_top;
    if (mode == PTO2ScopeMode::MANUAL && !already_in_manual_scope) {
        orch->manual_begin_depth = orch->scope_stack_top;
    }
#if SIMPLER_DFX
    // Gate via is_scope_stats_enabled() (weak-false in host builds) BEFORE the
    // collector call: when disabled we pay nothing. Sample the current ring's
    // task/heap start-end and tensormap usage at the scope boundary.
    if (is_scope_stats_enabled()) {
        auto &alloc = orch->task_allocator;
        scope_stats_begin(
            0, alloc.task_tail(), alloc.task_head(), alloc.heap_tail(), alloc.heap_top(), 1, orch->dep_pool.top,
            orch->tensor_map.current_used()
        );
    }
#endif
}

void PTO2OrchestratorState::end_scope() {
    auto *orch = this;
    if (orch->fatal) {
        return;
    }
    assert(orch->scope_stack_top >= 0 && "Scope stack underflow");

    // Snapshot the active graph-arena bump positions at the scope boundary.
    // Closing a scope never publishes or reclaims an arena.
#if SIMPLER_DFX
    // Gate via is_scope_stats_enabled() (see begin_scope). One collector call
    // emits the end-boundary record and tears down bookkeeping.
    if (is_scope_stats_enabled()) {
        auto &alloc = orch->task_allocator;
        scope_stats_end(
            0, alloc.task_tail(), alloc.task_head(), alloc.heap_tail(), alloc.heap_top(), 1, orch->dep_pool.top,
            orch->tensor_map.current_used()
        );
    }
#endif

    bool ending_manual_scope = orch->scope_stack_top == orch->manual_begin_depth;
    orch->scope_stack_top--;
    if (ending_manual_scope) {
        orch->manual_begin_depth = PTO2_MAX_SCOPE_DEPTH;
    }
}

// =============================================================================
// Task Submission
// =============================================================================

static bool ensure_tensormap_capacity(PTO2OrchestratorState *orch, int32_t needed) {
    PTO2TensorMap &tm = orch->tensor_map;
    if (tm.free_entries() >= needed) return true;
    orch->report_fatal(
        PTO2_ERROR_TENSORMAP_OVERFLOW, __FUNCTION__,
        "orchestration exceeds TensorMap pool (used=%d needed=%d capacity=%d)", tm.current_used(), needed,
        tm.pool_capacity()
    );
    return false;
}

// Shared body for submit_task / submit_dummy_task. Caller has already validated
// args.has_error, decided active_mask (empty for dummy), and resolved the per-slot
// kernel_ids (all INVALID_KERNEL_ID for dummy). Builds every dependency edge
// inline, registers outputs, initializes the slot, and seeds initial readiness.
static TaskOutputTensors submit_task_common(
    PTO2OrchestratorState *orch, const L0TaskArgs &args, ActiveMask active_mask, int32_t aic_kernel_id,
    int32_t aiv0_kernel_id, int32_t aiv1_kernel_id
) {
    CYCLE_COUNT_START();
    TaskOutputTensors result;
    PTO2OutputLayout layout = calculate_output_layout(args);
    PTO2PreparedTask prepared;
    if (!prepare_task(orch, args, layout.total_output_size, active_mask, &prepared)) {
        return result;
    }
    PTO2TaskId task_id = prepared.task_id;
    PTO2TaskSlotState &cur_slot_state = *prepared.slot_state;
    PTO2TaskDescriptor &task = *prepared.task;
    PTO2TaskPayload &payload = *prepared.payload;
    result.set_task_id(task_id);
    graph_record_begin_task(task_id);

    // dep_gen capture point: snapshot the orch submit_task inputs while the
    // tensormap is still in its pre-lookup state for this task. Replay reads
    // these records offline to reconstruct the complete dep graph — the sole
    // source of truth for fanout now that the swimlane hot path no longer
    // records it.
#if SIMPLER_DFX
    if (is_dep_gen_enabled()) {
        const void *tensor_ptrs[MAX_TENSOR_ARGS];
        // TensorArgType is `enum class : int32_t` (4 bytes); the on-disk record
        // packs arg_types as uint8_t[16] (5-value enum fits in a byte). Narrow
        // each tag here rather than letting the AICPU writer reinterpret a
        // 4×-wider array as bytes — that path silently lost two of every three
        // tags on little-endian and synthesized phantom self-edges in replay.
        uint8_t arg_types_u8[MAX_TENSOR_ARGS];
        // Clamp to MAX_TENSOR_ARGS even though the Arg builder caps adds at
        // MAX_TENSOR_ARGS: defensive against any future builder bypass /
        // shared-memory bit-flip that could otherwise overrun the two
        // MAX_TENSOR_ARGS-sized stack buffers above.
        const int tc_raw = args.tensor_count();
        const int tc = tc_raw > MAX_TENSOR_ARGS ? MAX_TENSOR_ARGS : tc_raw;
        for (int i = 0; i < tc; i++) {
            // OUTPUT slots carry create_info (not yet a Tensor); skip them —
            // they have no producer to look up and replay's per-tensor loop
            // also skips OUTPUT.
            tensor_ptrs[i] = (args.tag(i) == TensorArgType::OUTPUT) ? nullptr : &args.tensor(i).ref();
            arg_types_u8[i] = static_cast<uint8_t>(args.tag(i));
        }
        const int32_t kernel_ids_capture[3] = {aic_kernel_id, aiv0_kernel_id, aiv1_kernel_id};
        dep_gen_aicpu_record_submit(
            task_id.raw, orch->in_manual_scope(), args.allow_early_resolve(), tc, tensor_ptrs, arg_types_u8,
            static_cast<int>(args.explicit_dep_count()), reinterpret_cast<const uint64_t *>(args.explicit_deps_data()),
            args.launch_spec.block_num(), kernel_ids_capture
        );
    }
#endif

    uint32_t seen_epoch = next_fanin_seen_epoch(orch);

    CYCLE_COUNT_LAP(g_orch_alloc_cycle);

#if SIMPLER_DFX
    if (layout.total_output_size > 0) {
        orch->buffers_allocated++;
        orch->bytes_allocated += layout.total_output_size;
    }
#endif

    for (uint32_t i = 0; i < args.explicit_dep_count(); i++) {
        PTO2TaskId dep_task_id = args.explicit_dep(i);
        if (!dep_task_id.is_valid()) {
            orch->report_fatal(
                PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "Arg.set_dependencies(...) requires valid task ids"
            );
            return result;
        }
        if (dep_task_id.ring() != 0) {
            orch->report_fatal(
                PTO2_ERROR_INVALID_ARGS, __FUNCTION__,
                "replay_graph only accepts ring 0 task ids (raw=%llu ring=%u local=%u current=%u)",
                static_cast<unsigned long long>(dep_task_id.raw), static_cast<unsigned>(dep_task_id.ring()),
                dep_task_id.local(), task_id.local()
            );
            return result;
        }
        int32_t dep_local_task_id = static_cast<int32_t>(dep_task_id.local());
        if (dep_local_task_id < 0 || dep_local_task_id >= static_cast<int32_t>(task_id.local())) {
            orch->report_fatal(PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "dependency must name an earlier task");
            return result;
        }
        if (g_graph_recording.active && dep_local_task_id < g_graph_recording.start_local_task_id) {
            graph_mark_unsupported(PTO2_GRAPH_UNSUPPORTED_EXTERNAL_EXPLICIT_DEP, g_graph_recording.current_task_index);
        }
        int32_t dep_slot = orch->sm_header->get_slot_by_task_id(dep_local_task_id);
        PTO2TaskSlotState *producer = orch->sm_header->find_live_slot_state(dep_task_id);
        if (producer == nullptr) continue;
        if (!append_fanin_or_fail(orch, &cur_slot_state, dep_slot, producer, seen_epoch)) {
            return result;
        }
    }

    // === STEP 3: Lookup inputs (creator retention + tensormap modifier lookup) ===
    DepInputs dep_inputs{
        args.tensor_count(),       args.tensor_data(), args.tag_data(), static_cast<int32_t>(args.explicit_dep_count()),
        args.explicit_deps_data(),
    };

    auto runtime_emit = [&](PTO2TaskId producer_task_id) -> bool {
        int32_t producer_local = static_cast<int32_t>(producer_task_id.local());
        int32_t prod_slot = orch->sm_header->get_slot_by_task_id(producer_local);
        PTO2TaskSlotState *producer = orch->sm_header->find_live_slot_state(producer_task_id);
        if (producer == nullptr) return true;
        return append_fanin_or_fail(orch, &cur_slot_state, prod_slot, producer, seen_epoch);
    };

    if (!compute_task_fanin(dep_inputs, orch->tensor_map, orch->in_manual_scope(), runtime_emit)) {
        return result;
    }

    CYCLE_COUNT_LAP(g_orch_lookup_cycle);

    // === STEP 4: Register outputs/inouts in TensorMap (must be separate from lookup) ===
    // TensorMap entries do not retire during one orchestration invocation.
    int32_t tensormap_needed = count_registrable_outputs(dep_inputs, orch->in_manual_scope());
    if (tensormap_needed > 0 && !ensure_tensormap_capacity(orch, tensormap_needed)) {
        return result;
    }
    register_task_outputs(dep_inputs, task_id, orch->tensor_map, orch->in_manual_scope());

    CYCLE_COUNT_LAP(g_orch_insert_cycle);

    // === STEP 5: Batch-write to GM (single cache line burst) ===
    // Deferred from allocation phase to avoid scattered GM writes that get
    // evicted by TensorMap lookup/insert cache pressure.
    __builtin_prefetch(&task, 1, 1);
    task.task_id = task_id;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIC)] = aic_kernel_id;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV0)] = aiv0_kernel_id;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV1)] = aiv1_kernel_id;
    task.packed_buffer_base = prepared.alloc_result.packed_base;
    task.packed_buffer_end = prepared.alloc_result.packed_end;

    if (cur_slot_state.fanin_count > PTO2_DEP_DEGREE_WARN_THRESHOLD) {
        LOG_WARN(
            "dense dependency: task id=%u fanin>%d [orch submit]", task_id.local(), PTO2_DEP_DEGREE_WARN_THRESHOLD
        );
    }

    payload.init(args, result, prepared.alloc_result, layout);
    cur_slot_state.set_allow_early_resolve(args.allow_early_resolve());
    graph_record_task(prepared, args, /*completed_inline=*/false);
#if SIMPLER_DFX
    if (is_dump_args_enabled()) {
        if (args.scalar_count() > 0) {
            set_dump_args_task_scalar_dtypes(
                task_id.raw, static_cast<uint32_t>(args.scalar_count()), args.scalar_dtypes()
            );
        }
        // Selective vs full dump is latched at dump_args_init from DumpDataHeader
        // (host-decided before any dispatch), so it is race-free regardless of
        // submission order. Here we only record each marked task's arg mask and
        // metadata flags, which selective collection consults.
        if (args.dump_arg_mask() != 0) {
            set_dump_args_task_mask(task_id.raw, args.dump_arg_mask(), args.dump_arg_index_ambiguous_mask());
        }
    }
#endif

    CYCLE_COUNT_LAP(g_orch_args_cycle);

    CYCLE_COUNT_LAP(g_orch_fanin_cycle);
    CYCLE_COUNT_ORCH_SUBMIT_RECORD(task_id.raw);

#if SIMPLER_DFX
    orch->tasks_submitted++;
#if SIMPLER_ORCH_PROFILING
    g_orch_submit_count++;
#endif
    g_orch_submit_idx++;
#endif
    return result;
}

TaskOutputTensors PTO2OrchestratorState::submit_task(const MixedKernels &mixed_kernels, const L0TaskArgs &args) {
    auto *orch = this;

    // Orchestration API should short-circuit after fatal, but keep this entry
    // robust as a no-op in case a caller reaches it directly.
    if (orch->fatal) {
        return TaskOutputTensors{};
    }

    // Validate Arg construction (errors recorded by add_input/add_output/etc.)
    if (args.has_error) {
        LOG_ERROR("========================================");
        LOG_ERROR("FATAL: Invalid Arg Detected!");
        LOG_ERROR("========================================");
        LOG_ERROR("Error: %s", args.error_msg ? args.error_msg : "(unknown)");
        LOG_ERROR("  tensor_count: %d, scalar_count: %d", args.tensor_count(), args.scalar_count());
        LOG_ERROR("This is a bug in the orchestration code.");
        LOG_ERROR("========================================");
        orch_mark_fatal(orch, PTO2_ERROR_INVALID_ARGS);
        return TaskOutputTensors{};
    }
    always_assert(orch->scheduler != nullptr);
    // === Validate submit inputs ===
    ActiveMask active_mask = mixed_kernels.to_active_mask();
    always_assert(static_cast<bool>(active_mask) && "MixedKernels must have at least one active slot");

    int16_t block_num = args.launch_spec.block_num();
    always_assert(block_num >= 1 && "block_num must be >= 1");

    // Normalize single-AIV tasks: if only aiv1 is set (no aic, no aiv0), move
    // it to the aiv0 slot.  This guarantees the dispatch path can always use
    // PTO2SubtaskSlot::AIV0 for single-AIV shapes without inspecting active_mask.
    // Mixed tasks (AIC+AIV) keep their original AIV identity so the correct
    // hardware channel (AIV0→AIC vs AIV1→AIC) is used at dispatch time.
    MixedKernels normalized = mixed_kernels;
    bool has_aic = active_mask.has_mask(PTO2_SUBTASK_MASK_AIC);
    bool has_aiv0 = active_mask.has_mask(PTO2_SUBTASK_MASK_AIV0);
    bool has_aiv1 = active_mask.has_mask(PTO2_SUBTASK_MASK_AIV1);
    if (!has_aic && has_aiv1 && !has_aiv0) {
        normalized.aiv0_kernel_id = normalized.aiv1_kernel_id;
        normalized.aiv1_kernel_id = INVALID_KERNEL_ID;
        active_mask = normalized.to_active_mask();
    }

    // Encode require_sync_start into active_mask bit 3 (only meaningful for tasks with block_num > 1)
    if (block_num > 1 && args.launch_spec.require_sync_start()) {
        // Deadlock check: block_num >= total available slots of the required type.
        // For MIX/AIC: limit is total_cluster_count (one AIC per cluster).
        // For AIV:     limit is total_aiv_count.
        PTO2ResourceShape shape = active_mask.to_shape();
        int32_t limit = (shape == PTO2ResourceShape::AIV) ? orch->total_aiv_count : orch->total_cluster_count;
        if (limit > 0 && block_num > limit) {
            report_fatal(
                PTO2_ERROR_REQUIRE_SYNC_START_INVALID, __FUNCTION__,
                "require_sync_start block_num=%d > limit=%d (deadlock guaranteed)", block_num, limit
            );
            return TaskOutputTensors{};
        }
        active_mask.set_sync_start();
    }

    return submit_task_common(
        orch, args, active_mask, normalized.aic_kernel_id, normalized.aiv0_kernel_id, normalized.aiv1_kernel_id
    );
}

// Submit a dependency-only task: full dependency graph participation
// (tensormap lookup/insert, explicit_deps, manual_dep, manual_scope) but no
// AICore dispatch. Empty active_mask routes the slot to the DUMMY ready
// bucket; dispatch loop short-circuits to completion. Accepts the same Arg
// shape as submit_task; scalars are permitted but never consumed.
TaskOutputTensors PTO2OrchestratorState::submit_dummy_task(const L0TaskArgs &args) {
    auto *orch = this;

    if (orch->fatal) {
        return TaskOutputTensors{};
    }

    if (args.has_error) {
        LOG_ERROR("========================================");
        LOG_ERROR("FATAL: Invalid Arg in submit_dummy_task!");
        LOG_ERROR("========================================");
        LOG_ERROR("Error: %s", args.error_msg ? args.error_msg : "(unknown)");
        LOG_ERROR("  tensor_count: %d, scalar_count: %d", args.tensor_count(), args.scalar_count());
        LOG_ERROR("========================================");
        orch_mark_fatal(orch, PTO2_ERROR_INVALID_ARGS);
        return TaskOutputTensors{};
    }
    always_assert(orch->scheduler != nullptr);

    return submit_task_common(orch, args, ActiveMask{}, INVALID_KERNEL_ID, INVALID_KERNEL_ID, INVALID_KERNEL_ID);
}

TaskOutputTensors PTO2OrchestratorState::alloc_tensors(const L0TaskArgs &args) {
    auto *orch = this;
    // Orchestration API should short-circuit after fatal, but keep this entry
    // robust as a no-op in case a caller reaches it directly.
    if (orch->fatal) {
        return TaskOutputTensors{};
    }

    if (args.tensor_count() <= 0) {
        report_fatal(PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "alloc_tensors requires at least one TensorCreateInfo");
        return TaskOutputTensors{};
    }
    if (args.scalar_count() != 0) {
        report_fatal(PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "alloc_tensors only accepts output TensorCreateInfo args");
        return TaskOutputTensors{};
    }
    for (int32_t i = 0; i < args.tensor_count(); i++) {
        if (args.tag(i) != TensorArgType::OUTPUT) {
            report_fatal(
                PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "alloc_tensors only accepts output TensorCreateInfo args"
            );
            return TaskOutputTensors{};
        }
    }

    CYCLE_COUNT_START();

    if (args.has_error) {
        report_fatal(
            PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "%s",
            args.error_msg ? args.error_msg : "alloc_tensors failed to construct output-only Arg"
        );
        return TaskOutputTensors{};
    }

    PTO2OutputLayout layout = calculate_output_layout(args);
    PTO2PreparedTask prepared;
    if (!prepare_task(orch, args, layout.total_output_size, ActiveMask{}, &prepared)) {
        return TaskOutputTensors{};
    }

    PTO2TaskDescriptor &task = *prepared.task;
    PTO2TaskPayload &payload = *prepared.payload;
    graph_record_begin_task(prepared.task_id);

    CYCLE_COUNT_LAP(g_orch_alloc_cycle);

#if SIMPLER_DFX
    if (layout.total_output_size > 0) {
        orch->buffers_allocated++;
        orch->bytes_allocated += layout.total_output_size;
    }
#endif

    task.task_id = prepared.task_id;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIC)] = INVALID_KERNEL_ID;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV0)] = INVALID_KERNEL_ID;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV1)] = INVALID_KERNEL_ID;
    task.packed_buffer_base = prepared.alloc_result.packed_base;
    task.packed_buffer_end = prepared.alloc_result.packed_end;

    TaskOutputTensors outputs;
    outputs.set_task_id(prepared.task_id);
    payload.init(args, outputs, prepared.alloc_result, layout);
    CYCLE_COUNT_LAP(g_orch_args_cycle);

    if (prepared.slot_state != nullptr) {
        // Hidden alloc tasks complete inline in the orchestrator before any
        // consumer can exist, so they have no fanout to notify and no worker
        // subtasks to retire. Consumers see COMPLETED during inline wiring and
        // omit the dependency edge entirely.
        prepared.slot_state->mark_completed();
    }
    graph_record_task(prepared, args, /*completed_inline=*/true);
    orch->inline_completed_tasks++;

    CYCLE_COUNT_LAP(g_orch_fanin_cycle);
    CYCLE_COUNT_ORCH_SUBMIT_RECORD(prepared.task_id.raw);

#if SIMPLER_DFX
    orch->tasks_submitted++;
#if SIMPLER_ORCH_PROFILING
    g_orch_submit_count++;
#endif
    g_orch_submit_idx++;
#endif

    return outputs;
}

PTO2GraphScopeResult
PTO2OrchestratorState::graph_begin(uint64_t graph_key, const PTO2GraphBindings &bindings, uint64_t callable_hash) {
    PTO2GraphScopeResult result;
    auto *orch = this;
    if (orch->fatal || bindings.overflow) return result;
    if (g_graph_recording.active) {
        graph_mark_unsupported(PTO2_GRAPH_UNSUPPORTED_NESTED_SCOPE);
        return result;
    }

    uint64_t full_key = graph_full_key(callable_hash, graph_key);
    PTO2GraphTemplate *cached = find_graph_template(full_key);
    if (cached != nullptr) {
#if SIMPLER_DFX
        bool record_replay_orch = orch->l2_swimlane_level >= L2SwimlaneLevel::ORCH_PHASES;
        uint64_t replay_start_ts = record_replay_orch ? get_sys_cnt_aicpu() : 0;
#endif
        uint64_t replay_orch_task_id = PTO2TaskId::invalid().raw;
        if (graph_replay_template(orch, *cached, bindings, &replay_orch_task_id)) {
#if SIMPLER_DFX
            if (record_replay_orch) {
                l2_swimlane_aicpu_record_orch_phase(
                    replay_start_ts, get_sys_cnt_aicpu(), replay_orch_task_id, g_orch_submit_idx++
                );
            }
#endif
            LOG_INFO_V0(
                "[GraphCache] replay key=0x%llx tasks=%d", static_cast<unsigned long long>(full_key), cached->task_count
            );
            result.execute_block = false;
            result.recording = false;
            return result;
        }
        if (orch->fatal) return result;
    }

    reset_graph_recording();
    g_graph_recording.active = true;
    g_graph_recording.full_key = full_key;
    g_graph_recording.start_local_task_id = orch->task_allocator.active_count();
    g_graph_recording.bindings = bindings;
    reset_graph_template_header(&g_graph_recording.temp);
    g_graph_recording.temp.full_key = full_key;
    result.execute_block = true;
    result.recording = true;
    return result;
}

void PTO2OrchestratorState::graph_end(PTO2GraphCacheStats *stats) {
    if (!g_graph_recording.active) return;
    if (g_graph_recording.unsupported || g_graph_recording.temp.task_count <= 0) {
        if (stats != nullptr && g_graph_recording.unsupported) stats->overflow++;
        if (g_graph_recording.unsupported) {
            LOG_WARN(
                "graph cache record skipped: reason=%d task_index=%d tensor_index=%d recorded_tasks=%d",
                g_graph_recording.unsupported_reason, g_graph_recording.unsupported_task_index,
                g_graph_recording.unsupported_tensor_index, g_graph_recording.temp.task_count
            );
        }
        reset_graph_recording();
        return;
    }
    g_graph_recording.temp.in_use = true;
    store_graph_template(g_graph_recording.temp);
    if (stats != nullptr) stats->recorded++;
    LOG_INFO_V0(
        "[GraphCache] record key=0x%llx tasks=%d", static_cast<unsigned long long>(g_graph_recording.full_key),
        g_graph_recording.temp.task_count
    );
    reset_graph_recording();
}

// =============================================================================
// Flow Control
// =============================================================================

void PTO2OrchestratorState::mark_done() {
    auto *orch = this;
    int32_t total_tasks = orch->task_allocator.active_count();
    if (total_tasks > 0) {
        LOG_INFO_V0("=== [Orchestrator] total_tasks=%d ===", total_tasks);
    }
    if (orch->dep_pool.used() > 0) {
        LOG_INFO_V0(
            "=== [DepPool] used=%d high_water=%d capacity=%d ===", orch->dep_pool.used(), orch->dep_pool.high_water,
            orch->dep_pool.capacity
        );
    }
    orch->sm_header->orchestrator_done.store(1, std::memory_order_release);
    orch->scope_stack_top = -1;
    orch->manual_begin_depth = PTO2_MAX_SCOPE_DEPTH;
#if !SIMPLER_ORCH_PROFILING && SIMPLER_DFX
    g_orch_submit_idx = 0;
#endif
}

#if SIMPLER_ORCH_PROFILING
PTO2OrchProfilingData orchestrator_get_profiling() {
    PTO2OrchProfilingData d;
    d.alloc_cycle = g_orch_alloc_cycle;
    d.args_cycle = g_orch_args_cycle;
    d.lookup_cycle = g_orch_lookup_cycle;
    d.insert_cycle = g_orch_insert_cycle;
    d.fanin_cycle = g_orch_fanin_cycle;
    d.submit_count = g_orch_submit_count;

    // Reset
    g_orch_alloc_cycle = g_orch_args_cycle = 0;
    g_orch_lookup_cycle = g_orch_insert_cycle = 0;
    g_orch_fanin_cycle = 0;
    g_orch_submit_count = 0;
    g_orch_submit_idx = 0;
    return d;
}
#endif
