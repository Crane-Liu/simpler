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
 * Host/AICPU shared runtime-arena layout, init_data and wire implementations.
 *
 * Lives under runtime/shared/ so it is included in both the host_runtime.so
 * build (host pre-populates the prebuilt arena image) and the aicpu_runtime
 * build (AICPU runs wire_arena_pointers + destroy after attach). The
 * device-only parts of pto_runtime2.cpp / pto_orchestrator.cpp / pto_scheduler.cpp
 * (ops table, scope/submit/dispatch business logic, profiling) stay in their
 * original files and the aicpu build only.
 */

#include <stdlib.h>
#include <string.h>

#include <new>

#include "pto_orchestrator.h"
#include "pto_runtime2.h"
#include "pto_ring_buffer.h"
#include "pto_shared_memory.h"
#include "pto_tensormap.h"
#include "scheduler/pto_scheduler.h"

// =============================================================================
// Ready queue
// =============================================================================

size_t ready_queue_reserve_layout(DeviceArena &arena, uint64_t capacity) {
    // Align the slots[] base to a full cache line so MPMC CAS traffic on the
    // first slot cannot false-share with whatever region sits in front of us
    // (e.g. orchestrator tensormap heads written by the orch thread).
    return arena.reserve(capacity * sizeof(PTO2ReadyQueueSlot), PTO2_ALIGN_SIZE);
}

bool ready_queue_init_data_from_layout(PTO2ReadyQueue *queue, DeviceArena &arena, size_t slots_off, uint64_t capacity) {
    // Address the slots region for data writes without storing the pointer in
    // queue->slots — that field is set by ready_queue_wire_arena_pointers.
    auto *slots_arena = static_cast<PTO2ReadyQueueSlot *>(arena.region_ptr(slots_off));
    queue->capacity = capacity;
    queue->mask = capacity - 1;
    queue->enqueue_pos.store(0, std::memory_order_relaxed);
    queue->dequeue_pos.store(0, std::memory_order_relaxed);

    for (uint64_t i = 0; i < capacity; i++) {
        slots_arena[i].sequence.store((int64_t)i, std::memory_order_relaxed);
        slots_arena[i].slot_state = nullptr;
    }

    return true;
}

void ready_queue_wire_arena_pointers(PTO2ReadyQueue *queue, DeviceArena &arena, size_t slots_off) {
    queue->slots = static_cast<PTO2ReadyQueueSlot *>(arena.region_ptr(slots_off));
}

void ready_queue_destroy(PTO2ReadyQueue *queue) {
    // Arena owns the slots[] buffer; just forget the pointer.
    queue->slots = nullptr;
}

// =============================================================================
// Scheduler
// =============================================================================

PTO2SchedulerLayout PTO2SchedulerState::reserve_layout(DeviceArena &arena) {
    PTO2SchedulerLayout layout{};
    layout.ready_queue_capacity = PTO2_READY_QUEUE_SIZE;

    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        layout.off_ready_queue_slots[i] = ready_queue_reserve_layout(arena, PTO2_READY_QUEUE_SIZE);
    }
    layout.off_dummy_ready_queue_slots = ready_queue_reserve_layout(arena, PTO2_READY_QUEUE_SIZE);
    layout.off_early_dispatch_queue_slots = ready_queue_reserve_layout(arena, PTO2_EARLY_DISPATCH_QUEUE_SIZE);
    // dep_pool entries + wiring SPSC buffer are reserved by the orchestrator now
    // (replay_graph stage 1).
    return layout;
}

bool PTO2SchedulerState::init_data_from_layout(
    const PTO2SchedulerLayout &layout, DeviceArena &arena, void *sm_dev_base
) {
    PTO2SchedulerState *sched = this;
    // sm_header is the SM device base. Pure pointer math, no SM load — per-slot
    // SM-side init (bind_ring + dynamic-field reset + fanin_count/active_mask
    // zero) lives in PTO2SharedMemoryHandle::init_header so the AICPU performs
    // it during SM reset; host prebuilt-arena init skips SM access here.
    sched->sm_header = reinterpret_cast<PTO2SharedMemoryHeader *>(sm_dev_base);
#if PTO2_SCHED_PROFILING
    sched->tasks_completed.store(0, std::memory_order_relaxed);
#endif

    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        if (!ready_queue_init_data_from_layout(
                &sched->ready_queues[i], arena, layout.off_ready_queue_slots[i], layout.ready_queue_capacity
            )) {
            return false;
        }
    }
    if (!ready_queue_init_data_from_layout(
            &sched->dummy_ready_queue, arena, layout.off_dummy_ready_queue_slots, layout.ready_queue_capacity
        )) {
        return false;
    }
    if (!ready_queue_init_data_from_layout(
            &sched->early_dispatch_queue, arena, layout.off_early_dispatch_queue_slots, PTO2_EARLY_DISPATCH_QUEUE_SIZE
        )) {
        return false;
    }

    return true;
}

void PTO2SchedulerState::wire_arena_pointers(const PTO2SchedulerLayout &layout, DeviceArena &arena) {
    PTO2SchedulerState *sched = this;
    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        ready_queue_wire_arena_pointers(&sched->ready_queues[i], arena, layout.off_ready_queue_slots[i]);
    }
    ready_queue_wire_arena_pointers(&sched->dummy_ready_queue, arena, layout.off_dummy_ready_queue_slots);
    ready_queue_wire_arena_pointers(&sched->early_dispatch_queue, arena, layout.off_early_dispatch_queue_slots);
}

void PTO2SchedulerState::destroy() {
    PTO2SchedulerState *sched = this;
    sched->sm_header = nullptr;
    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        ready_queue_destroy(&sched->ready_queues[i]);
    }
    ready_queue_destroy(&sched->dummy_ready_queue);
    ready_queue_destroy(&sched->early_dispatch_queue);
}

// =============================================================================
// Orchestrator
// =============================================================================

PTO2OrchestratorLayout
PTO2OrchestratorState::reserve_layout(DeviceArena &arena, int32_t task_window_size, int32_t dep_pool_capacity) {
    PTO2OrchestratorLayout layout{};
    layout.scope_stack_capacity = PTO2_MAX_SCOPE_DEPTH;
    // Initial-ready upper bound: every task could be initially ready. Bound it by
    // the whole-graph task upper bound (PTO2_SCOPE_TASKS_CAP = window size).
    layout.initial_ready_cap = PTO2_SCOPE_TASKS_CAP;
    layout.dep_pool_capacity = dep_pool_capacity;

    // Fanout dep_pool entries, moved off the scheduler layout (replay_graph
    // stage 1). Cache-line aligned base so the single-writer orch wiring does
    // not false-share with neighboring regions.
    layout.off_dep_pool_entries =
        arena.reserve(static_cast<size_t>(dep_pool_capacity) * sizeof(PTO2DepListEntry), PTO2_ALIGN_SIZE);

    always_assert(task_window_size > 0 && (task_window_size & (task_window_size - 1)) == 0);
    const size_t seen_epoch_bytes =
        PTO2_ALIGN_UP(static_cast<size_t>(task_window_size) * sizeof(uint32_t), PTO2_ALIGN_SIZE);
    layout.off_fanin_seen_epoch = arena.reserve(seen_epoch_bytes, PTO2_ALIGN_SIZE);

    layout.off_initial_ready = arena.reserve(
        static_cast<size_t>(layout.initial_ready_cap) * sizeof(PTO2TaskSlotState *), alignof(PTO2TaskSlotState *)
    );
    layout.tensor_map = PTO2TensorMap::reserve_layout_default(arena);
    return layout;
}

bool PTO2OrchestratorState::init_data_from_layout(
    const PTO2OrchestratorLayout &layout, DeviceArena &arena, void *sm_dev_base, void *gm_heap, uint64_t heap_size,
    uint64_t task_window_size
) {
    auto *orch = this;
    // Default-construct in place so member default initializers (e.g.
    // fanin_seen_current_epoch{1}) are applied.
    new (orch) PTO2OrchestratorState{};

    orch->sm_header = reinterpret_cast<PTO2SharedMemoryHeader *>(sm_dev_base);
    orch->gm_heap_base = gm_heap;
    orch->gm_heap_size = heap_size;
    orch->fatal = false;

    auto *orch_err = pto2_sm_layout::orch_error_code_addr(sm_dev_base);
    auto *task_count_dev = pto2_sm_layout::task_count_addr(sm_dev_base);

    int32_t *task_slot_map_dev = pto2_sm_layout::task_slot_map_addr(sm_dev_base, task_window_size);
    orch->task_allocator.init(
        static_cast<int32_t>(task_window_size), task_count_dev, task_slot_map_dev, gm_heap, heap_size, orch_err
    );

    // Fanout dep_pool, moved off the scheduler (replay_graph stage 1).
    auto *dep_entries = static_cast<PTO2DepListEntry *>(arena.region_ptr(layout.off_dep_pool_entries));
    memset(dep_entries, 0, static_cast<size_t>(layout.dep_pool_capacity) * sizeof(PTO2DepListEntry));
    orch->dep_pool.init(dep_entries, layout.dep_pool_capacity, orch_err);

    const size_t seen_epoch_bytes =
        PTO2_ALIGN_UP(static_cast<size_t>(task_window_size) * sizeof(uint32_t), PTO2_ALIGN_SIZE);
    auto *seen_epoch = static_cast<uint32_t *>(arena.region_ptr(layout.off_fanin_seen_epoch));
    memset(seen_epoch, 0, seen_epoch_bytes);
    orch->fanin_seen_epoch = seen_epoch;

    if (!orch->tensor_map.init_data_from_layout(layout.tensor_map, arena)) {
        return false;
    }

    orch->scope_stack_top = -1;
    orch->scope_stack_capacity = layout.scope_stack_capacity;
    orch->manual_begin_depth = PTO2_MAX_SCOPE_DEPTH;

    orch->initial_ready_count = 0;
    orch->initial_ready_capacity = layout.initial_ready_cap;

    return true;
}

void PTO2OrchestratorState::wire_arena_pointers(
    const PTO2OrchestratorLayout &layout, DeviceArena &arena, PTO2SchedulerState *scheduler_arg
) {
    auto *orch = this;
    orch->dep_pool.base = static_cast<PTO2DepListEntry *>(arena.region_ptr(layout.off_dep_pool_entries));
    orch->fanin_seen_epoch = static_cast<uint32_t *>(arena.region_ptr(layout.off_fanin_seen_epoch));
    orch->initial_ready = static_cast<PTO2TaskSlotState **>(arena.region_ptr(layout.off_initial_ready));
    orch->tensor_map.wire_arena_pointers(layout.tensor_map, arena);
    orch->scheduler = scheduler_arg;
}

void PTO2OrchestratorState::destroy() {
    auto *orch = this;
    orch->tensor_map.destroy();
    orch->dep_pool.base = nullptr;
    orch->fanin_seen_epoch = nullptr;
    orch->initial_ready = nullptr;
}

void PTO2OrchestratorState::set_scheduler(PTO2SchedulerState *scheduler) { this->scheduler = scheduler; }

// =============================================================================
// Top-level runtime arena
// =============================================================================

PTO2RuntimeArenaLayout
runtime_reserve_layout(DeviceArena &arena, uint64_t task_window_size, int32_t dep_pool_capacity) {
    return runtime_reserve_layout(arena, task_window_size, /*heap_size=*/0, dep_pool_capacity);
}

PTO2RuntimeArenaLayout
runtime_reserve_layout(DeviceArena &arena, uint64_t task_window_size, uint64_t heap_size, int32_t dep_pool_capacity) {
    PTO2RuntimeArenaLayout layout{};

    layout.task_window_size = task_window_size;
    layout.heap_size = heap_size;
    layout.dep_pool_capacity = dep_pool_capacity;

    layout.off_sm_handle = arena.reserve(sizeof(PTO2SharedMemoryHandle), alignof(PTO2SharedMemoryHandle));
    layout.orch =
        PTO2OrchestratorState::reserve_layout(arena, static_cast<int32_t>(task_window_size), dep_pool_capacity);
    layout.sched = PTO2SchedulerState::reserve_layout(arena);
    layout.off_runtime = arena.reserve(sizeof(PTO2Runtime), PTO2_ALIGN_SIZE);
    layout.off_mailbox = arena.reserve(sizeof(AICoreCompletionMailbox), alignof(AICoreCompletionMailbox));

    layout.arena_size = arena.total_size();
    return layout;
}

PTO2Runtime *runtime_init_data_from_layout(
    DeviceArena &arena, const PTO2RuntimeArenaLayout &layout, PTO2RuntimeMode mode, void *sm_dev_base,
    uint64_t /*sm_size*/, void *gm_heap_dev_base, uint64_t heap_size
) {
    PTO2Runtime *rt = static_cast<PTO2Runtime *>(arena.region_ptr(layout.off_runtime));
    memset(rt, 0, sizeof(*rt));

    auto *sm_wrap = static_cast<PTO2SharedMemoryHandle *>(arena.region_ptr(layout.off_sm_handle));
    memset(sm_wrap, 0, sizeof(*sm_wrap));

    // rt->ops is filled by the AICPU at boot.
    rt->mode = mode;
    rt->gm_heap = gm_heap_dev_base;
    rt->gm_heap_size = heap_size;
    rt->gm_heap_owned = false;
    rt->total_cycles = 0;
    rt->graph_pipeline.enabled.store(0, std::memory_order_relaxed);
    rt->graph_pipeline.all_done.store(0, std::memory_order_relaxed);
    rt->graph_pipeline.published_task_count.store(0, std::memory_order_relaxed);
    rt->graph_pipeline.active_buffer = 0;
    rt->graph_pipeline.graph_count = 0;
    rt->graph_pipeline.published_initial_ready_count = 0;
    rt->graph_pipeline.current_graph_epoch = 0;
    for (int32_t i = 0; i < PTO2_REPLAY_GRAPH_BUFFER_COUNT; i++) {
        PTO2ReplayGraphBufferControl &buf = rt->graph_pipeline.buffers[i];
        buf.state.store(i == 0 ? PTO2_GRAPH_BUFFER_BUILDING : PTO2_GRAPH_BUFFER_FREE, std::memory_order_relaxed);
        buf.exec_done.store(i == 0 ? 0 : 1, std::memory_order_relaxed);
        buf.dep_closed.store(i == 0 ? 0 : 1, std::memory_order_relaxed);
        buf.completed_count.store(0, std::memory_order_relaxed);
        buf.buffer_id = i;
        buf.graph_epoch = i == 0 ? 0 : static_cast<uint64_t>(i);
        buf.task_begin = 0;
        buf.task_count = 0;
    }

    if (!rt->orchestrator.init_data_from_layout(
            layout.orch, arena, sm_dev_base, gm_heap_dev_base, heap_size, layout.task_window_size
        )) {
        return nullptr;
    }
    if (!rt->scheduler.init_data_from_layout(layout.sched, arena, sm_dev_base)) {
        return nullptr;
    }

    auto *mailbox = static_cast<AICoreCompletionMailbox *>(arena.region_ptr(layout.off_mailbox));
    memset(mailbox, 0, sizeof(*mailbox));

    return rt;
}

void runtime_wire_arena_pointers(DeviceArena &arena, const PTO2RuntimeArenaLayout &layout, PTO2Runtime *rt) {
    rt->sm_handle = static_cast<PTO2SharedMemoryHandle *>(arena.region_ptr(layout.off_sm_handle));
    rt->aicore_mailbox = static_cast<AICoreCompletionMailbox *>(arena.region_ptr(layout.off_mailbox));
    rt->orchestrator.wire_arena_pointers(layout.orch, arena, &rt->scheduler);
    rt->scheduler.wire_arena_pointers(layout.sched, arena);
}

void runtime_destroy(PTO2Runtime *rt, DeviceArena & /*arena*/) {
    // Arena buffer is pooled across runs by DeviceRunner — never freed here.
    if (!rt) return;
    rt->scheduler.destroy();
    rt->orchestrator.destroy();
    rt->aicore_mailbox = nullptr;
    rt->sm_handle = nullptr;
}
