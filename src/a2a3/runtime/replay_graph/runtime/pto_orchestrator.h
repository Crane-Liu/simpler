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
 * PTO Runtime2 - Orchestrator Interface
 *
 * The Orchestrator is responsible for:
 * 1. Executing the orchestration function (Turing-complete control flow)
 * 2. Allocating intermediate buffers from the heap
 * 3. Submitting tasks via async InCore function calls
 * 4. Building the dependency graph using TensorMap
 * 5. Managing buffer scopes for lifecycle control
 *
 * The Orchestrator can run on either:
 * - Host CPU (lower latency for complex control, easier debugging)
 * - Device AI_CPU (lower latency for task submission)
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#ifndef PTO_ORCHESTRATOR_H
#define PTO_ORCHESTRATOR_H

#include "common/l2_swimlane_profiling.h"
#include "utils/device_arena.h"
#include "pto_ring_buffer.h"
#include "pto_runtime2_types.h"
#include "pto_submit_types.h"
#include "scheduler/pto_scheduler.h"
#include "pto_shared_memory.h"
#include "pto_tensormap.h"
#include "pto_types.h"

/**
 * Layout descriptor produced by PTO2OrchestratorState::reserve_layout(). Holds
 * arena offsets for every sub-region the orchestrator owns (per-ring fanin
 * pools, scope arrays, plus the nested PTO2TensorMap layout).
 */
struct PTO2OrchestratorLayout {
    size_t off_fanin_seen_epoch;
    // Orchestrator-owned wiring sub-regions: the fanout dep_pool entries and the
    // initial-ready handoff array. submit_task builds the fanout lists in
    // dep_pool in-line and seeds initial_ready; the scheduler consumes both.
    size_t off_dep_pool_entries;
    size_t off_initial_ready;
    PTO2TensorMapLayout tensor_map;
    int32_t dep_pool_capacity;
    int32_t initial_ready_cap;
    uint64_t scope_stack_capacity;
};

// =============================================================================
// Orchestrator State
// =============================================================================

/**
 * Orchestrator state structure (private to Orchestrator)
 *
 * Contains all state needed for task graph construction and buffer management.
 */
struct PTO2OrchestratorState {
    // === SHARED MEMORY ACCESS ===
    PTO2SharedMemoryHeader *sm_header;

    // === ALLOCATOR RESOURCES (single-shot bump allocators) ===
    PTO2TaskAllocator task_allocator;
    // Fanout dependency-list pool. Owned by the orchestrator: submit_task builds
    // the fanout linked list here in-line during the orch phase; the scheduler
    // only reads it (read-only traversal in on_task_complete).
    PTO2DepListPool dep_pool;
    uint32_t *fanin_seen_epoch;
    uint32_t fanin_seen_current_epoch{1};

    // === TENSOR MAP (Private) ===
    PTO2TensorMap tensor_map;  // Producer lookup

    // === SCOPE STACK (Private) ===
    // Depth-only bookkeeping: the single-shot replay model tracks scope nesting
    // for manual-scope semantics, but no longer collects per-scope task lists
    // (slot reclaim was dropped).
    int32_t scope_stack_top;        // Current top of stack (-1 = no scope open)
    uint64_t scope_stack_capacity;  // Max nesting depth (PTO2_MAX_SCOPE_DEPTH)
    int32_t manual_begin_depth{PTO2_MAX_SCOPE_DEPTH};

    // === SCHEDULER REFERENCE ===
    // Note: In simulated mode, orchestrator and scheduler share address space
    // In real mode, they communicate via shared memory only
    PTO2SchedulerState *scheduler;  // For simulated mode only

    // Initial-ready handoff (orchestrator → scheduler, single direction).
    // submit_task appends every task whose fanin is already satisfied when it
    // returns; the scheduler seeds these into its ready_queues before dispatch.
    // A construction-time pure-function product, reusable across future
    // multi-pass scheduling.
    PTO2TaskSlotState **initial_ready;
    int32_t initial_ready_count;
    int32_t initial_ready_capacity;

    // Total core counts set once at executor init; used for submit-time deadlock detection.
    int32_t total_cluster_count{0};  // AIC cores = MIX clusters
    int32_t total_aiv_count{0};      // AIV cores (= 2 × clusters on standard hardware)
#if PTO2_PROFILING
    // L2 swimlane_level copied from get_l2_swimlane_level().
    L2SwimlaneLevel l2_swimlane_level{L2SwimlaneLevel::DISABLED};
#endif

    // === GM HEAP (for output buffers) ===
    void *gm_heap_base;     // Base address of GM heap
    uint64_t gm_heap_size;  // Total size of GM heap (all rings)

    // === FATAL ERROR ===
    // Fatal error flag (single-thread access by orchestrator, no atomic needed)
    // Cross-thread notification uses shared memory orch_error_code (atomic)
    bool fatal;

    // Hidden alloc tasks complete synchronously inside the orchestrator and
    // therefore bypass the executor's normal worker-completion counter path.
    // The executor adds this count into its completed_tasks_ progress counter
    // after orchestration finishes so shutdown/profiling totals remain closed.
    int64_t inline_completed_tasks{0};

    // === STATISTICS ===
#if PTO2_PROFILING
    int64_t tasks_submitted;
    int64_t buffers_allocated;
    int64_t bytes_allocated;
#endif

    bool in_manual_scope() const { return scope_stack_top >= manual_begin_depth; }

    // Append a task whose fanin is already satisfied when submit_task returns to
    // the initial-ready handoff array (seeded into the scheduler's ready_queues
    // before dispatch).
    void push_initial_ready(PTO2TaskSlotState *ws) {
        if (initial_ready_count >= initial_ready_capacity) {
            report_fatal(
                PTO2_ERROR_DEP_POOL_OVERFLOW, __FUNCTION__, "initial_ready overflow (count=%d cap=%d)",
                initial_ready_count, initial_ready_capacity
            );
            return;
        }
        initial_ready[initial_ready_count++] = ws;
    }

    // === Cold-path API (defined in pto_orchestrator.cpp) ===

    // Phase 1: declare every sub-region (fanin pool, scope arrays,
    // tensor_map sub-layout) on the supplied arena. task_window_size feeds
    // the nested tensor_map layout. Returned layout is consumed by
    // init_from_layout.
    static PTO2OrchestratorLayout
    reserve_layout(DeviceArena &arena, int32_t task_window_size, int32_t dep_pool_capacity = PTO2_DEP_LIST_POOL_SIZE);

    // Phase 3a: write everything *except* arena-internal pointer fields.
    // sm_dev_base is the SM device address (only stored, never dereferenced);
    // task_window_size feeds the per-ring SM address arithmetic. Safe to call
    // on a host arena that holds the prebuilt image.
    bool init_data_from_layout(
        const PTO2OrchestratorLayout &layout, DeviceArena &arena, void *sm_dev_base, void *gm_heap, uint64_t heap_size,
        uint64_t task_window_size
    );

    // Phase 3b: write the arena-internal pointer fields (dep_pool.base,
    // initial_ready, tensor_map.{buckets,entry_pool,free_entry_list},
    // scheduler reference).
    // Idempotent — host runs once on the image, AICPU runs once after attach.
    void wire_arena_pointers(const PTO2OrchestratorLayout &layout, DeviceArena &arena, PTO2SchedulerState *scheduler);

    // Forget pointers; arena owns the backing buffers.
    void destroy();
    void set_scheduler(PTO2SchedulerState *scheduler);
    void report_fatal(int32_t error_code, const char *func, const char *fmt, ...);
    void begin_scope(PTO2ScopeMode mode = PTO2ScopeMode::AUTO);
    void end_scope();
    TaskOutputTensors submit_task(const MixedKernels &mixed_kernels, const L0TaskArgs &args);
    TaskOutputTensors submit_dummy_task(const L0TaskArgs &args);
    TaskOutputTensors alloc_tensors(const L0TaskArgs &args);
    void mark_done();
};

// =============================================================================
// Orchestrator Profiling Data
// =============================================================================

#if PTO2_ORCH_PROFILING
struct PTO2OrchProfilingData {
    uint64_t alloc_cycle;  // Combined task slot + heap allocation
    uint64_t args_cycle;
    uint64_t lookup_cycle;
    uint64_t insert_cycle;
    uint64_t fanin_cycle;
    uint64_t scope_end_cycle;
    int64_t submit_count;
    // Wait time tracking for blocking phases
    uint64_t fanin_wait_cycle;  // Cycles spent waiting on fanin (was fanout_lock spin)
    // Atomic operation counts per phase
    uint64_t alloc_atomic_count;
};

PTO2OrchProfilingData orchestrator_get_profiling();
#endif

#endif  // PTO_ORCHESTRATOR_H
