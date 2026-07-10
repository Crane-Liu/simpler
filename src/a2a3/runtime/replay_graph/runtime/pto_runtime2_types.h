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
 * PTO Runtime2 - Core Type Definitions
 *
 * This header defines all fundamental types used by the PTO Runtime2 system:
 * - Configuration constants
 * - Worker types and task states
 * - Tensor regions and task parameters
 * - Task descriptors with fanin/fanout tracking
 * - Dependency list entries
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#ifndef SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_RUNTIME2_TYPES_H_
#define SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_RUNTIME2_TYPES_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <atomic>

#include "profiling_config.h"
#include "pto_constants.h"
#include "pto_runtime_status.h"
#include "pto2_dispatch_payload.h"
#include "aicore_completion_mailbox.h"
#include "pto_submit_types.h"
#include "pto_task_id.h"
#include "pto_types.h"

// Spin-wait hint for AICPU threads.  On real hardware the AICPU has dedicated
// ARM A55 cores — no OS yield is needed, so the hint is a no-op.  In simulation
// all threads share host CPU cores, so we yield to prevent starvation.
// This header is also compiled into the Host .so (for struct definitions only),
// where the hint is never called — the fallback no-op keeps Host builds clean.
#if __has_include("spin_hint.h")
#include "spin_hint.h"
#else
#define SPIN_WAIT_HINT() ((void)0)
#endif

#if PTO2_ORCH_PROFILING || PTO2_SCHED_PROFILING
#include "aicpu/device_time.h"
#endif

// =============================================================================
// Configuration Constants
// =============================================================================

// Task management
// NOTE: PTO2_TASK_WINDOW_SIZE is now a per-ring default value.
// Actual window size is passed at runtime to runtime_create_from_sm().
// Use pto2_task_slot(sched, task_id) for slot calculation.
#define PTO2_TASK_WINDOW_SIZE 16384  // Default task window size (power of 2)

// Memory pools
#define PTO2_HEAP_SIZE (256 * 1024 * 1024)  // 256MB heap
#define PTO2_DEP_LIST_POOL_SIZE 16384       // Dependency list pool entries
#define PTO2_TENSORMAP_POOL_SIZE (65536)    // TensorMap entry pool
#define PTO2_TENSORMAP_NUM_BUCKETS 4096     // Power of 2 for fast hash (4096×8B=32KB fits L1)

inline constexpr int32_t PTO2_REPLAY_GRAPH_BUFFER_COUNT = 2;

// Scope management
#define PTO2_MAX_SCOPE_DEPTH 64  // Maximum nesting depth
// Upper bound for the initial_ready handoff array. Equals the whole-graph task
// count upper bound = the total in-flight slot budget (PTO2_TASK_WINDOW_SIZE):
// every submitted task may be initially ready, so the array can never hold more
// than the window's worth of tasks.
#define PTO2_SCOPE_TASKS_CAP (PTO2_TASK_WINDOW_SIZE)

// Ready queue
#define PTO2_READY_QUEUE_SIZE 65536  // Per-shape queue size

// Cross-thread early-dispatch work queue (power of two)
#define PTO2_EARLY_DISPATCH_QUEUE_SIZE 64

// get_tensor_data/set_tensor_data spin wait timeout in cycles.
// ~10s on hardware (1.5 GHz counter), ~10s on simulation (chrono-based).
constexpr uint64_t PTO2_TENSOR_DATA_TIMEOUT_CYCLES = 15 * 1000 * 1000 * 1000ULL;

// =============================================================================
// Task States
// =============================================================================

/**
 * Task state enumeration
 *
 * State transitions:
 *   PENDING -> COMPLETED (terminal)
 *
 * The slot stays in PENDING from submit through "ready in queue" and "running
 * on a worker"; readiness and running-vs-idle are derived from fanin_refcount
 * and per-core running_slot_state respectively, not from task_state itself.
 *
 * Conditions:
 *   PENDING->COMPLETED:   all subtasks finish (set by scheduler) or task is a
 *                         hidden alloc completed inline by the orchestrator
 */
typedef enum {
    PTO2_TASK_PENDING = 0,   // Submitted; awaiting fanin, queued, or dispatched
    PTO2_TASK_COMPLETED = 1  // Execution finished (terminal in the single-shot replay model)
} PTO2TaskState;

/**
 * Result of a unified task allocation.
 */
struct PTO2TaskAllocResult {
    int32_t task_id;    // Absolute task ID (not wrapped)
    int32_t slot;       // task_id & (window_size - 1)
    void *packed_base;  // Heap allocation result (nullptr if failure)
    void *packed_end;   // packed_base + aligned output_size

    bool failed() const { return task_id < 0; }
};

struct PTO2OutputLayout {
    uint64_t offsets[MAX_TENSOR_ARGS] = {};
    uint64_t buffer_sizes[MAX_TENSOR_ARGS] = {};
    int32_t total_output_size = 0;
};

// =============================================================================
// Dependency List Entry
// =============================================================================

struct PTO2TaskSlotState;  // Forward declaration

/**
 * Dependency list entry (singly-linked list node)
 * Stored in DepListPool ring buffer.
 */
struct PTO2DepListEntry {
    PTO2TaskSlotState *slot_state;  // Consumer slot state (direct pointer)
    PTO2DepListEntry *next;         // next entry
};

inline PTO2DepListEntry *pto2_fanout_closed_sentinel() {
    return reinterpret_cast<PTO2DepListEntry *>(static_cast<uintptr_t>(1));
}

inline bool pto2_is_fanout_closed(PTO2DepListEntry *ptr) { return ptr == pto2_fanout_closed_sentinel(); }

// =============================================================================
// Task Descriptor
// =============================================================================

/**
 * Task descriptor structure (shared memory)
 *
 * Stored in the TaskDescriptor ring buffer in shared memory.
 * Contains static identification and buffer pointers only.
 * Dynamic scheduling state (fanin/fanout/task_state) is in PTO2TaskSlotState.
 *
 * Fields set by Orchestrator at submission, read by Scheduler for dispatch.
 */
struct PTO2TaskDescriptor {
    // Mixed-task identification (encodes ring_id in upper 32 bits)
    PTO2TaskId task_id;  // raw: (ring_id << 32) | local_id

    // Per-slot kernel IDs (INVALID_KERNEL_ID = inactive)
    int32_t kernel_id[PTO2_SUBTASK_SLOT_COUNT];

    // Packed output buffer (all outputs packed into single contiguous buffer)
    void *packed_buffer_base;  // Start of packed buffer in GM Heap
    void *packed_buffer_end;   // End of packed buffer (for heap reclamation)
};

// =============================================================================
// Per-Slot Scheduling State
// =============================================================================

/**
 * Task payload data (cold path - only accessed during orchestration and dispatch)
 *
 * Layout: metadata + early-dispatch spec fields packed in cache line 0, followed
 * by bulk tensor and scalar data (tensors[] is 64B-aligned, so it starts at
 * cache line 1). The fanin producer list no longer lives in the payload —
 * submit_task builds the fanout graph in-line via the slot's fanout_head /
 * fanin_count, so the payload only carries the dispatch-time tensor/scalar data
 * and the early-dispatch counters.
 */
// Speculative early-dispatch claim states for PTO2TaskPayload::spec_state.
enum PTO2SpecState : uint8_t {
    PTO2_SPEC_NONE = 0,       // not pre-staged
    PTO2_SPEC_STAGING = 1,    // Hook 1 claimed it; staging in progress
    PTO2_SPEC_STAGED = 2,     // staged on a core, gated; staged_* fields valid
    PTO2_SPEC_DISPATCHED = 3  // routed via the normal dispatch path (no pre-stage)
};

// A pre-staged consumer occupies one core per gated subtask block. WHICH cores
// it occupies is recorded as a bitmask (staged_core_mask, 1 bit per global
// core_id); the completion-path release iterates the set bits and rings each
// core's doorbell from the scheduler's per-core doorbell table. Bounded by the
// chip's core count (RUNTIME_MAX_WORKER = 72; no two-level pre-dispatch means
// gated cores in flight <= core count), NOT by block_num — so a wide SPMD
// consumer can pre-stage all its idle cores. 2 words = 128 bits >= 72.
inline constexpr int PTO2_SPEC_CORE_MASK_WORDS = 2;

struct PTO2TaskPayload {
    // === Cache line 0 — metadata + early-dispatch spec fields ===
    int32_t tensor_count{0};
    int32_t scalar_count{0};
    // Speculative early-dispatch metadata (AICPU-side only). Ordered by descending
    // alignment (8B mask, 4B fanin, then 1B flags) so the block packs with no
    // internal padding. Shares cache line 0 with tensor_count/scalar_count; the
    // whole block fits in the 56B between the counts and the 64B-aligned
    // tensors[] (offset 64).
    //
    // Bitmask of global core_ids this consumer is pre-staged (gated) on. Set with
    // atomic fetch_or by concurrent stagers; read by release. (Re)initialized in
    // PTO2TaskPayload::init before the slot can be staged again.
    std::atomic<uint64_t> staged_core_mask[PTO2_SPEC_CORE_MASK_WORDS]{};
    // Early-dispatch CANDIDATE detection (event-driven, dual of fanin_refcount):
    // starts at 0 (already-complete producers are not counted), then a flagged
    // producer's DISPATCH bumps each consumer's dispatch_fanin. dispatch_fanin ==
    // fanin_count  <=>  every pending producer is flagged-and-dispatched  =>  this
    // task is an early-dispatch candidate (push early_dispatch_queue).
    std::atomic<int32_t> dispatch_fanin{0};  // CONSUMER side: count of flagged-dispatched pending producers
    bool allow_early_resolve{false};         // codegen hint copied from Arg in PTO2TaskPayload::init
    // Lock-free claim state shared by the stagers (Hook 1, possibly several AICPU
    // threads concurrently) and the completion-path release: 0=NONE, 1=STAGING,
    // 3=DISPATCHED (2=STAGED is unused now). STAGING is the STABLE gated state —
    // many threads stage blocks concurrently while it holds, each claiming a block
    // via the atomic next_block_idx and OR-ing its cores into staged_core_mask.
    // Release does STAGING->DISPATCHED then rings the mask; a thread that stages a
    // block AFTER release flipped DISPATCHED rings that block's doorbell itself
    // (self-ring), so no doorbell is ever missed.
    std::atomic<uint8_t> spec_state{0};
    std::atomic<uint8_t> dispatch_propagated{0};  // PRODUCER side: once-guard for fanout propagation
    std::atomic<uint8_t> spec_chain_active{0};    // inherited early-dispatch flag (auto-chain past codegen flag)
    uint8_t spec_chain_depth{0};                  // auto-chain depth; inherited = parent+1, capped
    // === Cache lines 1-64 (4096B) — tensors (alignas(64) forces alignment) ===
    Tensor tensors[MAX_TENSOR_ARGS];
    // === Cache lines 65-66 (128B) — scalars ===
    uint64_t scalars[MAX_SCALAR_ARGS];

    // Layout verification (size checks that don't need offsetof).
    static_assert(sizeof(Tensor) == 128, "Tensor must be 2 cache lines");
    static_assert(MAX_SCALAR_ARGS * sizeof(uint64_t) == 128, "scalar region must be 128B (2 cache lines)");

    /**
     * Prefetch (for write) the regions init() is about to fill so the stores land
     * in warm cache. tensor_count/scalar_count come from the Arg — the payload's
     * own counts are not set until init(). Cache line 0 (`this`) holds both the
     * counts and the early-dispatch spec block init() rewrites. A member fn lowers
     * to the same prefetch instructions as a free function (`this` is just a
     * register), no cache impact.
     */
    void prefetch(int32_t tensor_count, int32_t scalar_count) const {
        for (int32_t i = 0; i < tensor_count; i++) {
            __builtin_prefetch(&tensors[i], 1, 3);
            __builtin_prefetch(reinterpret_cast<const char *>(&tensors[i]) + 64, 1, 3);
        }
        for (int32_t i = 0; i < scalar_count; i += 8) {
            __builtin_prefetch(&scalars[i], 1, 3);
        }
        __builtin_prefetch(this, 1, 3);  // cache line 0: counts + spec fields
    }

    /**
     * Initialize payload: copy tensors, store scalars.
     *
     * For each param slot, the tensor source is determined by TensorArgType:
     * - OUTPUT -> use materialized_outputs.output_ptr(out_idx++)
     * - INPUT / INOUT -> use refs[i].tensor
     *
     * @param args                Task arguments (tensors + scalars)
     * @param result  Materialized output tensors (from TensorCreateInfo path)
     */
    void init(
        const L0TaskArgs &args, TaskOutputTensors &result, PTO2TaskAllocResult &alloc_result, PTO2OutputLayout &layout
    ) {
        tensor_count = args.tensor_count();
        scalar_count = args.scalar_count();

        // int32_t out_idx = 0;
        for (int32_t i = 0; i < args.tensor_count(); i++) {
            if (args.tag(i) != TensorArgType::OUTPUT) {
                tensors[i].copy(args.tensor(i).ref());
            } else {
                init_tensor_from_create_info(
                    tensors[i], args.tensor(i).create_info(),
                    reinterpret_cast<void *>(reinterpret_cast<char *>(alloc_result.packed_base) + layout.offsets[i]),
                    layout.buffer_sizes[i]
                );
                tensors[i].owner_task_id = result.task_id();
                result.materialize_output(tensors[i]);
            }
        }
        // Round up to cache line boundary. Both arrays are 128B so no overrun.
        // Eliminates branches; extra bytes within the same CL have zero additional cost.
        memcpy(scalars, args.scalars(), PTO2_ALIGN_UP(args.scalar_count() * sizeof(uint64_t), 64));

        // Speculative early-dispatch metadata — the single init point for these
        // fields. The SM-init slot reset skips the payload; prepare_task only
        // allocates/binds. prefetch() warms cache line 0 so these writes land in
        // warm cache. submit_task runs init() BEFORE its fanin loop, so the
        // dispatch_fanin reset here precedes (does not clobber) the per-producer
        // seeding append_fanin_or_fail does for already-completed producers.
        //
        // spec_state / staged_core_mask / dispatch_fanin / spec_chain_* are all
        // CONSUMER-side: a task with allow_early_resolve == false still has them
        // touched when one of ITS producers is flagged (propagate_dispatch_fanin
        // bumps dispatch_fanin and may CAS spec_state / set the auto-chain flag on
        // any consumer, independent of the consumer's own hint). So they MUST be
        // zeroed here unconditionally — no per-task allow_early_resolve gating.
        allow_early_resolve = args.allow_early_resolve();
        spec_state.store(PTO2_SPEC_NONE, std::memory_order_relaxed);
        for (int w = 0; w < PTO2_SPEC_CORE_MASK_WORDS; w++)
            staged_core_mask[w].store(0, std::memory_order_relaxed);
        dispatch_fanin.store(0, std::memory_order_relaxed);
        dispatch_propagated.store(0, std::memory_order_relaxed);
        spec_chain_active.store(0, std::memory_order_relaxed);
        spec_chain_depth = 0;
    }
};

// PTO2TaskPayload layout verification (offsetof requires complete type).
// Metadata (tensor_count/scalar_count) + the early-dispatch spec block occupy
// cache line 0; tensors[] is alignas(64) so it starts at cache line 1 (byte 64).
static_assert(offsetof(PTO2TaskPayload, tensors) == 64, "tensors must start at byte 64 (cache line 1)");
static_assert(
    offsetof(PTO2TaskPayload, scalars) == 64 + MAX_TENSOR_ARGS * sizeof(Tensor),
    "scalars must immediately follow tensors"
);
static_assert(
    sizeof(PTO2TaskPayload) == 64 + MAX_TENSOR_ARGS * sizeof(Tensor) + MAX_SCALAR_ARGS * sizeof(uint64_t),
    "PTO2TaskPayload size must stay on the baseline cache-line footprint"
);

/**
 * Per-task slot scheduling state (scheduler-private, NOT in shared memory)
 *
 * Consolidates all hot-path scheduling fields into a single cache-friendly
 * structure (one 64-byte cache line; alignas(64) below). Accessing any field
 * of a task's slot state brings all related fields into the same cache line.
 *
 * Concurrency notes:
 * - fanout_head is an atomic intrusive stack. The orchestrator appends
 *   consumers while the scheduler may complete producers; completion closes the
 *   stack with pto2_fanout_closed_sentinel() to make late appends observe that
 *   the dependency is already satisfied.
 * - fanin_count set once at submission, read-only after (hot path for ready check)
 * - task_state, fanin_refcount updated atomically
 */
struct alignas(64) PTO2TaskSlotState {
    std::atomic<PTO2DepListEntry *> fanout_head;  // nullptr = empty, CLOSED sentinel = producer completed

    // Task state (completion, ready check)
    std::atomic<PTO2TaskState> task_state;  // PENDING/COMPLETED

    // Fanin (accessed together in release_fanin_and_check_ready)
    std::atomic<int32_t> fanin_refcount;  // Dynamic: counts completed producers
    int32_t fanin_count;                  // Number of producer dependencies (set once by wiring)

    // --- Per-slot constant, re-bound by orch::prepare_task each submit ---
    // Value is the same on every reuse (&task_payloads[slot] / &task_descriptors[slot]),
    // but written here per-submit instead of in an O(window_size) init loop —
    // these are the only "scale-dependent" pointers in this struct, so moving
    // them out of init makes startup cost independent of task_window_size.
    PTO2TaskPayload *payload;
    PTO2TaskDescriptor *task;

    // --- Set per-submit (depend on task inputs) ---
    ActiveMask active_mask;  // Bitmask of active subtask slots (set once)
    uint8_t ring_id;         // Ring layer (immutable after init)
    // Set by any subtask FIN that pushed deferred-completion CONDITIONs to
    // the runtime mailbox; read by the last subtask FIN to decide whether
    // the task needs MPSC-deferred completion or can complete inline on this
    // thread. The write is sequenced before on_subtask_complete's acq_rel
    // fetch_add and the read after, so all earlier subtasks' writes are
    // visible to the last subtask.
    std::atomic<bool> any_subtask_deferred{false};

    std::atomic<int16_t> completed_subtasks{0};  // Each core completion increments by 1
    int16_t total_required_subtasks{0};          // = logical_block_num * popcount(active_mask)
    int16_t logical_block_num{1};                // Total logical blocks (set by orchestrator)
    // Next block to dispatch. Atomic so concurrent speculative stagers can each
    // claim a distinct block via CAS; normal dispatch (ready-queue serialized)
    // uses plain relaxed load/store. The two phases never overlap in time (staging
    // happens before release; normal dispatch of the remainder happens after).
    std::atomic<int16_t> next_block_idx{0};

    /**
     * Bind the slot-invariant ring id. Called once per slot during
     * PTO2SharedMemoryHandle::init_header(); ring_id never changes.
     */
    void bind_ring(uint8_t rid) { ring_id = rid; }

    /**
     * Re-bind the per-slot payload/task pointers. Called by
     * orch::prepare_task on every submit. Value is constant for a given
     * slot, but we pay the cheap re-write each submit (both fields land on
     * the same 64B slot_state cache line that prepare_task is already
     * dirtying) to avoid the init-time per-slot loop.
     */
    void bind_buffers(PTO2TaskPayload *p, PTO2TaskDescriptor *t) {
        payload = p;
        task = t;
    }

    // Completion atomically exchanges fanout_head with CLOSED before walking the
    // captured list. A concurrent orchestrator append either lands before that
    // exchange and is released by completion, or observes CLOSED and treats the
    // dependency as already satisfied.
};

// alignas(64) keeps each slot on its own cache line (avoids false sharing
// between sched threads touching distinct slots); it also rounds the struct
// up to a full cache line even though the live fields sum to less.
static_assert(sizeof(PTO2TaskSlotState) == 64);

#endif  // SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_RUNTIME2_TYPES_H_
