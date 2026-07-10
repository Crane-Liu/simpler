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
 * PTO Runtime2 - Scheduler Interface
 *
 * The Scheduler is responsible for:
 * 1. Maintaining per-resource-shape ready queues
 * 2. Tracking task state (PENDING -> COMPLETED)
 * 3. Managing fanin refcounts for dependency resolution
 * 4. Two-stage mixed-task completion (subtask done bits → mixed-task complete)
 *
 * The Scheduler runs on Device AI_CPU and processes:
 * - Task state transitions based on fanin_refcount
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#pragma once

#include <atomic>

#include "common/core_type.h"
#include "utils/device_arena.h"
#include "aicpu/platform_regs.h"  // get_reg_ptr / RegId for the speculative doorbell
#include "pto_async_wait.h"
#include "pto_ring_buffer.h"
#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"

#include "aicpu/device_time.h"  // get_sys_cnt_aicpu (weak; used by spec doorbell timing too)
#if PTO2_SCHED_PROFILING
#define PTO2_SCHED_CYCLE_START() uint64_t _st0 = get_sys_cnt_aicpu(), _st1
#define PTO2_SCHED_CYCLE_LAP(acc)   \
    do {                            \
        _st1 = get_sys_cnt_aicpu(); \
        acc += (_st1 - _st0);       \
        _st0 = _st1;                \
    } while (0)
#endif

// =============================================================================
// Ready Queue (Lock-free bounded MPMC — Vyukov design)
// =============================================================================

/**
 * Per-slot entry: sequence counter for ABA safety + task payload
 */
struct PTO2ReadyQueueSlot {
    std::atomic<int64_t> sequence;
    PTO2TaskSlotState *slot_state;
};

/**
 * Thread-local ready buffer for local-first dispatch optimization.
 *
 * Two buffers per scheduling thread, one per CoreType (AIC=0, AIV=1).
 * Initialized once before the scheduling loop; must be empty at
 * the start of each iteration (verified by always_assert).
 *
 * Phase 1 fills per-CoreType buffers via on_task_complete().
 * The dispatch stage drains them local-first via get_ready_tasks_batch,
 * with any remaining tasks pushed to the global ready queue.
 */
// Number of CoreType values eligible for local dispatch (AIC=0, AIV=1)
static constexpr int PTO2_LOCAL_DISPATCH_TYPE_NUM = 2;

struct PTO2LocalReadyBuffer {
    PTO2TaskSlotState **slot_states = nullptr;
    int count = 0;
    int capacity = 0;

    void reset(PTO2TaskSlotState **buf, int cap) {
        slot_states = buf;
        count = 0;
        capacity = cap;
    }

    bool try_push(PTO2TaskSlotState *s) {
        if (slot_states && count < capacity) {
            slot_states[count++] = s;
            return true;
        }
        return false;
    }

    PTO2TaskSlotState *pop() { return (count > 0) ? slot_states[--count] : nullptr; }
};

/**
 * Lock-free bounded MPMC queue (Dmitry Vyukov design)
 *
 * Key properties:
 * - enqueue_pos and dequeue_pos on separate cache lines (no false sharing)
 * - Per-slot sequence counter prevents ABA problem
 * - Empty queue pop returns immediately (single atomic load, no lock)
 * - CAS contention is split: producers only touch enqueue_pos,
 *   consumers only touch dequeue_pos
 */
struct alignas(64) PTO2ReadyQueue {
    PTO2ReadyQueueSlot *slots;
    uint64_t capacity;
    uint64_t mask;        // capacity - 1
    char _pad0[64 - 24];  // Pad to own cache line

    std::atomic<uint64_t> enqueue_pos;
    char _pad1[64 - sizeof(std::atomic<uint64_t>)];  // Own cache line

    std::atomic<uint64_t> dequeue_pos;
    char _pad2[64 - sizeof(std::atomic<uint64_t>)];  // Own cache line

    uint64_t size() {
        uint64_t e = enqueue_pos.load(std::memory_order_relaxed);
        uint64_t d = dequeue_pos.load(std::memory_order_relaxed);
        return (e >= d) ? (e - d) : 0;
    }

    bool push(PTO2TaskSlotState *slot_state) {
        uint64_t pos;
        PTO2ReadyQueueSlot *slot;
        while (true) {
            pos = enqueue_pos.load(std::memory_order_relaxed);
            slot = &slots[pos & mask];
            int64_t seq = slot->sequence.load(std::memory_order_acquire);
            int64_t diff = seq - static_cast<int64_t>(pos);
            if (diff == 0) {
                if (enqueue_pos.compare_exchange_weak(
                        pos, pos + 1, std::memory_order_relaxed, std::memory_order_relaxed
                    )) {
                    break;
                }
            } else if (diff < 0) {
                return false;  // Queue full
            }
        }

        slot->slot_state = slot_state;
        slot->sequence.store(static_cast<int64_t>(pos + 1), std::memory_order_release);
        return true;
    }

    // Batch push: reserve count slots with a single CAS after confirming
    // every target slot is available under the usual Vyukov sequence check.
    void push_batch(PTO2TaskSlotState **items, int count) {
        if (count == 0) return;

        uint64_t pos;
        while (true) {
            pos = enqueue_pos.load(std::memory_order_relaxed);
            bool ready = true;
            for (int i = 0; i < count; i++) {
                PTO2ReadyQueueSlot *slot = &slots[(pos + i) & mask];
                int64_t seq = slot->sequence.load(std::memory_order_acquire);
                int64_t diff = seq - static_cast<int64_t>(pos + i);
                if (diff != 0) {
                    ready = false;
                    break;
                }
            }
            if (!ready) {
                continue;
            }
            if (enqueue_pos.compare_exchange_weak(
                    pos, pos + count, std::memory_order_relaxed, std::memory_order_relaxed
                )) {
                break;
            }
        }

        for (int i = 0; i < count; i++) {
            PTO2ReadyQueueSlot *slot = &slots[(pos + i) & mask];
            slot->slot_state = items[i];
            slot->sequence.store(static_cast<int64_t>(pos + i + 1), std::memory_order_release);
        }
    }

#if PTO2_ORCH_PROFILING || PTO2_SCHED_PROFILING
    bool push(PTO2TaskSlotState *slot_state, uint64_t &atomic_count, uint64_t &wait_cycle) {
        uint64_t pos;
        PTO2ReadyQueueSlot *slot;
        uint64_t t0 = get_sys_cnt_aicpu();
        bool contended = false;
        uint32_t atomic_ops = 0;
        while (true) {
            pos = enqueue_pos.load(std::memory_order_relaxed);
            slot = &slots[pos & mask];
            int64_t seq = slot->sequence.load(std::memory_order_acquire);
            int64_t diff = seq - static_cast<int64_t>(pos);
            atomic_ops += 2;  // enqueue_pos.load + sequence.load
            if (diff == 0) {
                if (enqueue_pos.compare_exchange_weak(
                        pos, pos + 1, std::memory_order_relaxed, std::memory_order_relaxed
                    )) {
                    atomic_ops++;  // successful CAS
                    break;
                }
                contended = true;
                atomic_ops++;  // failed CAS
            } else if (diff < 0) {
                return false;  // Queue full
            } else {
                contended = true;  // diff > 0: slot not yet released, spin
            }
        }
        atomic_ops++;  // final sequence.store
        atomic_count += atomic_ops;
        if (contended) {
            wait_cycle += (get_sys_cnt_aicpu() - t0);
        }

        slot->slot_state = slot_state;
        slot->sequence.store(static_cast<int64_t>(pos + 1), std::memory_order_release);
        return true;
    }
#endif

    PTO2TaskSlotState *pop() {
        // Fast-path: skip slot load when queue is clearly empty
        uint64_t d = dequeue_pos.load(std::memory_order_relaxed);
        uint64_t e = enqueue_pos.load(std::memory_order_relaxed);
        if (d >= e) {
            return nullptr;
        }

        uint64_t pos;
        PTO2ReadyQueueSlot *slot;
        while (true) {
            pos = dequeue_pos.load(std::memory_order_relaxed);
            slot = &slots[pos & mask];
            int64_t seq = slot->sequence.load(std::memory_order_acquire);
            int64_t diff = seq - static_cast<int64_t>(pos + 1);
            if (diff == 0) {
                if (dequeue_pos.compare_exchange_weak(
                        pos, pos + 1, std::memory_order_relaxed, std::memory_order_relaxed
                    ))
                    break;
            } else if (diff < 0) {
                return nullptr;  // Queue empty
            }
        }

        PTO2TaskSlotState *result = slot->slot_state;
        slot->sequence.store(static_cast<int64_t>(pos + mask + 1), std::memory_order_release);
        return result;
    }

#if PTO2_SCHED_PROFILING
    PTO2TaskSlotState *pop(uint64_t &atomic_count, uint64_t &wait_cycle) {
        // Fast-path: skip slot load when queue is clearly empty
        uint64_t d = dequeue_pos.load(std::memory_order_relaxed);
        uint64_t e = enqueue_pos.load(std::memory_order_relaxed);
        atomic_count += 2;  // dequeue_pos.load + enqueue_pos.load
        if (d >= e) {
            return nullptr;
        }

        uint64_t pos;
        PTO2ReadyQueueSlot *slot;
        uint64_t t0 = get_sys_cnt_aicpu();
        bool contended = false;
        uint32_t atomic_ops = 0;
        while (true) {
            pos = dequeue_pos.load(std::memory_order_relaxed);
            slot = &slots[pos & mask];
            int64_t seq = slot->sequence.load(std::memory_order_acquire);
            int64_t diff = seq - static_cast<int64_t>(pos + 1);
            atomic_ops += 2;  // dequeue_pos.load + sequence.load
            if (diff == 0) {
                if (dequeue_pos.compare_exchange_weak(
                        pos, pos + 1, std::memory_order_relaxed, std::memory_order_relaxed
                    )) {
                    atomic_ops++;  // successful CAS
                    break;
                }
                contended = true;
                atomic_ops++;  // failed CAS
            } else if (diff < 0) {
                atomic_count += atomic_ops;
                return nullptr;  // Queue empty
            } else {
                contended = true;
            }
        }
        atomic_ops++;  // final sequence.store
        atomic_count += atomic_ops;
        if (contended) {
            wait_cycle += (get_sys_cnt_aicpu() - t0);
        }

        PTO2TaskSlotState *result = slot->slot_state;
        slot->sequence.store(static_cast<int64_t>(pos + mask + 1), std::memory_order_release);
        return result;
    }
#endif

    // Batch pop: reserve a contiguous run of ready slots with a single CAS.
    // Returns actual number of items popped (may be less than max_count).
    int pop_batch(PTO2TaskSlotState **out, int max_count) {
        uint64_t pos;
        int count;
        while (true) {
            pos = dequeue_pos.load(std::memory_order_relaxed);
            count = 0;
            while (count < max_count) {
                PTO2ReadyQueueSlot *slot = &slots[(pos + count) & mask];
                int64_t seq = slot->sequence.load(std::memory_order_acquire);
                int64_t diff = seq - static_cast<int64_t>(pos + count + 1);
                if (diff == 0) {
                    count++;
                    continue;
                }
                if (diff < 0) {
                    break;
                }
                count = -1;
                break;
            }
            if (count == 0) return 0;
            if (count < 0) continue;
            if (dequeue_pos.compare_exchange_weak(
                    pos, pos + count, std::memory_order_relaxed, std::memory_order_relaxed
                )) {
                break;
            }
        }

        for (int i = 0; i < count; i++) {
            PTO2ReadyQueueSlot *slot = &slots[(pos + i) & mask];
            out[i] = slot->slot_state;
            slot->sequence.store(static_cast<int64_t>(pos + i + mask + 1), std::memory_order_release);
        }
        return count;
    }

#if PTO2_SCHED_PROFILING
    int pop_batch(PTO2TaskSlotState **out, int max_count, uint64_t &atomic_count, uint64_t &wait_cycle) {
        uint64_t pos;
        int count;
        uint64_t t0 = get_sys_cnt_aicpu();
        bool contended = false;
        uint32_t atomic_ops = 0;
        while (true) {
            pos = dequeue_pos.load(std::memory_order_relaxed);
            atomic_ops++;  // dequeue_pos.load
            count = 0;
            while (count < max_count) {
                PTO2ReadyQueueSlot *slot = &slots[(pos + count) & mask];
                int64_t seq = slot->sequence.load(std::memory_order_acquire);
                int64_t diff = seq - static_cast<int64_t>(pos + count + 1);
                atomic_ops++;  // sequence.load
                if (diff == 0) {
                    count++;
                    continue;
                }
                if (diff < 0) {
                    break;
                }
                contended = true;
                count = -1;
                break;
            }
            if (count == 0) {
                atomic_count += atomic_ops;
                return 0;
            }
            if (count < 0) {
                continue;
            }
            if (dequeue_pos.compare_exchange_weak(
                    pos, pos + count, std::memory_order_relaxed, std::memory_order_relaxed
                )) {
                atomic_ops++;  // successful CAS
                break;
            }
            contended = true;
            atomic_ops++;  // failed CAS
        }

        for (int i = 0; i < count; i++) {
            PTO2ReadyQueueSlot *slot = &slots[(pos + i) & mask];
            out[i] = slot->slot_state;
            slot->sequence.store(static_cast<int64_t>(pos + i + mask + 1), std::memory_order_release);
            atomic_ops++;  // sequence.store
        }
        atomic_count += atomic_ops;
        if (contended) {
            wait_cycle += (get_sys_cnt_aicpu() - t0);
        }
        return count;
    }
#endif
};

// Cold-path ready queue operations (defined in pto_scheduler.cpp). Declared
// as non-member so PTO2ReadyQueue stays a POD-like struct with cache-line
// alignment. Storage is owned by the caller-supplied arena.
//   reserve_layout: declare the slots[] region on the arena (must precede commit)
//   init_from_layout: bind slots pointer from arena.region_ptr(off) and
//                     initialize sequence counters
//   destroy: forget the slots pointer (arena owns the buffer)
size_t ready_queue_reserve_layout(DeviceArena &arena, uint64_t capacity);
// Writes everything *except* the arena-internal `slots` pointer field
// (sequences/positions on the slot array, capacity, mask). Uses
// arena.region_ptr(slots_off) only to address the slot array for writes;
// does NOT store the pointer in `queue->slots`. Call
// `ready_queue_wire_arena_pointers` afterwards to set the field itself.
bool ready_queue_init_data_from_layout(PTO2ReadyQueue *queue, DeviceArena &arena, size_t slots_off, uint64_t capacity);
// Stores queue->slots = arena.region_ptr(slots_off). Idempotent.
void ready_queue_wire_arena_pointers(PTO2ReadyQueue *queue, DeviceArena &arena, size_t slots_off);
void ready_queue_destroy(PTO2ReadyQueue *queue);

/**
 * Statistics returned by mixed-task completion processing
 */
struct CompletionStats {
    int32_t fanout_edges;       // Number of fanout edges traversed (notify consumers)
    int32_t tasks_enqueued;     // Number of consumers that became READY
    int32_t fanin_edges;        // Number of fanin edges traversed (release producers)
    bool mixed_task_completed;  // True only when this callback completed a mixed task
};

/**
 * Layout descriptor produced by PTO2SchedulerState::reserve_layout(). Holds
 * the arena offsets of every sub-region the scheduler needs plus the
 * capacities used at layout time (init_from_layout reuses them).
 */
struct PTO2SchedulerLayout {
    size_t off_ready_queue_slots[PTO2_NUM_RESOURCE_SHAPES];
    size_t off_dummy_ready_queue_slots;
    size_t off_early_dispatch_queue_slots;
    uint64_t ready_queue_capacity;
    // dep_pool entries + wiring SPSC buffer moved to PTO2OrchestratorLayout
    // (replay_graph stage 1): wiring is owned by the orchestrator now.
};

/**
 * Scheduler state structure
 *
 * Contains dynamic state updated during task execution.
 * Separated from shared memory for cache efficiency.
 * Hot-path methods are defined inline (implicitly inline as member functions).
 */
struct PTO2SchedulerState {
    // Shared memory access
    PTO2SharedMemoryHeader *sm_header;

    // Ready queues remain global (scheduling is ring-agnostic)
    PTO2ReadyQueue ready_queues[PTO2_NUM_RESOURCE_SHAPES];

    // Dependency-only tasks (active_mask is empty, shape == DUMMY). Drained by
    // the dispatch loop and completed inline -- never goes to AICore.
    PTO2ReadyQueue dummy_ready_queue;

    // Wiring (dep_pool + fanout list construction) lives in the orchestrator:
    // submit_task builds the graph in-line. The scheduler only consumes the
    // orchestrator's products: it seeds initial_ready into ready_queues and
    // traverses fanout linked lists read-only during completion.

    alignas(64) AsyncWaitList async_wait_list;

    // Statistics (cold path, isolated from hot-path fields)
#if PTO2_SCHED_PROFILING
    alignas(64) std::atomic<int64_t> tasks_completed;
#endif
    // =========================================================================
    // Inline hot-path methods
    // =========================================================================

    // Route a ready slot to the right global queue. Dummy tasks (empty
    // active_mask) live in dummy_ready_queue; everything else goes to the
    // per-shape ready_queues[]. Used by paths that do not have a thread-local
    // ready buffer (e.g. wiring). See push_ready_routed_local for the
    // dispatch-time fast path.
    void push_ready_routed(PTO2TaskSlotState *slot_state) {
        PTO2ResourceShape shape = slot_state->active_mask.to_shape();
        if (shape == PTO2ResourceShape::DUMMY) {
            dummy_ready_queue.push(slot_state);
        } else {
            ready_queues[static_cast<int32_t>(shape)].push(slot_state);
        }
    }

    // Speculative early-dispatch release. If the now-ready task was pre-staged
    // (gated on a core), ring its DATA_MAIN_BASE high-32 doorbell RIGHT HERE in
    // the completion path — the moment its last producer's FIN satisfies fanin —
    // instead of routing it through the ready queue and waiting for the dispatch
    // pass to pop it. Returns true if the task is fully handled (caller must NOT
    // push to the ready queue). Returns false when the caller must route C
    // normally: either it was never pre-staged, OR it is a SPMD consumer only
    // PARTIALLY pre-staged — the gated blocks are released by the doorbells rung
    // here, and the remaining (next_block_idx .. logical_block_num) blocks
    // dispatch normally off the ready queue. Lock-free claim shared with Hook 1
    // (the stager): CAS NONE->DISPATCHED wins => not pre-staged; lose => STAGED
    // (spin past the brief STAGING window so the mask is visible), then ring.

    // Per-core speculative doorbell table. Hook 1 records each gated core's
    // (reg_addr, dispatch token) here at stage time; the completion-path release
    // reads it back for the cores set in the consumer's staged_core_mask. One
    // global table indexed by core_id (not per-task): gated cores in flight are
    // bounded by the chip's core count (no two-level pre-dispatch), so this is the
    // natural capacity and removes the old per-task 3-doorbell cap.
    struct SpecDoorbell {
        uint64_t addr{0};
        uint32_t token{0};
    };
    SpecDoorbell spec_doorbell_table[PTO2_SPEC_CORE_MASK_WORDS * 64]{};

    // Cross-thread early-dispatch work queue (a PTO2ReadyQueue MPMC instance,
    // arena-backed — reserved/wired in pto_runtime2_init alongside the ready queues).
    // A consumer's SPMD blocks span cores owned by several AICPU threads, but only a
    // thread RUNNING the consumer's producer discovers it (via the producer's
    // fanout). When that producer is thread-local (e.g. a 16-block AIV op filling one
    // thread's cores), the other threads never see the consumer and its blocks on
    // their cores can't pre-stage. The first claimer pushes the partially-staged
    // consumer here; every idle thread's early_dispatch pass pops one, stages a range onto
    // ITS OWN cores (range-claim via next_block_idx), and re-pushes if blocks remain
    // — exactly mirroring how a partially-dispatched SPMD task is re-pushed to the
    // ready queue (scheduler_dispatch: pop -> claim -> re-push). A stale/released
    // entry fails the STAGING check on pop and is dropped; a push that overflows is
    // logged and the consumer's blocks fall back to normal dispatch.
    PTO2ReadyQueue early_dispatch_queue;

    static inline void ring_one_doorbell(uint64_t reg_addr, uint32_t token) {
        volatile uint64_t *dmb = reinterpret_cast<volatile uint64_t *>(get_reg_ptr(reg_addr, RegId::DATA_MAIN_BASE));
        uint64_t tk = static_cast<uint64_t>(token);
        *dmb = (tk << 32) | tk;  // 64-bit STR: high=low=token releases the gated AICore
    }

    // auto-chain depth cap: a candidate inherits the flag only while depth < this.
    static constexpr uint8_t PTO2_SPEC_CHAIN_MAX = 4;

    // Event-driven candidate detection (the dual of fanin_refcount/ready). Call when a
    // FLAGGED producer `p` DISPATCHES (starts running): walk its fanout and bump each
    // consumer's dispatch_fanin. A consumer whose dispatch_fanin reaches
    // fanin_count (every pending producer is flagged-and-dispatched; already-
    // complete producers are not counted) is an early-dispatch candidate:
    // CAS NONE->STAGING (exactly-once) and push to early_dispatch_queue for the idle drain to
    // pre-stage. Once-guarded per producer so an SPMD producer's block-by-block
    // dispatch propagates once. Replaces the old per-iteration pass-1 PULL scan.
    void propagate_dispatch_fanin(PTO2TaskSlotState &p) {
        if (!(p.payload->allow_early_resolve || p.payload->spec_chain_active.load(std::memory_order_acquire)))
            return;  // only flagged (codegen or inherited) producers propagate
        if (p.payload->dispatch_propagated.exchange(1, std::memory_order_acq_rel) != 0)
            return;  // already propagated once
        uint8_t child_depth = static_cast<uint8_t>(p.payload->spec_chain_depth + 1);
        // Early-dispatch propagation is opportunistic: it may miss consumers
        // appended after this snapshot, but completion will still release them.
        PTO2DepListEntry *edge = p.fanout_head.load(std::memory_order_acquire);
        if (pto2_is_fanout_closed(edge)) return;
        for (; edge != nullptr; edge = edge->next) {
            PTO2TaskSlotState *c = edge->slot_state;
            // dispatch_fanin (CONSUMER-side early-dispatch counter) starts at 0:
            // submit-time wiring no longer seeds it, because already-complete
            // producers are not counted at all (they gate nothing). It is bumped
            // here by each flagged producer's dispatch; reaching fanin_count (both
            // counters track only pending producers) means every pending producer
            // is flagged-dispatched, so the consumer is an early-dispatch
            // candidate. Read the count off the slot's hot fanin_count, not the
            // cold payload cache line.
            int32_t nf = c->payload->dispatch_fanin.fetch_add(1, std::memory_order_acq_rel) + 1;
            if (nf != c->fanin_count) continue;
            if (c->active_mask.requires_sync_start()) continue;  // sync_start can't be block-by-block pre-staged
            PTO2ResourceShape shape = c->active_mask.to_shape();
            if (shape != PTO2ResourceShape::AIC && shape != PTO2ResourceShape::AIV && shape != PTO2ResourceShape::MIX)
                continue;
            uint8_t expect = PTO2_SPEC_NONE;  // exactly-once: only the CAS winner enqueues
            if (!c->payload->spec_state.compare_exchange_strong(
                    expect, PTO2_SPEC_STAGING, std::memory_order_seq_cst, std::memory_order_seq_cst
                ))
                continue;
            if (child_depth < PTO2_SPEC_CHAIN_MAX) {  // auto-chain: C propagates to ITS consumers
                c->payload->spec_chain_depth = child_depth;
                c->payload->spec_chain_active.store(1, std::memory_order_release);
            }
            early_dispatch_queue.push(c);
        }
    }

    // Collects consumers released via the speculative-doorbell path during a
    // single on_task_complete fanout walk, so their dispatch_fanin
    // propagation runs AFTER the walk — never between two siblings' doorbells.
    struct SpecReleaseSink {
        static constexpr int CAP = 32;
        PTO2TaskSlotState *items[CAP];
        int n = 0;
        inline bool push(PTO2TaskSlotState *s) {
            if (n >= CAP) return false;
            items[n++] = s;
            return true;
        }
    };

    inline bool try_speculative_release(PTO2TaskSlotState &slot_state, SpecReleaseSink *sink = nullptr) {
        // Never staged => CAS NONE->DISPATCHED wins => dispatch normally.
        uint8_t expect = PTO2_SPEC_NONE;
        if (slot_state.payload->spec_state.compare_exchange_strong(
                expect, PTO2_SPEC_DISPATCHED, std::memory_order_seq_cst, std::memory_order_seq_cst
            )) {
            return false;
        }
        // Staged (STAGING). Flip STAGING->DISPATCHED, THEN read the mask. seq_cst
        // gives a total order with the concurrent stagers, each of which OR-s its
        // core into the mask and THEN loads spec_state: a stager whose bit lands
        // before this CAS is read here and rung; a stager whose bit lands after
        // sees DISPATCHED and rings that core itself (self-ring in
        // stage_consumer_blocks). Either way every gated core's doorbell fires once
        // (a double-ring is harmless — the AICore already matched). This replaces
        // the old transient-STAGING spin: STAGING is now the stable gated state.
        expect = PTO2_SPEC_STAGING;
        slot_state.payload->spec_state.compare_exchange_strong(
            expect, PTO2_SPEC_DISPATCHED, std::memory_order_seq_cst, std::memory_order_seq_cst
        );
        for (int w = 0; w < PTO2_SPEC_CORE_MASK_WORDS; w++) {
            uint64_t bits = slot_state.payload->staged_core_mask[w].load(std::memory_order_seq_cst);
            while (bits != 0) {
                int core_id = w * 64 + __builtin_ctzll(bits);
                bits &= bits - 1;
                ring_one_doorbell(spec_doorbell_table[core_id].addr, spec_doorbell_table[core_id].token);
            }
        }
        // This pre-staged consumer was just released by its doorbell — it starts
        // running NOW, so propagate dispatch_fanin to ITS consumers (auto-chain,
        // knob A). Defer it via the sink so it runs after the whole fanout walk:
        // doing it inline here would delay the doorbells of later consumers in the
        // same producer's fanout. Fallback to inline if no sink / sink full.
        if (sink == nullptr || !sink->push(&slot_state)) {
            propagate_dispatch_fanin(slot_state);
        }
        // No explicit removal from the cross-thread queue: a still-queued entry for
        // this consumer is now DISPATCHED and is dropped when a peer pops it.
        // Fully pre-staged => skip the ready queue. Partially staged SPMD consumer =>
        // fall through so the caller pushes C; dispatch resumes from next_block_idx.
        return slot_state.next_block_idx.load(std::memory_order_seq_cst) >= slot_state.logical_block_num;
    }

    bool release_fanin_and_check_ready(
        PTO2TaskSlotState &slot_state, PTO2LocalReadyBuffer *local_bufs = nullptr, SpecReleaseSink *sink = nullptr
    ) {
        // Atomically increment fanin_refcount and check if all producers are done
        // ACQ_REL on fanin_refcount already synchronizes with the orchestrator's
        // init release, making fanin_count visible — plain load suffices.
        int32_t new_refcount = slot_state.fanin_refcount.fetch_add(1, std::memory_order_acq_rel) + 1;

        if (new_refcount == slot_state.fanin_count) {
            // Speculative early-dispatch: pre-staged tasks are released by doorbell
            // here, skipping the ready-queue round-trip entirely.
            if (try_speculative_release(slot_state, sink)) return true;
            // Local-first: try per-CoreType thread-local buffer before global queue
            // Route by active_mask: AIC-containing tasks → buf[0], AIV-only → buf[1]
            // DUMMY shape is out of range for local_bufs (sized PTO2_NUM_RESOURCE_SHAPES);
            // dummy slots bypass the local fast path and go straight to dummy_ready_queue.
            PTO2ResourceShape shape = slot_state.active_mask.to_shape();
            if (shape == PTO2ResourceShape::DUMMY) {
                dummy_ready_queue.push(&slot_state);
            } else if (!local_bufs || !local_bufs[static_cast<int32_t>(shape)].try_push(&slot_state)) {
                ready_queues[static_cast<int32_t>(shape)].push(&slot_state);
            }
            return true;
        }
        return false;
    }

#if PTO2_ORCH_PROFILING || PTO2_SCHED_PROFILING
    bool release_fanin_and_check_ready(
        PTO2TaskSlotState &slot_state, uint64_t &atomic_count, uint64_t &push_wait,
        PTO2LocalReadyBuffer *local_bufs = nullptr, SpecReleaseSink *sink = nullptr
    ) {
        int32_t new_refcount = slot_state.fanin_refcount.fetch_add(1, std::memory_order_acq_rel) + 1;
        atomic_count += 1;  // fanin_refcount.fetch_add

        if (new_refcount == slot_state.fanin_count) {
            // Speculative early-dispatch: pre-staged tasks are released by doorbell
            // here, skipping the ready-queue round-trip entirely.
            if (try_speculative_release(slot_state, sink)) return true;
            // Local-first: try per-CoreType thread-local buffer before global queue.
            // Dummy slots bypass local_bufs (out-of-range for PTO2_NUM_RESOURCE_SHAPES)
            // and go straight to dummy_ready_queue; use the profiling-aware push so
            // atomic_count / push_wait stay consistent with the non-dummy path.
            PTO2ResourceShape shape = slot_state.active_mask.to_shape();
            if (shape == PTO2ResourceShape::DUMMY) {
                dummy_ready_queue.push(&slot_state, atomic_count, push_wait);
            } else if (!local_bufs || !local_bufs[static_cast<int32_t>(shape)].try_push(&slot_state)) {
                ready_queues[static_cast<int32_t>(shape)].push(&slot_state, atomic_count, push_wait);
            }
            return true;
        }
        return false;
    }
#endif

    int get_ready_tasks_batch(
        PTO2ResourceShape shape, PTO2LocalReadyBuffer &local_buf, PTO2TaskSlotState **out, int max_count
    ) {
        int count = 0;
        while (count < max_count && local_buf.count > 0) {
            out[count++] = local_buf.slot_states[--local_buf.count];
        }
        int remaining = max_count - count;
        if (remaining > 0) {
            count += ready_queues[static_cast<int32_t>(shape)].pop_batch(out + count, remaining);
        }
        return count;
    }

#if PTO2_SCHED_PROFILING
    int get_ready_tasks_batch(
        PTO2ResourceShape shape, PTO2LocalReadyBuffer &local_buf, PTO2TaskSlotState **out, int max_count,
        uint64_t &atomic_count, uint64_t &wait_cycle
    ) {
        int count = 0;
        while (count < max_count && local_buf.count > 0) {
            out[count++] = local_buf.slot_states[--local_buf.count];
        }
        int remaining = max_count - count;
        if (remaining > 0) {
            count +=
                ready_queues[static_cast<int32_t>(shape)].pop_batch(out + count, remaining, atomic_count, wait_cycle);
        }
        return count;
    }
#endif

    /**
     * Subtask completion: atomic counter model.
     * Called when a single subtask (AIC, AIV0, or AIV1) finishes on any block.
     * Atomically increments completed_subtasks and checks whether all subtasks
     * across all blocks are done.
     *
     * @return true if this was the last subtask, completing the entire task.
     */
    bool on_subtask_complete(PTO2TaskSlotState &slot_state) {
        int16_t prev = slot_state.completed_subtasks.fetch_add(1, std::memory_order_acq_rel);
        return (prev + 1) == slot_state.total_required_subtasks;
    }

    /**
     * Two-stage completion: second stage.
     * Called exactly once when all subtasks of a task are done (i.e.,
     * on_subtask_complete returned true). Walks the consumer (fanout) list,
     * decrements each consumer's fanin, pushes newly-ready ones, and rings
     * doorbells for speculative hits.
     *
     * Non-PROFILING returns the consumer-walk count (= edges traversed). The
     * Resolve swimlane bar reads it to label the bar with how many successors
     * actually got resolved. PROFILING returns the richer CompletionStats
     * whose `fanout_edges` carries the same number.
     */
#if PTO2_SCHED_PROFILING
    CompletionStats
#else
    uint32_t
#endif
    on_task_complete(
        PTO2TaskSlotState &slot_state,
#if PTO2_SCHED_PROFILING
        int thread_idx,
#endif

        PTO2LocalReadyBuffer *local_bufs = nullptr
    ) {
#if PTO2_SCHED_PROFILING
        CompletionStats stats = {0, 0, 0, true};
#else
        uint32_t consumer_walk_count = 0;
#endif
#if PTO2_SCHED_PROFILING
        extern uint64_t g_sched_lock_cycle[], g_sched_fanout_cycle[];
        extern uint64_t g_sched_lock_atomic_count[], g_sched_lock_wait_cycle[];
        extern uint64_t g_sched_fanout_atomic_count[], g_sched_push_wait_cycle[];
        PTO2_SCHED_CYCLE_START();
#endif

        slot_state.task_state.store(PTO2_TASK_COMPLETED, std::memory_order_release);
        PTO2DepListEntry *current =
            slot_state.fanout_head.exchange(pto2_fanout_closed_sentinel(), std::memory_order_acq_rel);
        if (pto2_is_fanout_closed(current)) {
            current = nullptr;
        }

#if PTO2_SCHED_PROFILING
        g_sched_lock_atomic_count[thread_idx] += 1;  // task_state.store
        g_sched_lock_wait_cycle[thread_idx] += 0;
        PTO2_SCHED_CYCLE_LAP(g_sched_lock_cycle[thread_idx]);
#endif

        // Fanout: notify consumers. A pre-staged consumer that becomes ready has
        // its doorbell rung INLINE (db = nullptr) the moment its node is reached,
        // not batched to after the whole walk — so a flagged consumer near the
        // front of the list starts immediately and overlaps the remaining
        // release_fanin work for the other consumers, instead of waiting for the
        // full O(fanout-degree) walk (~5us for a 50-consumer producer).
        //
        // Safe on silicon: the producer's slot is already COMPLETED here — every
        // SPMD block has FIN'd AND dcci-flushed its output to HBM before
        // on_task_complete runs — so a released consumer never reads stale
        // producer output. (Batching used to align the released wave, but pushed
        // every doorbell to the end of the walk, defeating the whole point of
        // speculative early-dispatch: minimal producer-end -> consumer-start.)
#if PTO2_SCHED_PROFILING
        uint64_t fanout_atomics = 0, push_wait = 0;
#endif
        // Doorbells for released pre-staged consumers fire INLINE in the walk
        // below; their dispatch_fanin propagation is collected here and replayed
        // after the walk, so no consumer's doorbell waits on a sibling's propagate.
        SpecReleaseSink rel_sink;
        while (current != nullptr) {
            PTO2TaskSlotState &consumer_slot = *current->slot_state;
#if PTO2_SCHED_PROFILING
            stats.fanout_edges++;
            if (release_fanin_and_check_ready(consumer_slot, fanout_atomics, push_wait, local_bufs, &rel_sink)) {
                stats.tasks_enqueued++;
            }
#else
            consumer_walk_count++;
            release_fanin_and_check_ready(consumer_slot, local_bufs, &rel_sink);
#endif
            current = current->next;
        }
        for (int i = 0; i < rel_sink.n; i++) {
            propagate_dispatch_fanin(*rel_sink.items[i]);
        }

#if PTO2_SCHED_PROFILING
        g_sched_fanout_atomic_count[thread_idx] += fanout_atomics;
        g_sched_push_wait_cycle[thread_idx] += push_wait;
        PTO2_SCHED_CYCLE_LAP(g_sched_fanout_cycle[thread_idx]);
        return stats;
#else
        return consumer_walk_count;
#endif
    }

    // === Cold-path API (defined in pto_scheduler.cpp) ===

    // Phase 1: declare every sub-region (ready_queue slots, dummy queue slots)
    // on the supplied arena. dep_pool entries + wiring SPSC buffer are reserved
    // by the orchestrator now (replay_graph stage 1).
    static PTO2SchedulerLayout reserve_layout(DeviceArena &arena);

    // Phase 3a: write everything *except* arena-internal pointer fields.
    // `sm_dev_base` is the device address of the SM (only stored, never
    // dereferenced here). Safe to call on a host arena that holds the
    // prebuilt image buffer. (The orchestrator counterpart takes
    // task_window_size for the task_descriptors address arithmetic; the
    // scheduler only needs the SM header base address,
    // window-size-independent.)
    bool init_data_from_layout(const PTO2SchedulerLayout &layout, DeviceArena &arena, void *sm_dev_base);

    // Phase 3b: write the arena-internal pointer fields
    // (ready_queues[].slots, dummy_ready_queue.slots). Called on both host and
    // device sides. (dep_pool + wiring SPSC buffer moved to the orchestrator.)
    void wire_arena_pointers(const PTO2SchedulerLayout &layout, DeviceArena &arena);

    // Forget per-region pointers; arena owns the backing memory.
    void destroy();
    void print_stats();
    void print_queues();
};

// Scheduler cold-path API is declared as PTO2SchedulerState member functions.
// See init()/destroy()/print_stats()/print_queues() below the struct definition.

// try_inline_complete_locked: short-circuit NotDeferred completions seen during
// drain so they don't grow entries[]. Defined here (not in pto_async_wait.h)
// because PTO2SchedulerState's on_task_complete signature is only known
// after its full definition above.
inline bool
AsyncWaitList::try_inline_complete_locked(AsyncWaitList::DrainCompletionSink &sink, PTO2TaskSlotState &slot_state) {
    // Return value (CompletionStats / consumer-walk count) discarded:
    // async-wait drain path has no Resolve swimlane bar attached.
#if PTO2_SCHED_PROFILING
    (void)sink.sched->on_task_complete(slot_state, sink.thread_idx, sink.local_bufs);
#else
    (void)sink.sched->on_task_complete(slot_state, sink.local_bufs);
#endif
    sink.inline_completed++;
    return true;
}

template <bool Profiling>
inline AsyncPollResult AsyncWaitList::poll_and_complete(
    AICoreCompletionMailbox *aicore_mailbox, PTO2SchedulerState *sched, PTO2LocalReadyBuffer *local_bufs
#if PTO2_SCHED_PROFILING
    ,
    int thread_idx
#endif
) {
    AsyncPollResult result;
    if (!try_lock()) return result;

    AsyncWaitList::DrainCompletionSink sink{};
    sink.sched = sched;
    sink.local_bufs = local_bufs;
#if PTO2_SCHED_PROFILING
    sink.thread_idx = thread_idx;
#endif

    int32_t drain_err = PTO2_ERROR_NONE;
    drain_aicore_completion_mailbox_locked(aicore_mailbox, sink, drain_err);
    if (drain_err != PTO2_ERROR_NONE) {
        result.error_code = drain_err;
        unlock();
        return result;
    }
    result.completed += sink.inline_completed;

    for (int32_t i = count - 1; i >= 0; --i) {
        AsyncWaitEntry &entry = entries[i];
        uintptr_t last_invalidated_counter_line = static_cast<uintptr_t>(-1);
        for (int32_t c = 0; c < entry.condition_count; c++) {
            CompletionCondition &cond = entry.conditions[c];
            if (cond.satisfied) continue;
            if (cond.completion_type == COMPLETION_TYPE_COUNTER && cond.counter_addr != nullptr) {
                uintptr_t counter_line = mailbox_cache_line(cond.counter_addr);
                if (counter_line != last_invalidated_counter_line) {
                    cache_invalidate_range(reinterpret_cast<const void *>(counter_line), sizeof(uint32_t));
                    last_invalidated_counter_line = counter_line;
                }
            }
            CompletionPollResult poll = cond.test();
            if (poll.state == CompletionPollState::FAILED) {
                result.error_code = poll.error_code;
                result.failed_slot_state = entry.slot_state;
                unlock();
                return result;
            }
            if (poll.state == CompletionPollState::READY) {
                cond.satisfied = true;
                cond.retire();
                entry.waiting_completion_count--;
            }
        }

        if (entry.normal_done && entry.waiting_completion_count <= 0) {
            // Return value (CompletionStats / consumer-walk count) discarded:
            // deferred-completion drain has no Resolve swimlane bar attached.
#if PTO2_SCHED_PROFILING
            (void)sched->on_task_complete(*entry.slot_state, thread_idx, local_bufs);
#else
            (void)sched->on_task_complete(*entry.slot_state, local_bufs);
#endif
            result.completed++;

            int32_t last = count - 1;
            if (i != last) entries[i] = entries[last];
            count = last;
        }
    }

    unlock();
    return result;
}

// =============================================================================
// Scheduler Profiling Data
// =============================================================================

#if PTO2_SCHED_PROFILING
struct PTO2SchedProfilingData {
    // Sub-phase cycle breakdown within on_task_complete
    uint64_t lock_cycle;    // task_state store + head snapshot (was lock+store+unlock)
    uint64_t fanout_cycle;  // fanout traversal
    uint64_t fanin_cycle;   // fanin traversal

    // Wait times
    uint64_t lock_wait_cycle;  // always 0 now (was spin-wait in fanout_lock)
    uint64_t push_wait_cycle;  // CAS contention in push()
    uint64_t pop_wait_cycle;   // CAS contention in pop()

    // Atomic counts per sub-phase
    uint64_t lock_atomic_count;
    uint64_t fanout_atomic_count;
    uint64_t fanin_atomic_count;
    uint64_t pop_atomic_count;

    int64_t complete_count;
};

/**
 * Get and reset scheduler profiling data for a specific thread.
 * Returns accumulated profiling data and resets counters.
 */
PTO2SchedProfilingData scheduler_get_profiling(int thread_idx);
#endif
