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
 * PTO Runtime2 - Shared Memory Layout
 *
 * Defines the shared memory structure for Orchestrator-Scheduler communication.
 *
 * Memory Layout (single ring):
 *   +---------------------------+
 *   | SharedMemoryHeader        |  (flow control + sync)
 *   +---------------------------+
 *   | TaskDescriptor[]          |
 *   | TaskPayload[]             |
 *   | TaskSlotState[]           |
 *   +---------------------------+
 *
 * Design principles:
 * - Only data needed for Orchestrator<->Scheduler communication is here
 * - TensorMap, scope_stack, ready_queues, dep_pool are in private memory
 * - Flow control via atomic counters/flags (no locks needed for single-word R/W)
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#pragma once

#include "utils/device_arena.h"
#include "pto_runtime2_types.h"

// =============================================================================
// Shared Memory Header
// =============================================================================

struct PTO2SharedMemoryHandle;

/**
 * Flow control state in shared memory.
 * Written/read by Orchestrator and Scheduler for synchronization. Kept in its
 * own 64B cache line so the Orchestrator-write / Scheduler-read of task_count
 * does not false-share with the layout metadata in the rest of the header.
 */
struct alignas(64) PTO2FlowControl {
    // Written by Orchestrator, Read by Scheduler. Frozen after the orch phase
    // completes to the total task count (single-shot: tasks are bump-allocated
    // once and never reclaimed).
    alignas(64) std::atomic<int32_t> task_count;

    // Per-boot SM reset. PTO2TaskAllocator::init() seeds its private
    // local_task_id_ from initial_local_task_id (default 0 in production)
    // *without* dereferencing task_count — it relies on this reset
    // running on every AICPU boot so 0 stays in sync. If you ever change
    // the initial fc value or the boot ordering, update the default in
    // PTO2TaskAllocator::init (pto_ring_buffer.h) in the same change, or
    // submit IDs will be off by the divergence.
    void init() { task_count.store(0, std::memory_order_relaxed); }

    bool validate(PTO2SharedMemoryHandle *handle) const;
};

static_assert(sizeof(PTO2FlowControl) == 64, "PTO2FlowControl must be exactly 1 cache line (64B)");

/**
 * Shared memory header structure
 *
 * Holds flow control, layout metadata, per-task data pointers, and global
 * sync/error fields. fc is kept first (its own cache line) so the
 * Orchestrator-write / Scheduler-read of task_count stays isolated.
 *
 * Pointers (task_descriptors/task_payloads/slot_states) are host-side only
 * (set by setup_pointers, invalid on device).
 */
struct alignas(PTO2_ALIGN_SIZE) PTO2SharedMemoryHeader {
    // === FLOW CONTROL (own cache line) ===
    PTO2FlowControl fc;

    // === LAYOUT METADATA (set once at init) ===
    uint64_t task_window_size;
    int32_t task_window_mask;
    uint64_t heap_size;
    uint64_t task_descriptors_offset;  // Offset from SM base, in bytes

    // === DATA POINTERS (host-side, set by setup_pointers) ===
    PTO2TaskDescriptor *task_descriptors;
    PTO2TaskPayload *task_payloads;
    PTO2TaskSlotState *slot_states;
    int32_t *task_slot_map;

    int32_t get_slot_by_task_id(int32_t local_task_id) {
        int32_t map_slot = local_task_id & task_window_mask;
        return task_slot_map ? task_slot_map[map_slot] : map_slot;
    }

    PTO2TaskDescriptor &get_task_by_slot(int32_t slot) { return task_descriptors[slot]; }

    PTO2TaskDescriptor &get_task_by_task_id(int32_t local_id) {
        return task_descriptors[get_slot_by_task_id(local_id)];
    }

    PTO2TaskPayload &get_payload_by_slot(int32_t slot) { return task_payloads[slot]; }

    PTO2TaskPayload &get_payload_by_task_id(int32_t local_id) { return task_payloads[get_slot_by_task_id(local_id)]; }

    PTO2TaskSlotState &get_slot_state_by_slot(int32_t slot) { return slot_states[slot]; }

    PTO2TaskSlotState &get_slot_state_by_task_id(int32_t local_id) {
        return slot_states[get_slot_by_task_id(local_id)];
    }

    // === GLOBAL FIELDS ===
    // alignas(64) preserves the historical layout: the fc + layout/pointer block
    // formerly lived inside an alignas(64) ring sub-struct that rounded up to 128
    // bytes, so the global fields started at offset 128. Flattening removed that
    // sub-struct; this alignas restores the same 128-byte boundary. The SM image
    // is shared between the host-prebuilt arena and the AICPU runtime, so these
    // offsets must not move (a mismatch surfaces as an onboard 507018).
    alignas(64) std::atomic<int32_t> orchestrator_done;  // Flag: orchestration complete

    // Total shared memory size (for validation)
    uint64_t total_size;

    // Graph output for copy-back (set by orchestrator when using packed buffer)
    // Host finalize copies from this address instead of dev_ptr when non-zero
    std::atomic<uint64_t> graph_output_ptr;   // Address where final output was written (packed buffer)
    std::atomic<uint64_t> graph_output_size;  // Size in bytes

    // === ERROR REPORTING ===

    // Orchestrator fatal error code (Orchestrator → Scheduler, AICPU → Host)
    // Non-zero signals fatal error. Written by orchestrator, read by scheduler and host.
    std::atomic<int32_t> orch_error_code;

    // Scheduler error state (Scheduler → Host, independent of orchestrator)
    // Written by scheduler threads on timeout; read by orchestrator and host.
    std::atomic<uint32_t> sched_error_bitmap;  // Bit X set = thread X had error
    std::atomic<int32_t> sched_error_code;     // Last scheduler error code (last-writer-wins)
    std::atomic<int32_t> sched_error_thread;   // Thread index of last error writer
};

static_assert(
    (sizeof(PTO2SharedMemoryHeader) % PTO2_ALIGN_SIZE == 0) && (sizeof(PTO2SharedMemoryHeader) < 4096),
    "PTO2SharedMemoryHeader should be reasonably sized"
);

// SM image is shared verbatim between the host-prebuilt arena and the AICPU
// runtime, so flattening the former PTO2SharedMemoryRingHeader sub-struct into
// this header MUST NOT move any field offset (a mismatch surfaces as an onboard
// 507018). Anchor the layout that both sides depend on: fc/task_count at 0,
// task_descriptors_offset at 88, and the global block starting at 128.
static_assert(offsetof(PTO2SharedMemoryHeader, fc) == 0, "fc must be the first member (offset 0)");
static_assert(
    offsetof(PTO2SharedMemoryHeader, fc) + offsetof(PTO2FlowControl, task_count) == 0,
    "task_count must sit at SM offset 0 (host/AICPU task_count_addr depends on it)"
);
static_assert(offsetof(PTO2SharedMemoryHeader, task_descriptors_offset) == 88, "task_descriptors_offset moved");
static_assert(
    offsetof(PTO2SharedMemoryHeader, orchestrator_done) == 128, "global field block must start at offset 128"
);
static_assert(
    offsetof(PTO2SharedMemoryHeader, orch_error_code) == 160,
    "orch_error_code moved (orch_error_code_addr depends on it)"
);

// =============================================================================
// Shared Memory Handle
// =============================================================================

/**
 * Handle for shared memory lifecycle management (create/destroy).
 * Runtime components (orchestrator, scheduler) use PTO2SharedMemoryHeader* directly.
 */
struct PTO2SharedMemoryHandle {
    void *sm_base;     // Base address of shared memory
    uint64_t sm_size;  // Total size of shared memory

    PTO2SharedMemoryHeader *header;

    // Ownership flag
    bool is_owner;  // True if this handle allocated the memory

    // === Static helpers ===

    static uint64_t calculate_size(uint64_t task_window_size);

    // UT convenience: reserve wrapper + sm_base on `arena`, commit, and init
    // using default PTO2_TASK_WINDOW_SIZE / PTO2_HEAP_SIZE. Only valid when the
    // arena is otherwise empty (the call performs the single commit). All
    // memory is owned by the arena — caller must not call destroy().
    static PTO2SharedMemoryHandle *create_and_init_default(DeviceArena &arena);

    // === Instance methods ===

    // In-place init for caller-provided wrapper storage (e.g. a region carved
    // out of a DeviceArena). Sets is_owner = false, calls setup_pointers and
    // init_header. Returns false when `sm_size` is too small for the requested
    // `task_window_size`.
    bool init(void *sm_base, uint64_t sm_size, uint64_t task_window_size, uint64_t heap_size);

    void destroy();
    void print_layout();
    bool validate();

private:
    void init_header(uint64_t task_window_size, uint64_t heap_size);
    void setup_pointers(uint64_t task_window_size);
};

// =============================================================================
// SM Device Layout Helpers
// =============================================================================
//
// When the host pre-builds a runtime-arena image, it needs the device-side
// addresses of several SM sub-fields (flow-control counter,
// task_descriptors arrays, orch_error_code) so it can wire them into the
// orchestrator / scheduler init_data path without dereferencing the SM —
// the SM lives in device memory and cannot be touched from host.
//
// These helpers compute those addresses by offset arithmetic on the SM
// device base. Pure pointer math, no loads/stores; safe to call from host.
// The same arithmetic happens on AICPU too (via PTO2SharedMemoryHandle's
// own setup_pointers), so values are guaranteed consistent across sides.
namespace pto2_sm_layout {

inline std::atomic<int32_t> *orch_error_code_addr(void *sm_dev_base) noexcept {
    return reinterpret_cast<std::atomic<int32_t> *>(
        static_cast<char *>(sm_dev_base) + offsetof(PTO2SharedMemoryHeader, orch_error_code)
    );
}

inline std::atomic<int32_t> *task_count_addr(void *sm_dev_base) noexcept {
    return reinterpret_cast<std::atomic<int32_t> *>(
        static_cast<char *>(sm_dev_base) + offsetof(PTO2SharedMemoryHeader, fc) + offsetof(PTO2FlowControl, task_count)
    );
}

inline int32_t *task_slot_map_addr(void *sm_dev_base, uint64_t task_window_size) noexcept {
    uint64_t off = PTO2_ALIGN_UP(sizeof(PTO2SharedMemoryHeader), PTO2_ALIGN_SIZE);
    off += PTO2_ALIGN_UP(task_window_size * sizeof(PTO2TaskDescriptor), PTO2_ALIGN_SIZE);
    off += PTO2_ALIGN_UP(task_window_size * sizeof(PTO2TaskPayload), PTO2_ALIGN_SIZE);
    off += PTO2_ALIGN_UP(task_window_size * sizeof(PTO2TaskSlotState), PTO2_ALIGN_SIZE);
    return reinterpret_cast<int32_t *>(static_cast<char *>(sm_dev_base) + off);
}

}  // namespace pto2_sm_layout
