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
 * PTO Runtime2 - Shared Memory Implementation
 *
 * Implements shared memory allocation, initialization, and management
 * for Orchestrator-Scheduler communication.
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#include "pto_shared_memory.h"
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include "common/unified_log.h"

// =============================================================================
// Size Calculation
// =============================================================================

uint64_t PTO2SharedMemoryHandle::calculate_size(uint64_t task_window_size) {
    uint64_t size = 0;

    // Header (aligned to cache line)
    size += PTO2_ALIGN_UP(sizeof(PTO2SharedMemoryHeader), PTO2_ALIGN_SIZE);

    // Task descriptors, payloads, and slot states
    size += PTO2_ALIGN_UP(task_window_size * sizeof(PTO2TaskDescriptor), PTO2_ALIGN_SIZE);
    size += PTO2_ALIGN_UP(task_window_size * sizeof(PTO2TaskPayload), PTO2_ALIGN_SIZE);
    size += PTO2_ALIGN_UP(task_window_size * sizeof(PTO2TaskSlotState), PTO2_ALIGN_SIZE);
    size += PTO2_ALIGN_UP(task_window_size * sizeof(int32_t), PTO2_ALIGN_SIZE);

    return size;
}

// =============================================================================
// Creation and Destruction
// =============================================================================

void PTO2SharedMemoryHandle::setup_pointers(uint64_t task_window_size) {
    char *ptr = (char *)sm_base;

    // Header
    header = (PTO2SharedMemoryHeader *)ptr;
    ptr += PTO2_ALIGN_UP(sizeof(PTO2SharedMemoryHeader), PTO2_ALIGN_SIZE);

    // Task descriptors, payloads, and slot states
    header->task_descriptors = (PTO2TaskDescriptor *)ptr;
    ptr += PTO2_ALIGN_UP(task_window_size * sizeof(PTO2TaskDescriptor), PTO2_ALIGN_SIZE);

    header->task_payloads = (PTO2TaskPayload *)ptr;
    ptr += PTO2_ALIGN_UP(task_window_size * sizeof(PTO2TaskPayload), PTO2_ALIGN_SIZE);

    header->slot_states = (PTO2TaskSlotState *)ptr;
    ptr += PTO2_ALIGN_UP(task_window_size * sizeof(PTO2TaskSlotState), PTO2_ALIGN_SIZE);

    header->task_slot_map = (int32_t *)ptr;
    ptr += PTO2_ALIGN_UP(task_window_size * sizeof(int32_t), PTO2_ALIGN_SIZE);
}

bool PTO2SharedMemoryHandle::init(
    void *sm_base_arg, uint64_t sm_size_arg, uint64_t task_window_size, uint64_t heap_size
) {
    if (!sm_base_arg || sm_size_arg == 0) return false;
    if (sm_size_arg < calculate_size(task_window_size)) return false;

    sm_base = sm_base_arg;
    sm_size = sm_size_arg;
    is_owner = false;
    setup_pointers(task_window_size);
    init_header(task_window_size, heap_size);
    return true;
}

PTO2SharedMemoryHandle *PTO2SharedMemoryHandle::create_and_init_default(DeviceArena &arena) {
    const uint64_t buffer_size = calculate_size(PTO2_TASK_WINDOW_SIZE);
    const size_t off_handle = arena.reserve(sizeof(PTO2SharedMemoryHandle), alignof(PTO2SharedMemoryHandle));
    const size_t off_buffer = arena.reserve(static_cast<size_t>(buffer_size), PTO2_ALIGN_SIZE);
    if (arena.commit() == nullptr) return nullptr;

    auto *handle = static_cast<PTO2SharedMemoryHandle *>(arena.region_ptr(off_handle));
    memset(handle, 0, sizeof(*handle));
    void *buffer = arena.region_ptr(off_buffer);
    memset(buffer, 0, static_cast<size_t>(buffer_size));
    if (!handle->init(buffer, buffer_size, PTO2_TASK_WINDOW_SIZE, PTO2_HEAP_SIZE)) return nullptr;
    return handle;
}

void PTO2SharedMemoryHandle::destroy() {
    // Arena-owned wrappers (is_owner == false) are reclaimed by arena.release();
    // calling destroy on them is a no-op so existing callers stay safe.
    if (is_owner && sm_base) {
        free(sm_base);
        free(this);
    }
}

// =============================================================================
// Initialization
// =============================================================================
//
// no need init data in pool, init pool data when used
void PTO2SharedMemoryHandle::init_header(uint64_t task_window_size, uint64_t heap_size) {
    // Flow control (start at 0)
    header->fc.init();

    header->orchestrator_done.store(0, std::memory_order_relaxed);

    // Layout info
    uint64_t offset = PTO2_ALIGN_UP(sizeof(PTO2SharedMemoryHeader), PTO2_ALIGN_SIZE);
    header->task_window_size = task_window_size;
    header->task_window_mask = static_cast<int32_t>(task_window_size - 1);
    header->heap_size = heap_size;
    header->task_descriptors_offset = offset;

    header->total_size = sm_size;
    header->graph_output_ptr.store(0, std::memory_order_relaxed);
    header->graph_output_size.store(0, std::memory_order_relaxed);

    // Error reporting
    header->orch_error_code.store(PTO2_ERROR_NONE, std::memory_order_relaxed);
    header->sched_error_bitmap.store(0, std::memory_order_relaxed);
    header->sched_error_code.store(PTO2_ERROR_NONE, std::memory_order_relaxed);
    header->sched_error_thread.store(-1, std::memory_order_relaxed);

    // Per-slot one-time init. Lives here (not in scheduler init) because it
    // writes SM-side slot_states[], so host-side prebuilt-arena init skips
    // all SM dereferences. The single-shot replay model fills the window exactly
    // once and never reuses a slot, so this one-time init of the dynamic
    // scheduling fields leaves each slot ready for its single allocation.
    // bind_ring() pins ring_id (slot-invariant); payload spec fields are (re)set
    // by PTO2TaskPayload::init on every submit.
    for (uint64_t i = 0; i < task_window_size; i++) {
        PTO2TaskSlotState &s = header->slot_states[i];
        s.bind_ring(0);
        s.fanout_head.store(nullptr, std::memory_order_relaxed);
        s.fanin_refcount.store(0, std::memory_order_relaxed);
        s.completed_subtasks.store(0, std::memory_order_relaxed);
        s.next_block_idx.store(0, std::memory_order_relaxed);
        s.any_subtask_deferred.store(false, std::memory_order_relaxed);
        s.fanin_count = 0;
        s.active_mask = ActiveMask{};
        s.task_state.store(PTO2_TASK_PENDING, std::memory_order_relaxed);
        header->task_slot_map[i] = static_cast<int32_t>(i);
    }
}

// =============================================================================
// Debug Utilities
// =============================================================================

void PTO2SharedMemoryHandle::print_layout() {
    if (!header) return;

    PTO2SharedMemoryHeader *h = header;

    LOG_INFO_V0("=== PTO2 Shared Memory Layout ===");
    LOG_INFO_V0("Base address:       %p", sm_base);
    LOG_INFO_V0("Total size:         %" PRIu64 " bytes", h->total_size);
    LOG_INFO_V0("Ring:");
    LOG_INFO_V0("  task_window_size: %" PRIu64, h->task_window_size);
    LOG_INFO_V0("  heap_size:        %" PRIu64 " bytes", h->heap_size);
    LOG_INFO_V0(
        "  descriptors_off:  %" PRIu64 " (0x%" PRIx64 ")", h->task_descriptors_offset, h->task_descriptors_offset
    );
    LOG_INFO_V0("  task_count:       %d", h->fc.task_count.load(std::memory_order_acquire));
    LOG_INFO_V0("orchestrator_done:  %d", h->orchestrator_done.load(std::memory_order_acquire));
    LOG_INFO_V0("Error state:");
    LOG_INFO_V0("  orch_error_code:    %d", h->orch_error_code.load(std::memory_order_relaxed));
    LOG_INFO_V0("  sched_error_bitmap: 0x%x", h->sched_error_bitmap.load(std::memory_order_relaxed));
    LOG_INFO_V0("  sched_error_code:   %d", h->sched_error_code.load(std::memory_order_relaxed));
    LOG_INFO_V0("  sched_error_thread: %d", h->sched_error_thread.load(std::memory_order_relaxed));
    LOG_INFO_V0("================================");
}

bool PTO2SharedMemoryHandle::validate() {
    if (!sm_base) return false;
    if (!header) return false;

    PTO2SharedMemoryHeader *h = header;

    if (!h->fc.validate(this)) return false;

    return true;
}

bool PTO2FlowControl::validate(PTO2SharedMemoryHandle *handle) const {
    if (!handle) return false;
    if (!handle->header) return false;

    const PTO2SharedMemoryHeader *h = handle->header;

    // Check that offsets are within bounds
    if (h->task_descriptors_offset >= h->total_size) return false;

    // Check pointer alignment
    if ((uintptr_t)h->task_descriptors % PTO2_ALIGN_SIZE != 0) return false;

    // Check flow control pointer sanity
    int32_t current = task_count.load(std::memory_order_acquire);
    if (current < 0) return false;

    return true;
}
