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

#ifndef PTO_RING_BUFFER_H
#define PTO_RING_BUFFER_H

#include <inttypes.h>

#include "common/unified_log.h"
#include "pto_runtime2_types.h"

// Historical filename: replay_graph allocates linearly and never wraps.
class PTO2TaskAllocator {
public:
    void init(
        int32_t window_size, std::atomic<int32_t> *task_count_ptr, void *heap_base, uint64_t heap_size,
        std::atomic<int32_t> *error_code_ptr, int32_t initial_task_id = 0
    ) {
        window_size_ = window_size;
        task_count_ptr_ = task_count_ptr;
        heap_base_ = heap_base;
        heap_size_ = heap_size;
        error_code_ptr_ = error_code_ptr;
        task_count_ = initial_task_id;
        heap_top_ = 0;
    }

    PTO2TaskAllocResult alloc(int32_t output_size) {
        uint64_t aligned = output_size > 0 ? PTO2_ALIGN_UP(static_cast<uint64_t>(output_size), PTO2_ALIGN_SIZE) : 0;
        if (task_count_ + 1 >= window_size_) {
            report_task_overflow();
            return {-1, -1, nullptr, nullptr};
        }
        if (heap_top_ + aligned > heap_size_) {
            report_heap_overflow(output_size);
            return {-1, -1, nullptr, nullptr};
        }

        void *base = static_cast<char *>(heap_base_) + heap_top_;
        heap_top_ += aligned;
        int32_t task_id = task_count_++;
        task_count_ptr_->store(task_count_, std::memory_order_release);
        return {task_id, task_id, base, static_cast<char *>(base) + aligned};
    }

    int32_t active_count() const { return task_count_; }
    int32_t task_tail() const { return 0; }
    int32_t task_head() const { return task_count_; }
    int32_t window_size() const { return window_size_; }
    uint64_t heap_available() const { return heap_size_ - heap_top_; }
    uint64_t heap_top() const { return heap_top_; }
    uint64_t heap_tail() const { return 0; }
    uint64_t heap_capacity() const { return heap_size_; }
    uint64_t heap_used_bytes() const { return heap_top_; }

private:
    int32_t window_size_{0};
    std::atomic<int32_t> *task_count_ptr_{nullptr};
    void *heap_base_{nullptr};
    uint64_t heap_size_{0};
    std::atomic<int32_t> *error_code_ptr_{nullptr};
    int32_t task_count_{0};
    uint64_t heap_top_{0};

    void report_task_overflow() {
        LOG_ERROR("========================================");
        LOG_ERROR("FATAL: Replay Graph Task Window Overflow!");
        LOG_ERROR("========================================");
        LOG_ERROR("Whole graph needs more task slots: current=%d, capacity=%d.", task_count_, window_size_ - 1);
        LOG_ERROR("Increase PTO2_RING_TASK_WINDOW to a power of two larger than the complete graph.");
        LOG_ERROR("========================================");
        if (error_code_ptr_) {
            error_code_ptr_->store(PTO2_ERROR_FLOW_CONTROL_DEADLOCK, std::memory_order_release);
        }
    }

    void report_heap_overflow(int32_t requested) {
        LOG_ERROR("========================================");
        LOG_ERROR("FATAL: Replay Graph Heap Overflow!");
        LOG_ERROR("========================================");
        LOG_ERROR(
            "Whole graph output storage exceeds heap: top=%" PRIu64 ", size=%" PRIu64 ", requested=%d.", heap_top_,
            heap_size_, requested
        );
        LOG_ERROR("Increase PTO2_RING_HEAP to hold all graph outputs.");
        LOG_ERROR("========================================");
        if (error_code_ptr_) {
            error_code_ptr_->store(PTO2_ERROR_HEAP_RING_DEADLOCK, std::memory_order_release);
        }
    }
};

struct PTO2DepListPool {
    PTO2DepListEntry *base{nullptr};
    int32_t capacity{0};
    int32_t top{1};
    int32_t high_water{0};
    std::atomic<int32_t> *error_code_ptr{nullptr};

    void init(PTO2DepListEntry *in_base, int32_t in_capacity, std::atomic<int32_t> *in_error_code_ptr) {
        base = in_base;
        capacity = in_capacity;
        top = 1;
        high_water = 0;
        error_code_ptr = in_error_code_ptr;
        base[0].slot_state = nullptr;
        base[0].next = nullptr;
    }

    PTO2DepListEntry *alloc() {
        if (top >= capacity) {
            LOG_ERROR("========================================");
            LOG_ERROR("FATAL: Replay Graph Dependency Pool Overflow!");
            LOG_ERROR("========================================");
            LOG_ERROR("Whole graph needs at least %d fanout entries; capacity=%d.", top, capacity - 1);
            LOG_ERROR("Increase PTO2_RING_DEP_POOL to hold every graph edge.");
            LOG_ERROR("========================================");
            if (error_code_ptr) {
                error_code_ptr->store(PTO2_ERROR_DEP_POOL_OVERFLOW, std::memory_order_release);
            }
            return nullptr;
        }
        PTO2DepListEntry *entry = &base[top++];
        if (used() > high_water) high_water = used();
        return entry;
    }

    PTO2DepListEntry *prepend(PTO2DepListEntry *head, PTO2TaskSlotState *slot_state) {
        PTO2DepListEntry *entry = alloc();
        if (!entry) return nullptr;
        entry->slot_state = slot_state;
        entry->next = head;
        return entry;
    }

    int32_t used() const { return top - 1; }
    int32_t available() const { return capacity - top; }
};

#endif  // PTO_RING_BUFFER_H
