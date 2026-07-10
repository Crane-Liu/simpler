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
 * Unit tests for PTO2TaskAllocator from pto_ring_buffer.h
 *
 * replay_graph is single-shot: the arena is filled exactly once and never
 * wraps, and the scheduler does not run during the orch phase so nothing is
 * ever reclaimed. The allocator is therefore a pure bump — running out of
 * either the task window or the heap is a fatal sizing error, not a
 * back-pressure / spin condition.
 *
 * Design contracts (pure bump):
 *
 * - Task slots and heap bytes are handed out monotonically. task_id grows by
 *   1 per alloc; slot == task_id (single-shot never wraps, so no modulo).
 *
 * - heap_available() is simply heap_size - heap_top: no wrap, no reclaim.
 *
 * - Zero-size allocation is a no-op for the heap top, returning the current
 *   position. Two consecutive zero-size allocs return the SAME pointer.
 *
 * - Exceeding the task window -> fatal (PTO2_ERROR_FLOW_CONTROL_DEADLOCK).
 *   Exceeding the heap -> fatal (PTO2_ERROR_HEAP_RING_DEADLOCK).
 */

#include <gtest/gtest.h>

#include <atomic>
#include <climits>
#include <cstring>
#include <set>

#include "pto_ring_buffer.h"

// =============================================================================
// Fixture
// =============================================================================

class TaskAllocatorTest : public ::testing::Test {
protected:
    static constexpr int32_t WINDOW_SIZE = 16;
    static constexpr uint64_t HEAP_SIZE = 4096;

    alignas(64) uint8_t heap_buf[HEAP_SIZE]{};
    std::atomic<int32_t> current_index{0};
    std::atomic<int32_t> error_code{PTO2_ERROR_NONE};
    PTO2TaskAllocator allocator{};

    void SetUp() override {
        std::memset(heap_buf, 0, sizeof(heap_buf));
        current_index.store(0);
        error_code.store(PTO2_ERROR_NONE);
        allocator.init(WINDOW_SIZE, &current_index, heap_buf, HEAP_SIZE, &error_code);
    }
};

// =============================================================================
// Normal path
// =============================================================================

TEST_F(TaskAllocatorTest, InitialState) {
    EXPECT_EQ(allocator.window_size(), WINDOW_SIZE);
    EXPECT_EQ(allocator.active_count(), 0);
    EXPECT_EQ(allocator.heap_top(), 0u);
    EXPECT_EQ(allocator.heap_capacity(), HEAP_SIZE);
    EXPECT_EQ(allocator.heap_available(), HEAP_SIZE);
    EXPECT_EQ(allocator.task_tail(), 0);
    EXPECT_EQ(allocator.task_head(), 0);
    EXPECT_EQ(allocator.heap_tail(), 0u);
}

TEST_F(TaskAllocatorTest, AllocNonZeroSize) {
    auto result = allocator.alloc(100);
    ASSERT_FALSE(result.failed());
    EXPECT_EQ(result.task_id, 0);
    EXPECT_EQ(result.slot, 0);
    EXPECT_NE(result.packed_base, nullptr);
    // 100 bytes aligned up to PTO2_ALIGN_SIZE (64) = 128
    uint64_t expected_aligned = PTO2_ALIGN_UP(100u, PTO2_ALIGN_SIZE);
    EXPECT_EQ(expected_aligned, 128u);
    EXPECT_EQ(allocator.heap_top(), expected_aligned);
    EXPECT_EQ(
        static_cast<char *>(result.packed_end) - static_cast<char *>(result.packed_base),
        static_cast<ptrdiff_t>(expected_aligned)
    );
}

TEST_F(TaskAllocatorTest, SequentialTaskIds) {
    int32_t prev_id = -1;
    for (int i = 0; i < 5; i++) {
        auto result = allocator.alloc(0);
        ASSERT_FALSE(result.failed()) << "Alloc failed at i=" << i;
        EXPECT_EQ(result.task_id, prev_id + 1) << "Task IDs must be monotonically increasing";
        EXPECT_EQ(result.slot, result.task_id) << "slot == task_id (single-shot never wraps)";
        prev_id = result.task_id;
    }
    EXPECT_EQ(allocator.active_count(), 5);
    EXPECT_EQ(allocator.task_head(), 5);
    EXPECT_EQ(current_index.load(), 5) << "current_index published to shared memory";
}

TEST_F(TaskAllocatorTest, OutputSizeAlignment) {
    // 1 byte -> aligned to 64
    auto r1 = allocator.alloc(1);
    ASSERT_FALSE(r1.failed());
    EXPECT_EQ(allocator.heap_top(), 64u);

    // Another 33 bytes -> aligned to 64, total 128
    auto r2 = allocator.alloc(33);
    ASSERT_FALSE(r2.failed());
    EXPECT_EQ(allocator.heap_top(), 128u);

    // Exactly 64 bytes -> stays 64, total 192
    auto r3 = allocator.alloc(64);
    ASSERT_FALSE(r3.failed());
    EXPECT_EQ(allocator.heap_top(), 192u);
}

TEST_F(TaskAllocatorTest, SlotEqualsTaskId) {
    // Single-shot fill: every slot is the task id itself, distinct across the
    // window (no modulo, no reuse).
    std::set<int32_t> slots;
    for (int i = 0; i < WINDOW_SIZE - 1; i++) {
        auto r = allocator.alloc(0);
        ASSERT_FALSE(r.failed());
        EXPECT_EQ(r.slot, r.task_id);
        slots.insert(r.slot);
    }
    EXPECT_EQ(slots.size(), static_cast<size_t>(WINDOW_SIZE - 1)) << "Every slot is allocated exactly once";
}

// Heap bump advances monotonically; heap_available shrinks by the aligned size.
TEST_F(TaskAllocatorTest, HeapBumpMonotonic) {
    EXPECT_EQ(allocator.heap_available(), HEAP_SIZE);

    auto r1 = allocator.alloc(256);
    ASSERT_FALSE(r1.failed());
    EXPECT_EQ(allocator.heap_top(), 256u);
    EXPECT_EQ(allocator.heap_available(), HEAP_SIZE - 256u);

    auto r2 = allocator.alloc(128);
    ASSERT_FALSE(r2.failed());
    EXPECT_EQ(allocator.heap_top(), 384u);
    EXPECT_EQ(allocator.heap_available(), HEAP_SIZE - 384u);
    // Buffers do not overlap and are contiguous.
    EXPECT_EQ(r2.packed_base, static_cast<char *>(r1.packed_end));
}

// =============================================================================
// Boundary conditions
// =============================================================================

TEST_F(TaskAllocatorTest, HeapExactFitAtEnd) {
    // Allocate 4032 bytes to leave exactly 64 at end.
    auto r1 = allocator.alloc(HEAP_SIZE - 64);
    ASSERT_FALSE(r1.failed());
    EXPECT_EQ(allocator.heap_top(), HEAP_SIZE - 64u);

    auto r2 = allocator.alloc(64);
    ASSERT_FALSE(r2.failed());
    EXPECT_EQ(allocator.heap_top(), HEAP_SIZE);
    EXPECT_EQ(static_cast<char *>(r2.packed_base), reinterpret_cast<char *>(heap_buf) + HEAP_SIZE - 64);
    EXPECT_EQ(allocator.heap_available(), 0u);
}

// Zero-size allocs return the same address and don't advance the top.
TEST_F(TaskAllocatorTest, ZeroSizeAllocationAliased) {
    auto r1 = allocator.alloc(0);
    auto r2 = allocator.alloc(0);
    ASSERT_FALSE(r1.failed());
    ASSERT_FALSE(r2.failed());

    EXPECT_EQ(r1.packed_base, r2.packed_base) << "Zero-size allocs return same address";
    EXPECT_EQ(r1.packed_base, r1.packed_end) << "packed_end == packed_base for zero-size";
    EXPECT_EQ(allocator.heap_top(), 0u) << "top doesn't advance for zero-size allocs";
}

// =============================================================================
// Overflow is fatal (no back-pressure, no spin)
// =============================================================================

TEST_F(TaskAllocatorTest, AllocExactlyHeapSize) {
    auto r1 = allocator.alloc(HEAP_SIZE);
    ASSERT_FALSE(r1.failed());
    EXPECT_EQ(r1.packed_base, static_cast<void *>(heap_buf));
    EXPECT_EQ(allocator.heap_top(), HEAP_SIZE);

    auto r2 = allocator.alloc(64);
    EXPECT_TRUE(r2.failed()) << "No space after full allocation";
    EXPECT_EQ(error_code.load(), PTO2_ERROR_HEAP_RING_DEADLOCK);
}

TEST_F(TaskAllocatorTest, AllocLargerThanHeap) {
    auto r = allocator.alloc(HEAP_SIZE * 2);
    EXPECT_TRUE(r.failed()) << "Cannot allocate more than heap size";
    EXPECT_EQ(error_code.load(), PTO2_ERROR_HEAP_RING_DEADLOCK);
}

TEST_F(TaskAllocatorTest, HeapOverflowDoesNotAdvanceTop) {
    auto r1 = allocator.alloc(HEAP_SIZE - 64);
    ASSERT_FALSE(r1.failed());
    uint64_t top_before = allocator.heap_top();

    // 128 needed but only 64 left -> fatal, top unchanged.
    auto r2 = allocator.alloc(128);
    EXPECT_TRUE(r2.failed());
    EXPECT_EQ(error_code.load(), PTO2_ERROR_HEAP_RING_DEADLOCK);
    EXPECT_EQ(allocator.heap_top(), top_before) << "Failed alloc must not advance the heap top";
}

TEST_F(TaskAllocatorTest, TaskWindowSaturates) {
    for (int i = 0; i < WINDOW_SIZE - 1; i++) {
        auto r = allocator.alloc(0);
        ASSERT_FALSE(r.failed()) << "Alloc failed at i=" << i;
        EXPECT_EQ(r.task_id, i);
    }
    EXPECT_EQ(allocator.active_count(), WINDOW_SIZE - 1);

    auto overflow = allocator.alloc(0);
    EXPECT_TRUE(overflow.failed());
    EXPECT_EQ(error_code.load(), PTO2_ERROR_FLOW_CONTROL_DEADLOCK);
    EXPECT_EQ(allocator.active_count(), WINDOW_SIZE - 1) << "Failed alloc must not bump the task counter";
}

// Single-shot bounds task_id by the window, so a seed must stay below
// window_size. Allocation resumes from the seed and slot mirrors task_id
// directly (no modulo). The window guard still fires once head + 1 == window.
TEST_F(TaskAllocatorTest, TaskIdNonZeroSeed) {
    constexpr int32_t SEED = 10;
    current_index.store(SEED);
    allocator.init(
        WINDOW_SIZE, &current_index, heap_buf, HEAP_SIZE, &error_code,
        /*initial_local_task_id=*/SEED
    );

    auto r1 = allocator.alloc(0);
    ASSERT_FALSE(r1.failed());
    EXPECT_EQ(r1.task_id, SEED);
    EXPECT_EQ(r1.slot, SEED) << "slot == task_id (single-shot never wraps)";
    EXPECT_EQ(current_index.load(), SEED + 1);

    auto r2 = allocator.alloc(0);
    ASSERT_FALSE(r2.failed());
    EXPECT_EQ(r2.task_id, SEED + 1);
    EXPECT_EQ(r2.slot, SEED + 1);

    // head is now SEED+2 == 12; the guard rejects once head + 1 >= 16,
    // i.e. when head reaches 15. Allocate up to head==15, the last success.
    for (int32_t head = 12; head < 15; head++) {
        auto r = allocator.alloc(0);  // head 12->13, 13->14, 14->15
        ASSERT_FALSE(r.failed()) << "alloc failed early at head=" << head;
    }
    EXPECT_EQ(allocator.task_head(), 15);

    auto overflow = allocator.alloc(0);  // head 15, 15 + 1 >= 16 -> fatal
    EXPECT_TRUE(overflow.failed());
    EXPECT_EQ(error_code.load(), PTO2_ERROR_FLOW_CONTROL_DEADLOCK);
}
