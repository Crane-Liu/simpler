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

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <vector>

#include "utils/device_arena.h"
#include "pto_orchestrator.h"
#include "pto_ring_buffer.h"
#include "pto_shared_memory.h"

TEST(ReplayGraphAllocator, HeapOverflowIsFatalWithoutWrap) {
    alignas(64) uint8_t heap[128]{};
    std::atomic<int32_t> task_count{0};
    std::atomic<int32_t> error{PTO2_ERROR_NONE};
    PTO2TaskAllocator allocator;
    allocator.init(8, &task_count, heap, sizeof(heap), &error);

    auto first = allocator.alloc(128);
    ASSERT_FALSE(first.failed());
    EXPECT_EQ(first.slot, first.task_id);
    EXPECT_EQ(allocator.heap_top(), sizeof(heap));

    auto overflow = allocator.alloc(64);
    EXPECT_TRUE(overflow.failed());
    EXPECT_EQ(error.load(), PTO2_ERROR_HEAP_RING_DEADLOCK);
    EXPECT_EQ(allocator.heap_top(), sizeof(heap));
    EXPECT_EQ(task_count.load(), 1);
}

TEST(ReplayGraphAllocator, TaskWindowIsNeverReused) {
    alignas(64) uint8_t heap[64]{};
    std::atomic<int32_t> task_count{0};
    std::atomic<int32_t> error{PTO2_ERROR_NONE};
    PTO2TaskAllocator allocator;
    allocator.init(4, &task_count, heap, sizeof(heap), &error);

    for (int32_t expected = 0; expected < 3; expected++) {
        auto result = allocator.alloc(0);
        ASSERT_FALSE(result.failed());
        EXPECT_EQ(result.task_id, expected);
        EXPECT_EQ(result.slot, expected);
    }
    EXPECT_TRUE(allocator.alloc(0).failed());
    EXPECT_EQ(error.load(), PTO2_ERROR_FLOW_CONTROL_DEADLOCK);
    EXPECT_EQ(task_count.load(), 3);
}

TEST(ReplayGraphDepPool, CapacityMustHoldTheCompleteGraph) {
    PTO2DepListEntry entries[4]{};
    std::atomic<int32_t> error{PTO2_ERROR_NONE};
    PTO2DepListPool pool;
    pool.init(entries, 4, &error);

    EXPECT_NE(pool.alloc(), nullptr);
    EXPECT_NE(pool.alloc(), nullptr);
    EXPECT_NE(pool.alloc(), nullptr);
    EXPECT_EQ(pool.used(), 3);
    EXPECT_EQ(pool.alloc(), nullptr);
    EXPECT_EQ(error.load(), PTO2_ERROR_DEP_POOL_OVERFLOW);
    EXPECT_EQ(pool.used(), 3);
}

class ReplayGraphOrchestratorTest : public ::testing::Test {
protected:
    static constexpr uint64_t kTaskWindow = 16;
    static constexpr uint64_t kHeapSize = 4096;
    static constexpr int32_t kDepPoolCapacity = 64;

    DeviceArena sm_arena;
    DeviceArena runtime_arena;
    PTO2SharedMemoryHandle *sm_handle{nullptr};
    PTO2OrchestratorState orch{};
    PTO2SchedulerState sched{};
    PTO2OrchestratorLayout orch_layout{};
    PTO2SchedulerLayout sched_layout{};
    std::vector<char> gm_heap;

    void SetUp() override {
        size_t handle_off = sm_arena.reserve(sizeof(PTO2SharedMemoryHandle), alignof(PTO2SharedMemoryHandle));
        uint64_t sm_size = PTO2SharedMemoryHandle::calculate_size(kTaskWindow);
        size_t buffer_off = sm_arena.reserve(sm_size, PTO2_ALIGN_SIZE);
        ASSERT_NE(sm_arena.commit(), nullptr);
        sm_handle = static_cast<PTO2SharedMemoryHandle *>(sm_arena.region_ptr(handle_off));
        ASSERT_TRUE(sm_handle->init(sm_arena.region_ptr(buffer_off), sm_size, kTaskWindow, kHeapSize));

        gm_heap.resize(kHeapSize);
        orch_layout = PTO2OrchestratorState::reserve_layout(runtime_arena, kTaskWindow, kDepPoolCapacity);
        sched_layout = PTO2SchedulerState::reserve_layout(runtime_arena);
        ASSERT_NE(runtime_arena.commit(), nullptr);
        ASSERT_TRUE(orch.init_data_from_layout(
            orch_layout, runtime_arena, sm_handle->sm_base, gm_heap.data(), kHeapSize, kTaskWindow
        ));
        ASSERT_TRUE(sched.init_data_from_layout(sched_layout, runtime_arena, sm_handle->sm_base));
        sched.wire_arena_pointers(sched_layout, runtime_arena);
        orch.wire_arena_pointers(orch_layout, runtime_arena, &sched);
    }

    void TearDown() override {
        orch.destroy();
        sched.destroy();
        runtime_arena.release();
        sm_arena.release();
    }
};

TEST_F(ReplayGraphOrchestratorTest, SubmitBuildsExactFrozenFaninWithoutSentinel) {
    orch.begin_scope();

    L0TaskArgs producer_args;
    TaskOutputTensors producer = orch.submit_dummy_task(producer_args);
    ASSERT_TRUE(producer.task_id().is_valid());
    EXPECT_EQ(orch.initial_ready_count, 1);

    PTO2TaskId duplicate_deps[] = {producer.task_id(), producer.task_id()};
    L0TaskArgs consumer_args;
    consumer_args.set_dependencies(duplicate_deps, 2);
    TaskOutputTensors consumer = orch.submit_dummy_task(consumer_args);
    ASSERT_TRUE(consumer.task_id().is_valid());

    auto &producer_slot = sm_handle->header->get_slot_state_by_task_id(producer.task_id().local());
    auto &consumer_slot = sm_handle->header->get_slot_state_by_task_id(consumer.task_id().local());
    EXPECT_EQ(producer_slot.fanin_count, 0);
    EXPECT_EQ(consumer_slot.fanin_count, 1);
    EXPECT_EQ(consumer_slot.fanin_refcount.load(), 0);
    EXPECT_EQ(orch.initial_ready_count, 1);
    ASSERT_NE(producer_slot.fanout_head, nullptr);
    EXPECT_EQ(producer_slot.fanout_head->slot_state, &consumer_slot);
    EXPECT_EQ(producer_slot.fanout_head->next, nullptr);
}
