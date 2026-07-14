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
#include "pto_graph_cache.h"
#include "pto_orchestrator.h"
#include "pto_ring_buffer.h"
#include "pto_shared_memory.h"

TEST(ReplayGraphCacheKey, TracksMetadataAndScalarsButNotBoundaryAddress) {
    uint32_t shape[] = {2, 4};
    PTO2GraphBindings first;
    first.tensor_count = 1;
    first.scalar_count = 1;
    first.tensors[0] = make_tensor_external(reinterpret_cast<void *>(0x1000), shape, 2, DataType::FLOAT16);
    first.scalars[0] = 7;

    PTO2GraphBindings relocated = first;
    relocated.tensors[0].buffer.addr = 0x2000;
    uint64_t namespace_hash = PTO2_GRAPH_KEY("ut_graph_cache_key_v1");
    uint64_t key = rt_graph_make_key(namespace_hash, first);
    EXPECT_EQ(key, rt_graph_make_key(namespace_hash, relocated));

    PTO2GraphBindings different_scalar = first;
    different_scalar.scalars[0] = 8;
    EXPECT_NE(key, rt_graph_make_key(namespace_hash, different_scalar));

    uint32_t different_shape[] = {4, 4};
    PTO2GraphBindings different_metadata = first;
    different_metadata.tensors[0] =
        make_tensor_external(reinterpret_cast<void *>(0x1000), different_shape, 2, DataType::FLOAT16);
    EXPECT_NE(key, rt_graph_make_key(namespace_hash, different_metadata));
    EXPECT_NE(key, rt_graph_make_key(PTO2_GRAPH_KEY("ut_graph_cache_key_v2"), first));
}

TEST(ReplayGraphAllocator, HeapOverflowIsFatalWithoutWrap) {
    alignas(64) uint8_t heap[256]{};
    int32_t task_slot_map[8]{};
    std::atomic<int32_t> task_count{0};
    std::atomic<int32_t> error{PTO2_ERROR_NONE};
    PTO2TaskAllocator allocator;
    allocator.init(8, &task_count, task_slot_map, heap, sizeof(heap), &error);

    auto first = allocator.alloc(128);
    ASSERT_FALSE(first.failed());
    EXPECT_EQ(first.slot, first.task_id);
    EXPECT_EQ(allocator.heap_top(), sizeof(heap) / 2);

    auto overflow = allocator.alloc(64);
    EXPECT_TRUE(overflow.failed());
    EXPECT_EQ(error.load(), PTO2_ERROR_HEAP_RING_DEADLOCK);
    EXPECT_EQ(allocator.heap_top(), sizeof(heap) / 2);
    EXPECT_EQ(task_count.load(), 1);
}

TEST(ReplayGraphAllocator, MapsDenseTaskIdsAcrossPingPongArenas) {
    alignas(64) uint8_t heap[64]{};
    int32_t task_slot_map[4]{};
    std::atomic<int32_t> task_count{0};
    std::atomic<int32_t> error{PTO2_ERROR_NONE};
    PTO2TaskAllocator allocator;
    allocator.init(4, &task_count, task_slot_map, heap, sizeof(heap), &error);

    for (int32_t expected = 0; expected < 2; expected++) {
        auto result = allocator.alloc(0);
        ASSERT_FALSE(result.failed());
        EXPECT_EQ(result.task_id, expected);
        EXPECT_EQ(result.slot, expected);
    }
    EXPECT_TRUE(allocator.alloc(0).failed());
    EXPECT_EQ(error.load(), PTO2_ERROR_FLOW_CONTROL_DEADLOCK);
    EXPECT_EQ(task_count.load(), 2);

    error.store(PTO2_ERROR_NONE);
    allocator.begin_buffer(1);
    auto third = allocator.alloc(0);
    auto fourth = allocator.alloc(0);
    ASSERT_FALSE(third.failed());
    ASSERT_FALSE(fourth.failed());
    EXPECT_EQ(third.task_id, 2);
    EXPECT_EQ(third.slot, 2);
    EXPECT_EQ(fourth.task_id, 3);
    EXPECT_EQ(fourth.slot, 3);
    EXPECT_EQ(task_slot_map[2], 2);
    EXPECT_EQ(task_slot_map[3], 3);
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

TEST_F(ReplayGraphOrchestratorTest, SubmitBuildsFaninWithPublishGate) {
    orch.begin_scope();

    L0TaskArgs producer_args;
    TaskOutputTensors producer = orch.submit_dummy_task(producer_args);
    ASSERT_TRUE(producer.task_id().is_valid());

    PTO2TaskId duplicate_deps[] = {producer.task_id(), producer.task_id()};
    L0TaskArgs consumer_args;
    consumer_args.set_dependencies(duplicate_deps, 2);
    TaskOutputTensors consumer = orch.submit_dummy_task(consumer_args);
    ASSERT_TRUE(consumer.task_id().is_valid());

    auto &producer_slot = sm_handle->header->get_slot_state_by_task_id(producer.task_id().local());
    auto &consumer_slot = sm_handle->header->get_slot_state_by_task_id(consumer.task_id().local());
    EXPECT_EQ(producer_slot.fanin_count, 1);
    EXPECT_EQ(consumer_slot.fanin_count, 2);
    EXPECT_EQ(consumer_slot.fanin_refcount.load(), 0);
    PTO2DepListEntry *fanout = producer_slot.fanout_head.load();
    ASSERT_NE(fanout, nullptr);
    EXPECT_EQ(fanout->slot_state, &consumer_slot);
    EXPECT_EQ(fanout->next, nullptr);
}

TEST_F(ReplayGraphOrchestratorTest, LiveTaskLookupRejectsReusedPhysicalSlot) {
    constexpr int32_t logical_task_id = 0;
    constexpr int32_t reused_task_id = static_cast<int32_t>(kTaskWindow);
    auto &task = sm_handle->header->get_task_by_slot(0);
    auto &slot = sm_handle->header->get_slot_state_by_slot(0);
    slot.task = &task;
    sm_handle->header->task_slot_map[logical_task_id] = 0;

    task.task_id = PTO2TaskId::make(0, logical_task_id);
    EXPECT_EQ(sm_handle->header->find_live_slot_state(task.task_id), &slot);

    task.task_id = PTO2TaskId::make(0, reused_task_id);
    EXPECT_EQ(sm_handle->header->find_live_slot_state(PTO2TaskId::make(0, logical_task_id)), nullptr);
}

TEST_F(ReplayGraphOrchestratorTest, ReplayRestoresFrozenTaskDag) {
    orch.begin_scope();
    PTO2GraphBindings bindings;
    uint64_t graph_key = rt_graph_make_key(PTO2_GRAPH_KEY("ut_frozen_task_dag_v1"), bindings);
    constexpr uint64_t callable_hash = 0x8d5e52b41f62d9a3ULL;

    PTO2GraphScopeResult record = orch.graph_begin(graph_key, bindings, callable_hash);
    ASSERT_TRUE(record.execute_block);
    ASSERT_TRUE(record.recording);

    L0TaskArgs producer_args;
    TaskOutputTensors producer = orch.submit_dummy_task(producer_args);
    ASSERT_TRUE(producer.task_id().is_valid());
    PTO2TaskId dependency[] = {producer.task_id()};
    L0TaskArgs consumer_args;
    consumer_args.set_dependencies(dependency, 1);
    TaskOutputTensors consumer = orch.submit_dummy_task(consumer_args);
    ASSERT_TRUE(consumer.task_id().is_valid());

    PTO2GraphCacheStats stats;
    orch.graph_end(&stats);
    EXPECT_EQ(stats.recorded, 1);
    ASSERT_EQ(orch.task_allocator.active_count(), 2);

    PTO2GraphScopeResult replay = orch.graph_begin(graph_key, bindings, callable_hash);
    EXPECT_FALSE(replay.execute_block);
    EXPECT_FALSE(replay.recording);
    ASSERT_FALSE(orch.fatal);
    ASSERT_EQ(orch.task_allocator.active_count(), 4);

    auto &replayed_producer = sm_handle->header->get_slot_state_by_task_id(2);
    auto &replayed_consumer = sm_handle->header->get_slot_state_by_task_id(3);
    EXPECT_EQ(replayed_producer.task->task_id.local(), 2);
    EXPECT_EQ(replayed_consumer.task->task_id.local(), 3);
    EXPECT_EQ(replayed_producer.fanin_count, 1);
    EXPECT_EQ(replayed_consumer.fanin_count, 2);
    PTO2DepListEntry *fanout = replayed_producer.fanout_head.load();
    ASSERT_NE(fanout, nullptr);
    EXPECT_EQ(fanout->slot_state, &replayed_consumer);
    EXPECT_EQ(fanout->next, nullptr);
}
