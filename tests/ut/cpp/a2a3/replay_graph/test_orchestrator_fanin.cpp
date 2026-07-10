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

#include <cstdint>
#include <vector>

#include "utils/device_arena.h"
#include "pto_orchestrator.h"
#include "pto_shared_memory.h"

class OrchestratorFaninTest : public ::testing::Test {
protected:
    DeviceArena sm_arena;
    DeviceArena runtime_arena;
    PTO2SharedMemoryHandle *sm_handle = nullptr;
    PTO2OrchestratorState orch{};
    PTO2SchedulerState sched{};
    PTO2OrchestratorLayout orch_layout{};
    PTO2SchedulerLayout sched_layout{};
    std::vector<char> gm_heap;

    void SetUp() override {
        sm_handle = PTO2SharedMemoryHandle::create_and_init_default(sm_arena);
        ASSERT_NE(sm_handle, nullptr);
        gm_heap.resize(4096);

        orch_layout = PTO2OrchestratorState::reserve_layout(runtime_arena, static_cast<int32_t>(PTO2_TASK_WINDOW_SIZE));
        sched_layout = PTO2SchedulerState::reserve_layout(runtime_arena);
        ASSERT_NE(runtime_arena.commit(), nullptr);

        ASSERT_TRUE(orch.init_data_from_layout(
            orch_layout, runtime_arena, sm_handle->sm_base, gm_heap.data(), 4096, PTO2_TASK_WINDOW_SIZE
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

TEST_F(OrchestratorFaninTest, DuplicateExplicitProducerAddsOneFanin) {
    orch.begin_scope();

    L0TaskArgs producer_args;
    TaskOutputTensors producer = orch.submit_dummy_task(producer_args);
    ASSERT_TRUE(producer.task_id().is_valid());

    PTO2TaskId deps[] = {producer.task_id(), producer.task_id()};
    L0TaskArgs consumer_args;
    consumer_args.set_dependencies(deps, 2);
    TaskOutputTensors consumer = orch.submit_dummy_task(consumer_args);
    ASSERT_TRUE(consumer.task_id().is_valid());

    auto &producer_slot = sm_handle->header->get_slot_state_by_task_id(producer.task_id().local());
    auto &consumer_slot = sm_handle->header->get_slot_state_by_task_id(consumer.task_id().local());

    // The duplicate explicit dep is deduped: exactly one producer edge.
    EXPECT_EQ(consumer_slot.fanin_count, 1);
    // The producer is still PENDING (dummy task), so the consumer was wired as a
    // fanout successor rather than counted as already-satisfied.
    EXPECT_EQ(consumer_slot.fanin_refcount.load(), 0);
    ASSERT_NE(producer_slot.fanout_head, nullptr);
    EXPECT_EQ(producer_slot.fanout_head->slot_state, &consumer_slot);
    // Deduped: only one fanout entry on the producer.
    EXPECT_EQ(producer_slot.fanout_head->next, nullptr);
}
