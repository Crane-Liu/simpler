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
 * Unit tests for the dependency-graph construction and completion paths:
 *
 * 1. submit-time graph wiring — fanout edges, already-completed-producer
 *    detection, fanin_count/fanin_refcount seeding, initial-ready push.
 *    Wiring is no longer a deferred pass (wire_task / drain_wiring_queue / the
 *    SPSC queue are gone): submit_task builds the graph in-line, so these tests
 *    drive the public submit API (submit_dummy_task / alloc_tensors) and assert
 *    on the resulting slot state instead of calling a wiring entry point.
 * 2. on_task_complete() — COMPLETED transition, fanout traversal,
 *                         consumer fanin release.
 */

#include <gtest/gtest.h>

#include <atomic>
#include <cstring>
#include <vector>

#include "utils/device_arena.h"
#include "pto_orchestrator.h"
#include "pto_shared_memory.h"
#include "scheduler/pto_scheduler.h"

// =============================================================================
// Fixture: sets up an orchestrator + scheduler over shared memory.
// =============================================================================

class WiringTest : public ::testing::Test {
protected:
    // submit_task builds the dependency graph in-line in the orchestrator, so
    // the wiring tests drive `orch`. The completion-path test (on_task_complete)
    // drives `sched`; both are set up here, mirroring test_orchestrator_fanin.
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
        orch.begin_scope();
    }

    void TearDown() override {
        orch.destroy();
        sched.destroy();
        runtime_arena.release();
        sm_arena.release();
    }

    // Submit a PENDING producer (dependency-only task; never completes inline).
    PTO2TaskId submit_pending_producer() {
        L0TaskArgs args;
        TaskOutputTensors out = orch.submit_dummy_task(args);
        EXPECT_TRUE(out.task_id().is_valid());
        return out.task_id();
    }

    // Allocate an inline-COMPLETED producer (alloc_tensors completes the slot in
    // the orchestrator before any consumer can exist).
    PTO2TaskId alloc_completed_producer() {
        L0TaskArgs args;
        const uint32_t shape[] = {16};
        TensorCreateInfo ci(shape, 1, DataType::FLOAT32);
        args.add_output(ci);
        TaskOutputTensors out = orch.alloc_tensors(args);
        EXPECT_TRUE(out.task_id().is_valid());
        return out.task_id();
    }

    PTO2TaskSlotState &slot_of(PTO2TaskId id) { return sm_handle->header->get_slot_state_by_task_id(id.local()); }
};

// =============================================================================
// submit: no fanin (independent task) -> initially ready
// =============================================================================

TEST_F(WiringTest, NoFaninBecomesReady) {
    int32_t before = orch.initial_ready_count;

    L0TaskArgs args;
    TaskOutputTensors out = orch.submit_dummy_task(args);
    ASSERT_TRUE(out.task_id().is_valid());

    auto &slot = slot_of(out.task_id());
    EXPECT_EQ(slot.fanin_count, 0);
    EXPECT_EQ(slot.fanin_refcount.load(), 0);

    // No producers -> appended to the initial-ready handoff.
    ASSERT_EQ(orch.initial_ready_count, before + 1);
    EXPECT_EQ(orch.initial_ready[orch.initial_ready_count - 1], &slot);
}

// =============================================================================
// submit: all producers already inline-completed (alloc) -> ready
// =============================================================================

TEST_F(WiringTest, AllProducersEarlyFinished) {
    PTO2TaskId p0 = alloc_completed_producer();
    PTO2TaskId p1 = alloc_completed_producer();

    int32_t before = orch.initial_ready_count;

    PTO2TaskId deps[] = {p0, p1};
    L0TaskArgs consumer_args;
    consumer_args.set_dependencies(deps, 2);
    TaskOutputTensors consumer = orch.submit_dummy_task(consumer_args);
    ASSERT_TRUE(consumer.task_id().is_valid());

    auto &cslot = slot_of(consumer.task_id());
    // Both producers already complete -> not counted at all: fanin_count and
    // fanin_refcount stay 0, so the consumer is still initially ready (0 == 0).
    EXPECT_EQ(cslot.fanin_count, 0);
    EXPECT_EQ(cslot.fanin_refcount.load(), 0);
    // dispatch_fanin is not seeded for already-complete producers.
    ASSERT_NE(cslot.payload, nullptr);
    EXPECT_EQ(cslot.payload->dispatch_fanin.load(), 0);

    // No fanout edge built on completed producers.
    EXPECT_EQ(slot_of(p0).fanout_head, nullptr);
    EXPECT_EQ(slot_of(p1).fanout_head, nullptr);

    // Every producer already satisfied -> consumer is initially ready.
    ASSERT_EQ(orch.initial_ready_count, before + 1);
    EXPECT_EQ(orch.initial_ready[orch.initial_ready_count - 1], &cslot);
}

// =============================================================================
// submit: producers still pending -> consumer NOT ready, fanout edges built
// =============================================================================

TEST_F(WiringTest, ProducersPendingTaskNotReady) {
    PTO2TaskId p0 = submit_pending_producer();
    PTO2TaskId p1 = submit_pending_producer();

    int32_t before = orch.initial_ready_count;

    PTO2TaskId deps[] = {p0, p1};
    L0TaskArgs consumer_args;
    consumer_args.set_dependencies(deps, 2);
    TaskOutputTensors consumer = orch.submit_dummy_task(consumer_args);
    ASSERT_TRUE(consumer.task_id().is_valid());

    auto &cslot = slot_of(consumer.task_id());
    EXPECT_EQ(cslot.fanin_count, 2);
    // Pending producers -> refcount untouched -> not ready.
    EXPECT_EQ(cslot.fanin_refcount.load(), 0);
    EXPECT_LT(cslot.fanin_refcount.load(), cslot.fanin_count);

    // Not ready -> nothing appended to initial-ready.
    EXPECT_EQ(orch.initial_ready_count, before);

    // Each pending producer carries the consumer in its fanout chain.
    auto &p0slot = slot_of(p0);
    auto &p1slot = slot_of(p1);
    ASSERT_NE(p0slot.fanout_head, nullptr);
    EXPECT_EQ(p0slot.fanout_head->slot_state, &cslot);
    ASSERT_NE(p1slot.fanout_head, nullptr);
    EXPECT_EQ(p1slot.fanout_head->slot_state, &cslot);
}

// =============================================================================
// submit: mixed completed (alloc) and pending producers
// =============================================================================

TEST_F(WiringTest, MixedProducerStates) {
    PTO2TaskId pa = alloc_completed_producer();  // inline-completed
    PTO2TaskId pb = submit_pending_producer();   // pending
    PTO2TaskId pc = alloc_completed_producer();  // inline-completed

    int32_t before = orch.initial_ready_count;

    PTO2TaskId deps[] = {pa, pb, pc};
    L0TaskArgs consumer_args;
    consumer_args.set_dependencies(deps, 3);
    TaskOutputTensors consumer = orch.submit_dummy_task(consumer_args);
    ASSERT_TRUE(consumer.task_id().is_valid());

    auto &cslot = slot_of(consumer.task_id());
    // Only the pending producer is counted: fanin_count == 1, refcount stays 0 ->
    // not ready. The two completed producers contribute nothing.
    EXPECT_EQ(cslot.fanin_count, 1);
    EXPECT_EQ(cslot.fanin_refcount.load(), 0);
    EXPECT_EQ(orch.initial_ready_count, before);  // not ready

    // Only the pending producer carries the consumer in its fanout chain.
    EXPECT_EQ(slot_of(pa).fanout_head, nullptr);
    ASSERT_NE(slot_of(pb).fanout_head, nullptr);
    EXPECT_EQ(slot_of(pb).fanout_head->slot_state, &cslot);
    EXPECT_EQ(slot_of(pc).fanout_head, nullptr);
}

// =============================================================================
// on_task_complete: notifies consumers via fanout chain
// =============================================================================

TEST_F(WiringTest, OnMixedTaskCompleteNotifiesConsumers) {
    alignas(64) PTO2TaskSlotState producer;
    alignas(64) PTO2TaskSlotState consumer1, consumer2;
    alignas(64) PTO2TaskPayload prod_payload, c1_payload, c2_payload;
    memset(&prod_payload, 0, sizeof(prod_payload));
    memset(&c1_payload, 0, sizeof(c1_payload));
    memset(&c2_payload, 0, sizeof(c2_payload));
    PTO2TaskDescriptor desc{};

    auto init_slot = [](PTO2TaskSlotState &slot, PTO2TaskState state, int32_t fanin_count, PTO2TaskPayload &pl) {
        memset(&slot, 0, sizeof(slot));
        slot.task_state.store(state);
        slot.fanin_count = fanin_count;
        slot.fanin_refcount.store(0);
        slot.fanout_head = nullptr;
        slot.active_mask = ActiveMask(PTO2_SUBTASK_MASK_AIC);
        slot.completed_subtasks.store(0);
        slot.total_required_subtasks = 1;
        slot.logical_block_num = 1;
        slot.payload = &pl;
    };

    // Producer in flight (PENDING) with 2 consumers in its fanout chain.
    init_slot(producer, PTO2_TASK_PENDING, 1, prod_payload);
    producer.task = &desc;

    // Consumer1: needs 1 more fanin to become ready.
    init_slot(consumer1, PTO2_TASK_PENDING, 2, c1_payload);
    consumer1.fanin_refcount.store(1);  // 1 of 2 satisfied

    // Consumer2: this release will make it ready.
    init_slot(consumer2, PTO2_TASK_PENDING, 2, c2_payload);
    consumer2.fanin_refcount.store(1);  // 1 of 2 satisfied

    // Build fanout chain: producer -> consumer2 -> consumer1
    PTO2DepListEntry dep_entries[2];
    dep_entries[0].slot_state = &consumer1;
    dep_entries[0].next = nullptr;
    dep_entries[1].slot_state = &consumer2;
    dep_entries[1].next = &dep_entries[0];
    producer.fanout_head = &dep_entries[1];

    sched.on_task_complete(producer);

    EXPECT_EQ(producer.task_state.load(), PTO2_TASK_COMPLETED);

    EXPECT_EQ(consumer1.fanin_refcount.load(), 2);
    EXPECT_EQ(consumer2.fanin_refcount.load(), 2);

    // Both consumers should be ready (fanin_refcount == fanin_count).
    PTO2ResourceShape shape = consumer1.active_mask.to_shape();
    auto *r1 = sched.ready_queues[static_cast<int32_t>(shape)].pop();
    auto *r2 = sched.ready_queues[static_cast<int32_t>(shape)].pop();
    EXPECT_TRUE((r1 == &consumer1 && r2 == &consumer2) || (r1 == &consumer2 && r2 == &consumer1));
}
