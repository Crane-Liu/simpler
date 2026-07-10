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
 * Unit tests for PTO2SharedMemory layout from pto_shared_memory.h
 *
 * Tests creation, validation, per-ring independence, alignment, size
 * calculation, and error handling under the DeviceArena-backed init model:
 *   - Wrapper and SM buffer both live in a caller-supplied DeviceArena.
 *   - handle->init(...) writes fields in place; arena.release() reclaims.
 */

#include <gtest/gtest.h>
#include <cstring>
#include <limits>
#include <vector>
#include "pto_runtime2.h"
#include "pto_shared_memory.h"

namespace {

// Reserve + commit a fresh handle + sm_base on `arena` and run init.
// Returns the wrapper pointer (arena-owned) or nullptr on init failure.
PTO2SharedMemoryHandle *make_handle(DeviceArena &arena, uint64_t task_window_size, uint64_t heap_size) {
    const uint64_t sm_size = PTO2SharedMemoryHandle::calculate_size(task_window_size);
    const size_t off_handle = arena.reserve(sizeof(PTO2SharedMemoryHandle), alignof(PTO2SharedMemoryHandle));
    const size_t off_buffer = arena.reserve(static_cast<size_t>(sm_size), PTO2_ALIGN_SIZE);
    if (arena.commit() == nullptr) return nullptr;

    auto *handle = static_cast<PTO2SharedMemoryHandle *>(arena.region_ptr(off_handle));
    std::memset(handle, 0, sizeof(*handle));
    void *buffer = arena.region_ptr(off_buffer);
    std::memset(buffer, 0, static_cast<size_t>(sm_size));
    if (!handle->init(buffer, sm_size, task_window_size, heap_size)) return nullptr;
    return handle;
}

}  // namespace

// =============================================================================
// Fixture (default-sized, libc-backed arena)
// =============================================================================

class SharedMemoryTest : public ::testing::Test {
protected:
    DeviceArena arena;
    PTO2SharedMemoryHandle *handle = nullptr;

    void SetUp() override {
        handle = PTO2SharedMemoryHandle::create_and_init_default(arena);
        ASSERT_NE(handle, nullptr);
    }

    void TearDown() override {
        handle = nullptr;
        arena.release();
    }
};

// =============================================================================
// Normal path
// =============================================================================

TEST_F(SharedMemoryTest, CreateDefaultReturnsNonNull) {
    EXPECT_NE(handle->sm_base, nullptr);
    EXPECT_GT(handle->sm_size, 0u);
}

TEST_F(SharedMemoryTest, NotOwnerOfArenaBackedHandle) {
    // The arena owns both the wrapper and the SM buffer; the handle must
    // not try to free them in destroy().
    EXPECT_FALSE(handle->is_owner);
}

TEST_F(SharedMemoryTest, HeaderInitValues) {
    auto *hdr = handle->header;
    EXPECT_EQ(hdr->orchestrator_done.load(), 0);
    EXPECT_EQ(hdr->orch_error_code.load(), 0);
    EXPECT_EQ(hdr->sched_error_bitmap.load(), 0);
    EXPECT_EQ(hdr->sched_error_code.load(), 0);

    auto &fc = hdr->fc;
    EXPECT_EQ(fc.task_count.load(), 0);
}

TEST_F(SharedMemoryTest, Validate) { EXPECT_TRUE(handle->validate()); }

TEST_F(SharedMemoryTest, RingPointersInitialized) {
    EXPECT_NE(handle->header->task_descriptors, nullptr);
    EXPECT_NE(handle->header->task_payloads, nullptr);
}

TEST_F(SharedMemoryTest, PointerAlignment) {
    auto addr = reinterpret_cast<uintptr_t>(handle->header->task_descriptors);
    EXPECT_EQ(addr % PTO2_ALIGN_SIZE, 0u) << "descriptors not aligned";
}

TEST_F(SharedMemoryTest, HeaderAlignment) {
    uintptr_t header_addr = (uintptr_t)handle->header;
    EXPECT_EQ(header_addr % PTO2_ALIGN_SIZE, 0u) << "Header must be cache-line aligned";
}

// Descriptor and payload regions don't overlap.
TEST(SharedMemoryLayout, RegionsNonOverlapping) {
    DeviceArena arena;
    PTO2SharedMemoryHandle *h = make_handle(arena, /*ws=*/64, /*heap=*/4096);
    ASSERT_NE(h, nullptr);

    uintptr_t desc_start = (uintptr_t)h->header->task_descriptors;
    uintptr_t desc_end = desc_start + 64 * sizeof(PTO2TaskDescriptor);
    uintptr_t payload_start = (uintptr_t)h->header->task_payloads;

    EXPECT_GE(payload_start, desc_end) << "payload region should not overlap descriptors";
}

// =============================================================================
// Size calculation
// =============================================================================

TEST(SharedMemoryCalcSize, NonZero) {
    uint64_t size = PTO2SharedMemoryHandle::calculate_size(PTO2_TASK_WINDOW_SIZE);
    EXPECT_GT(size, 0u);
}

TEST(SharedMemoryCalcSize, LargerWindowGivesLargerSize) {
    uint64_t small_size = PTO2SharedMemoryHandle::calculate_size(64);
    uint64_t large_size = PTO2SharedMemoryHandle::calculate_size(256);
    EXPECT_GT(large_size, small_size);
}

TEST(SharedMemoryCalcSize, HeaderAligned) { EXPECT_EQ(sizeof(PTO2SharedMemoryHeader) % PTO2_ALIGN_SIZE, 0u); }

TEST(SharedMemoryLayout, InitWritesHeaderValues) {
    const uint64_t ws = 64;
    const uint64_t heap = 20 * 1024;
    const uint64_t sm_size = PTO2SharedMemoryHandle::calculate_size(ws);

    DeviceArena arena;
    const size_t off_handle = arena.reserve(sizeof(PTO2SharedMemoryHandle), alignof(PTO2SharedMemoryHandle));
    const size_t off_buffer = arena.reserve(static_cast<size_t>(sm_size), PTO2_ALIGN_SIZE);
    ASSERT_NE(arena.commit(), nullptr);

    auto *handle = static_cast<PTO2SharedMemoryHandle *>(arena.region_ptr(off_handle));
    std::memset(handle, 0, sizeof(*handle));
    void *buffer = arena.region_ptr(off_buffer);
    std::memset(buffer, 0, static_cast<size_t>(sm_size));
    ASSERT_TRUE(handle->init(buffer, sm_size, ws, heap));

    EXPECT_EQ(handle->header->task_window_size, ws);
    EXPECT_EQ(handle->header->heap_size, heap);
    EXPECT_EQ(handle->header->task_window_mask, static_cast<int32_t>(ws - 1));
}

TEST(RuntimeArenaLayout, ConfigInitializesRuntimeComponents) {
    const uint64_t ws = 64;
    const uint64_t heap = 20 * 1024;
    const int32_t dep_cap = 16;
    const uint64_t sm_size = PTO2SharedMemoryHandle::calculate_size(ws);

    DeviceArena runtime_arena;
    PTO2RuntimeArenaLayout layout = runtime_reserve_layout(runtime_arena, ws, heap, dep_cap);
    ASSERT_NE(runtime_arena.commit(DeviceArena::kDefaultBaseAlign), nullptr);

    DeviceArena sm_arena;
    const size_t sm_off = sm_arena.reserve(static_cast<size_t>(sm_size), PTO2_ALIGN_SIZE);
    ASSERT_NE(sm_arena.commit(), nullptr);
    void *sm = sm_arena.region_ptr(sm_off);
    std::memset(sm, 0, static_cast<size_t>(sm_size));

    std::vector<char> gm(static_cast<size_t>(heap));
    PTO2Runtime *rt =
        runtime_init_data_from_layout(runtime_arena, layout, PTO2_MODE_EXECUTE, sm, sm_size, gm.data(), heap);
    ASSERT_NE(rt, nullptr);
    runtime_wire_arena_pointers(runtime_arena, layout, rt);

    EXPECT_EQ(rt->gm_heap_size, heap);
    EXPECT_EQ(layout.task_window_size, ws);
    EXPECT_EQ(layout.heap_size, heap);
    EXPECT_EQ(layout.dep_pool_capacity, dep_cap);
    EXPECT_EQ(rt->orchestrator.task_allocator.window_size(), static_cast<int32_t>(ws));
    EXPECT_EQ(rt->orchestrator.task_allocator.heap_capacity(), heap);
    EXPECT_EQ(rt->orchestrator.dep_pool.capacity, dep_cap);
}

// =============================================================================
// Boundary conditions
// =============================================================================

// Zero window size: SM collapses to just the header region.
TEST(SharedMemoryBoundary, ZeroWindowSize) {
    uint64_t size = PTO2SharedMemoryHandle::calculate_size(0);
    uint64_t header_size = PTO2_ALIGN_UP(sizeof(PTO2SharedMemoryHeader), PTO2_ALIGN_SIZE);
    EXPECT_EQ(size, header_size);
}

TEST(SharedMemoryBoundary, ValidateDetectsCorruption) {
    DeviceArena arena;
    PTO2SharedMemoryHandle *h = make_handle(arena, /*ws=*/256, /*heap=*/4096);
    ASSERT_NE(h, nullptr);
    EXPECT_TRUE(h->validate());

    h->header->fc.task_count.store(-1);
    EXPECT_FALSE(h->validate());
}

TEST(SharedMemoryBoundary, InitRejectsUndersizedBuffer) {
    // init() must refuse an SM buffer smaller than calculate_size(window_size).
    PTO2SharedMemoryHandle handle{};
    char buf[64]{};
    EXPECT_FALSE(handle.init(buf, sizeof(buf), /*task_window_size=*/256, /*heap=*/4096));
}
