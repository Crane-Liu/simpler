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
 * PTO Runtime2 - TensorMap Implementation
 *
 * Implements TensorMap with ring buffer pool, lazy invalidation,
 * and chain truncation optimization.
 *
 * Key features:
 * 1. O(1) insert at bucket head
 * 2. O(valid_entries) lookup with chain truncation
 * 3. Automatic stale entry cleanup during lookup
 * 4. Periodic explicit cleanup for long chains
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#include "pto_tensormap.h"

#include <stdlib.h>
#include <string.h>

#include "common.h"

// =============================================================================
// TensorMap Lookup Chain Length Statistics (compile-time toggle)
// =============================================================================
#if PTO2_TENSORMAP_PROFILING
uint64_t g_lookup_chain_total = 0;
uint64_t g_lookup_count = 0;
int32_t g_lookup_chain_max = 0;
uint64_t g_lookup_overlap_checks = 0;
uint64_t g_lookup_overlap_hits = 0;
uint64_t g_insert_count = 0;
#endif

// =============================================================================
// Initialization and Destruction
// =============================================================================

PTO2TensorMapLayout PTO2TensorMap::reserve_layout(DeviceArena &arena, int32_t new_num_buckets, int32_t new_pool_size) {
    // num_buckets must be a power of two for the hash truncation to work.
    always_assert((new_num_buckets & (new_num_buckets - 1)) == 0);

    PTO2TensorMapLayout layout{};
    layout.num_buckets = new_num_buckets;
    layout.pool_size = new_pool_size;

    layout.off_buckets = arena.reserve(
        static_cast<size_t>(new_num_buckets) * sizeof(PTO2TensorMapEntry *), alignof(PTO2TensorMapEntry *)
    );
    layout.off_entry_pool =
        arena.reserve(static_cast<size_t>(new_pool_size) * sizeof(PTO2TensorMapEntry), alignof(PTO2TensorMapEntry));
    layout.off_free_entry_list =
        arena.reserve(static_cast<size_t>(new_pool_size) * sizeof(PTO2TensorMapEntry *), alignof(PTO2TensorMapEntry *));
    return layout;
}

PTO2TensorMapLayout PTO2TensorMap::reserve_layout_default(DeviceArena &arena) {
    return reserve_layout(arena, PTO2_TENSORMAP_NUM_BUCKETS, PTO2_TENSORMAP_POOL_SIZE);
}

bool PTO2TensorMap::init_data_from_layout(const PTO2TensorMapLayout &layout, DeviceArena &arena) {
    num_buckets = layout.num_buckets;
    pool_size = layout.pool_size;

    // Address arena regions for data writes; do not store these in struct
    // fields (wire_arena_pointers does that).
    auto *buckets_arena = static_cast<PTO2TensorMapEntry **>(arena.region_ptr(layout.off_buckets));
    auto *entry_pool_arena = static_cast<PTO2TensorMapEntry *>(arena.region_ptr(layout.off_entry_pool));
    auto *free_list_arena = static_cast<PTO2TensorMapEntry **>(arena.region_ptr(layout.off_free_entry_list));

    // buckets[]: empty == nullptr.
    for (int32_t i = 0; i < num_buckets; i++) {
        buckets_arena[i] = nullptr;
    }

    // entry_pool: zero-init equivalent to the previous calloc(entry_pool, ...).
    // The pool's persistent invariant after init is "bucket_index == -1 means
    // not linked", set explicitly below.
    memset(entry_pool_arena, 0, static_cast<size_t>(pool_size) * sizeof(PTO2TensorMapEntry));
    for (int32_t i = 0; i < pool_size; i++) {
        entry_pool_arena[i].bucket_index = -1;
        entry_pool_arena[i].next_in_bucket = nullptr;
        entry_pool_arena[i].prev_in_bucket = nullptr;
        entry_pool_arena[i].next_in_task = nullptr;
        entry_pool_arena[i].prev_in_task = nullptr;
        entry_pool_arena[i].producer_task_id = PTO2TaskId{};
    }

    // free_entry_list: zeroed (was calloc'd before); contents become meaningful
    // only after entries are freed back, so the body of the array stays as 0.
    memset(free_list_arena, 0, static_cast<size_t>(pool_size) * sizeof(PTO2TensorMapEntry *));

    next_entry_idx = 0;
    free_num = 0;

    return true;
}

void PTO2TensorMap::wire_arena_pointers(const PTO2TensorMapLayout &layout, DeviceArena &arena) {
    buckets = static_cast<PTO2TensorMapEntry **>(arena.region_ptr(layout.off_buckets));
    entry_pool = static_cast<PTO2TensorMapEntry *>(arena.region_ptr(layout.off_entry_pool));
    free_entry_list = static_cast<PTO2TensorMapEntry **>(arena.region_ptr(layout.off_free_entry_list));
}

void PTO2TensorMap::destroy() {
    // Arena owns the backing memory; here we only forget our pointers so any
    // stray post-destroy access trips a nullptr dereference instead of reading
    // a recycled allocation.
    buckets = nullptr;
    entry_pool = nullptr;
    free_entry_list = nullptr;
}

// =============================================================================
// TensorMap Lookup Profiling
// =============================================================================
#if PTO2_TENSORMAP_PROFILING
PTO2TensorMapProfilingData pto2_tensormap_get_profiling() {
    PTO2TensorMapProfilingData d;
    d.lookup_chain_total = g_lookup_chain_total;
    d.lookup_count = g_lookup_count;
    d.lookup_chain_max = g_lookup_chain_max;
    d.overlap_checks = g_lookup_overlap_checks;
    d.overlap_hits = g_lookup_overlap_hits;
    d.insert_count = g_insert_count;

    // Reset
    g_lookup_chain_total = 0;
    g_lookup_count = 0;
    g_lookup_chain_max = 0;
    g_lookup_overlap_checks = 0;
    g_lookup_overlap_hits = 0;
    g_insert_count = 0;
    return d;
}
#endif
