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

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "../runtime/pto_runtime2.h"

inline constexpr uint32_t PTO2_HOST_GRAPH_MAGIC = 0x48544752u;
inline constexpr uint16_t PTO2_HOST_GRAPH_VERSION = 2;

struct HostGraphHeader {
    uint32_t magic{PTO2_HOST_GRAPH_MAGIC};
    uint16_t version{PTO2_HOST_GRAPH_VERSION};
    uint16_t flags{0};
    uint32_t header_size{sizeof(HostGraphHeader)};
    uint32_t task_record_size{0};
    uint32_t boundary_record_size{0};
    uint32_t payload_size{sizeof(PTO2TaskPayload)};
    int32_t first_task_id{0};
    int32_t task_count{0};
    int32_t edge_count{0};
    int32_t boundary_count{0};
    int32_t inline_completed_tasks{0};
    uint64_t graph_output_ptr{0};
    uint64_t graph_output_size{0};
};

struct alignas(64) HostGraphTaskRecord {
    uint64_t task_id_raw{PTO2TaskId::invalid().raw};
    int32_t kernel_id[PTO2_SUBTASK_SLOT_COUNT]{INVALID_KERNEL_ID, INVALID_KERNEL_ID, INVALID_KERNEL_ID};
    uint64_t packed_buffer_base{0};
    uint64_t packed_buffer_end{0};
    uint8_t active_mask{0};
    uint8_t task_state{PTO2_TASK_PENDING};
    uint8_t allow_early_resolve{0};
    uint8_t reserved0{0};
    int16_t total_required_subtasks{0};
    int16_t logical_block_num{1};
    uint32_t reserved1{0};
    alignas(64) uint8_t payload[sizeof(PTO2TaskPayload)]{};
};

struct HostGraphEdgeRecord {
    int32_t producer_task_id{-1};
    int32_t consumer_task_id{-1};
};

struct HostGraphBoundaryRecord {
    int32_t first_task_id{0};
    int32_t task_count{0};
    uint64_t graph_epoch{0};
};

static_assert(std::is_standard_layout_v<HostGraphHeader>);
static_assert(std::is_standard_layout_v<HostGraphTaskRecord>);
static_assert(std::is_standard_layout_v<HostGraphEdgeRecord>);
static_assert(std::is_standard_layout_v<HostGraphBoundaryRecord>);
static_assert(alignof(HostGraphTaskRecord) >= alignof(PTO2TaskPayload));

struct HostGraph {
    HostGraphHeader header;
    std::vector<HostGraphTaskRecord> tasks;
    std::vector<HostGraphEdgeRecord> edges;
    std::vector<HostGraphBoundaryRecord> boundaries;
};

bool capture_host_graph_range(PTO2Runtime *source, int32_t task_begin, int32_t task_end, HostGraph *out);
bool import_host_graph(const HostGraph &graph, PTO2Runtime *target, bool finalize = true);
