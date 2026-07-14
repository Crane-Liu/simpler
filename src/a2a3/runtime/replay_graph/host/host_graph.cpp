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

#include "host_graph.h"

#include <cstring>

#include "common/unified_log.h"

namespace {

bool validate_host_graph(const HostGraph &graph) {
    const HostGraphHeader &h = graph.header;
    if (h.magic != PTO2_HOST_GRAPH_MAGIC || h.version != PTO2_HOST_GRAPH_VERSION ||
        h.header_size != sizeof(HostGraphHeader) || h.task_record_size != sizeof(HostGraphTaskRecord) ||
        h.boundary_record_size != sizeof(HostGraphBoundaryRecord) || h.payload_size != sizeof(PTO2TaskPayload)) {
        LOG_ERROR("HostGraph ABI header mismatch");
        return false;
    }
    if (h.first_task_id < 0 || h.task_count < 0 || h.edge_count < 0 || h.boundary_count <= 0 ||
        static_cast<size_t>(h.task_count) != graph.tasks.size() ||
        static_cast<size_t>(h.edge_count) != graph.edges.size() ||
        static_cast<size_t>(h.boundary_count) != graph.boundaries.size()) {
        LOG_ERROR("HostGraph counts or task range are invalid");
        return false;
    }
    int32_t expected_begin = h.first_task_id;
    for (int32_t i = 0; i < h.boundary_count; i++) {
        const HostGraphBoundaryRecord &boundary = graph.boundaries[static_cast<size_t>(i)];
        if (boundary.first_task_id != expected_begin || boundary.task_count < 0 ||
            boundary.graph_epoch != static_cast<uint64_t>(i)) {
            LOG_ERROR("HostGraph boundary %d is not a contiguous ordered range", i);
            return false;
        }
        expected_begin += boundary.task_count;
    }
    if (expected_begin != h.first_task_id + h.task_count) {
        LOG_ERROR("HostGraph boundaries do not cover the task range");
        return false;
    }
    return true;
}

}  // namespace

bool capture_host_graph_range(PTO2Runtime *source, int32_t task_begin, int32_t task_end, HostGraph *out) {
    if (source == nullptr || source->orchestrator.sm_header == nullptr || out == nullptr) return false;

    PTO2SharedMemoryHeader *header = source->orchestrator.sm_header;
    const int32_t total_tasks = header->fc.task_count.load(std::memory_order_acquire);
    const int32_t task_count = task_end - task_begin;
    if (task_begin < 0 || task_end < task_begin || task_end > total_tasks ||
        task_count > static_cast<int32_t>(header->task_window_size / PTO2_REPLAY_GRAPH_BUFFER_COUNT)) {
        LOG_ERROR("HostGraph task range [%d,%d) is invalid (total=%d)", task_begin, task_end, total_tasks);
        return false;
    }

    *out = HostGraph{};
    out->header.task_record_size = sizeof(HostGraphTaskRecord);
    out->header.boundary_record_size = sizeof(HostGraphBoundaryRecord);
    out->header.first_task_id = task_begin;
    out->header.task_count = task_count;
    out->header.graph_output_ptr = header->graph_output_ptr.load(std::memory_order_acquire);
    out->header.graph_output_size = header->graph_output_size.load(std::memory_order_acquire);
    out->tasks.resize(static_cast<size_t>(task_count));
    out->boundaries.push_back({task_begin, task_count, 0});
    out->header.boundary_count = 1;

    for (int32_t task_id = task_begin; task_id < task_end; task_id++) {
        PTO2TaskSlotState &slot = header->get_slot_state_by_task_id(task_id);
        if (slot.task == nullptr || slot.payload == nullptr) {
            LOG_ERROR("HostGraph task %d has incomplete slot bindings", task_id);
            return false;
        }
        HostGraphTaskRecord &record = out->tasks[static_cast<size_t>(task_id - task_begin)];
        record.task_id_raw = slot.task->task_id.raw;
        for (int i = 0; i < PTO2_SUBTASK_SLOT_COUNT; i++) record.kernel_id[i] = slot.task->kernel_id[i];
        record.packed_buffer_base = reinterpret_cast<uint64_t>(slot.task->packed_buffer_base);
        record.packed_buffer_end = reinterpret_cast<uint64_t>(slot.task->packed_buffer_end);
        record.active_mask = slot.active_mask.raw();
        record.task_state = static_cast<uint8_t>(slot.task_state.load(std::memory_order_acquire));
        if (record.task_state >= PTO2_TASK_COMPLETED) out->header.inline_completed_tasks++;
        record.allow_early_resolve = slot.allow_early_resolve ? 1 : 0;
        record.total_required_subtasks = slot.total_required_subtasks;
        record.logical_block_num = slot.logical_block_num;
        std::memcpy(record.payload, slot.payload, sizeof(PTO2TaskPayload));

        PTO2DepListEntry *edge = slot.fanout_head.load(std::memory_order_acquire);
        while (edge != nullptr && !pto2_is_fanout_closed(edge)) {
            if (edge->slot_state == nullptr || edge->slot_state->task == nullptr) {
                LOG_ERROR("HostGraph task %d has an invalid fanout edge", task_id);
                return false;
            }
            const int32_t consumer = static_cast<int32_t>(edge->slot_state->task->task_id.local());
            // Cross-boundary dependencies are represented by the publish
            // protocol's previous-graph completion barrier. Only internal
            // edges enter a range image, so no executing producer fanout is
            // ever patched after publication.
            if (consumer >= task_begin && consumer < task_end) out->edges.push_back({task_id, consumer});
            edge = edge->next;
        }
    }
    out->header.edge_count = static_cast<int32_t>(out->edges.size());
    return true;
}

bool import_host_graph(const HostGraph &graph, PTO2Runtime *target, bool finalize) {
    if (target == nullptr || target->orchestrator.sm_header == nullptr || !validate_host_graph(graph)) return false;
    if (graph.header.task_count >
            static_cast<int32_t>(target->orchestrator.sm_header->task_window_size / PTO2_REPLAY_GRAPH_BUFFER_COUNT) ||
        target->orchestrator.dep_pool.used() + graph.header.edge_count >= target->orchestrator.dep_pool.capacity) {
        LOG_ERROR("HostGraph exceeds target task or dependency capacity");
        return false;
    }

    PTO2SharedMemoryHeader *header = target->orchestrator.sm_header;
    for (int32_t i = 0; i < graph.header.task_count; i++) {
        const int32_t task_id = graph.header.first_task_id + i;
        const HostGraphTaskRecord &record = graph.tasks[static_cast<size_t>(i)];
        PTO2TaskId record_id{record.task_id_raw};
        if (!record_id.is_valid() || record_id.ring() != 0 ||
            static_cast<int32_t>(record_id.local()) != task_id) {
            LOG_ERROR("HostGraph task record %d has invalid id", task_id);
            return false;
        }

        PTO2TaskAllocResult alloc = target->orchestrator.task_allocator.alloc(0);
        if (alloc.failed() || alloc.task_id != task_id) return false;
        PTO2TaskDescriptor &task = header->get_task_by_slot(alloc.slot);
        PTO2TaskPayload &payload = header->get_payload_by_slot(alloc.slot);
        PTO2TaskSlotState &slot = header->get_slot_state_by_slot(alloc.slot);

        task.task_id = record_id;
        for (int k = 0; k < PTO2_SUBTASK_SLOT_COUNT; k++) task.kernel_id[k] = record.kernel_id[k];
        task.packed_buffer_base = reinterpret_cast<void *>(record.packed_buffer_base);
        task.packed_buffer_end = reinterpret_cast<void *>(record.packed_buffer_end);
        std::memcpy(&payload, record.payload, sizeof(PTO2TaskPayload));

        slot.bind_buffers(&payload, &task);
        slot.fanout_head.store(nullptr, std::memory_order_relaxed);
        slot.task_state.store(static_cast<PTO2TaskState>(record.task_state), std::memory_order_relaxed);
        slot.fanin_refcount.store(0, std::memory_order_relaxed);
        slot.fanin_count = 1;
        slot.active_mask = ActiveMask(record.active_mask);
        slot.bind_ring(0);
        slot.allow_early_resolve = record.allow_early_resolve != 0;
        slot.ready_state.store(
            record.task_state == PTO2_TASK_COMPLETED ? PTO2_COMPLETION_DONE : PTO2_READY_UNCLAIMED,
            std::memory_order_relaxed
        );
        slot.completed_subtasks.store(0, std::memory_order_relaxed);
        slot.total_required_subtasks = record.total_required_subtasks;
        slot.logical_block_num = record.logical_block_num;
        slot.next_block_idx.store(0, std::memory_order_relaxed);
    }

    const int32_t task_begin = graph.header.first_task_id;
    const int32_t task_end = task_begin + graph.header.task_count;
    for (const HostGraphEdgeRecord &record : graph.edges) {
        if (record.producer_task_id < task_begin || record.producer_task_id >= task_end ||
            record.consumer_task_id < task_begin || record.consumer_task_id >= task_end ||
            record.producer_task_id >= record.consumer_task_id) {
            LOG_ERROR("HostGraph contains an invalid dependency edge");
            return false;
        }
        PTO2TaskSlotState &producer = header->get_slot_state_by_task_id(record.producer_task_id);
        PTO2TaskSlotState &consumer = header->get_slot_state_by_task_id(record.consumer_task_id);
        PTO2DepListEntry *entry = target->orchestrator.dep_pool.alloc();
        if (entry == nullptr) return false;
        entry->slot_state = &consumer;
        entry->next = producer.fanout_head.load(std::memory_order_relaxed);
        producer.fanout_head.store(entry, std::memory_order_relaxed);
        consumer.fanin_count++;
    }

    target->orchestrator.inline_completed_tasks += graph.header.inline_completed_tasks;
    header->graph_output_ptr.store(graph.header.graph_output_ptr, std::memory_order_relaxed);
    header->graph_output_size.store(graph.header.graph_output_size, std::memory_order_relaxed);
    if (!finalize) return true;

    int32_t completed_count = 0;
    for (int32_t task_id = task_begin; task_id < task_end; task_id++) {
        PTO2TaskSlotState &slot = header->get_slot_state_by_task_id(task_id);
        if (slot.task_state.load(std::memory_order_relaxed) >= PTO2_TASK_COMPLETED) {
            completed_count++;
            continue;
        }
        slot.payload->dispatch_fanin.fetch_add(1, std::memory_order_relaxed);
        target->scheduler.release_fanin_and_check_ready(slot);
    }

    PTO2ReplayGraphPipelineState &pipeline = target->graph_pipeline;
    PTO2ReplayGraphBufferControl &buffer = pipeline.buffers[0];
    buffer.task_begin = task_begin;
    buffer.task_count = graph.header.task_count;
    buffer.completed_count.store(completed_count, std::memory_order_relaxed);
    buffer.exec_done.store(completed_count >= graph.header.task_count ? 1 : 0, std::memory_order_relaxed);
    buffer.dep_closed.store(1, std::memory_order_relaxed);
    buffer.state.store(
        completed_count >= graph.header.task_count ? PTO2ReplayGraphBufferState::DONE
                                                   : PTO2ReplayGraphBufferState::RUNNING,
        std::memory_order_relaxed
    );
    pipeline.all_done.store(1, std::memory_order_relaxed);
    pipeline.published_task_count.store(task_end, std::memory_order_relaxed);
    pipeline.graph_count = graph.header.boundary_count;
    pipeline.current_graph_epoch = static_cast<uint64_t>(graph.header.boundary_count);

    target->orchestrator.mark_done();
    return true;
}
