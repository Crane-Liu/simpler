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

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "graph_cache_stats.h"
#include "tensor.h"

inline constexpr uint64_t PTO2_GRAPH_CACHE_SCHEMA_VERSION = 1;
inline constexpr uint32_t PTO2_GRAPH_MAX_BOUNDARY_TENSORS = 32;
inline constexpr uint32_t PTO2_GRAPH_MAX_BOUNDARY_SCALARS = 32;

struct PTO2GraphBindings {
    uint32_t tensor_count{0};
    uint32_t scalar_count{0};
    bool overflow{false};
    Tensor tensors[PTO2_GRAPH_MAX_BOUNDARY_TENSORS];
    uint64_t scalars[PTO2_GRAPH_MAX_BOUNDARY_SCALARS];
};

struct PTO2GraphScopeResult {
    bool execute_block{true};
    bool recording{false};
};

constexpr uint64_t pto2_graph_hash_byte(uint64_t h, uint8_t b) {
    return (h ^ static_cast<uint64_t>(b)) * 1099511628211ULL;
}

inline uint64_t pto2_graph_hash_bytes(uint64_t h, const void *data, size_t bytes) {
    const auto *p = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < bytes; ++i) {
        h = pto2_graph_hash_byte(h, p[i]);
    }
    return h;
}

constexpr uint64_t pto2_graph_const_hash_impl(const char *s, uint64_t h) {
    return (*s == '\0') ? h : pto2_graph_const_hash_impl(s + 1, pto2_graph_hash_byte(h, static_cast<uint8_t>(*s)));
}

constexpr uint64_t PTO2_GRAPH_KEY(const char *s) { return pto2_graph_const_hash_impl(s, 1469598103934665603ULL); }

inline uint64_t rt_graph_make_key(uint64_t namespace_hash, const PTO2GraphBindings &bindings) {
    uint64_t h = 1469598103934665603ULL;
    h = pto2_graph_hash_bytes(h, &PTO2_GRAPH_CACHE_SCHEMA_VERSION, sizeof(PTO2_GRAPH_CACHE_SCHEMA_VERSION));
    h = pto2_graph_hash_bytes(h, &namespace_hash, sizeof(namespace_hash));
    h = pto2_graph_hash_bytes(h, &bindings.tensor_count, sizeof(bindings.tensor_count));
    h = pto2_graph_hash_bytes(h, &bindings.scalar_count, sizeof(bindings.scalar_count));
    for (uint32_t i = 0; i < bindings.tensor_count; ++i) {
        const Tensor &t = bindings.tensors[i];
        h = pto2_graph_hash_bytes(h, &t.buffer.size, sizeof(t.buffer.size));
        h = pto2_graph_hash_bytes(h, &t.start_offset, sizeof(t.start_offset));
        h = pto2_graph_hash_bytes(h, &t.version, sizeof(t.version));
        h = pto2_graph_hash_bytes(h, &t.ndims, sizeof(t.ndims));
        h = pto2_graph_hash_bytes(h, &t.dtype, sizeof(t.dtype));
        h = pto2_graph_hash_bytes(h, &t.manual_dep, sizeof(t.manual_dep));
        h = pto2_graph_hash_bytes(h, &t.is_contiguous, sizeof(t.is_contiguous));
        h = pto2_graph_hash_bytes(h, t.shapes, sizeof(uint32_t) * t.ndims);
        h = pto2_graph_hash_bytes(h, t.strides, sizeof(uint32_t) * t.ndims);
    }
    for (uint32_t i = 0; i < bindings.scalar_count; ++i) {
        h = pto2_graph_hash_bytes(h, &bindings.scalars[i], sizeof(bindings.scalars[i]));
    }
    h = pto2_graph_hash_bytes(h, &bindings.overflow, sizeof(bindings.overflow));
    return h;
}
