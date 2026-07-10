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
 * PTO Runtime2 - Ring Buffer Implementation
 *
 * DepListPool and TaskAllocator are single-shot pure-bump
 * allocators in the replay_graph model: the arena fills exactly once, never
 * wraps, and nothing is reclaimed during the orch phase. All allocation and
 * traversal logic is therefore inline in pto_ring_buffer.h; this translation
 * unit holds no out-of-line definitions.
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#include "pto_ring_buffer.h"
