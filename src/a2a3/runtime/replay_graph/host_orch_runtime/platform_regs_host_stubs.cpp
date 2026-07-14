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

#include <stdint.h>

#include "common/platform_config.h"

// Host orchestration does not access device registers. Keep a valid address
// for code paths that take the register pointer without reading the device.
static volatile uint32_t g_host_reg = 0;

__attribute__((weak, visibility("hidden"))) volatile uint32_t *get_reg_ptr(uint64_t, RegId) {
    return &g_host_reg;
}

__attribute__((weak, visibility("hidden"))) uint64_t read_reg(uint64_t, RegId) { return 0; }
