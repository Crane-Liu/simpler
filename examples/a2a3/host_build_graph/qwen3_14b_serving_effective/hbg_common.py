#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared slot state and fixture geometry for the Qwen HBG runners."""

from __future__ import annotations

from typing import Any

import torch

BATCH = 16
SAMPLED_IDS_PAD = 8


def shared_slot_state(fixture: Any) -> list[dict[str, torch.Tensor]]:
    block_table = fixture.metadata["block_table"].reshape(-1)
    return [
        {
            "seq_lens": fixture.metadata["seq_lens_after_first_token"].clone().share_memory_(),
            "block_table": block_table.clone().share_memory_(),
            "initial_block_table": block_table.clone(),
            "slot_mapping": fixture.metadata["next_slot_mapping"].clone().share_memory_(),
            "sampled_ids_host": torch.zeros((BATCH, SAMPLED_IDS_PAD), dtype=torch.int32).share_memory_(),
        }
        for _ in range(2)
    ]


def update_slot(
    slot: dict[str, torch.Tensor],
    golden: dict[str, torch.Tensor],
    step: int,
    *,
    page_size: int,
    blocks_per_row: int,
) -> None:
    slot["seq_lens"].copy_(golden["seq_lens"][step])
    slot["slot_mapping"].copy_(golden["slot_mapping"][step])
    positions = slot["seq_lens"] - 1
    if not torch.equal(slot["slot_mapping"].remainder(page_size), positions.remainder(page_size)):
        raise RuntimeError(f"slot mapping offset mismatch at decode step {step}")
    logical_blocks = positions.div(page_size, rounding_mode="floor")
    page_ids = slot["slot_mapping"].div(page_size, rounding_mode="floor")
    if int(logical_blocks.min()) < 0 or int(logical_blocks.max()) >= blocks_per_row:
        raise RuntimeError(f"logical block exceeds the block table row at decode step {step}")
    if slot["block_table"].numel() < BATCH * blocks_per_row:
        raise RuntimeError("block table is smaller than the fixture geometry")
    for row in range(BATCH):
        slot["block_table"][row * blocks_per_row + int(logical_blocks[row])] = page_ids[row]
    slot["sampled_ids_host"].zero_()
