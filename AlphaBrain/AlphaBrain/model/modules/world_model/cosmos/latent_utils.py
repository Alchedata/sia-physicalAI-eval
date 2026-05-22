# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Ported from cosmos_policy.models.policy_text2world_model (lines 45-173)
# Pure tensor ops, no model dependencies.

import torch


def replace_latent_with_action_chunk(
    x0: torch.Tensor, action_chunk: torch.Tensor, action_indices: torch.Tensor
) -> torch.Tensor:
    """
    Replace image latent at action_indices with the action chunk.

    x0: (B, C', T', H', W')
    action_chunk: (B, chunk_size, action_dim)  e.g. (B, 16, 7)
    action_indices: (B,) — per-batch temporal index to overwrite

    The action chunk is flattened and tiled to fill the full (C', H', W') volume.
    """
    batch_indices = torch.arange(x0.shape[0], device=x0.device)
    action_image_latent = x0[batch_indices, :, action_indices, :, :]

    result = torch.zeros_like(action_image_latent)
    batch_size, latent_channels, latent_h, latent_w = action_image_latent.shape

    flat_action = action_chunk.reshape(batch_size, -1)
    num_action_elements = flat_action.shape[1]
    latent_elements = latent_channels * latent_h * latent_w

    assert num_action_elements <= latent_elements, (
        f"Not enough room in the latent tensor for the full action chunk: "
        f"{num_action_elements} action elements > {latent_elements} latent elements!"
    )

    num_repeats = (latent_elements + num_action_elements - 1) // num_action_elements
    repeated_action = flat_action.repeat(1, num_repeats)[:, :latent_elements]

    flat_result = result.reshape(batch_size, -1)
    flat_result[:, :] = repeated_action
    result = flat_result.reshape(batch_size, latent_channels, latent_h, latent_w)

    new_x0 = x0
    new_x0[batch_indices, :, action_indices, :, :] = result
    return new_x0


def replace_latent_with_proprio(
    x0: torch.Tensor, proprio: torch.Tensor, proprio_indices: torch.Tensor
) -> torch.Tensor:
    """
    Replace image latent at proprio_indices with the proprio vector.

    x0: (B, C', T', H', W')
    proprio: (B, proprio_dim)  e.g. (B, 9)
    proprio_indices: (B,) — per-batch temporal index to overwrite

    The proprio is tiled to fill the full (C', H', W') volume.
    """
    batch_indices = torch.arange(x0.shape[0], device=x0.device)
    proprio_image_latent = x0[batch_indices, :, proprio_indices, :, :]

    result = torch.zeros_like(proprio_image_latent)
    batch_size, latent_channels, latent_h, latent_w = proprio_image_latent.shape

    num_proprio_elements = proprio.shape[1]
    latent_elements = latent_channels * latent_h * latent_w

    assert num_proprio_elements <= latent_elements, (
        f"Not enough room in the latent tensor for the full proprio: "
        f"{num_proprio_elements} proprio elements > {latent_elements} latent elements!"
    )

    num_repeats = (latent_elements + num_proprio_elements - 1) // num_proprio_elements
    repeated_proprio = proprio.repeat(1, num_repeats)[:, :latent_elements]

    flat_result = result.reshape(batch_size, -1)
    flat_result[:, :] = repeated_proprio
    result = flat_result.reshape(batch_size, latent_channels, latent_h, latent_w)

    new_x0 = x0
    new_x0[batch_indices, :, proprio_indices, :, :] = result
    return new_x0
