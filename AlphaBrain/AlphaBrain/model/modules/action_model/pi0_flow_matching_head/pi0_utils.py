"""
Utility functions for Pi0 flow matching, extracted from openpi.
"""

import math
import torch
from torch import Tensor


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    """Sample from Beta distribution for flow matching time scheduling."""
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """
    Build 2D attention masks from padding masks and autoregressive masks.
    
    Tokens can attend to valid input tokens which have a cumulative mask_ar
    smaller or equal to theirs.
    
    Examples:
      [[0 0 0 1 1 1]]: prefix-lm attention (first 3 bidirectional, last 3 causal)
      [[1 1 1 1 1 1]]: pure causal attention
    """
    if att_masks.ndim != 2:
        raise ValueError(f"att_masks.ndim={att_masks.ndim}, expected 2")
    if pad_masks.ndim != 2:
        raise ValueError(f"pad_masks.ndim={pad_masks.ndim}, expected 2")

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks
