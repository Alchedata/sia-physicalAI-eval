"""
GPU-architecture-invariant random noise generation.

Matches original cosmos-policy's arch_invariant_rand from
cosmos_policy._src.imaginaire.utils.misc.
Uses NumPy CPU RNG for deterministic, reproducible noise across all GPU types.
"""

import numpy as np
import torch


def arch_invariant_rand(shape, seed=1):
    """
    Generate deterministic random noise that is invariant to GPU architecture.

    Args:
        shape: tuple, output tensor shape
        seed: int, random seed for reproducibility

    Returns:
        torch.Tensor of shape `shape`, float32, on CPU
    """
    rng = np.random.RandomState(seed)
    return torch.from_numpy(rng.standard_normal(shape).astype(np.float32))
