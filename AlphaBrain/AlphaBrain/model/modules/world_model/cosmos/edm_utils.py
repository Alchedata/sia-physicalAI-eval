# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Ported from cosmos_policy._src.imaginaire.modules.edm_sde
# Pure PyTorch, no megatron/imaginaire/hydra dependencies.

import numpy as np
import torch
from scipy.stats import norm as _norm


class EDMSDE:
    """
    EDM (Elucidating the Design Space of Diffusion-Based Generative Models) SDE base class.

    Implements log-normal sigma sampling and marginal probability for EDM-style diffusion.
    """

    def __init__(
        self,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        sigma_max: float = 80.0,
        sigma_min: float = 0.002,
    ):
        self.p_mean = p_mean
        self.p_std = p_std
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min

    def sample_t(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        """Sample sigma values from log-normal distribution."""
        cdf_vals = np.random.uniform(size=(batch_size,))
        # inv_cdf of N(p_mean, p_std): ppf(u) = p_mean + p_std * ppf_standard(u)
        log_sigmas = _norm.ppf(cdf_vals, loc=self.p_mean, scale=self.p_std)
        log_sigma = torch.tensor(log_sigmas, dtype=torch.float32, device=device)
        return torch.exp(log_sigma)

    def marginal_prob(self, x0: torch.Tensor, sigma: torch.Tensor) -> tuple:
        """
        Compute marginal probability parameters.
        Returns (mean, std) — trivial in base class (identity).
        """
        return x0, sigma
