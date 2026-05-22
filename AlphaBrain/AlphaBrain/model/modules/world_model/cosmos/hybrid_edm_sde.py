# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Ported from cosmos_policy.modules.hybrid_edm_sde
# Pure PyTorch, no megatron/imaginaire/hydra dependencies.

import numpy as np
import torch
from scipy.stats import norm as _norm

from .edm_utils import EDMSDE


class HybridEDMSDE(EDMSDE):
    """
    Extended EDMSDE for Cosmos Policy with hybrid sigma distribution.

    Combines:
    - 70% samples from log-normal distribution
    - 30% samples from uniform distribution over [uniform_lower, uniform_upper]

    LIBERO defaults: p_mean=1.3862943611198906, p_std=1.2, sigma_max=200, sigma_min=0.01
    """

    def __init__(
        self,
        p_mean: float = 1.3862943611198906,
        p_std: float = 1.2,
        sigma_max: float = 200.0,
        sigma_min: float = 0.01,
        hybrid_sigma_distribution: bool = False,
        uniform_lower: float = 1.0,
        uniform_upper: float = 85.0,
    ):
        super().__init__(p_mean=p_mean, p_std=p_std, sigma_max=sigma_max, sigma_min=sigma_min)
        self.hybrid_sigma_distribution = hybrid_sigma_distribution
        self.uniform_lower = uniform_lower
        self.uniform_upper = uniform_upper

    def sample_t(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        """
        Sample sigma values.

        When hybrid_sigma_distribution=True: 70% log-normal + 30% uniform.
        Otherwise: pure log-normal (base class).
        """
        if self.hybrid_sigma_distribution:
            samples = torch.zeros(batch_size, device=device)

            # 70/30 split
            distribution_choice = torch.rand(batch_size, device=device) < 0.7

            num_lognormal = distribution_choice.sum().item()
            num_uniform = batch_size - num_lognormal

            if num_lognormal > 0:
                cdf_vals = np.random.uniform(size=(num_lognormal,))
                log_sigmas_np = _norm.ppf(cdf_vals, loc=self.p_mean, scale=self.p_std)
                log_sigmas = torch.tensor(log_sigmas_np, dtype=torch.float32, device=device)
                samples[distribution_choice] = torch.exp(log_sigmas)

            if num_uniform > 0:
                samples[~distribution_choice] = torch.empty(num_uniform, device=device).uniform_(
                    self.uniform_lower, self.uniform_upper
                )

            return samples

        return super().sample_t(batch_size, device=device)
