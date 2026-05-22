# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Ported from cosmos_policy._src.imaginaire.utils.denoise_prediction
# Pure PyTorch, no megatron/imaginaire/hydra dependencies.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class DenoisePrediction:
    """
    Container for denoiser model output.

    Attributes:
        x0: Clean data prediction (denoised output).
        eps: Noise prediction (optional).
        logvar: Log variance of noise prediction — can be used as confidence/uncertainty (optional).
    """

    x0: torch.Tensor
    eps: Optional[torch.Tensor] = None
    logvar: Optional[torch.Tensor] = None
