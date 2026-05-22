"""LoRA injection: freeze backbone + wrap with PEFT + freeze extras.

Extracted verbatim from the (more complete) implementation that previously
lived in `AlphaBrain/training/continual_learning/train.py`. The simpler
implementation in `AlphaBrain/training/train_alphabrain.py` is replaced by
this version (a strict superset — the auto-detect / freeze_extras paths are
no-op when the relevant yaml fields are absent, so QwenGR00T behavior is
identical).
"""
from __future__ import annotations

import logging
from typing import Any

import torch.nn as nn

from AlphaBrain.training.trainer_utils.peft.config import LoRASpec, is_lora_enabled

logger = logging.getLogger(__name__)


def apply_lora(
    model: nn.Module,
    cfg: Any,
    *,
    print_summary: bool = True,
) -> nn.Module:
    """Apply LoRA in-place per spec.

    Steps:
      1. Resolve VLM interface (from `lora.vlm_module` or auto-detect via
         `_VLM_REGISTRY`).
      2. Freeze ALL params of the VLM interface wrapper.
      3. Replace `vlm_interface.model = get_peft_model(...)` so PEFT injects
         LoRA layers (their params are trainable, base remains frozen).
      4. Freeze each module listed in `lora.freeze_extra_modules`.
      5. Modules not touched by the above stay with their original
         `requires_grad` (typically full-FT — e.g. action_model, dino).

    Returns the same `model` instance (mutated in place).
    """
    if not is_lora_enabled(cfg):
        return model

    from peft import get_peft_model
    from AlphaBrain.model.framework.base_framework import _detect_vlm_interface

    spec = LoRASpec.from_omega(cfg)
    lora_config = spec.peft_config()

    # 1. Resolve VLM interface
    if spec.vlm_module:
        if not hasattr(model, spec.vlm_module):
            raise AttributeError(
                f"lora.vlm_module='{spec.vlm_module}' not found on model "
                f"(available: {[n for n, _ in model.named_children()]})"
            )
        vlm_interface = getattr(model, spec.vlm_module)
    else:
        vlm_interface = _detect_vlm_interface(model)
    assert vlm_interface is not None, (
        "No VLM interface found for LoRA injection. "
        "Set lora.vlm_module explicitly in config."
    )

    # 2 + 3. Freeze backbone, inject PEFT
    for p in vlm_interface.parameters():
        p.requires_grad = False
    vlm_interface.model = get_peft_model(vlm_interface.model, lora_config)

    # 4. Freeze extras
    for module_name in spec.freeze_extra_modules:
        if hasattr(model, module_name):
            extra_module = getattr(model, module_name)
            n = sum(1 for _ in extra_module.parameters())
            for p in extra_module.parameters():
                p.requires_grad = False
            logger.info(f"Froze extra module '{module_name}' ({n} param tensors)")
        else:
            logger.warning(f"freeze_extra_modules: '{module_name}' not found, skipping")

    # 5. Logging
    if print_summary:
        logger.info("LoRA enabled on VLM backbone")
        try:
            vlm_interface.model.print_trainable_parameters()
        except Exception:
            pass
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Total trainable: {trainable / 1e6:.1f}M / {total / 1e6:.1f}M "
            f"({100 * trainable / max(total, 1):.2f}%)"
        )

    return model
