"""LoRA / PEFT helpers shared across all trainers.

Public API
----------
    is_lora_enabled(cfg)                            -> bool
    apply_lora(model, cfg)                          -> model (in-place)
    save_lora_checkpoint(accelerator, model, base_path, cfg)
    load_and_merge(base_model_factory, lora_adapter_dir,
                   action_model_pt, output_path, vlm_module=None)

Schema
------
The `lora:` block in training yaml is parsed by `LoRASpec.from_omega`.
See `config.py` for the recognized fields. Backward-compatible with all
existing yaml configs under `configs/continual_learning/`.

Checkpoint layout (unchanged from previous inline implementation):
    <base>_lora_adapter/      PEFT adapter directory
    <base>_action_model.pt    Non-VLM trainable weights
"""
from AlphaBrain.training.trainer_utils.peft.config import LoRASpec, is_lora_enabled
from AlphaBrain.training.trainer_utils.peft.injector import apply_lora
from AlphaBrain.training.trainer_utils.peft.checkpoint import (
    save_lora_checkpoint,
    load_and_merge,
)

__all__ = [
    "LoRASpec",
    "is_lora_enabled",
    "apply_lora",
    "save_lora_checkpoint",
    "load_and_merge",
]
