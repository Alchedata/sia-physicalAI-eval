"""LoRA checkpoint save / load+merge.

File-name conventions are kept identical to the previous inline code, so
existing checkpoints (5d / 5h / 5l etc.) remain merge-and-eval compatible:

    <base_path>_lora_adapter/        ← PEFT adapter directory
      adapter_config.json
      adapter_model.safetensors
    <base_path>_action_model.pt      ← non-VLM weights (action_model + extras
                                       like layer_qformer / edit_model / dino)
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)


def _resolve_vlm_interface(model, vlm_module: str | None = None):
    """Return the VLM interface submodule (auto-detect if `vlm_module` is None)."""
    from AlphaBrain.model.framework.base_framework import _detect_vlm_interface
    if vlm_module:
        return getattr(model, vlm_module)
    iface = _detect_vlm_interface(model)
    assert iface is not None, "No VLM interface found on model"
    return iface


def _vlm_attr_prefixes(model) -> tuple[str, ...]:
    """All possible VLM-interface attribute prefixes that exist on `model`.

    Used to filter out VLM weights from the state_dict (they are recovered
    via LoRA merge at eval time).
    """
    from AlphaBrain.model.framework.base_framework import _VLM_REGISTRY
    return tuple(f"{attr}." for _, attr in _VLM_REGISTRY if hasattr(model, attr))


def save_lora_checkpoint(
    *,
    accelerator,
    model,
    base_path: str,
    cfg: Any,
) -> None:
    """Save LoRA adapter + non-VLM weights for a checkpoint.

    Creates:
      <base_path>_lora_adapter/        (PEFT adapter)
      <base_path>_action_model.pt      (all keys NOT starting with a VLM attr prefix)
    """
    unwrapped = accelerator.unwrap_model(model)
    vlm_module = (
        cfg.get("lora", {}).get("vlm_module")
        if hasattr(cfg, "get")
        else getattr(getattr(cfg, "lora", None), "vlm_module", None)
    )

    # 1. Adapter
    vlm_interface = _resolve_vlm_interface(unwrapped, vlm_module)
    adapter_path = base_path + "_lora_adapter"
    vlm_interface.model.save_pretrained(adapter_path)

    # 2. Non-VLM weights
    vlm_prefixes = _vlm_attr_prefixes(unwrapped)
    state_dict = accelerator.get_state_dict(model)
    non_vlm_state = {k: v for k, v in state_dict.items() if not k.startswith(vlm_prefixes)}
    torch.save(non_vlm_state, base_path + "_action_model.pt")

    logger.info(
        f"LoRA checkpoint saved: {adapter_path} + non-VLM weights "
        f"({len(non_vlm_state)} keys)"
    )


def load_and_merge(
    *,
    base_model_factory: Callable[[], "torch.nn.Module"],
    lora_adapter_dir: str,
    action_model_pt: str,
    output_path: str,
    vlm_module: str | None = None,
) -> None:
    """Build base model, attach LoRA adapter, merge, load extras, save full ckpt.

    The output is a single `.pt` file usable by `BaseFramework.from_pretrained`,
    suitable for the standard server_policy + eval_libero pipeline.
    """
    from peft import PeftModel

    print(f"[1/4] Build base model")
    vla = base_model_factory()

    print(f"[2/4] Attach + merge LoRA adapter from {lora_adapter_dir}")
    vlm_interface = _resolve_vlm_interface(vla, vlm_module)
    vlm_interface.model = PeftModel.from_pretrained(
        vlm_interface.model,
        lora_adapter_dir,
    )
    vlm_interface.model = vlm_interface.model.merge_and_unload()
    print("  LoRA merged into VLM backbone")

    print(f"[3/4] Load non-VLM weights from {action_model_pt}")
    non_vlm_state = torch.load(action_model_pt, map_location="cpu")
    missing, unexpected = vla.load_state_dict(non_vlm_state, strict=False)
    if unexpected:
        print(f"  WARNING: unexpected keys: {unexpected[:5]}...")
    print(
        f"  Loaded {len(non_vlm_state)} non-VLM keys "
        f"(missing {len(missing)} VLM keys as expected — recovered via LoRA merge)"
    )

    print(f"[4/4] Save merged checkpoint to {output_path}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    full_state = vla.state_dict()
    torch.save(full_state, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Done! Merged checkpoint: {size_mb:.0f} MB")
