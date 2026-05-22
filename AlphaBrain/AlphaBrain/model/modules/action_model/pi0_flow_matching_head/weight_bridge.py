"""
Weight Bridge: Load openpi π₀/π₀.₅ PyTorch checkpoints into AlphaBrain's decomposed modules.

Supports two sources:
1. openpi converted safetensors (from convert_jax_model_to_pytorch.py)
   - Keys look like: paligemma_with_expert.paligemma.xxx / paligemma_with_expert.gemma_expert.xxx
2. Direct JAX→AlphaBrain conversion (requires orbax + numpy, for advanced users)

Usage:
    from AlphaBrain.model.modules.action_model.pi0_flow_matching_head.weight_bridge import load_pi0_weights
    load_pi0_weights(paligemma_oft_model, "/path/to/openpi_pytorch_checkpoint/model.safetensors")
"""

import logging
import re
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# ── Key mapping: openpi PyTorch → AlphaBrain decomposed ──

def _map_openpi_key_to_alphabrain(openpi_key: str) -> Optional[str]:
    """
    Map openpi PyTorch state_dict key to AlphaBrain PaliGemmaOFT key.
    
    openpi structure:
        paligemma_with_expert.paligemma.model.vision_tower.xxx  → vlm_interface.model.vision_tower.xxx
        paligemma_with_expert.paligemma.model.language_model.xxx → vlm_interface.model.language_model.xxx
        paligemma_with_expert.paligemma.model.multi_modal_projector.xxx → vlm_interface.model.multi_modal_projector.xxx
        paligemma_with_expert.gemma_expert.model.xxx → flow_matching_head.action_expert.model.model.xxx
        action_in_proj.xxx → flow_matching_head.action_in_proj.xxx
        action_out_proj.xxx → flow_matching_head.action_out_proj.xxx
        state_proj.xxx → flow_matching_head.state_proj.xxx
        action_time_mlp_in/out.xxx → flow_matching_head.action_time_mlp_in/out.xxx
        time_mlp_in/out.xxx → flow_matching_head.time_mlp_in/out.xxx
    """
    # VLM (PaliGemma) weights
    if openpi_key.startswith("paligemma_with_expert.paligemma.model."):
        suffix = openpi_key[len("paligemma_with_expert.paligemma.model."):]
        return f"vlm_interface.model.{suffix}"
    
    if openpi_key.startswith("paligemma_with_expert.paligemma."):
        # e.g. paligemma_with_expert.paligemma.lm_head.weight
        suffix = openpi_key[len("paligemma_with_expert.paligemma."):]
        return f"vlm_interface.model.{suffix}"
    
    # Action Expert (Gemma) weights
    if openpi_key.startswith("paligemma_with_expert.gemma_expert.model."):
        suffix = openpi_key[len("paligemma_with_expert.gemma_expert.model."):]
        return f"flow_matching_head.action_expert.model.model.{suffix}"
    
    if openpi_key.startswith("paligemma_with_expert.gemma_expert."):
        suffix = openpi_key[len("paligemma_with_expert.gemma_expert."):]
        return f"flow_matching_head.action_expert.model.{suffix}"
    
    # Projection layers (directly on PI0Pytorch → flow_matching_head)
    projection_prefixes = [
        "action_in_proj", "action_out_proj",
        "state_proj",
        "action_time_mlp_in", "action_time_mlp_out",
        "time_mlp_in", "time_mlp_out",
    ]
    for prefix in projection_prefixes:
        if openpi_key.startswith(f"{prefix}."):
            return f"flow_matching_head.{openpi_key}"
    
    logger.warning(f"Unmapped openpi key: {openpi_key}")
    return None


def _fixup_vlm_keys(mapped_state: dict) -> dict:
    """Fix VLM key paths for PaliGemmaVLM model structure."""
    result = {}
    for k, v in mapped_state.items():
        if not k.startswith("vlm_interface."):
            result[k] = v
            continue
        key = k
        key = key.replace("multi_modal_projector.linear.", "multi_modal_projector.")
        result[key] = v
    return result


def load_pi0_weights(
    model,
    checkpoint_path: str,
    strict: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Load openpi PyTorch checkpoint into AlphaBrain PaliGemmaOFT model.
    
    Args:
        model: PaliGemma_OFT instance
        checkpoint_path: path to model.safetensors or .pt file
        strict: if True, raise on missing/unexpected keys
        verbose: print loading summary
        
    Returns:
        dict with 'matched', 'missing', 'unexpected' key lists
    """
    checkpoint_path = Path(checkpoint_path)
    
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file
        openpi_state = load_file(str(checkpoint_path))
    else:
        openpi_state = torch.load(str(checkpoint_path), map_location="cpu")
    
    # Map keys
    mapped_state = {}
    unmapped_keys = []
    
    for openpi_key, value in openpi_state.items():
        alphabrain_key = _map_openpi_key_to_alphabrain(openpi_key)
        if alphabrain_key is not None:
            mapped_state[alphabrain_key] = value
        else:
            unmapped_keys.append(openpi_key)
    
    # Fix VLM keys for PaliGemmaVLM structure
    mapped_state = _fixup_vlm_keys(mapped_state)

    # Load into model
    model_state = model.state_dict()
    
    matched = []
    missing = []
    shape_mismatch = []
    
    for key in model_state:
        if key in mapped_state:
            if model_state[key].shape == mapped_state[key].shape:
                matched.append(key)
            else:
                shape_mismatch.append(
                    f"{key}: model={model_state[key].shape} vs ckpt={mapped_state[key].shape}"
                )
        else:
            missing.append(key)
    
    unexpected = [k for k in mapped_state if k not in model_state]
    
    # Actually load the matched weights
    load_dict = {k: mapped_state[k] for k in matched}
    model.load_state_dict(load_dict, strict=False)
    
    if verbose:
        logger.info(f"Weight loading summary:")
        logger.info(f"  Matched:        {len(matched)}/{len(model_state)}")
        logger.info(f"  Missing:        {len(missing)}")
        logger.info(f"  Unexpected:     {len(unexpected)}")
        logger.info(f"  Shape mismatch: {len(shape_mismatch)}")
        logger.info(f"  Unmapped openpi keys: {len(unmapped_keys)}")
        
        if shape_mismatch:
            for s in shape_mismatch[:5]:
                logger.warning(f"  Shape mismatch: {s}")
    
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"Strict loading failed.\n"
            f"Missing keys ({len(missing)}): {missing[:10]}\n"
            f"Unexpected keys ({len(unexpected)}): {unexpected[:10]}"
        )
    
    return {
        "matched": matched,
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatch": shape_mismatch,
        "unmapped": unmapped_keys,
    }


def convert_openpi_jax_to_alphabrain(
    jax_checkpoint_dir: str,
    output_path: str,
    pi05: bool = True,
    precision: str = "bfloat16",
):
    """
    Convert openpi JAX checkpoint directly to AlphaBrain format.
    
    Requires: orbax-checkpoint, numpy, flax (for reading JAX checkpoints)
    
    This is a standalone conversion script that produces a safetensors file
    with AlphaBrain-compatible keys. Run once, then use load_pi0_weights().
    
    Args:
        jax_checkpoint_dir: path to openpi JAX checkpoint (contains params/)
        output_path: path to save converted model.safetensors
        pi05: True for π₀.₅, False for π₀
        precision: "float32" or "bfloat16"
    """
    try:
        import numpy as np
        import orbax.checkpoint as ocp
        from flax.nnx import traversals
    except ImportError:
        raise ImportError(
            "JAX→AlphaBrain conversion requires: orbax-checkpoint, flax, numpy.\n"
            "Install with: pip install orbax-checkpoint flax numpy\n"
            "Alternatively, use openpi's convert_jax_model_to_pytorch.py first, "
            "then use load_pi0_weights() to load the result."
        )
    
    import json
    from safetensors.torch import save_file
    
    logger.info(f"Converting JAX checkpoint: {jax_checkpoint_dir}")
    logger.info(f"Mode: {'pi0.5' if pi05 else 'pi0'}, precision: {precision}")
    
    # Step 1: Load JAX params
    # This is complex and depends on the exact openpi checkpoint format.
    # For most users, it's easier to:
    # 1. Use openpi's convert_jax_model_to_pytorch.py to get safetensors
    # 2. Then use load_pi0_weights() above
    
    raise NotImplementedError(
        "Direct JAX→AlphaBrain conversion is complex. Recommended workflow:\n"
        "1. Use openpi's script: python examples/convert_jax_model_to_pytorch.py \\\n"
        "     --checkpoint_dir <jax_ckpt> --config_name pi05_base --output_path <output>\n"
        "2. Then load into AlphaBrain: load_pi0_weights(model, '<output>/model.safetensors')\n"
    )
