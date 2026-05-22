# Copyright 2026 VLA-Engine. All rights reserved.
#
# Licensed under the VLA-Engine License. You may not use this file except
# in compliance with the License.

"""
V-JEPA 2 / V-JEPA 2.1 Encoder Integration for VLA-Engine.

Wraps Meta's V-JEPA 2 VisionTransformer into the BaseWorldModelEncoder
interface, producing dense patch tokens [B, N, D] from single-frame
images [B, 3, H, W].

Supports all model sizes (ViT-B through ViT-G) and both V-JEPA 2 and
V-JEPA 2.1 encoder variants.
"""

import logging
import os
import re
from contextlib import nullcontext
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from AlphaBrain.model.modules.world_model.base import BaseWorldModelEncoder, WorldModelEncoderConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Checkpoint key priorities: try these in order when loading weights
_CHECKPOINT_KEYS = ("target_encoder", "ema_encoder", "encoder")

# Map from human-readable model size to (factory_function_name, embed_dim, version)
# version: "v2" uses src/models, "v2.1" uses app/vjepa_2_1/models
_MODEL_REGISTRY = {
    # V-JEPA 2 models
    "vit_base":              ("vit_base",              768,  "v2"),
    "vit_large":             ("vit_large",             1024, "v2"),
    "vit_huge":              ("vit_huge",              1280, "v2"),
    "vit_giant":             ("vit_giant_xformers",    1408, "v2"),
    "vit_gigantic":          ("vit_gigantic_xformers", 1664, "v2"),
    # V-JEPA 2.1 models
    "vjepa2_1_vit_base":     ("vit_base",              768,  "v2.1"),
    "vjepa2_1_vit_large":    ("vit_large",             1024, "v2.1"),
    "vjepa2_1_vit_giant":    ("vit_giant_xformers",    1408, "v2.1"),
    "vjepa2_1_vit_gigantic": ("vit_gigantic_xformers", 1664, "v2.1"),
}

# Patterns used to auto-detect model size from checkpoint filename
_SIZE_PATTERNS = [
    (r"vitG",   "vjepa2_1_vit_gigantic"),
    (r"vitg",   "vjepa2_1_vit_giant"),
    (r"vitl",   "vjepa2_1_vit_large"),
    (r"vitb",   "vjepa2_1_vit_base"),
    (r"vith",   "vit_huge"),
    (r"giant",  "vit_giant"),
    (r"huge",   "vit_huge"),
    (r"large",  "vit_large"),
    (r"base",   "vit_base"),
]


def _auto_detect_model_size(checkpoint_path: str) -> str:
    """Attempt to infer model size from the checkpoint filename."""
    basename = os.path.basename(checkpoint_path)
    # Check for v2.1 pattern first (has "vjepa2_1" in name)
    is_v21 = "vjepa2_1" in basename

    for pattern, model_key in _SIZE_PATTERNS:
        if re.search(pattern, basename):
            # If filename indicates v2.1 and key is a v2 key, upgrade
            if is_v21 and not model_key.startswith("vjepa2_1"):
                v21_key = "vjepa2_1_" + model_key
                if v21_key in _MODEL_REGISTRY:
                    return v21_key
            return model_key

    raise ValueError(
        f"Cannot auto-detect model size from checkpoint filename: {basename}. "
        f"Please set the model_size explicitly. Available: {list(_MODEL_REGISTRY.keys())}"
    )


def _clean_state_dict(state_dict: dict) -> dict:
    """Strip 'module.' and 'backbone.' prefixes from state dict keys."""
    cleaned = {}
    for key, val in state_dict.items():
        new_key = key.replace("module.", "").replace("backbone.", "")
        cleaned[new_key] = val
    return cleaned


def _extract_encoder_state_dict(checkpoint: dict) -> dict:
    """Extract and clean the encoder state dict from a V-JEPA checkpoint."""
    for ck in _CHECKPOINT_KEYS:
        if ck in checkpoint:
            logger.info("Using checkpoint key '%s' for encoder weights.", ck)
            return _clean_state_dict(checkpoint[ck])

    # Fallback: maybe the checkpoint IS the state dict already
    logger.warning(
        "No standard encoder key found in checkpoint (%s). "
        "Attempting to use checkpoint dict directly.",
        list(checkpoint.keys())[:10],
    )
    return _clean_state_dict(checkpoint)


# ---------------------------------------------------------------------------
# V-JEPA Encoder
# ---------------------------------------------------------------------------

class VJEPAEncoder(BaseWorldModelEncoder):
    """V-JEPA 2 / 2.1 visual encoder for the VLA-Engine pipeline.

    Produces dense patch tokens [B, N_patches, embed_dim] from single-frame
    images [B, 3, H, W].  No CLS token is emitted.
    """

    def __init__(
        self,
        config: WorldModelEncoderConfig,
        model_size: Optional[str] = None,
    ):
        super().__init__(config)
        self.wm_config = config
        self.model_size = model_size
        self._embed_dim: int = 0
        self.encoder: Optional[nn.Module] = None

        self._build_encoder()

    # ------------------------------------------------------------------ #
    #  Build
    # ------------------------------------------------------------------ #

    def _build_encoder(self) -> None:
        """Build and load the V-JEPA encoder."""

        # --- Resolve model size -------------------------------------------
        model_size = self.model_size
        if model_size is None and self.wm_config.checkpoint_path:
            model_size = _auto_detect_model_size(self.wm_config.checkpoint_path)
            logger.info("Auto-detected model size: %s", model_size)
        if model_size is None:
            model_size = "vjepa2_1_vit_gigantic"
            logger.info("No checkpoint path; defaulting to %s", model_size)

        if model_size not in _MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model_size '{model_size}'. "
                f"Choose from: {list(_MODEL_REGISTRY.keys())}"
            )

        factory_name, embed_dim, version = _MODEL_REGISTRY[model_size]
        self._embed_dim = embed_dim
        self.model_size = model_size

        # --- Import from local self-contained vjepa2 module ----------------
        from AlphaBrain.model.modules.world_model.vjepa import vision_transformer as vit_module

        img_size = self.wm_config.image_size or 384

        # Geometry hyperparams (config-driven; defaults preserve legacy values).
        patch_size = getattr(self.wm_config, "vjepa_patch_size", 16)
        num_frames = getattr(self.wm_config, "vjepa_num_frames", 16)
        tubelet_size = getattr(self.wm_config, "vjepa_tubelet_size", 2)
        use_rope = getattr(self.wm_config, "vjepa_use_rope", True)
        interpolate_rope = getattr(self.wm_config, "vjepa_interpolate_rope", True)

        encoder_kwargs = dict(
            patch_size=patch_size,
            img_size=(img_size, img_size),
            num_frames=num_frames,       # must be >1 so patch_embed uses PatchEmbed3D matching checkpoint
            tubelet_size=tubelet_size,
            use_sdpa=True,
            use_SiLU=False,
            wide_SiLU=True,
            uniform_power=False,
            use_rope=use_rope,
            img_temporal_dim_size=1,
            interpolate_rope=interpolate_rope,
        )

        # Instantiate encoder via factory function
        if not hasattr(vit_module, factory_name):
            raise AttributeError(
                f"Vision transformer module ({version}) has no factory '{factory_name}'. "
                f"Available: {[k for k in dir(vit_module) if k.startswith('vit_')]}"
            )
        factory_fn = getattr(vit_module, factory_name)
        self.encoder = factory_fn(**encoder_kwargs)
        logger.info(
            "Built V-JEPA %s encoder (%s): embed_dim=%d, image_size=%d",
            version, model_size, embed_dim, img_size,
        )

        # --- Load pretrained weights --------------------------------------
        ckpt_path = self.wm_config.checkpoint_path
        if ckpt_path and os.path.isfile(ckpt_path):
            self._load_weights(ckpt_path)
        elif ckpt_path:
            logger.warning(
                "Checkpoint path specified but file not found: %s. "
                "Encoder initialized with random weights.",
                ckpt_path,
            )
        else:
            logger.info("No checkpoint_path provided; using random initialization.")

        # --- Freeze if requested ------------------------------------------
        if self.wm_config.freeze_encoder:
            self._freeze()
            # Cast entire encoder to bf16 to match checkpoint dtype
            # (patch_embed_img is randomly init in fp32 but checkpoint weights are bf16)
            self.encoder = self.encoder.to(dtype=torch.bfloat16)
        else:
            # Trainable mode: also cast to bf16 for memory efficiency and
            # consistency (patch_embed_img is randomly init in fp32 but
            # checkpoint weights are bf16, causing dtype mismatch in forward).
            self.encoder = self.encoder.to(dtype=torch.bfloat16)
            logger.info("V-JEPA encoder set to bf16 (trainable, freeze_encoder=False).")

        # --- Log parameter count ------------------------------------------
        total_params = sum(p.numel() for p in self.encoder.parameters())
        trainable_params = sum(
            p.numel() for p in self.encoder.parameters() if p.requires_grad
        )
        logger.info(
            "V-JEPA encoder params: total=%.2fM, trainable=%.2fM",
            total_params / 1e6,
            trainable_params / 1e6,
        )

    def _load_weights(self, checkpoint_path: str) -> None:
        """Load encoder weights from a V-JEPA checkpoint file."""
        logger.info("Loading V-JEPA weights from %s", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        encoder_sd = _extract_encoder_state_dict(checkpoint)

        # Load with strict=False to tolerate missing pos_embed when using RoPE
        missing, unexpected = self.encoder.load_state_dict(encoder_sd, strict=False)

        if missing:
            # pos_embed is expected to be missing when use_rope=True
            non_trivial_missing = [
                k for k in missing if "pos_embed" not in k
            ]
            if non_trivial_missing:
                logger.warning(
                    "Missing keys (non-positional): %s", non_trivial_missing
                )
            else:
                logger.info(
                    "All missing keys are positional embeddings (expected with RoPE): %s",
                    missing,
                )
        if unexpected:
            logger.warning("Unexpected keys in checkpoint: %s", unexpected)

        logger.info("Successfully loaded V-JEPA encoder weights.")

    def _freeze(self) -> None:
        """Freeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        logger.info("V-JEPA encoder frozen.")

    # ------------------------------------------------------------------ #
    #  Encode
    # ------------------------------------------------------------------ #

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode preprocessed images into dense patch tokens.

        Args:
            pixel_values: [B, 3, H, W] -- already preprocessed and normalised.

        Returns:
            [B, N_patches, embed_dim] dense visual tokens (no CLS token).
        """
        if pixel_values.ndim != 4:
            raise ValueError(
                f"Expected 4D input [B, 3, H, W], got shape {pixel_values.shape}"
            )

        # Unsqueeze to 5D [B,3,1,H,W] so VisionTransformer uses patch_embed_img
        # (check_temporal_dim checks shape[2]==img_temporal_dim_size)
        x = pixel_values.unsqueeze(2)

        # Match encoder device and dtype
        encoder_device = next(self.encoder.parameters()).device
        encoder_dtype = next(self.encoder.parameters()).dtype
        x = x.to(device=encoder_device, dtype=encoder_dtype)

        ctx = torch.no_grad() if self.wm_config.freeze_encoder else nullcontext()
        with ctx:
            out = self.encoder(x)

        # V-JEPA 2 base returns [B, N, D] from self.norm(x)
        # V-JEPA 2.1 returns [B, N, D] from self.norms_block[-1](x) in eval
        # Both are already [B, N, D] tensors
        if isinstance(out, list):
            # If out_layers was set, we get a list -- take the last one
            out = out[-1]

        return out


    # NOTE: V-JEPA intentionally does NOT implement encode_to_latent /
    # encode_images_with_video_loss. The JEPA objective is trained in the
    # pretrained checkpoint and its ViT features carry no generative /
    # reconstruction signal, so an MSE between frame_t and frame_{t+1}
    # features provides no useful supervision (and risks representation
    # collapse). The base WorldModelVLMInterface.forward_with_video_loss
    # detects the absence of these methods via hasattr() and falls back to
    # a plain encode_images() path with video_loss=0.

    # ------------------------------------------------------------------ #
    #  Preprocess
    # ------------------------------------------------------------------ #

    def preprocess(
        self,
        images: Union[List[Image.Image], List[np.ndarray], torch.Tensor],
    ) -> torch.Tensor:
        """Preprocess raw images for the V-JEPA encoder.

        Accepts:
            - list of PIL.Image.Image
            - list of np.ndarray (H, W, 3) uint8
            - torch.Tensor already in [B, 3, H, W] format (returned as-is)

        Returns:
            [B, 3, H, W] float32 tensor, normalised with ImageNet stats.
        """
        if isinstance(images, torch.Tensor):
            return images

        # Handle list of tensors from dataloader
        if isinstance(images, list):
            if len(images) > 0 and isinstance(images[0], torch.Tensor):
                return torch.stack(images, dim=0)
            # List of PIL/ndarray — fall through to per-image processing below

        img_size = self.wm_config.image_size or 384

        transform = transforms.Compose([
            transforms.Resize(
                (img_size, img_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        tensors = []
        for img in images:
            if isinstance(img, np.ndarray):
                # Expect (H, W, 3) uint8 BGR or RGB
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
                img = Image.fromarray(img)
            elif not isinstance(img, Image.Image):
                raise TypeError(
                    f"Unsupported image type: {type(img)}. "
                    f"Expected PIL.Image, np.ndarray, or torch.Tensor."
                )
            # Ensure RGB
            img = img.convert("RGB")
            tensors.append(transform(img))

        return torch.stack(tensors, dim=0)

    # ------------------------------------------------------------------ #
    #  Properties
    # ------------------------------------------------------------------ #

    @property
    def encoder_dim(self) -> int:
        """Native hidden dimension of the V-JEPA encoder."""
        return self._embed_dim
