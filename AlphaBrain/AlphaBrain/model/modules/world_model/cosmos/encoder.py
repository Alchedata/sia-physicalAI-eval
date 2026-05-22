# Copyright 2026 VLA-Engine. All rights reserved.
#
# Licensed under the VLA-Engine License. You may not use this file except
# in compliance with the License.

"""
Cosmos Predict2 Encoder (diffusers-native) for VLA-Engine World Model Interface.

Uses the frozen Cosmos DiT (CosmosTransformer3DModel from diffusers, 2B) as a
visual feature extractor:
  - WAN VAE encodes single images to latent space
  - DiT processes the latent at sigma_min (clean data limit) with text conditioning
  - Intermediate DiT block features are extracted via forward hooks

Key differences from world_model_cosmos.py (MiniTrainDIT backend):
  - 17 input channels (16 latent + 1 condition_mask), NOT 18 like MiniTrainDIT
  - Uses diffusers API (hidden_states, timestep, encoder_hidden_states, ...)
  - Intermediate features captured via register_forward_hook instead of
    intermediate_feature_ids parameter
  - Scheduler: FlowMatchEulerDiscreteScheduler
"""

import logging
import math
import os
from contextlib import nullcontext
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from AlphaBrain.model.modules.world_model.base import BaseWorldModelEncoder, WorldModelEncoderConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# NOTE: these module-level constants are kept as the *factory defaults* only.
# Each encoder instance reads its effective values from WorldModelEncoderConfig
# in __init__ (self._sigma_min_cfg / self._feature_layer etc.) so runs can
# ablate them from yaml without code edits. Do NOT reference these constants
# inside methods -- use the self.* attributes.
_SIGMA_MIN = 0.002       # from scheduler_config.json
_SIGMA_DATA = 1.0        # sigma_data = 1.0
_SIGMA_MAX = 80.0        # from scheduler_config.json
_SIGMA_CONDITIONAL = 0.0001  # condition frame sigma (matches original repo)
_MODEL_CHANNELS = 2048   # num_attention_heads(16) * attention_head_dim(128)
_NUM_BLOCKS = 28
_TEXT_EMBED_DIM = 1024    # T5 d_model for DiT cross-attention
_FEATURE_LAYER = 18      # Layer 18 for action features (DiT4DiT ablation: mid-layer > last layer)


# ---------------------------------------------------------------------------
# Cosmos2DiffusersEncoder
# ---------------------------------------------------------------------------

class Cosmos2DiffusersEncoder(BaseWorldModelEncoder):
    """
    Cosmos Predict2 DiT (diffusers-native) as a frozen visual feature extractor.

    Architecture:
      image -> VAE encode -> latent -> DiT(sigma_min, text) -> layer 18 features

    The DiT is CosmosTransformer3DModel loaded via diffusers, with 17 input
    channels (16 latent + 1 condition_mask binary channel). Intermediate features
    are extracted from layer 18 via a forward hook on transformer_blocks[18].

    Config fields used:
      - checkpoint_path: path to the Cosmos-Predict2-2B-Video2World directory
      - pretrained_dir: directory containing tokenizer/ for VAE weights
      - image_size: input image resolution (default 224)
    """

    def __init__(self, config: WorldModelEncoderConfig):
        super().__init__(config)
        self._model_channels = _MODEL_CHANNELS
        self._intermediate_features = {}

        # Pull diffusion + feature-layer hyperparams from config (with the
        # module-level constants as defaults so unconfigured runs preserve
        # legacy behaviour).  Using distinct attribute names to avoid a
        # collision with the registered buffer `self._sigma_min` below.
        self._sigma_min_cfg = getattr(config, "sigma_min", _SIGMA_MIN)
        self._sigma_max_cfg = getattr(config, "sigma_max", _SIGMA_MAX)
        self._sigma_data_cfg = getattr(config, "sigma_data", _SIGMA_DATA)
        self._sigma_conditional_cfg = getattr(
            config, "sigma_conditional", _SIGMA_CONDITIONAL,
        )
        self._feature_layer = (
            getattr(config, "feature_layer_id", None) or _FEATURE_LAYER
        )

        self._build_encoder()

    # -----------------------------------------------------------------
    # Build
    # -----------------------------------------------------------------

    def _build_encoder(self) -> None:
        from diffusers.models import CosmosTransformer3DModel
        from AlphaBrain.model.modules.world_model.cosmos.wan_vae import WanVAEWrapper

        # --- DiT backbone (diffusers native) ---
        pretrained_dir = self.config.checkpoint_path
        if not pretrained_dir:
            pretrained_dir = self.config.pretrained_dir

        logger.info(
            "Building CosmosTransformer3DModel (diffusers) from %s ...",
            pretrained_dir,
        )
        self.dit = CosmosTransformer3DModel.from_pretrained(
            pretrained_dir,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )

        # Disable concat_padding_mask - patch_embed expects 18ch (17+condition), not 19
        self.dit.config.concat_padding_mask = False

        # Register forward hook on the configured feature layer for
        # intermediate feature extraction.
        self._register_feature_hook(self._feature_layer)

        logger.info("Cosmos2 DiT encoder is TRAINABLE (always).")

        # --- VAE tokenizer (same WanVAE architecture as Cosmos 2.5) ---
        logger.info("Building WanVAEWrapper for Cosmos2 encoder ...")
        vae_dir = self._find_vae_dir(pretrained_dir)
        self.vae = WanVAEWrapper(
            pretrained_dir=vae_dir,
            dtype=torch.bfloat16,
            device="cpu",
            temporal_window=16,
        )
        self.vae.eval()
        self.vae.requires_grad_(False)

        # --- Native text encoder (T5 or precomputed) ---
        self._init_native_text_encoder()

        # Pre-register sigma_min buffer (tensor form used elsewhere as a
        # device-aware scalar; value mirrors the config-driven scalar).
        self.register_buffer(
            "_sigma_min",
            torch.tensor([self._sigma_min_cfg], dtype=torch.float32),
            persistent=False,
        )

        # Log parameter counts
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "Cosmos2DiffusersEncoder built: model_channels=%d, num_blocks=%d, "
            "feature_layer=%d, sigma=[%.4f, %.4f], total=%.2fM, trainable=%.2fM",
            self._model_channels,
            _NUM_BLOCKS,
            self._feature_layer,
            self._sigma_min_cfg,
            self._sigma_max_cfg,
            total_params / 1e6,
            trainable_params / 1e6,
        )

    def _register_feature_hook(self, layer_idx: int) -> None:
        """Register a forward hook on transformer_blocks[layer_idx] to capture
        intermediate features during the DiT forward pass."""
        def hook_fn(module, input, output):
            # CosmosTransformerBlock returns a single tensor (hidden_states)
            if isinstance(output, tuple):
                self._intermediate_features[layer_idx] = output[0]
            else:
                self._intermediate_features[layer_idx] = output
        self.dit.transformer_blocks[layer_idx].register_forward_hook(hook_fn)
        logger.info("Registered forward hook on transformer_blocks[%d]", layer_idx)

    def _find_vae_dir(self, pretrained_dir: str) -> str:
        """Locate a directory suitable for WanVAEWrapper loading."""
        # Check if pretrained_dir has tokenizer/tokenizer.pth
        if os.path.isfile(os.path.join(pretrained_dir, "tokenizer", "tokenizer.pth")):
            return pretrained_dir

        # Check config pretrained_dir
        if self.config.pretrained_dir and os.path.isfile(
            os.path.join(self.config.pretrained_dir, "tokenizer", "tokenizer.pth")
        ):
            return self.config.pretrained_dir

        # Common fallback paths
        base = os.environ.get("PRETRAINED_MODELS_DIR", "data/pretrained_models")
        for name in ["Cosmos-Predict2-2B-Video2World", "Cosmos-Predict2.5-2B"]:
            fb = os.path.join(base, name)
            if os.path.isfile(os.path.join(fb, "tokenizer", "tokenizer.pth")):
                return fb

        logger.warning(
            "Could not locate VAE tokenizer.pth; WanVAEWrapper will attempt "
            "to load from %s",
            pretrained_dir,
        )
        return pretrained_dir

    # -----------------------------------------------------------------
    # Native text encoder
    # -----------------------------------------------------------------

    def _init_native_text_encoder(self) -> None:
        """Load the native text encoder for DiT cross-attention conditioning.

        Backend-aware routing (decided at runtime from ``self.config.backend``,
        NOT from file path):
          - Cosmos 2.5 (predict2.5): NVIDIA pretrains with Reason1 (Qwen2.5-VL-7B)
            28-layer concat -> Linear(100352, 1024). Prefer the 28-layer pkl.
          - Cosmos 2.0 (predict2): NVIDIA pretrains with T5. Prefer T5 pkl, then
            legacy pooled Reason1 pkl (dim=3584) for backward compatibility.
        """
        self._text_encoder_type = None

        backend = (getattr(self.config, "backend", "") or "").lower().strip()
        is_cosmos25 = backend in (
            "cosmos2.5-diffusers", "cosmos25-diff", "predict2.5-diffusers",
            "cosmos2.5", "predict2.5",
        )

        # =========================================================
        # Cosmos 2.5 path: Reason1 28-layer concat (dim=100352)
        # =========================================================
        if is_cosmos25:
            for candidate in [
                "data/pretrained_models/text_embeddings/reason1_projected_text_embeddings.pkl",
                "data/datasets/libero_datasets/reason1_projected_text_embeddings.pkl",
                os.path.join(
                    self.config.pretrained_dir or "",
                    "reason1_projected_text_embeddings.pkl",
                ),
                os.path.join(
                    self.config.checkpoint_path or "",
                    "reason1_projected_text_embeddings.pkl",
                ),
            ]:
                if candidate and os.path.isfile(candidate):
                    import pickle
                    logger.info(
                        "[Cosmos2.5] Loading Reason1 28-layer embeddings from %s",
                        candidate,
                    )
                    with open(candidate, "rb") as f:
                        object.__setattr__(
                            self, "_precomputed_text_cache", pickle.load(f),
                        )
                    logger.info(
                        "[Cosmos2.5] Reason1 28-layer cache: %d instructions",
                        len(self._precomputed_text_cache),
                    )
                    self.native_text_encoder = True
                    self.native_tokenizer = True
                    self._text_encoder_type = "reason1_precomputed"
                    return

            logger.warning(
                "[Cosmos2.5] No reason1_projected_text_embeddings.pkl found. "
                "Run scripts/run_world_model/preprocess/precompute_text_embeddings/precompute_reason1.py first. "
                "Falling back to dummy zero conditioning."
            )
            self.native_text_encoder = None
            self.native_tokenizer = None
            return

        # =========================================================
        # Cosmos 2.0 path: T5 preferred, legacy Reason1 pooled fallback
        # =========================================================
        # --- Branch 1: Precomputed T5 embeddings ---
        for candidate in [
            "data/pretrained_models/text_embeddings/t5_text_embeddings.pkl",
            "data/datasets/libero_datasets/t5_text_embeddings.pkl",
        ]:
            if os.path.isfile(candidate):
                import pickle
                logger.info("Loading precomputed T5 embeddings from %s", candidate)
                with open(candidate, "rb") as f:
                    object.__setattr__(
                        self, "_precomputed_text_cache", pickle.load(f),
                    )
                logger.info(
                    "Precomputed T5 cache: %d instructions",
                    len(self._precomputed_text_cache),
                )
                self.native_text_encoder = True
                self.native_tokenizer = True
                self._text_encoder_type = "t5_precomputed"
                return

        # --- Branch 2: Legacy Reason1 pooled embeddings (dim=3584) ---
        for candidate in [
            "data/pretrained_models/text_embeddings/reason1_text_embeddings.pkl",
            "data/datasets/libero_datasets/reason1_text_embeddings.pkl",
            os.path.join(
                self.config.pretrained_dir or "", "reason1_text_embeddings.pkl",
            ),
            os.path.join(
                self.config.checkpoint_path or "", "reason1_text_embeddings.pkl",
            ),
        ]:
            if os.path.isfile(candidate):
                import pickle
                logger.info(
                    "Loading precomputed Reason1 (pooled) embeddings from %s",
                    candidate,
                )
                with open(candidate, "rb") as f:
                    object.__setattr__(
                        self, "_precomputed_text_cache", pickle.load(f),
                    )
                logger.info(
                    "Precomputed Reason1 cache: %d instructions",
                    len(self._precomputed_text_cache),
                )
                self.reason1_proj = nn.Linear(3584, 1024).to(dtype=torch.bfloat16)
                self._reason1_concat_dim = 3584
                self.native_text_encoder = True
                self.native_tokenizer = True
                self._text_encoder_type = "reason1_precomputed"
                return

        # --- Branch 3: No text encoder available ---
        logger.warning(
            "No precomputed text embeddings found. "
            "Will use dummy zero conditioning."
        )
        self.native_text_encoder = None
        self.native_tokenizer = None

    def encode_text(self, instructions, device):
        """Encode text using precomputed embeddings.

        Returns:
            [B, L, 1024] text embeddings, or None if not available.
        """
        if not self.native_text_encoder or self.native_tokenizer is None:
            return None

        if self._text_encoder_type == "t5_precomputed":
            return self._encode_text_t5_precomputed(instructions, device)
        elif self._text_encoder_type in (
            "reason1_precomputed", "reason1_28layer_precomputed",
        ):
            return self._encode_text_reason1_precomputed(instructions, device)
        return None

    def _encode_text_t5_precomputed(self, instructions, device):
        """Look up precomputed T5 embeddings (1024-dim, matches DiT cross-attn)."""
        batch_embeds = []
        for inst in instructions:
            if inst in self._precomputed_text_cache:
                emb = self._precomputed_text_cache[inst].to(dtype=torch.bfloat16)
            else:
                logger.warning(
                    "Instruction not in T5 precomputed cache: %s", inst[:60],
                )
                emb = torch.zeros(512, 1024, dtype=torch.bfloat16)
            batch_embeds.append(emb)
        return torch.stack(batch_embeds).to(device)  # [B, 512, 1024]

    def _encode_text_reason1_precomputed(self, instructions, device):
        batch_embeds = []
        for inst in instructions:
            if inst in self._precomputed_text_cache:
                emb = self._precomputed_text_cache[inst].to(dtype=torch.bfloat16)
            else:
                emb = torch.zeros(512, 1024, dtype=torch.bfloat16)
            batch_embeds.append(emb)
        return torch.stack(batch_embeds).to(device)
    # -----------------------------------------------------------------
    # Preprocess
    # -----------------------------------------------------------------

    def preprocess(self, images) -> torch.Tensor:
        """Preprocess raw images for encoding.

        Args:
            images: [B, 3, H, W] tensor, or list of PIL.Image / np.ndarray.

        Returns:
            [B, 3, H, W] float tensor in [-1, 1], resized to config.image_size.
        """
        target_size = self.config.image_size

        if isinstance(images, (list, tuple)):
            import numpy as np
            from PIL import Image
            tensors = []
            for img in images:
                if isinstance(img, Image.Image):
                    img = (
                        torch.from_numpy(np.array(img.convert("RGB")))
                        .permute(2, 0, 1)
                        .float()
                        / 255.0
                    )
                elif isinstance(img, np.ndarray):
                    if img.ndim == 3:
                        img = (
                            torch.from_numpy(img)
                            .permute(2, 0, 1)
                            .float()
                            / 255.0
                        )
                    else:
                        img = torch.from_numpy(img).float() / 255.0
                elif isinstance(img, torch.Tensor):
                    pass
                tensors.append(img)
            images = torch.stack(tensors, dim=0)

        if images.ndim == 4 and images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)

        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        elif images.max() > 1.5:
            images = images / 255.0

        if images.shape[-2] != target_size or images.shape[-1] != target_size:
            images = F.interpolate(
                images.float(),
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            )

        # Scale from [0, 1] to [-1, 1] (VAE input range)
        images = images * 2.0 - 1.0
        return images

    # -----------------------------------------------------------------
    # Core encoding methods
    # -----------------------------------------------------------------

    def encode_to_latent(self, images: torch.Tensor) -> torch.Tensor:
        """Encode preprocessed images to VAE latent space only (no DiT forward).

        Args:
            images: [B, 3, H, W] float tensor in [-1, 1].

        Returns:
            [B, 16, 1, 28, 28] VAE latent tensor (bf16).
        """
        encoder_device = next(self.dit.parameters()).device
        images = images.to(device=encoder_device, dtype=torch.bfloat16)
        video = images.unsqueeze(2)  # [B, 3, 1, H, W]
        with torch.no_grad():
            latent = self.vae.encode(video)  # [B, 16, 1, H/8, W/8]
        return latent

    def _build_condition_mask(self, B, T, H, W, device, dtype,
                              condition_frames=None):
        """Build the 1-channel condition_mask for the diffusers DiT.

        Cosmos Predict2 (diffusers) uses 17 input channels:
          - channels 0..15: VAE latent (16 channels)
          - channel 16: condition_mask (1 = condition frame, 0 = prediction frame)

        This is different from MiniTrainDIT which uses 18 channels
        (16 + 2 zero-padding).

        Args:
            B: batch size
            T: number of temporal frames
            H, W: spatial dimensions of latent
            device: target device
            dtype: target dtype
            condition_frames: list/tuple of frame indices that are condition
                              frames. If None, all frames are treated as
                              condition frames.

        Returns:
            condition_mask: [B, 1, T, H, W] binary mask
        """
        condition_mask = torch.zeros(B, 1, T, H, W, device=device, dtype=dtype)
        if condition_frames is not None:
            for t in condition_frames:
                condition_mask[:, :, t, :, :] = 1.0
        else:
            condition_mask.fill_(1.0)
        return condition_mask

    def _pad_to_17ch(self, latent: torch.Tensor,
                     condition_mask: torch.Tensor) -> torch.Tensor:
        """Concatenate 16-ch latent with 1-ch condition_mask to get 17 channels.

        Args:
            latent: [B, 16, T, H, W]
            condition_mask: [B, 1, T, H, W]

        Returns:
            [B, 17, T, H, W]
        """
        return torch.cat([latent, condition_mask], dim=1)

    def _dit_forward(self, hidden_states, timestep, encoder_hidden_states,
                     condition_mask=None, padding_mask=None, fps=None):
        """Call the diffusers CosmosTransformer3DModel forward.

        Args:
            hidden_states: [B, C, T, H, W] -- 17-channel input
            timestep: [B] -- noise level per sample
            encoder_hidden_states: [B, L, 1024] -- text embeddings
            condition_mask: [B, 1, T, H, W] -- binary condition mask
            padding_mask: optional padding mask
            fps: optional FPS tensor

        Returns:
            output: denoised tensor [B, out_channels, T, H, W]
        """
        # Clear any previous intermediate features
        self._intermediate_features.clear()

        # Build padding_mask if not provided (required by diffusers when concat_padding_mask=True)
        # diffusers expects padding_mask as [1, H, W] spatial mask (gets resized + repeated across frames)
        # All regions valid -> all ones
        # padding_mask not needed (concat_padding_mask=False)

        output = self.dit(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            condition_mask=condition_mask,
            padding_mask=padding_mask,
            fps=fps,
            return_dict=False,
        )

        # diffusers returns a tuple when return_dict=False
        if isinstance(output, tuple):
            output = output[0]

        return output

    def encode_images(self, pixel_values: torch.Tensor,
                      text_embeds=None) -> torch.Tensor:
        """Encode preprocessed images into visual token features.

        Uses sigma_min (clean data limit) for deterministic feature extraction.
        Captures layer 18 intermediate features via forward hook.

        Args:
            pixel_values: [B, 3, H, W] float tensor in [-1, 1].
            text_embeds: [B, L, 1024] text embeddings for DiT cross-attention.

        Returns:
            [B, N, 2048] visual feature tokens from layer 18.
        """
        return self._encode_images_inner(pixel_values, text_embeds)

    def _encode_images_inner(self, pixel_values, text_embeds=None):
        """2-frame action inference: frame0 (clean) + frame1 (noise).

        Matches training self-attention context where frame0 tokens attend to
        both frame0 and frame1 tokens. Returns only frame0's layer-18 features.

        During training, frame1 has random sigma ~ LogNormal(0,1); the
        Flow-Matching convention for pure-noise frames is sigma -> sigma_max
        (t -> 1). We therefore use sigma = _SIGMA_MAX (80.0) => t = 80/81
        ~= 0.988 at inference time, aligned with the high-sigma tail of the
        training distribution. We also set the noise latent to zeros so the
        encoder output is fully deterministic (the frame1 contribution to
        c_in * (x0 + sigma * eps) with eps=0 is just 0, paired with the
        sigma_max timestep embedding).
        """
        encoder_device = next(self.dit.parameters()).device
        pixel_values = pixel_values.to(
            device=encoder_device, dtype=torch.bfloat16,
        )

        B = pixel_values.shape[0]
        device = pixel_values.device
        dtype = torch.bfloat16

        # --- 1. VAE encode: [B, 3, H, W] -> [B, 16, 1, H/8, W/8] ---
        video = pixel_values.unsqueeze(2)  # [B, 3, 1, H, W]
        latent = self.vae.encode(video.to(dtype))  # [B, 16, 1, H/8, W/8]
        H_lat = latent.shape[3]
        W_lat = latent.shape[4]

        # --- 2. Build 2-frame input: frame0 (clean) + frame1 (noise) ---
        # Flow-Matching convention: at inference the future frame is pure
        # noise, so use sigma = _SIGMA_MAX (80.0) => t = 80/81 ~= 0.988,
        # matching the high-sigma tail of the LogNormal(0,1) training
        # distribution. Use zeros for the noise latent to make encoding
        # deterministic (c_in * sigma * 0 == 0; the timestep embedding at
        # t ~= 0.988 still signals "pure noise frame" to the DiT).
        sigma_noise = self._sigma_max_cfg
        t_noise = sigma_noise / (sigma_noise + 1.0)  # 80/81 ~= 0.988
        c_in_noise = 1.0 - t_noise                    # 1/81  ~= 0.012
        noise_frame = torch.zeros(
            B, 16, 1, H_lat, W_lat, device=device, dtype=dtype,
        )  # deterministic zero latent; sigma_max timestep carries the "noise" signal
        two_frame_latent = torch.cat([latent, noise_frame], dim=2)  # [B, 16, 2, ...]

        # --- 3. Build condition_mask: frame0=1 (cond), frame1=0 (pred) ---
        condition_mask = self._build_condition_mask(
            B, 2, H_lat, W_lat, device, dtype, condition_frames=[0],
        )  # [B, 1, 2, H, W]

        # --- 4. Apply preconditioning ---
        # c_in for prediction frame (frame1), condition frame gets overwritten
        net_input = two_frame_latent.float() * c_in_noise
        net_input[:, :, :1, :, :] = latent.float() / self._sigma_data_cfg  # clean condition
        hidden_states = self._pad_to_17ch(
            net_input.to(dtype), condition_mask,
        )  # [B, 17, 2, H, W]

        # --- 5. Per-frame timestep [B, 1, 2, 1, 1] ---
        t_cond = self._sigma_conditional_cfg / (self._sigma_conditional_cfg + 1.0)  # ~0.0001
        t_cond_B = torch.full((B,), t_cond, device=device, dtype=dtype)
        t_noise_B = torch.full((B,), t_noise, device=device, dtype=dtype)
        timestep = torch.stack([t_cond_B, t_noise_B], dim=1).reshape(B, 1, 2, 1, 1)

        # --- 6. Text conditioning ---
        if text_embeds is not None:
            encoder_hidden_states = text_embeds.to(device=device, dtype=dtype)
        else:
            encoder_hidden_states = torch.zeros(
                B, 1, _TEXT_EMBED_DIM, device=device, dtype=dtype,
            )

        # --- 7. DiT forward (hook captures layer 18 features) ---
        _ = self._dit_forward(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            condition_mask=condition_mask,
        )

        # --- 8. Extract layer 18 features (frame0-only) ---
        visual_tokens = self._intermediate_features.get(self._feature_layer)
        if visual_tokens is None:
            raise RuntimeError(
                f"Forward hook on layer {self._feature_layer} did not capture "
                "features. Check that the hook is properly registered."
            )

        # visual_tokens: [B, N_total, 2048] where N_total = 2 * (H/p_h) * (W/p_w)
        # Return only frame0 tokens (first half)
        p_h, p_w = self.dit.config.patch_size[1], self.dit.config.patch_size[2]
        n_tokens_per_frame = (H_lat // p_h) * (W_lat // p_w)
        visual_tokens = visual_tokens[:, :n_tokens_per_frame, :]  # [B, 196, 2048]

        return visual_tokens

    def encode_images_with_video_loss(
        self,
        latent_t: torch.Tensor,
        latent_t1: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single DiT forward that yields both action features and video loss.

        V2 single-forward design: runs the DiT ONCE to get both:
          - Layer 18 intermediate features -> for action head
          - Final denoised output           -> for video prediction loss

        Uses 17-channel input (16 latent + 1 condition_mask), where
        condition_mask is a 1-channel binary mask:
          - 1 for condition frames (frame 0)
          - 0 for prediction frames (frame 1)

        Args:
            latent_t:    [B, 16, 1, 28, 28] current frame VAE latent (clean).
            latent_t1:   [B, 16, 1, 28, 28] next frame VAE latent (clean,
                         target).
            text_embeds: [B, L, 1024] text conditioning for DiT cross-attention.

        Returns:
            visual_tokens: [B, N, 2048] layer-18 intermediate features.
            video_loss:    scalar       weighted MSE on predicted future frame.
        """
        device = latent_t.device
        dtype = torch.bfloat16
        B = latent_t.shape[0]
        H_lat, W_lat = latent_t.shape[3], latent_t.shape[4]

        # --- 1. Build 2-frame clean latent: [B, 16, 2, 28, 28] ---
        x0 = torch.cat([latent_t, latent_t1], dim=2)  # [B, 16, 2, 28, 28]

        # --- 2. Build condition_mask ---
        # frame 0 = condition (1), frame 1 = prediction (0)
        condition_mask = self._build_condition_mask(
            B, 2, H_lat, W_lat, device, dtype, condition_frames=[0],
        )  # [B, 1, 2, 28, 28]

        # --- 3. Sample sigma (LogNormal, matching cosmos-policy SDE) ---
        log_sigma = torch.randn(B, device=device)  # LogNormal(p_mean=0.0, p_std=1.0) matching original Cosmos repo
        sigma = log_sigma.exp().clamp(min=1e-4, max=self._sigma_max_cfg)  # [B]

        # --- 4. Add noise to the clean latent ---
        noise = torch.randn_like(x0)
        sigma_5d = sigma.view(B, 1, 1, 1, 1)
        xt = x0 + sigma_5d * noise  # [B, 16, 2, 28, 28]

        # --- 5. RectifiedFlow preconditioning scalars ---
        t = sigma / (sigma + 1.0)  # [B], in (0, 1)
        c_skip = 1.0 - t           # [B]
        c_out = -t                 # [B]
        c_in = 1.0 - t             # [B]

        # --- 6. Condition frame handling ---
        # Replace noisy frame 0 with clean condition
        xt[:, :, :1, :, :] = x0[:, :, :1, :, :]

        # --- 7. Scale net input by c_in, then restore condition frame ---
        net_input = xt.float() * c_in.view(B, 1, 1, 1, 1)
        # Condition frame is always passed as clean: gt / sigma_data
        net_input[:, :, :1, :, :] = (
            x0[:, :, :1, :, :].float() / self._sigma_data_cfg
        )

        # --- 8. Pad to 17 channels (16 latent + 1 condition_mask) ---
        hidden_states = self._pad_to_17ch(
            net_input.to(dtype), condition_mask,
        )  # [B, 17, 2, 28, 28]

        # --- 9. Per-frame timestep [B, 1, 2, 1, 1] ---
        # frame 0 (condition): sigma_conditional -> t_cond ~ 0.0001 (fixed)
        # frame 1 (prediction): random sigma -> t_pred (varies per sample)
        # This matches original Cosmos repo per-frame timestep design.
        # diffusers CosmosTransformer3DModel.forward() natively supports
        # timestep.ndim==5 with shape [B, 1, T, 1, 1].
        t_cond = self._sigma_conditional_cfg / (self._sigma_conditional_cfg + 1.0)
        t_cond_B = torch.full((B,), t_cond, device=device, dtype=dtype)
        t_pred_B = t.to(device=device, dtype=dtype)
        timestep = torch.stack([t_cond_B, t_pred_B], dim=1).reshape(B, 1, 2, 1, 1)

        # --- 10. Text conditioning ---
        if text_embeds is not None:
            encoder_hidden_states = text_embeds.to(device=device, dtype=dtype)
        else:
            encoder_hidden_states = torch.zeros(
                B, 1, _TEXT_EMBED_DIM, device=device, dtype=dtype,
            )

        # --- 11. DiT forward (hook captures layer 18 + final output) ---
        dit_device = next(self.dit.parameters()).device
        net_output = self._dit_forward(
            hidden_states=hidden_states.to(dit_device),
            timestep=timestep.to(dit_device),
            encoder_hidden_states=encoder_hidden_states.to(dit_device),
            condition_mask=condition_mask.to(dit_device),
        )

        # --- 12. Extract layer 18 visual tokens ---
        visual_tokens = self._intermediate_features.get(self._feature_layer)
        if visual_tokens is None:
            raise RuntimeError(
                f"Forward hook on layer {self._feature_layer} did not capture "
                "features."
            )

        # --- 13. Reconstruct x0_pred from net_output ---
        # DiT outputs 16 channels (out_channels=16)
        x0_pred = (
            c_skip.view(B, 1, 1, 1, 1) * xt[:, :16, :, :, :].float()
            + c_out.view(B, 1, 1, 1, 1) * net_output.float().to(xt.device)
        )  # [B, 16, 2, 28, 28]

        # --- 14. Video loss on predicted future frame only (frame 1) ---
        pred_future = x0_pred[:, :16, 1:2, :, :]           # [B, 16, 1, 28, 28]
        gt_future = (
            x0[:, :16, 1:2, :, :].float() * self._sigma_data_cfg
        )  # [B, 16, 1, 28, 28]

        # Loss weight: (1 + sigma)^2 / sigma^2
        weight = ((1.0 + sigma) ** 2 / (sigma ** 2)).view(B, 1, 1, 1, 1)
        video_loss = (
            weight * F.mse_loss(pred_future, gt_future, reduction="none")
        ).mean()

        return visual_tokens, video_loss

    def encode_images_all_layers(
        self,
        pixel_values: torch.Tensor,
        text_embeds=None,
    ) -> List[torch.Tensor]:
        """Encode images and return features from ALL 28 DiT layers (PI-style).

        Registers temporary hooks on all blocks, runs one forward pass,
        then removes the hooks and returns per-layer features.

        Args:
            pixel_values: [B, 3, H, W] float tensor in [-1, 1].
            text_embeds:  [B, L, 1024] text embeddings.

        Returns:
            List of 28 tensors, each [B, N, 2048].
        """
        return self._encode_images_all_layers_inner(
                pixel_values, text_embeds,
            )

    def _encode_images_all_layers_inner(self, pixel_values, text_embeds=None):
        """Register hooks on all 28 blocks, forward, collect frame0-only features."""
        all_features = {}
        handles = []

        # Register temporary hooks on ALL blocks
        for idx in range(_NUM_BLOCKS):
            def make_hook(layer_idx):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        all_features[layer_idx] = output[0]
                    else:
                        all_features[layer_idx] = output
                return hook_fn
            h = self.dit.transformer_blocks[idx].register_forward_hook(
                make_hook(idx),
            )
            handles.append(h)

        try:
            # _encode_images_inner already returns frame0-only tokens,
            # but hooks capture full 2-frame features -- we slice below.
            _ = self._encode_images_inner(pixel_values, text_embeds)
        finally:
            for h in handles:
                h.remove()

        # Determine tokens-per-frame for slicing (2-frame inference)
        ref_feat = next(iter(all_features.values()), None)
        if ref_feat is not None:
            n_tokens_per_frame = ref_feat.shape[1] // 2  # 2-frame -> half
        else:
            n_tokens_per_frame = None

        # Collect in order, slicing to frame0-only
        result = []
        for idx in range(_NUM_BLOCKS):
            if idx in all_features:
                feat = all_features[idx]
                if n_tokens_per_frame is not None:
                    feat = feat[:, :n_tokens_per_frame, :]  # frame0 only
                result.append(feat)
            else:
                logger.warning("Missing features for layer %d", idx)
                ref = all_features.get(self._feature_layer)
                if ref is not None:
                    result.append(torch.zeros_like(ref[:, :n_tokens_per_frame, :]))
                else:
                    raise RuntimeError(
                        "No features captured from any layer."
                    )
        return result

    # -----------------------------------------------------------------
    # V2: Future frame denoising (inference)
    # -----------------------------------------------------------------

    @torch.inference_mode()
    def denoise_future_frame(
        self,
        latent_t: torch.Tensor,
        text_embeds: torch.Tensor,
        num_steps: int = 35,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
        rho: float = 7.0,
    ) -> torch.Tensor:
        """Iterative denoising to generate future frame prediction.

        Uses 2nd-order Adams-Bashforth (2-AB) solver with Karras sigma schedule,
        matching the original Cosmos repo's RectifiedFlowAB2Scheduler:
          - Step 0: Euler (no previous x0 available)
          - Step 1+: 2-AB (uses current + previous x0 predictions)
          - Final: clean pass at sigma_min

        Args:
            latent_t: [B, 16, 1, 28, 28] current frame VAE latent (clean).
            text_embeds: [B, L, 1024] cross-attention text conditioning.
            num_steps: number of denoising steps (default 35, matching original).
            sigma_min: minimum sigma for denoising schedule.  Defaults to
                ``config.sigma_min`` (typically 0.002).
            sigma_max: maximum sigma for denoising schedule.  Defaults to
                ``config.sigma_max`` (typically 80.0).
            rho: Karras schedule order parameter (default 7.0).

        Returns:
            future_latent: [B, 16, 1, 28, 28] predicted future frame latent.
        """
        # Resolve schedule limits from config if caller didn't override.
        if sigma_min is None:
            sigma_min = self._sigma_min_cfg
        if sigma_max is None:
            sigma_max = self._sigma_max_cfg

        device = latent_t.device
        dtype = torch.bfloat16
        B = latent_t.shape[0]
        H_lat, W_lat = latent_t.shape[3], latent_t.shape[4]
        dit_device = next(self.dit.parameters()).device

        if text_embeds is None:
            text_embeds = torch.zeros(
                B, 1, _TEXT_EMBED_DIM, device=dit_device, dtype=dtype,
            )

        # ----------------------------------------------------------------
        # Karras sigma schedule: sigma_i = (sigma_max^(1/rho) + i/(L)*(sigma_min^(1/rho) - sigma_max^(1/rho)))^rho
        # Produces (num_steps + 1) sigma values from sigma_max to sigma_min
        # ----------------------------------------------------------------
        n_sigma = num_steps + 1
        i_vals = torch.arange(n_sigma, device=device, dtype=torch.float64)
        ramp = (
            sigma_max ** (1.0 / rho)
            + i_vals / (n_sigma - 1) * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
        )
        sigmas = (ramp ** rho).to(dtype=torch.float32)  # [num_steps + 1]

        # Initialize: prediction frame starts as pure noise at sigma_max
        xt_future = (
            torch.randn(B, 16, 1, H_lat, W_lat, device=device, dtype=torch.float32)
            * sigmas[0]
        )

        x0_prev = None  # cached x0 from previous step (for 2-AB)

        for i in range(num_steps):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]

            # --- x0 prediction at current sigma ---
            t = sigma / (sigma + 1.0)
            c_skip = 1.0 - t
            c_out = -t
            c_in = 1.0 - t

            # Build 2-frame input: [condition (clean), prediction (noisy)]
            condition_mask = self._build_condition_mask(
                B, 2, H_lat, W_lat, device, dtype, condition_frames=[0],
            )
            two_frame_latent = torch.cat(
                [latent_t.to(dtype), xt_future.to(dtype)], dim=2,
            )

            # Scale input
            net_input = two_frame_latent.float() * c_in
            net_input[:, :, :1, :, :] = latent_t.float() / self._sigma_data_cfg

            # Pad to 17 channels
            hidden_states = self._pad_to_17ch(net_input.to(dtype), condition_mask)

            # Per-frame timestep [B, 1, 2, 1, 1]
            t_cond_val = self._sigma_conditional_cfg / (self._sigma_conditional_cfg + 1.0)
            t_cond_B = torch.full((B,), t_cond_val, device=device, dtype=dtype)
            t_pred_B = (t.expand(B) if t.dim() == 0 else t).to(dtype=dtype)
            timestep = torch.stack([t_cond_B, t_pred_B], dim=1).reshape(
                B, 1, 2, 1, 1,
            ).to(device=device)

            # DiT forward
            net_output = self._dit_forward(
                hidden_states=hidden_states.to(dit_device),
                timestep=timestep.to(dit_device),
                encoder_hidden_states=text_embeds.to(dit_device, dtype=dtype),
                condition_mask=condition_mask.to(dit_device),
            )

            # x0 prediction (future frame only)
            x0_pred = (
                c_skip * two_frame_latent[:, :16].float()
                + c_out * net_output.float()
            )
            x0_future = x0_pred[:, :, 1:2, :, :]  # [B, 16, 1, H, W]

            # --- Solver step ---
            if x0_prev is None:
                # First step: Euler in x0-formulation
                # x_next = (sigma - sigma_next) / sigma * x0 + sigma_next / sigma * x_t
                coef_x0 = (sigma - sigma_next) / sigma
                coef_xt = sigma_next / sigma
                xt_future = (coef_x0 * x0_future + coef_xt * xt_future.float()).to(torch.float32)
            else:
                # Subsequent steps: 2nd-order Adams-Bashforth (2-AB)
                # Matches original repo: res_x0_rk2_step in runge_kutta.py
                sigma_prev = sigmas[i - 1]
                # Convert to log-space (note: -log(sigma) increases as sigma decreases)
                s_log = (-torch.log(sigma)).to(torch.float64)       # current
                t_log = (-torch.log(sigma_next)).to(torch.float64)  # next (larger in log-space)
                m_log = (-torch.log(sigma_prev)).to(torch.float64)  # previous (smaller in log-space)

                dt = t_log - s_log  # > 0 (sigma decreasing)
                c2 = (m_log - s_log) / dt

                # phi functions: phi1(x) = (exp(x)-1)/x, phi2(x) = (phi1(x)-1)/x
                neg_dt = -dt  # < 0
                phi1_val = (torch.expm1(neg_dt) / neg_dt)  # = (1-exp(-dt))/dt
                phi2_val = (phi1_val - 1.0) / neg_dt       # second-order correction

                b1 = phi1_val - 1.0 / c2 * phi2_val
                b2 = 1.0 / c2 * phi2_val

                # exp(-dt) is the decay factor (< 1 since dt > 0)
                xt_future = (
                    (torch.exp(neg_dt) * xt_future.to(torch.float64)
                     + dt * (b1 * x0_future.to(torch.float64) + b2 * x0_prev.to(torch.float64)))
                ).to(torch.float32)

            x0_prev = x0_future.clone()

        # --- Final clean pass at sigma_min ---
        sigma_final = sigmas[-1]
        t_final = sigma_final / (sigma_final + 1.0)
        c_skip_f = 1.0 - t_final
        c_out_f = -t_final
        c_in_f = 1.0 - t_final

        condition_mask = self._build_condition_mask(
            B, 2, H_lat, W_lat, device, dtype, condition_frames=[0],
        )
        two_frame_latent = torch.cat(
            [latent_t.to(dtype), xt_future.to(dtype)], dim=2,
        )
        net_input = two_frame_latent.float() * c_in_f
        net_input[:, :, :1, :, :] = latent_t.float() / self._sigma_data_cfg
        hidden_states = self._pad_to_17ch(net_input.to(dtype), condition_mask)

        t_cond_B = torch.full((B,), t_cond_val, device=device, dtype=dtype)
        t_pred_B = torch.full((B,), t_final, device=device, dtype=dtype).to(dtype=dtype)
        timestep = torch.stack([t_cond_B, t_pred_B], dim=1).reshape(B, 1, 2, 1, 1).to(device=device)

        net_output = self._dit_forward(
            hidden_states=hidden_states.to(dit_device),
            timestep=timestep.to(dit_device),
            encoder_hidden_states=text_embeds.to(dit_device, dtype=dtype),
            condition_mask=condition_mask.to(dit_device),
        )

        x0_pred = (
            c_skip_f * two_frame_latent[:, :16].float()
            + c_out_f * net_output.float()
        )
        final_latent = x0_pred[:, :, 1:2, :, :]

        return final_latent.to(dtype)  # [B, 16, 1, 28, 28]

    @torch.inference_mode()
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """VAE decode latent to pixel space.

        Args:
            latent: [B, 16, T, H, W] latent tensor.

        Returns:
            video: [B, 3, T*4, H*8, W*8] pixel tensor in [-1, 1].
        """
        return self.vae.decode(latent / self._sigma_data_cfg)

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def encoder_dim(self) -> int:
        """Native hidden dimension of the Cosmos DiT encoder (2048 for 2B)."""
        return self._model_channels
