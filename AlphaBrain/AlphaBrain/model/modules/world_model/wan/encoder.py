# Copyright 2026 VLA-Engine. All rights reserved.
#
# Licensed under the VLA-Engine License. You may not use this file except
# in compliance with the License.

"""
Wan 2.2 World-Model Encoder for VLA-Engine.

Wraps the Wan 2.2 VAE and DiT backbone (WanModel) as a visual encoder,
extracting intermediate DiT block features via forward hooks for rich
spatiotemporal representations.

Supports both ti2v-5B (dim=3072, 30 layers) and t2v-A14B (dim=5120, 40 layers)
model variants, with Wan 2.1 VAE (z_dim=16, 8x spatial) or Wan 2.2 VAE
(z_dim=48, 16x spatial).
"""

import logging
import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from AlphaBrain.model.modules.world_model.base import BaseWorldModelEncoder, WorldModelEncoderConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Wan 2.2 model variant configurations
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Global constants aligned with Cosmos 2.0 stable mode
# NOTE: module-level constants below are now *defaults only*; each WanEncoder
# instance reads its effective values from WorldModelEncoderConfig in __init__
# (self._sigma_min_cfg / self._feature_layer etc.) so yaml overrides work.
# ---------------------------------------------------------------------------

_SIGMA_MIN = 0.002
_SIGMA_DATA = 1.0
_SIGMA_MAX = 80.0
_SIGMA_CONDITIONAL = 0.0001
_FEATURE_LAYER = 14  # Layer 14 for WAN (30 layers, ~47% depth, matching Cosmos 18/28)

WAN_VARIANT_CONFIGS = {
    "ti2v-5B": dict(
        model_type="ti2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=48,          # Wan 2.2 VAE z_dim
        dim=3072,
        ffn_dim=14336,
        freq_dim=256,
        text_dim=4096,
        out_dim=48,
        num_heads=24,
        num_layers=30,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        # VAE settings
        vae_type="2.2",
        vae_z_dim=48,
        vae_stride=(4, 16, 16),
    ),
    "t2v-A14B": dict(
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,           # Wan 2.1 VAE z_dim
        dim=5120,
        ffn_dim=13824,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=40,
        num_layers=40,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        # VAE settings
        vae_type="2.1",
        vae_z_dim=16,
        vae_stride=(4, 8, 8),
    ),
}





# ---------------------------------------------------------------------------
# WanEncoder
# ---------------------------------------------------------------------------

class WanEncoder(BaseWorldModelEncoder):
    """Wan 2.2 world-model encoder for VLA-Engine.

    Encodes single images through the Wan video VAE (treating them as
    single-frame videos) and optionally extracts intermediate DiT block
    features via forward hooks for richer visual representations.

    Args:
        config: WorldModelEncoderConfig with the following relevant fields:
            - checkpoint_path: path to DiT checkpoint (.pth or directory)
            - pretrained_dir: directory containing VAE checkpoint
            - image_size: target spatial resolution (default 384)
            - use_intermediate_features: whether to run DiT for features
            - intermediate_layer_ids: which DiT block indices to hook
    """

    # Variant name extracted from checkpoint path or set explicitly
    SUPPORTED_VARIANTS = ("ti2v-5B", "t2v-A14B")

    def __init__(self, config: WorldModelEncoderConfig):
        super().__init__(config)

        # --- Resolve variant (config override > checkpoint path auto-detect) ---
        variant_override = getattr(config, "wan_variant", None)
        if variant_override:
            if variant_override not in WAN_VARIANT_CONFIGS:
                raise ValueError(
                    f"wan_variant={variant_override!r} not in "
                    f"WAN_VARIANT_CONFIGS ({list(WAN_VARIANT_CONFIGS.keys())})"
                )
            self.variant = variant_override
        else:
            self.variant = self._detect_variant(config.checkpoint_path)
        self.variant_cfg = WAN_VARIANT_CONFIGS[self.variant]
        logger.info("Wan encoder variant: %s", self.variant)

        # --- Pull diffusion + feature-layer hyperparams from config ---
        self._sigma_min_cfg = getattr(config, "sigma_min", _SIGMA_MIN)
        self._sigma_max_cfg = getattr(config, "sigma_max", _SIGMA_MAX)
        self._sigma_data_cfg = getattr(config, "sigma_data", _SIGMA_DATA)
        self._sigma_conditional_cfg = getattr(
            config, "sigma_conditional", _SIGMA_CONDITIONAL,
        )
        self._feature_layer = (
            getattr(config, "feature_layer_id", None) or _FEATURE_LAYER
        )

        # State for forward hooks
        self._hook_handles: List[torch.utils.hooks.RemovableHook] = []
        self._intermediate_features: Dict[int, torch.Tensor] = {}

        # Build encoder components
        self._build_encoder()

    # ------------------------------------------------------------------
    # Variant detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_variant(checkpoint_path: str) -> str:
        """Infer the model variant from the checkpoint path."""
        path_lower = checkpoint_path.lower()
        if "14b" in path_lower or "t2v" in path_lower:
            return "t2v-A14B"
        # Default to the smaller ti2v-5B
        return "ti2v-5B"

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_encoder(self) -> None:
        """Build VAE, DiT, and set up hooks."""

        cfg = self.variant_cfg

        # --- Load VAE ---
        self.vae = self._load_vae(cfg)

        # --- Load DiT (only if using intermediate features) ---
        self.dit = None
        if self.config.use_intermediate_features:
            self.dit = self._load_dit(cfg)
            self._register_hooks()

        # --- Native text encoder (UMT5-XXL from Wan pretrained dir) ---
        self._init_native_text_encoder()

        logger.info(
            "WanEncoder built: variant=%s, vae_type=%s, vae_z_dim=%d, "
            "dit_loaded=%s, use_intermediate=%s",
            self.variant,
            cfg["vae_type"],
            cfg["vae_z_dim"],
            self.dit is not None,
            self.config.use_intermediate_features,
        )

    # ------------------------------------------------------------------
    # Native text encoder
    # ------------------------------------------------------------------

    def _init_native_text_encoder(self) -> None:
        """Load UMT5-XXL text encoder using Wan's native T5 implementation.

        The UMT5-XXL model (5.7B params) is stored via object.__setattr__ to
        prevent DeepSpeed ZeRO-2 from traversing its parameters during init,
        mirroring the same pattern used for Cosmos T5.

        Priority:
          1. Precomputed pkl (data/pretrained_models/text_embeddings/umt5_text_embeddings.pkl)
             -- skips loading the 5.7B model entirely.
          2. Live UMT5-XXL inference -- loads model isolated from nn.Module tree.
        """
        # -- 1. Try precomputed embeddings first ----------------------
        # Search new canonical location first, fall back to old datasets location
        precomp_pkl = ""
        for _candidate in [
            os.path.join("data", "pretrained_models", "text_embeddings", "umt5_text_embeddings.pkl"),
            os.path.join("data", "datasets", "libero_datasets", "umt5_text_embeddings.pkl"),
        ]:
            if os.path.isfile(_candidate):
                precomp_pkl = _candidate
                break
        if precomp_pkl:
            import pickle
            logger.info(
                "Loading precomputed UMT5 embeddings from %s (skipping 5.7B model load)",
                precomp_pkl,
            )
            with open(precomp_pkl, "rb") as _f:
                _cache = pickle.load(_f)
            # Store cache via object.__setattr__ so DeepSpeed does not see it
            object.__setattr__(self, "_umt5_precomp_cache", _cache)
            object.__setattr__(self, "native_text_encoder", None)
            logger.info(
                "UMT5 precomputed cache loaded: %d instructions, dim=4096", len(_cache)
            )
            return

        # -- 2. Fall back to live UMT5-XXL inference ------------------
        from AlphaBrain.model.modules.world_model.wan.t5 import T5EncoderModel

        base_dir = self.config.pretrained_dir or os.path.dirname(self.config.checkpoint_path)

        # UMT5-XXL checkpoint (flat Wan-format keys)
        t5_pth = os.path.join(base_dir, "models_t5_umt5-xxl-enc-bf16.pth")
        # Tokenizer from google/umt5-xxl directory
        tok_dir = os.path.join(base_dir, "google", "umt5-xxl")

        object.__setattr__(self, "_umt5_precomp_cache", None)

        if not os.path.isfile(t5_pth):
            logger.warning(
                "Native UMT5-XXL text encoder not found at %s, "
                "will use dummy zero conditioning.", t5_pth,
            )
            object.__setattr__(self, "native_text_encoder", None)
            return

        logger.info("Loading native UMT5-XXL text encoder from %s", t5_pth)

        try:
            # T5EncoderModel handles model construction, weight loading,
            # and tokenizer initialization internally.
            # It calls umt5_xxl(encoder_only=True) -> T5Encoder with the
            # correct flat key structure (vocab_size=256384, dim=4096).
            # Use object.__setattr__ so DeepSpeed ZeRO-2 does NOT traverse
            # the 5.7B T5Encoder nn.Module during distributed init.
            _te = T5EncoderModel(
                text_len=512,
                dtype=torch.bfloat16,
                device="cpu",          # load on CPU, move to GPU during inference
                checkpoint_path=t5_pth,
                tokenizer_path=tok_dir,
            )
            object.__setattr__(self, "native_text_encoder", _te)
        except Exception as e:
            logger.warning("Failed to load native UMT5-XXL: %s. Will use dummy.", e)
            object.__setattr__(self, "native_text_encoder", None)
            return

        te_params = sum(p.numel() for p in self.native_text_encoder.model.parameters())
        logger.info(
            "Native UMT5-XXL text encoder loaded: %.2fM params (frozen, isolated from "
            "nn.Module via object.__setattr__), output_dim=4096",
            te_params / 1e6,
        )
    def encode_text(self, instructions, device):
        """Encode text using the native Wan UMT5-XXL text encoder.

        Priority:
          1. Precomputed embeddings cache (pkl) -- O(1) lookup, no GPU overhead.
          2. Live UMT5-XXL inference -- pads variable-length outputs to
             [B, L_max, 4096].

        Returns:
            torch.Tensor of shape [B, L_max, 4096] in bfloat16,
            or None if neither precomputed cache nor live encoder is available.
        """
        # -- Branch 1: precomputed lookup ----------------------------
        precomp = getattr(self, "_umt5_precomp_cache", None)
        if precomp is not None:
            return self._encode_text_umt5_precomputed(instructions, device, precomp)

        # -- Branch 2: live UMT5-XXL inference -----------------------
        if self.native_text_encoder is None:
            return None

        # Ensure the UMT5 model weights are on the target device.
        # native_text_encoder is a plain Python class (not nn.Module), so
        # vlm.cuda() does NOT move its weights automatically.
        te_model = self.native_text_encoder.model
        te_current_device = next(iter(te_model.parameters())).device
        if te_current_device != device:
            self.native_text_encoder.model = te_model.to(device)

        with torch.no_grad():
            # Returns List[Tensor[L_i, 4096]] -- variable length per sample
            embeddings = self.native_text_encoder(instructions, device)

        # Pad to uniform length: [B, L_max, 4096]
        L_max = max(e.shape[0] for e in embeddings)
        dim = embeddings[0].shape[1]
        padded = torch.zeros(len(embeddings), L_max, dim,
                             dtype=torch.bfloat16, device=device)
        for i, e in enumerate(embeddings):
            padded[i, :e.shape[0]] = e

        return padded

    def _encode_text_umt5_precomputed(self, instructions, device, cache: dict):
        """Look up precomputed UMT5 embeddings from cache dict.

        Args:
            instructions: List[str] -- raw task instruction strings.
            device: target torch.device for the returned tensor.
            cache: Dict[str, Tensor[512, 4096]] loaded from pkl.

        Returns:
            Tensor [B, 512, 4096] bfloat16 on ``device``.
            Falls back to zeros for instructions not in cache (with a warning).
        """
        TEXT_LEN = 512
        DIM = 4096
        B = len(instructions)
        out = torch.zeros(B, TEXT_LEN, DIM, dtype=torch.bfloat16, device=device)
        for i, instr in enumerate(instructions):
            if instr in cache:
                out[i] = cache[instr].to(device=device, dtype=torch.bfloat16)
            else:
                logger.warning(
                    "UMT5 precomputed cache miss for instruction: %r -- using zeros.", instr
                )
        return out
    def _load_vae(self, cfg: dict) -> object:
        """Load and freeze the Wan VAE."""
        vae_type = cfg["vae_type"]
        vae_z_dim = cfg["vae_z_dim"]

        # Determine VAE checkpoint path
        vae_pth = self._resolve_vae_path(vae_type)

        if vae_type == "2.2":
            from AlphaBrain.model.modules.world_model.wan.vae2_2 import Wan2_2_VAE
            vae_wrapper = Wan2_2_VAE(
                z_dim=vae_z_dim,
                vae_pth=vae_pth,
                dtype=torch.bfloat16,
                device="cpu",
            )
        else:
            from AlphaBrain.model.modules.world_model.wan.vae2_1 import Wan2_1_VAE
            vae_wrapper = Wan2_1_VAE(
                z_dim=vae_z_dim,
                vae_pth=vae_pth,
                dtype=torch.bfloat16,
                device="cpu",
            )

        # Freeze VAE — it is always frozen
        vae_wrapper.model.eval()
        vae_wrapper.model.requires_grad_(False)

        # Store normalization scale from VAE wrapper
        self._vae_scale = vae_wrapper.scale
        self._vae_z_dim = vae_z_dim
        self._vae_dtype = torch.bfloat16

        logger.info(
            "Loaded Wan %s VAE from %s (z_dim=%d)", vae_type, vae_pth, vae_z_dim
        )
        return vae_wrapper

    def _resolve_vae_path(self, vae_type: str) -> str:
        """Resolve the VAE checkpoint file path."""
        vae_filename = "Wan2.2_VAE.pth" if vae_type == "2.2" else "Wan2.1_VAE.pth"

        # Try pretrained_dir first
        if self.config.pretrained_dir:
            candidate = os.path.join(self.config.pretrained_dir, vae_filename)
            if os.path.isfile(candidate):
                return candidate

        # Try alongside checkpoint_path
        ckpt_dir = os.path.dirname(self.config.checkpoint_path)
        if ckpt_dir:
            candidate = os.path.join(ckpt_dir, vae_filename)
            if os.path.isfile(candidate):
                return candidate

        # Fallback: try pretrained_models dir
        candidate = os.path.join("data", "pretrained_models", "Wan2.2-TI2V-5B", vae_filename)

        logger.info("Resolved VAE path: %s", candidate)
        return candidate

    def _load_dit(self, cfg: dict) -> nn.Module:
        """Load the WanModel DiT backbone."""
        from AlphaBrain.model.modules.world_model.wan.model import WanModel

        # Build model from config
        dit = WanModel(
            model_type=cfg["model_type"],
            patch_size=cfg["patch_size"],
            text_len=cfg["text_len"],
            in_dim=cfg["in_dim"],
            dim=cfg["dim"],
            ffn_dim=cfg["ffn_dim"],
            freq_dim=cfg["freq_dim"],
            text_dim=cfg["text_dim"],
            out_dim=cfg["out_dim"],
            num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
            window_size=cfg["window_size"],
            qk_norm=cfg["qk_norm"],
            cross_attn_norm=cfg["cross_attn_norm"],
            eps=cfg["eps"],
        )

        # Load checkpoint weights (supports single .pt/.pth or safetensors shards)
        ckpt_path = self.config.checkpoint_path
        if ckpt_path and os.path.exists(ckpt_path):
            if os.path.isdir(ckpt_path):
                # Directory: look for safetensors shards or single file
                index_file = os.path.join(ckpt_path, "diffusion_pytorch_model.safetensors.index.json")
                if os.path.isfile(index_file):
                    # Load sharded safetensors
                    import json
                    from safetensors.torch import load_file
                    with open(index_file) as f:
                        index = json.load(f)
                    shard_files = sorted(set(index["weight_map"].values()))
                    state_dict = {}
                    for shard in shard_files:
                        shard_path = os.path.join(ckpt_path, shard)
                        logger.info("Loading DiT shard: %s", shard)
                        state_dict.update(load_file(shard_path))
                else:
                    # Fallback: look for .pt/.pth file
                    for ext in [".pt", ".pth"]:
                        for f in sorted(os.listdir(ckpt_path)):
                            if f.endswith(ext) and "vae" not in f.lower() and "t5" not in f.lower():
                                state_dict = torch.load(os.path.join(ckpt_path, f), map_location="cpu")
                                break
                        if "state_dict" in dir():
                            break
            else:
                # Single file
                if ckpt_path.endswith(".safetensors"):
                    from safetensors.torch import load_file
                    state_dict = load_file(ckpt_path)
                else:
                    state_dict = torch.load(ckpt_path, map_location="cpu")

            # Handle nested state dicts
            if isinstance(state_dict, dict):
                if "model" in state_dict:
                    state_dict = state_dict["model"]
                elif "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]

            missing, unexpected = dit.load_state_dict(state_dict, strict=False)
            logger.info("DiT loaded: %d keys, missing=%d, unexpected=%d",
                        len(state_dict), len(missing), len(unexpected))
            if missing:
                logger.warning("DiT missing keys (first 5): %s", missing[:5])
        else:
            logger.warning(
                "No DiT checkpoint found at '%s'; using random initialization.",
                self.config.checkpoint_path,
            )

        # Freeze DiT if configured
        if self.config.freeze_encoder:
            dit.eval()
            dit.requires_grad_(False)
            dit = dit.to(dtype=torch.bfloat16)

        return dit

    # ------------------------------------------------------------------
    # Forward Hooks for Intermediate Feature Extraction
    # ------------------------------------------------------------------

    def _register_hooks(self) -> None:
        """Register forward hooks on selected DiT blocks.

        WanModel has no built-in intermediate_feature_ids support,
        so we use PyTorch forward hooks to capture block outputs.
        """
        if self.dit is None:
            return

        self._clear_hooks()

        # Determine which layers to hook
        layer_ids = self.config.intermediate_layer_ids
        if layer_ids is None:
            # Default: sample evenly across the transformer depth
            num_layers = len(self.dit.blocks)
            # Pick ~4 layers spread across the model
            step = max(1, num_layers // 4)
            layer_ids = list(range(step - 1, num_layers, step))
            logger.info("Auto-selected intermediate layer ids: %s", layer_ids)

        for layer_id in layer_ids:
            if layer_id < 0 or layer_id >= len(self.dit.blocks):
                logger.warning(
                    "Skipping invalid layer_id %d (model has %d blocks)",
                    layer_id,
                    len(self.dit.blocks),
                )
                continue

            block = self.dit.blocks[layer_id]

            def _make_hook(lid: int):
                def hook_fn(module, input, output):
                    # WanAttentionBlock.forward returns x: [B, L, C]
                    # detach only when frozen to save memory; keep grad when training
                    if self.config.freeze_encoder:
                        self._intermediate_features[lid] = output.detach()
                    else:
                        self._intermediate_features[lid] = output
                return hook_fn

            handle = block.register_forward_hook(_make_hook(layer_id))
            self._hook_handles.append(handle)

        logger.info(
            "Registered %d forward hooks on DiT blocks: %s",
            len(self._hook_handles),
            layer_ids,
        )

    def _clear_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._intermediate_features.clear()

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode_images(self, pixel_values: torch.Tensor, text_embeds=None) -> torch.Tensor:
        """Encode preprocessed images into visual token features.

        Args:
            pixel_values: [B, 3, H, W] preprocessed images in [-1, 1].

        Returns:
            Visual features of shape [B, N, D] where:
                - If use_intermediate_features: D = dit.dim, features from
                  hooked DiT blocks are averaged.
                - Else: D = vae_z_dim, flattened VAE latent.
        """
        # Move to GPU, keep float32 for VAE (VAE is fp32, DiT is bf16)
        if self.dit is not None:
            encoder_device = next(self.dit.parameters()).device
        else:
            encoder_device = next(self.vae.model.parameters()).device
        pixel_values = pixel_values.to(device=encoder_device, dtype=torch.float32)

        B = pixel_values.shape[0]
        device = pixel_values.device

        # --- Step 0: Ensure VAE (model + scale) is on same device ---
        if next(self.vae.model.parameters()).device != device:
            self.vae.model = self.vae.model.to(device)
        # Move scale tensors to device
        if hasattr(self.vae, "scale") and isinstance(self.vae.scale, (list, tuple)):
            self.vae.scale = [s.to(device) if isinstance(s, torch.Tensor) else s for s in self.vae.scale]

        # --- Step 1: VAE encode ---
        # Reshape to single-frame video: [B, 3, 1, H, W]
        video_input = pixel_values.unsqueeze(2)  # [B, 3, 1, H, W]

        # Encode each sample through the VAE (expects list of [C, T, H, W])
        latents = []
        for i in range(B):
            frame = video_input[i]  # [3, 1, H, W]
            encoded = self.vae.encode([frame])  # list of [z_dim, T', H', W']
            latents.append(encoded[0])  # [z_dim, T', H', W']

        # Stack into batch: [B, z_dim, T', H', W']
        latent = torch.stack(latents, dim=0).to(device)

        if not self.config.use_intermediate_features or self.dit is None:
            # --- VAE-only mode: flatten spatial dims ---
            # latent: [B, z_dim, T', H', W'] -> [B, N, z_dim]
            B, C, T, H, W = latent.shape
            features = latent.reshape(B, C, -1)        # [B, z_dim, T'*H'*W']
            features = features.permute(0, 2, 1)       # [B, N, z_dim]
            return features

        # --- Step 2: Build 2-frame input matching training context ---
        # Training uses 2 frames [frame0_clean, frame1_noisy] with c_in preconditioning.
        # At inference, use sigma_max for frame1 (pure noise convention).
        B_lat, z_dim, T_lat, H_lat, W_lat = latent.shape
        sigma_noise = self._sigma_max_cfg
        t_rf = sigma_noise / (sigma_noise + 1.0)  # ~0.988
        c_in = 1.0 - t_rf  # ~0.012

        # frame1 = zeros (deterministic, c_in * sigma * 0 = 0)
        noise_frame = torch.zeros(B_lat, z_dim, 1, H_lat, W_lat, device=latent.device, dtype=latent.dtype)
        two_frame = torch.cat([latent, noise_frame], dim=2)  # [B, z_dim, 2, H', W']

        # c_in preconditioning (matching training)
        net_input = two_frame.float() * c_in
        net_input[:, :, :1, :, :] = latent.float() / self._sigma_data_cfg  # clean condition (matching Cosmos)

        # Per-frame timestep matching training
        patch_size_inf = self.variant_cfg["patch_size"]
        _Hp_inf = H_lat // patch_size_inf[1]
        _Wp_inf = W_lat // patch_size_inf[2]
        _tpf_inf = _Hp_inf * _Wp_inf  # tokens per frame
        t_noise_int = int(round(t_rf * 1000))
        t_wan = torch.cat([
            torch.zeros(B_lat, _tpf_inf, device=latent.device, dtype=torch.long),  # frame0: t=0
            torch.full((B_lat, _tpf_inf), t_noise_int, device=latent.device, dtype=torch.long),  # frame1: t=988
        ], dim=1)  # [B, seq_len]

        features = self._extract_dit_features_2frame(net_input.to(latent.dtype), text_embeds=text_embeds, t_wan=t_wan)
        return features

    def _extract_dit_features_2frame(self, latent_2f, text_embeds=None, t_wan=None):
        """Two frame DiT forward matching training. Returns frame0 only features."""
        B = latent_2f.shape[0]
        dit_device = next(self.dit.parameters()).device
        dit_dtype = next(self.dit.parameters()).dtype
        latent_2f = latent_2f.to(device=dit_device, dtype=dit_dtype)

        x_list = [latent_2f[i] for i in range(B)]
        if t_wan is None:
            t_wan = torch.zeros(B, device=dit_device, dtype=torch.long)
        t_wan = t_wan.to(dit_device)

        text_dim = self.variant_cfg["text_dim"]
        if text_embeds is not None:
            text_embeds = text_embeds.to(device=dit_device, dtype=dit_dtype)
            context_list = [text_embeds[i] for i in range(B)]
        else:
            context_list = [torch.zeros(1, text_dim, device=dit_device, dtype=dit_dtype) for _ in range(B)]

        patch_size = self.variant_cfg["patch_size"]
        _, F_lat, H_lat, W_lat = x_list[0].shape
        T_p = F_lat // patch_size[0]
        H_p = H_lat // patch_size[1]
        W_p = W_lat // patch_size[2]
        seq_len = T_p * H_p * W_p
        tokens_per_frame = H_p * W_p

        self.dit = self.dit.to(dit_device)
        self._intermediate_features.clear()

        from contextlib import nullcontext
        grad_ctx = torch.no_grad() if self.config.freeze_encoder else nullcontext()
        with grad_ctx, torch.autocast("cuda", dtype=torch.bfloat16):
            _ = self.dit(x=x_list, t=t_wan, context=context_list, seq_len=seq_len)

        if not self._intermediate_features:
            features = latent_2f.reshape(B, latent_2f.shape[1], -1).permute(0, 2, 1)
            return features

        layer_ids = sorted(self._intermediate_features.keys())
        stacked = torch.stack(
            [self._intermediate_features[lid][:, :seq_len, :] for lid in layer_ids], dim=0,
        )
        features = stacked.mean(dim=0)  # [B, seq_len, D]

        # Frame0-only: match training (only condition frame features)
        features = features[:, :tokens_per_frame, :]
        return features.to(latent_2f.device)

    def _extract_dit_features(self, latent: torch.Tensor, text_embeds=None) -> torch.Tensor:
        """Run DiT forward and collect hooked features.

        Args:
            latent: [B, z_dim, T', H', W'] — normalized VAE latent.
            text_embeds: [B, L, text_dim] — text embeddings for DiT cross-attention.

        Returns:
            Aggregated features: [B, N, D] where D = dit.dim.
        """
        B = latent.shape[0]
        device = latent.device
        dit_device = next(self.dit.parameters()).device
        dit_dtype = next(self.dit.parameters()).dtype

        # Move latent to DiT device
        latent = latent.to(device=dit_device, dtype=dit_dtype)

        # Prepare DiT inputs
        # x: list of [C, F, H, W] tensors (one per batch element)
        x_list = [latent[i] for i in range(B)]  # each [z_dim, T', H', W']

        # t: dummy timestep = 0 (we just want features, not denoising)
        t = torch.zeros(B, device=dit_device, dtype=torch.long)

        # context: text embeddings for DiT cross-attention
        text_dim = self.variant_cfg["text_dim"]
        if text_embeds is not None:
            # Use real text embeddings from native text encoder
            text_embeds = text_embeds.to(device=dit_device, dtype=dit_dtype)
            context_list = [text_embeds[i] for i in range(B)]  # each [L, text_dim]
        else:
            # Fallback: dummy zero conditioning
            context_list = [
                torch.zeros(1, text_dim, device=dit_device, dtype=dit_dtype)
                for _ in range(B)
            ]

        # Compute seq_len for positional encoding
        # After patch embedding: each spatial dim is divided by patch_size
        patch_size = self.variant_cfg["patch_size"]
        _, T_lat, H_lat, W_lat = x_list[0].shape
        T_p = T_lat // patch_size[0]
        H_p = H_lat // patch_size[1]
        W_p = W_lat // patch_size[2]
        seq_len = T_p * H_p * W_p

        # Ensure DiT is fully on device
        self.dit = self.dit.to(dit_device)

        # Clear collected features
        self._intermediate_features.clear()

        # Forward pass through DiT
        # Use no_grad when frozen to save memory; allow gradients when training backbone
        from contextlib import nullcontext
        grad_ctx = torch.no_grad() if self.config.freeze_encoder else nullcontext()
        with grad_ctx, torch.autocast("cuda", dtype=torch.bfloat16):
            _ = self.dit(
                x=x_list,
                t=t,
                context=context_list,
                seq_len=seq_len,
            )

        # Aggregate intermediate features
        if not self._intermediate_features:
            logger.warning(
                "No intermediate features collected; "
                "falling back to final DiT output."
            )
            # Fallback: flatten latent
            features = latent.reshape(B, latent.shape[1], -1).permute(0, 2, 1)
            return features

        # Average features across hooked layers: each is [B, L_padded, D]
        # We only take the valid seq_len tokens (rest is padding)
        layer_ids = sorted(self._intermediate_features.keys())
        stacked = torch.stack(
            [self._intermediate_features[lid][:, :seq_len, :] for lid in layer_ids],
            dim=0,
        )  # [num_layers, B, seq_len, D]
        features = stacked.mean(dim=0)  # [B, seq_len, D]

        return features.to(device)


    def encode_images_all_layers(self, pixel_values, text_embeds=None):
        """Return per-DiT-block visual features for layerwise PI cross-attention.

        Registers temporary hooks on ALL DiT blocks and returns the outputs as a
        list indexed by block id.  The PI action head cross-attends to a different
        block at each of its own transformer layers.

        Args:
            pixel_values: [B, 3, H, W] preprocessed images in [-1, 1].
            text_embeds: [B, L, text_dim] text embeddings for DiT cross-attention.

        Returns:
            List of num_blocks tensors, each [B, seq_len, dit_dim].
            List length == len(self.dit.blocks) == 30 (for Wan2.2-5B).
        """
        if self.dit is None:
            raise RuntimeError("WanEncoder: DiT not loaded; cannot extract per-layer features.")

        num_blocks = len(self.dit.blocks)
        all_layer_ids = list(range(num_blocks))

        # Temporarily register hooks on ALL blocks
        tmp_features = {}
        tmp_handles = []

        def _make_hook(lid):
            def hook_fn(module, input, output):
                if self.config.freeze_encoder:
                    tmp_features[lid] = output.detach()
                else:
                    tmp_features[lid] = output
            return hook_fn

        for lid in all_layer_ids:
            h = self.dit.blocks[lid].register_forward_hook(_make_hook(lid))
            tmp_handles.append(h)

        try:
            # Run encode_images (which calls _extract_dit_features internally)
            # We need to bypass the existing hooks and get per-block features.
            # Replicate _extract_dit_features logic inline.
            if next(self.dit.parameters()).device.type == 'cuda':
                encoder_device = next(self.dit.parameters()).device
            else:
                encoder_device = next(self.dit.parameters()).device

            pixel_values = pixel_values.to(device=encoder_device, dtype=torch.float32)
            B = pixel_values.shape[0]
            device = pixel_values.device

            # VAE encode
            if next(self.vae.model.parameters()).device != device:
                self.vae.model = self.vae.model.to(device)
            if hasattr(self.vae, "scale") and isinstance(self.vae.scale, (list, tuple)):
                self.vae.scale = [s.to(device) if isinstance(s, torch.Tensor) else s for s in self.vae.scale]

            video_input = pixel_values.unsqueeze(2)
            latents = []
            for i in range(B):
                frame = video_input[i]
                encoded = self.vae.encode([frame])
                latents.append(encoded[0])
            latent = torch.stack(latents, dim=0).to(device)

            dit_device = next(self.dit.parameters()).device
            dit_dtype = next(self.dit.parameters()).dtype
            latent = latent.to(device=dit_device, dtype=dit_dtype)

            x_list = [latent[i] for i in range(B)]
            t = torch.zeros(B, device=dit_device, dtype=torch.long)

            text_dim = self.variant_cfg["text_dim"]
            if text_embeds is not None:
                text_embeds = text_embeds.to(device=dit_device, dtype=dit_dtype)
                context_list = [text_embeds[i] for i in range(B)]
            else:
                context_list = [
                    torch.zeros(1, text_dim, device=dit_device, dtype=dit_dtype)
                    for _ in range(B)
                ]

            patch_size = self.variant_cfg["patch_size"]
            _, T_lat, H_lat, W_lat = x_list[0].shape
            T_p = T_lat // patch_size[0]
            H_p = H_lat // patch_size[1]
            W_p = W_lat // patch_size[2]
            seq_len = T_p * H_p * W_p

            self.dit = self.dit.to(dit_device)

            from contextlib import nullcontext
            grad_ctx = torch.no_grad() if self.config.freeze_encoder else nullcontext()
            with grad_ctx, torch.autocast("cuda", dtype=torch.bfloat16):
                _ = self.dit(
                    x=x_list,
                    t=t,
                    context=context_list,
                    seq_len=seq_len,
                )

            # Collect per-block features in order
            per_block = [
                tmp_features[lid][:, :seq_len, :].to(device)
                for lid in all_layer_ids
                if lid in tmp_features
            ]

        finally:
            for h in tmp_handles:
                h.remove()

        return per_block




    def encode_to_latent(self, images: torch.Tensor) -> torch.Tensor:
        """Encode preprocessed images to VAE latent space only (no DiT forward).

        Args:
            images: [B, 3, H, W] float tensor in [-1, 1] (output of preprocess()).

        Returns:
            [B, z_dim, 1, H', W'] VAE latent tensor (bf16).
        """
        if self.dit is not None:
            encoder_device = next(self.dit.parameters()).device
        else:
            encoder_device = next(self.vae.model.parameters()).device

        images = images.to(device=encoder_device, dtype=torch.float32)
        B = images.shape[0]
        device = images.device

        # Ensure VAE is on same device
        if next(self.vae.model.parameters()).device != device:
            self.vae.model = self.vae.model.to(device)
        if hasattr(self.vae, "scale") and isinstance(self.vae.scale, (list, tuple)):
            self.vae.scale = [s.to(device) if isinstance(s, torch.Tensor) else s for s in self.vae.scale]

        # Reshape to single-frame video: [B, 3, 1, H, W]
        video_input = images.unsqueeze(2)

        latents = []
        for i in range(B):
            frame = video_input[i]  # [3, 1, H, W]
            with torch.no_grad():
                encoded = self.vae.encode([frame])  # list of [z_dim, T', H', W']
            latents.append(encoded[0])

        # Stack into batch: [B, z_dim, T', H', W']
        latent = torch.stack(latents, dim=0).to(device)
        return latent

    def encode_images_with_video_loss(
        self,
        latent_t: torch.Tensor,
        latent_t1: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> 'Tuple[torch.Tensor, torch.Tensor]':
        """Single DiT forward that yields action features and video loss.

        Adapts the Cosmos-Policy V2 single-forward design for the WAN DiT:
          - Intermediate hook features -> for action head
          - Final denoised output      -> for video prediction loss

        Uses forward hooks (already registered) to capture intermediate
        block outputs, and RectifiedFlow preconditioning for denoising.

        Args:
            latent_t:    [B, z_dim, 1, H', W'] current frame VAE latent (clean).
            latent_t1:   [B, z_dim, 1, H', W'] next frame VAE latent (clean, target).
            text_embeds: [B, L, text_dim] text conditioning for DiT cross-attention,
                         or None for dummy zero conditioning.

        Returns:
            visual_tokens: [B, N, dit_dim]  intermediate features for action head.
            video_loss:    scalar            weighted MSE on predicted future frame.
        """
        from contextlib import nullcontext

        device = latent_t.device
        dtype = torch.bfloat16
        B = latent_t.shape[0]
        z_dim = self._vae_z_dim
        out_dim = self.variant_cfg["out_dim"]

        # --- 1. Build 2-frame clean latent: [B, z_dim, 2, H', W'] ---
        x0 = torch.cat([latent_t, latent_t1], dim=2)  # [B, z_dim, 2, H', W']

        # --- 2. Sample sigma (LogNormal, same as Cosmos) ---
        log_sigma = torch.randn(B, device=device)  # LogNormal(0, 1) matching Cosmos
        sigma = log_sigma.exp().clamp(min=1e-4, max=self._sigma_max_cfg)  # [B], float32

        # --- 3. Add noise to clean latent ---
        noise = torch.randn_like(x0)
        sigma_5d = sigma.view(B, 1, 1, 1, 1)
        xt = x0.float() + sigma_5d * noise.float()  # [B, z_dim, 2, H', W']

        # --- 4. RectifiedFlow preconditioning scalars ---
        t_rf = sigma / (sigma + 1.0)   # [B], in (0, 1)
        c_skip = 1.0 - t_rf            # [B]
        c_out = -t_rf                   # [B]
        c_in = 1.0 - t_rf              # [B]

        # --- 5. Condition frame handling (frame 0 = given clean frame) ---
        gt_cond = x0[:, :, :1, :, :].float()  # [B, z_dim, 1, H', W']
        xt[:, :, :1, :, :] = gt_cond          # no noise on condition frame

        # --- 6. Scale net input by c_in, restore condition frame ---
        net_input = xt * c_in.view(B, 1, 1, 1, 1)
        net_input[:, :, :1, :, :] = gt_cond / self._sigma_data_cfg   # condition frame passed as clean (matching Cosmos)

        # --- 7. Prepare DiT inputs ---
        dit_device = next(self.dit.parameters()).device
        dit_dtype = next(self.dit.parameters()).dtype
        net_input = net_input.to(device=dit_device, dtype=dit_dtype)

        # x: list of [C, F, H, W] tensors (one per batch element)
        x_list = [net_input[i] for i in range(B)]  # each [z_dim, 2, H', W']

        # Per-frame timestep: WAN DiT supports [B, seq_len] per-token timestep
        # frame0 (condition) = t_cond ~ 0, frame1 (prediction) = random t
        t_pred_int = (t_rf * 1000.0).long().to(dit_device)  # [B]
        patch_size_t = self.variant_cfg["patch_size"]
        _C, _F, _H, _W = net_input.shape[1], net_input.shape[2], net_input.shape[3], net_input.shape[4]
        _Tp = _F // patch_size_t[0]
        _Hp = _H // patch_size_t[1]
        _Wp = _W // patch_size_t[2]
        _tokens_per_frame = _Hp * _Wp
        t_wan = torch.cat([
            torch.zeros(B, _tokens_per_frame, device=dit_device, dtype=torch.long),  # frame0: t=0
            t_pred_int.unsqueeze(1).expand(B, _tokens_per_frame),  # frame1: random t
        ], dim=1)  # [B, seq_len]

        # Context: text embeddings for cross-attention
        text_dim = self.variant_cfg["text_dim"]
        if text_embeds is not None:
            text_embeds_dit = text_embeds.to(device=dit_device, dtype=dit_dtype)
            context_list = [text_embeds_dit[i] for i in range(B)]
        else:
            context_list = [
                torch.zeros(1, text_dim, device=dit_device, dtype=dit_dtype)
                for _ in range(B)
            ]

        # Compute seq_len for the 2-frame input
        patch_size = self.variant_cfg["patch_size"]
        C_lat, F_lat, H_lat, W_lat = x_list[0].shape
        T_p = F_lat // patch_size[0]
        H_p = H_lat // patch_size[1]
        W_p = W_lat // patch_size[2]
        seq_len = T_p * H_p * W_p

        # --- 8. DiT forward with hooks ---
        self.dit = self.dit.to(dit_device)
        self._intermediate_features.clear()

        grad_ctx = torch.no_grad() if self.config.freeze_encoder else nullcontext()
        with grad_ctx, torch.autocast("cuda", dtype=torch.bfloat16):
            dit_output = self.dit(
                x=x_list,
                t=t_wan,
                context=context_list,
                seq_len=seq_len,
            )

        # --- 9. Extract intermediate features for action head ---
        if self._intermediate_features:
            layer_ids = sorted(self._intermediate_features.keys())
            stacked = torch.stack(
                [self._intermediate_features[lid][:, :seq_len, :] for lid in layer_ids],
                dim=0,
            )  # [num_layers, B, seq_len, D]
            visual_tokens = stacked.mean(dim=0)  # [B, seq_len, D]
            # Frame0-only: match inference path (only condition frame features)
            _tokens_pf = H_p * W_p
            visual_tokens = visual_tokens[:, :_tokens_pf, :]  # [B, tokens_per_frame, D]
        else:
            # Fallback: use final DiT output reshaped
            logger.warning(
                "encode_images_with_video_loss: no intermediate features collected; "
                "using latent fallback."
            )
            visual_tokens = latent_t.reshape(B, latent_t.shape[1], -1).permute(0, 2, 1)
            visual_tokens = visual_tokens.to(device=device)

        # --- 10. Reconstruct x0_pred from DiT output ---
        # WanModel.forward returns list of per-sample tensors [C, T, H, W]
        if isinstance(dit_output, (list, tuple)):
            dit_output = torch.stack(dit_output, dim=0)  # [B, C, T, H, W]
        # WanModel.forward returns output after head + unpatchify: [B, C_out, F, H, W]
        if dit_output.dim() == 5:
            dit_out_spatial = dit_output  # [B, out_dim, F_lat, H_lat, W_lat]
        elif dit_output.dim() == 3:
            # [B, seq_len, patch_vol * out_dim] -> unpatchify manually
            patch_vol = patch_size[0] * patch_size[1] * patch_size[2]
            if dit_output.shape[-1] == patch_vol * out_dim:
                dit_out_spatial = dit_output.reshape(
                    B, T_p, H_p, W_p,
                    patch_size[0], patch_size[1], patch_size[2], out_dim
                )
                dit_out_spatial = dit_out_spatial.permute(0, 7, 1, 4, 2, 5, 3, 6)
                dit_out_spatial = dit_out_spatial.reshape(B, out_dim, F_lat, H_lat, W_lat)
            else:
                # Reshape [B, seq, out_dim] -> spatial
                dit_out_spatial = dit_output[:, :seq_len, :].reshape(
                    B, T_p, H_p, W_p, out_dim
                ).permute(0, 4, 1, 2, 3)
                if dit_out_spatial.shape[2:] != (F_lat, H_lat, W_lat):
                    dit_out_spatial = F.interpolate(
                        dit_out_spatial.reshape(B * out_dim, 1, T_p, H_p, W_p).float(),
                        size=(F_lat, H_lat, W_lat),
                        mode="nearest",
                    ).reshape(B, out_dim, F_lat, H_lat, W_lat)
        else:
            logger.warning(
                "Unexpected dit_output shape %s; using zero video loss.", dit_output.shape
            )
            video_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return visual_tokens.to(device), video_loss

        # x0_pred = c_skip * xt[:out_dim] + c_out * dit_output
        xt_out = xt[:, :out_dim, :, :, :].to(dit_out_spatial.device).float()
        x0_pred = (
            c_skip.view(B, 1, 1, 1, 1).to(dit_out_spatial.device) * xt_out
            + c_out.view(B, 1, 1, 1, 1).to(dit_out_spatial.device) * dit_out_spatial.float()
        )  # [B, out_dim, F_lat, H_lat, W_lat]

        # --- 11. Video loss on predicted future frame only (frame index 1) ---
        pred_future = x0_pred[:, :, 1:2, :, :]
        gt_future = x0[:, :out_dim, 1:2, :, :].to(x0_pred.device).float() * self._sigma_data_cfg

        # Loss weight: (1 + sigma)^2 / sigma^2
        sigma_f32 = sigma.to(x0_pred.device).float()
        weight = ((1.0 + sigma_f32) ** 2 / (sigma_f32 ** 2)).view(B, 1, 1, 1, 1)
        video_loss = (weight * F.mse_loss(pred_future, gt_future, reduction='none')).mean()

        # Debug: log shapes and values for first few steps
        if not hasattr(self, "_vloss_debug_count"):
            self._vloss_debug_count = 0
        if self._vloss_debug_count < 5:
            logger.info(
                "[WAN video_loss debug] x0_pred=%s, pred_future=%s, gt_future=%s, "
                "sigma=%.4f, weight=%.4f, video_loss=%.6f, F_lat=%d",
                x0_pred.shape, pred_future.shape, gt_future.shape,
                sigma_f32.mean().item(), weight.mean().item(), video_loss.item(), F_lat,
            )
            self._vloss_debug_count += 1

        return visual_tokens.to(device), video_loss


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
        matching the Cosmos repo RectifiedFlowAB2Scheduler:
          - Step 0: Euler (no previous x0 available)
          - Step 1+: 2-AB (uses current + previous x0 predictions)
          - Final: clean pass at sigma_min

        Adapted from Cosmos world_model_cosmos2_diffusers.py for WAN DiT interface
        (x_list, t, context, seq_len parameter form).

        Args:
            latent_t: [B, z_dim, 1, H', W'] current frame VAE latent (clean).
            text_embeds: [B, L, text_dim] cross-attention text conditioning.
            num_steps: number of denoising steps (default 35, matching Cosmos).
            sigma_min: minimum sigma for denoising schedule.  Defaults to
                ``config.sigma_min`` (typically 0.002).
            sigma_max: maximum sigma for denoising schedule.  Defaults to
                ``config.sigma_max`` (typically 80.0).
            rho: Karras schedule order parameter (default 7.0).

        Returns:
            future_latent: [B, z_dim, 1, H', W'] predicted future frame latent.
        """
        if sigma_min is None:
            sigma_min = self._sigma_min_cfg
        if sigma_max is None:
            sigma_max = self._sigma_max_cfg

        device = latent_t.device
        dtype = torch.bfloat16
        B = latent_t.shape[0]
        z_dim = latent_t.shape[1]
        H_lat, W_lat = latent_t.shape[3], latent_t.shape[4]
        dit_device = next(self.dit.parameters()).device
        dit_dtype = next(self.dit.parameters()).dtype
        out_dim = self.variant_cfg["out_dim"]
        text_dim = self.variant_cfg["text_dim"]
        patch_size = self.variant_cfg["patch_size"]

        if text_embeds is None:
            text_embeds = torch.zeros(
                B, 1, text_dim, device=dit_device, dtype=dit_dtype,
            )

        # ----------------------------------------------------------------
        # Karras sigma schedule
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
            torch.randn(B, z_dim, 1, H_lat, W_lat, device=device, dtype=torch.float32)
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
            two_frame_latent = torch.cat(
                [latent_t.to(dtype), xt_future.to(dtype)], dim=2,
            )  # [B, z_dim, 2, H', W']

            # Scale input by c_in, restore condition frame as clean
            net_input = two_frame_latent.float() * c_in
            net_input[:, :, :1, :, :] = latent_t.float() / self._sigma_data_cfg

            # WAN timestep: t -> integer range [0, 1000]
            t_wan = torch.full(
                (B,), int(round(float(t) * 1000)),
                device=dit_device, dtype=torch.long,
            )

            # Prepare x_list and context_list for WAN DiT
            net_input_dit = net_input.to(device=dit_device, dtype=dit_dtype)
            x_list = [net_input_dit[b] for b in range(B)]

            text_embeds_dit = text_embeds.to(device=dit_device, dtype=dit_dtype)
            context_list = [text_embeds_dit[b] for b in range(B)]

            # Compute seq_len
            F_lat = x_list[0].shape[1]
            T_p = F_lat // patch_size[0]
            H_p = H_lat // patch_size[1]
            W_p = W_lat // patch_size[2]
            seq_len = T_p * H_p * W_p

            # DiT forward
            with torch.autocast("cuda", dtype=torch.bfloat16):
                dit_output = self.dit(
                    x=x_list, t=t_wan, context=context_list, seq_len=seq_len,
                )

            # Unpatchify dit_output if needed
            dit_out_spatial = self._unpatchify_output(
                dit_output, B, out_dim, patch_size,
                T_p, H_p, W_p, F_lat, H_lat, W_lat, seq_len,
            )

            # x0 prediction: c_skip * xt[:out_dim] + c_out * dit_output
            xt_out = two_frame_latent[:, :out_dim, :, :, :].to(dit_out_spatial.device).float()
            x0_pred = c_skip * xt_out + c_out * dit_out_spatial.float()
            x0_future = x0_pred[:, :, 1:2, :, :]  # [B, out_dim, 1, H', W']

            # --- Solver step ---
            if x0_prev is None:
                # First step: Euler
                coef_x0 = (sigma - sigma_next) / sigma
                coef_xt = sigma_next / sigma
                xt_future = (
                    coef_x0 * x0_future + coef_xt * xt_future.float()
                ).to(torch.float32)
            else:
                # 2nd-order Adams-Bashforth (2-AB)
                sigma_prev = sigmas[i - 1]
                s_log = (-torch.log(sigma)).to(torch.float64)
                t_log = (-torch.log(sigma_next)).to(torch.float64)
                m_log = (-torch.log(sigma_prev)).to(torch.float64)

                dt = t_log - s_log
                c2 = (m_log - s_log) / dt

                neg_dt = -dt
                phi1_val = torch.expm1(neg_dt) / neg_dt
                phi2_val = (phi1_val - 1.0) / neg_dt

                b1 = phi1_val - 1.0 / c2 * phi2_val
                b2 = 1.0 / c2 * phi2_val

                xt_future = (
                    torch.exp(neg_dt) * xt_future.to(torch.float64)
                    + dt * (b1 * x0_future.to(torch.float64) + b2 * x0_prev.to(torch.float64))
                ).to(torch.float32)

            x0_prev = x0_future.clone()

        # --- Final clean pass at sigma_min ---
        sigma_final = sigmas[-1]
        t_final = sigma_final / (sigma_final + 1.0)
        c_skip_f = 1.0 - t_final
        c_out_f = -t_final
        c_in_f = 1.0 - t_final

        two_frame_latent = torch.cat(
            [latent_t.to(dtype), xt_future.to(dtype)], dim=2,
        )
        net_input = two_frame_latent.float() * c_in_f
        net_input[:, :, :1, :, :] = latent_t.float() / self._sigma_data_cfg

        t_wan = torch.full(
            (B,), int(round(float(t_final) * 1000)),
            device=dit_device, dtype=torch.long,
        )
        net_input_dit = net_input.to(device=dit_device, dtype=dit_dtype)
        x_list = [net_input_dit[b] for b in range(B)]
        text_embeds_dit = text_embeds.to(device=dit_device, dtype=dit_dtype)
        context_list = [text_embeds_dit[b] for b in range(B)]

        F_lat = x_list[0].shape[1]
        T_p = F_lat // patch_size[0]
        H_p = H_lat // patch_size[1]
        W_p = W_lat // patch_size[2]
        seq_len = T_p * H_p * W_p

        with torch.autocast("cuda", dtype=torch.bfloat16):
            dit_output = self.dit(
                x=x_list, t=t_wan, context=context_list, seq_len=seq_len,
            )

        dit_out_spatial = self._unpatchify_output(
            dit_output, B, out_dim, patch_size,
            T_p, H_p, W_p, F_lat, H_lat, W_lat, seq_len,
        )

        xt_out = two_frame_latent[:, :out_dim, :, :, :].to(dit_out_spatial.device).float()
        x0_pred = c_skip_f * xt_out + c_out_f * dit_out_spatial.float()
        final_latent = x0_pred[:, :, 1:2, :, :]

        return final_latent.to(dtype)  # [B, z_dim, 1, H', W']

    @staticmethod
    def _unpatchify_output(
        dit_output, B, out_dim, patch_size,
        T_p, H_p, W_p, F_lat, H_lat, W_lat, seq_len,
    ):
        """Convert DiT output to spatial [B, out_dim, F, H, W] tensor."""
        if dit_output.dim() == 5:
            return dit_output
        elif dit_output.dim() == 3:
            patch_vol = patch_size[0] * patch_size[1] * patch_size[2]
            if dit_output.shape[-1] == patch_vol * out_dim:
                out = dit_output.reshape(
                    B, T_p, H_p, W_p,
                    patch_size[0], patch_size[1], patch_size[2], out_dim,
                )
                out = out.permute(0, 7, 1, 4, 2, 5, 3, 6)
                return out.reshape(B, out_dim, F_lat, H_lat, W_lat)
            else:
                out = dit_output[:, :seq_len, :].reshape(
                    B, T_p, H_p, W_p, out_dim,
                ).permute(0, 4, 1, 2, 3)
                if out.shape[2:] != (F_lat, H_lat, W_lat):
                    out = F.interpolate(
                        out.reshape(B * out_dim, 1, T_p, H_p, W_p).float(),
                        size=(F_lat, H_lat, W_lat),
                        mode="nearest",
                    ).reshape(B, out_dim, F_lat, H_lat, W_lat)
                return out
        else:
            raise ValueError(f"Unexpected dit_output shape: {dit_output.shape}")

    # ------------------------------------------------------------------
    # Preprocess
    # ------------------------------------------------------------------

    def preprocess(self, images) -> torch.Tensor:
        """Preprocess raw images for the Wan encoder.

        Args:
            images: [B, 3, H, W] tensor, or list of PIL.Image / np.ndarray.

        Returns:
            Preprocessed tensor [B, 3, H, W] normalized to [-1, 1].
        """
        # Handle list of PIL/ndarray/tensor
        if isinstance(images, (list, tuple)):
            import numpy as np
            from PIL import Image
            tensors = []
            for img in images:
                if isinstance(img, Image.Image):
                    img = torch.from_numpy(np.array(img.convert("RGB"))).permute(2, 0, 1).float() / 255.0
                elif isinstance(img, np.ndarray):
                    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 if img.ndim == 3 else torch.from_numpy(img).float() / 255.0
                tensors.append(img if isinstance(img, torch.Tensor) else img)
            images = torch.stack(tensors, dim=0)

        x = images

        # Handle channel-last format
        if x.ndim == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)  # [B, H, W, 3] -> [B, 3, H, W]

        # Ensure float
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        elif x.max() > 1.5:
            # Likely [0, 255] range
            x = x.float() / 255.0

        # Resize to target image_size
        target_size = self.config.image_size
        if x.shape[-2] != target_size or x.shape[-1] != target_size:
            x = F.interpolate(
                x,
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            )

        # Normalize to [-1, 1] (standard for video VAEs)
        x = x * 2.0 - 1.0

        return x

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def encoder_dim(self) -> int:
        """Return the native hidden dimension of the encoder output.

        - With intermediate features: DiT hidden dim.
        - Without: VAE z_dim (latent channel count).
        """
        if self.config.use_intermediate_features and self.dit is not None:
            return self.variant_cfg["dim"]
        return self._vae_z_dim

    def __del__(self):
        """Clean up hooks on deletion."""
        self._clear_hooks()
