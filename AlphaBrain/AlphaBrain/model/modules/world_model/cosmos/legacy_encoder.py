# Copyright 2026 VLA-Engine. All rights reserved.
#
# Licensed under the VLA-Engine License. You may not use this file except
# in compliance with the License.

"""
Cosmos Predict2 / Predict2.5 Encoder for VLA-Engine World Model Interface.

Uses the frozen Cosmos DiT (MiniTrainDIT, 2B) as a visual feature extractor:
  - WAN 2.1 VAE encodes single images to latent space
  - DiT processes the latent at sigma_min (clean data limit) with dummy text conditioning
  - Intermediate DiT block features are extracted as rich visual representations

The DiT architecture is identical between Predict2 and Predict2.5 (same 2B config).
The only difference is the text encoder (T5 vs Qwen2.5-VL), which we bypass entirely
since we use our own lightweight text encoder in the fusion stage.

Reuses existing VLA-Engine modules:
  - AlphaBrain.model.modules.world_model.cosmos.mini_train_dit.MiniTrainDIT
  - AlphaBrain.model.modules.world_model.cosmos.wan_vae.WanVAEWrapper
"""

import logging
import math
from contextlib import nullcontext
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from AlphaBrain.model.modules.world_model.base import BaseWorldModelEncoder, WorldModelEncoderConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configs
# ---------------------------------------------------------------------------

# Cosmos Predict2/2.5 2B DiT configuration
_DIT_2B_CONFIG = dict(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=18,  # Predict2.5 checkpoint uses 18 (16 latent + 2 conditioning)
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    concat_padding_mask=False,
    model_channels=2048,
    num_blocks=28,
    num_heads=16,
    mlp_ratio=4.0,
    crossattn_emb_channels=1024,
    use_crossattn_projection=False,
    pos_emb_cls="rope3d",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    atten_backend="torch",
    use_wan_fp32_strategy=False,
)

# Evenly spaced block indices for intermediate feature extraction (28 blocks)
_DEFAULT_INTERMEDIATE_LAYER_IDS = [7, 14, 21, 27]

# Noise schedule constants (from cosmos-policy)
_SIGMA_MIN = 0.0002
_SIGMA_DATA = 1.0


# ---------------------------------------------------------------------------
# Weight loading helpers
# ---------------------------------------------------------------------------

def _load_dit_weights(
    dit: nn.Module,
    checkpoint_path: str,
    strict: bool = False,
) -> None:
    """Load DiT weights from a Cosmos Predict2/2.5 checkpoint.

    Handles both full model checkpoints (with 'model' or 'ema' keys)
    and bare state_dict files. Also strips common prefixes from key names
    (e.g. 'net.', 'model.net.') to match MiniTrainDIT parameter names.
    """
    logger.info("Loading DiT weights from %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract state dict from nested checkpoint structures
    if isinstance(ckpt, dict):
        # Try common keys in order of preference
        for key in ("ema", "model", "state_dict", "net"):
            if key in ckpt:
                ckpt = ckpt[key]
                break

    if not isinstance(ckpt, dict):
        raise ValueError(
            f"Cannot extract state_dict from checkpoint at {checkpoint_path}"
        )

    # Strip common prefixes that differ between checkpoint sources
    prefixes_to_strip = ["net.", "model.net.", "model."]
    cleaned = {}
    for k, v in ckpt.items():
        clean_k = k
        for prefix in prefixes_to_strip:
            if clean_k.startswith(prefix):
                clean_k = clean_k[len(prefix):]
                break
        cleaned[clean_k] = v

    # Filter to only real parameter keys (skip _extra_state TE metadata)
    # that exist in our model AND have matching shapes.
    model_state = dit.state_dict()
    model_keys = set(model_state.keys())
    filtered = {}
    skipped = []
    shape_mismatch = []
    extra_state_skipped = 0
    for k, v in cleaned.items():
        # _extra_state buffers are transformer-engine RMSNorm metadata,
        # not real weights.  They almost always have a checkpoint/model
        # shape mismatch (ckpt=[5] vs model=[0]) that is harmless.
        # Skip them entirely to avoid noisy warnings.
        if "_extra_state" in k:
            extra_state_skipped += 1
            continue
        if k in model_keys:
            model_shape = model_state[k].shape
            if v.shape == model_shape:
                filtered[k] = v
            else:
                shape_mismatch.append((k, v.shape, model_shape))
        else:
            skipped.append(k)

    if extra_state_skipped:
        logger.info(
            "Skipped %d _extra_state (TE metadata) checkpoint keys",
            extra_state_skipped,
        )

    if shape_mismatch:
        logger.warning(
            "Shape mismatch for %d real keys (skipped): %s",
            len(shape_mismatch),
            [(k, f"ckpt={cs} model={ms}") for k, cs, ms in shape_mismatch[:5]],
        )

    if skipped:
        logger.info(
            "Skipped %d checkpoint keys not in MiniTrainDIT (e.g. %s)",
            len(skipped),
            skipped[:5],
        )

    missing, unexpected = dit.load_state_dict(filtered, strict=False, assign=True)

    # Report only real missing keys (exclude _extra_state metadata buffers)
    real_missing = [k for k in missing if "_extra_state" not in k]
    if real_missing:
        logger.warning(
            "Missing %d real parameter keys in DiT: %s",
            len(real_missing), real_missing[:10],
        )
    if unexpected:
        logger.warning("Unexpected %d keys: %s", len(unexpected), unexpected[:10])

    # Count only real (non-_extra_state) model keys for the summary
    real_model_keys = [k for k in model_keys if "_extra_state" not in k]
    logger.info(
        "Loaded %d / %d real DiT parameters from checkpoint "
        "(excluded %d _extra_state metadata)",
        len(filtered),
        len(real_model_keys),
        len(missing) - len(real_missing),
    )


def _find_dit_checkpoint(pretrained_dir: str) -> str:
    """Locate the DiT checkpoint file within a pretrained model directory.

    Search order:
      1. base/pre-trained/model.pt  (Predict2.5 NGC layout)
      2. model-480p-10fps.pt        (Predict2 layout)
      3. model-480p-16fps.pt
      4. model-720p-10fps.pt
      5. Any .pt file at top level
    """
    candidates = [
        os.path.join(pretrained_dir, "base", "pre-trained", "model.pt"),
        os.path.join(pretrained_dir, "model-480p-10fps.pt"),
        os.path.join(pretrained_dir, "model-480p-16fps.pt"),
        os.path.join(pretrained_dir, "model-720p-10fps.pt"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path

    # Fallback: search for any .pt file
    if os.path.isdir(pretrained_dir):
        for fname in sorted(os.listdir(pretrained_dir)):
            if fname.endswith(".pt") and "tokenizer" not in fname.lower():
                return os.path.join(pretrained_dir, fname)

    raise FileNotFoundError(
        f"No DiT checkpoint found in {pretrained_dir}. "
        "Expected base/pre-trained/model.pt or model-*.pt"
    )


def _find_vae_dir(pretrained_dir: str, config_pretrained_dir: str = "") -> str:
    """Locate a directory suitable for WanVAEWrapper loading.

    WanVAEWrapper expects a dir containing either tokenizer/tokenizer.pth or vae/.
    We look in the provided pretrained_dir first, then fall back to common locations.
    """
    # Check if pretrained_dir itself has tokenizer/tokenizer.pth
    if os.path.isfile(os.path.join(pretrained_dir, "tokenizer", "tokenizer.pth")):
        return pretrained_dir

    # Check config_pretrained_dir (may point to Predict2 dir with tokenizer)
    if config_pretrained_dir and os.path.isfile(
        os.path.join(config_pretrained_dir, "tokenizer", "tokenizer.pth")
    ):
        return config_pretrained_dir

    # Common fallback paths
    base = os.environ.get("PRETRAINED_MODELS_DIR", "data/pretrained_models")
    fallbacks = [
        os.path.join(base, "Cosmos-Predict2-2B-Video2World"),
        os.path.join(base, "Cosmos-Predict2.5-2B"),
    ]
    for fb in fallbacks:
        if os.path.isfile(os.path.join(fb, "tokenizer", "tokenizer.pth")):
            return fb

    # Last resort: return pretrained_dir and let WanVAEWrapper raise its own error
    logger.warning(
        "Could not locate VAE tokenizer.pth; WanVAEWrapper will attempt to load from %s",
        pretrained_dir,
    )
    return pretrained_dir


# ---------------------------------------------------------------------------
# CosmosEncoder
# ---------------------------------------------------------------------------

class CosmosEncoder(BaseWorldModelEncoder):
    """
    Cosmos Predict2/2.5 DiT as a frozen visual feature extractor.

    Architecture:
      image -> VAE encode -> latent -> DiT(t=sigma_min, dummy_text) -> intermediate features

    The DiT is run at the clean-data limit (sigma_min) so it acts as a
    deterministic feature extractor rather than a denoiser. Intermediate
    block outputs provide multi-scale representations analogous to ViT
    intermediate layers.

    Config fields used:
      - checkpoint_path: path to DiT .pt file, or directory to search in
      - pretrained_dir: directory containing tokenizer/ for VAE weights
                        (falls back to Cosmos-Predict2-2B-Video2World)
      - image_size: input image resolution (default 224)
      - use_intermediate_features: if True, extract and concatenate multi-layer features
      - intermediate_layer_ids: which block indices to extract (default [7,14,21,27])
    """

    def __init__(self, config: WorldModelEncoderConfig):
        super().__init__(config)
        self._model_channels = _DIT_2B_CONFIG["model_channels"]  # 2048
        self._build_encoder()

    # -----------------------------------------------------------------
    # Build
    # -----------------------------------------------------------------

    def _build_encoder(self) -> None:
        from AlphaBrain.model.modules.world_model.cosmos.mini_train_dit import MiniTrainDIT
        from AlphaBrain.model.modules.world_model.cosmos.wan_vae import WanVAEWrapper

        # --- DiT backbone ---
        logger.info("Building MiniTrainDIT (2B) for Cosmos encoder ...")
        # Build on meta device first, then load weights (memory efficient)
        with torch.device("meta"):
            self.dit = MiniTrainDIT(**_DIT_2B_CONFIG)

        # Locate and load checkpoint
        ckpt_path = self.config.checkpoint_path
        if os.path.isdir(ckpt_path):
            ckpt_path = _find_dit_checkpoint(ckpt_path)
        elif not ckpt_path or not os.path.isfile(ckpt_path):
            # Try searching in pretrained_dir
            search_dir = self.config.pretrained_dir or ckpt_path
            if search_dir and os.path.isdir(search_dir):
                ckpt_path = _find_dit_checkpoint(search_dir)
            else:
                raise FileNotFoundError(
                    f"DiT checkpoint not found: checkpoint_path={self.config.checkpoint_path}, "
                    f"pretrained_dir={self.config.pretrained_dir}"
                )

        _load_dit_weights(self.dit, ckpt_path)

        # Freeze DiT if configured (default: True).
        # When freeze_encoder=False, DiT backbone becomes trainable.
        if self.config.freeze_encoder:
            self.dit.eval()
            self.dit.requires_grad_(False)
            self.dit = self.dit.to(dtype=torch.bfloat16)
            logger.info("Cosmos DiT encoder frozen (freeze_encoder=True).")
        else:
            logger.info("Cosmos DiT encoder TRAINABLE (freeze_encoder=False).")

        # --- VAE tokenizer ---
        logger.info("Building WanVAEWrapper for Cosmos encoder ...")
        vae_dir = _find_vae_dir(
            self.config.pretrained_dir or os.path.dirname(ckpt_path),
            config_pretrained_dir=self.config.pretrained_dir,
        )
        self.vae = WanVAEWrapper(
            pretrained_dir=vae_dir,
            dtype=torch.bfloat16,
            device="cpu",  # Will be moved to correct device later via .to()
            temporal_window=16,
        )
        # VAE tokenizer is always frozen (not affected by freeze_encoder).
        self.vae.eval()
        self.vae.requires_grad_(False)

        # --- Native text encoder (T5 from Cosmos Predict2 pretrained dir) ---
        self._init_native_text_encoder()

        # --- Intermediate feature config ---
        if self.config.intermediate_layer_ids is not None:
            self.intermediate_layer_ids = list(self.config.intermediate_layer_ids)
        else:
            self.intermediate_layer_ids = list(_DEFAULT_INTERMEDIATE_LAYER_IDS)

        # Whether to use intermediate features (multi-layer) or final output only
        self._use_intermediate = self.config.use_intermediate_features

        # feature_proj not used in WM V2 (single intermediate layer).
        self.feature_proj = None

        # Pre-register sigma_min buffer for timestep creation
        self.register_buffer(
            "_sigma_min",
            torch.tensor([_SIGMA_MIN], dtype=torch.float32),
            persistent=False,
        )

        # Log parameter counts
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "CosmosEncoder built: model_channels=%d, num_blocks=%d, "
            "intermediate_layers=%s, use_intermediate=%s, "
            "total=%.2fM, trainable=%.2fM",
            self._model_channels,
            _DIT_2B_CONFIG["num_blocks"],
            self.intermediate_layer_ids,
            self._use_intermediate,
            total_params / 1e6,
            trainable_params / 1e6,
        )

    # -----------------------------------------------------------------
    # Native text encoder
    # -----------------------------------------------------------------

    def _init_native_text_encoder(self) -> None:
        """Load the native text encoder for DiT cross-attention conditioning.

        Strategy:
          1. If <pretrained_dir>/text_encoder/ exists -> load T5 (Predict2 path).
          2. Else if config.reason1_path points to a valid Reason1 model -> load
             Cosmos-Reason1 (Qwen2.5-VL-7B) with MEAN_POOLING + projection
             (3584 -> 1024).
          3. Otherwise -> dummy zero conditioning.
        """
        base_dir = self.config.pretrained_dir or os.path.dirname(self.config.checkpoint_path)
        te_dir = os.path.join(base_dir, "text_encoder")
        tok_dir = os.path.join(base_dir, "tokenizer")

        # Track which encoder type is active
        self._text_encoder_type = None  # "t5" | "t5_precomputed" | "reason1" | "reason1_precomputed" | None

        # ------ Branch 1: T5 text encoder (Cosmos Predict2) ------
        if os.path.isdir(te_dir):
            # Check for precomputed T5 embeddings first (avoids loading 4.8B T5 model)
            t5_emb_path = ""
            for candidate in [
                "data/pretrained_models/text_embeddings/t5_text_embeddings.pkl",
                "data/datasets/libero_datasets/t5_text_embeddings.pkl",
            ]:
                if os.path.isfile(candidate):
                    t5_emb_path = candidate
                    break

            if t5_emb_path:
                import pickle
                logger.info("Loading precomputed T5 embeddings from %s", t5_emb_path)
                with open(t5_emb_path, "rb") as f:
                    object.__setattr__(self, "_precomputed_text_cache", pickle.load(f))
                logger.info(
                    "Precomputed T5 cache: %d instructions, shape=[512, 1024]",
                    len(self._precomputed_text_cache),
                )
                self.native_text_encoder = True
                self.native_tokenizer = True
                self._text_encoder_type = "t5_precomputed"
                return

            from transformers import T5EncoderModel, T5Tokenizer

            logger.info("Loading native T5 text encoder from %s", te_dir)
            # Store as plain Python attr (NOT nn.Module child) to prevent
            # DeepSpeed ZeRO-2 from managing this frozen 4.8B model during
            # optimizer init, which would take 15+ minutes.
            object.__setattr__(self, "_native_t5_model", T5EncoderModel.from_pretrained(
                te_dir, torch_dtype=torch.bfloat16,
            ))
            self._native_t5_model.eval()
            self._native_t5_model.requires_grad_(False)
            self.native_text_encoder = True  # flag: T5 available

            tok_path = tok_dir if os.path.isdir(tok_dir) else te_dir
            self.native_tokenizer = T5Tokenizer.from_pretrained(tok_path)

            te_params = sum(p.numel() for p in self._native_t5_model.parameters())
            logger.info(
                "Native T5 text encoder loaded: %.2fM params (frozen), output_dim=%d",
                te_params / 1e6,
                self._native_t5_model.config.d_model,
            )
            self._text_encoder_type = "t5"
            return

        # ------ Branch 2: Precomputed Reason1 embeddings (fast, recommended) ------
        reason1_emb_path = getattr(self.config, 'reason1_embeddings_path', '')
        if not reason1_emb_path:
            for candidate in [
                'data/pretrained_models/text_embeddings/reason1_text_embeddings.pkl',
                'data/datasets/libero_datasets/reason1_text_embeddings.pkl',
                os.path.join(self.config.pretrained_dir or '', 'reason1_text_embeddings.pkl'),
            ]:
                if os.path.isfile(candidate):
                    reason1_emb_path = candidate
                    break

        if reason1_emb_path and os.path.isfile(reason1_emb_path):
            import pickle
            logger.info("Loading precomputed Reason1 embeddings from %s", reason1_emb_path)
            with open(reason1_emb_path, 'rb') as f:
                object.__setattr__(self, '_precomputed_text_cache', pickle.load(f))
            logger.info(
                "Precomputed Reason1 cache: %d instructions, shape=[512, 3584]",
                len(self._precomputed_text_cache),
            )
            self.reason1_proj = nn.Linear(3584, 1024).to(dtype=torch.bfloat16)
            self.native_text_encoder = True
            self.native_tokenizer = True
            self._text_encoder_type = "reason1_precomputed"
            return

        # ------ Branch 3: Cosmos-Reason1 online (slow fallback) ------
        reason1_path = getattr(self.config, "reason1_path", "")
        if reason1_path and os.path.isdir(reason1_path):
            self._init_reason1_text_encoder(reason1_path)
            return

        # ------ Branch 3: No text encoder available ------
        logger.warning(
            "Native text encoder not found (no T5 at %s, no Reason1 path configured). "
            "Will use dummy zero conditioning.", te_dir,
        )
        self.native_text_encoder = None
        self.native_tokenizer = None

    def _init_reason1_text_encoder(self, reason1_path: str) -> None:
        """Load Cosmos-Reason1 (Qwen2.5-VL-7B) as a frozen text encoder.

        Uses only the LLM backbone (no vision encoder forward) with
        MEAN_POOLING across all 28 hidden layers, then a trainable
        linear projection from 3584 -> 1024 to match DiT crossattn_emb dim.
        """
        from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

        logger.info("Loading Cosmos-Reason1 text encoder from %s", reason1_path)

        # Load full model but we only use the LLM text path (no images)
        self.native_text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            reason1_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",  # avoid flash_attn dependency issues
        )
        self.native_text_encoder.eval()
        self.native_text_encoder.requires_grad_(False)

        self.native_tokenizer = AutoTokenizer.from_pretrained(reason1_path)

        # Read model dimensions from config
        hidden_size = self.native_text_encoder.config.hidden_size  # 3584
        num_layers = self.native_text_encoder.config.num_hidden_layers  # 28
        target_dim = _DIT_2B_CONFIG["crossattn_emb_channels"]  # 1024

        # Trainable projection: MEAN_POOLING(28 layers) -> 3584-dim -> 1024-dim
        self.reason1_proj = nn.Linear(hidden_size, target_dim).to(dtype=torch.bfloat16)
        # Initialize with small values for stable training start
        nn.init.xavier_uniform_(self.reason1_proj.weight)
        nn.init.zeros_(self.reason1_proj.bias)

        te_params = sum(p.numel() for p in self.native_text_encoder.parameters())
        proj_params = sum(p.numel() for p in self.reason1_proj.parameters())
        logger.info(
            "Cosmos-Reason1 text encoder loaded: %.2fM params (frozen), "
            "hidden_size=%d, num_layers=%d, "
            "projection %d->%d (%.4fM trainable params)",
            te_params / 1e6, hidden_size, num_layers,
            hidden_size, target_dim, proj_params / 1e6,
        )
        self._text_encoder_type = "reason1"

    def encode_text(self, instructions, device):
        """Encode text using the native text encoder (T5 or Reason1).

        Returns:
            [B, L, 1024] text embeddings, or None if not available.
            For T5: L=512, output_dim=1024 (T5 d_model).
            For Reason1: L=512, output_dim=1024 (after 3584->1024 projection).
        """
        if not self.native_text_encoder or self.native_tokenizer is None:
            return None

        if self._text_encoder_type == "t5_precomputed":
            return self._encode_text_t5_precomputed(instructions, device)
        elif self._text_encoder_type == "reason1_precomputed":
            return self._encode_text_reason1_precomputed(instructions, device)
        elif self._text_encoder_type == "reason1":
            return self._encode_text_reason1(instructions, device)
        else:
            return self._encode_text_t5(instructions, device)

    def _encode_text_reason1_precomputed(self, instructions, device):
        """Look up precomputed Reason1 embeddings and project to 1024-dim."""
        batch_embeds = []
        for inst in instructions:
            if inst in self._precomputed_text_cache:
                emb = self._precomputed_text_cache[inst].to(dtype=torch.bfloat16)
            else:
                logger.warning("Instruction not in precomputed cache: %s", inst[:60])
                emb = torch.zeros(512, 3584, dtype=torch.bfloat16)
            batch_embeds.append(emb)

        text_embeds = torch.stack(batch_embeds).to(device)
        # Ensure projection layer is on correct device
        if next(self.reason1_proj.parameters()).device != text_embeds.device:
            self.reason1_proj = self.reason1_proj.to(text_embeds.device)
        text_embeds = self.reason1_proj(text_embeds)  # [B, 512, 1024]
        return text_embeds

    def _encode_text_t5_precomputed(self, instructions, device):
        """Look up precomputed T5 embeddings (no projection needed).

        T5-XXL outputs 1024-dim, matching DiT crossattn_emb_channels directly.
        """
        batch_embeds = []
        for inst in instructions:
            if inst in self._precomputed_text_cache:
                emb = self._precomputed_text_cache[inst].to(dtype=torch.bfloat16)
            else:
                logger.warning("Instruction not in T5 precomputed cache: %s", inst[:60])
                emb = torch.zeros(512, 1024, dtype=torch.bfloat16)
            batch_embeds.append(emb)

        text_embeds = torch.stack(batch_embeds).to(device)  # [B, 512, 1024]
        return text_embeds

    def _encode_text_t5(self, instructions, device):
        """Encode text with T5 (Cosmos Predict2 path)."""
        # Ensure T5 is on the target device (GPU) for fast forward.
        te_device = next(self._native_t5_model.parameters()).device
        if str(te_device) != str(device):
            object.__setattr__(self, "_native_t5_model", self._native_t5_model.to(device))

        tokens = self.native_tokenizer(
            instructions,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            text_embeds = self._native_t5_model(**tokens).last_hidden_state

        # Zero out padding positions (matching original Cosmos Predict2 behavior)
        attn_mask = tokens["attention_mask"]  # [B, 512]
        lengths = attn_mask.sum(dim=1)  # [B]
        for i in range(text_embeds.shape[0]):
            text_embeds[i, lengths[i]:] = 0

        return text_embeds.to(device=device, dtype=torch.bfloat16)

    @staticmethod
    def _mean_normalize(tensor: torch.Tensor) -> torch.Tensor:
        """Mean-normalize per token (matching Cosmos Predict2.5 convention)."""
        return (tensor - tensor.mean(dim=-1, keepdim=True)) / (
            tensor.std(dim=-1, keepdim=True) + 1e-8
        )

    def _encode_text_reason1(self, instructions, device):
        """Encode text with Cosmos-Reason1 (Qwen2.5-VL-7B).

        Process:
          1. Tokenize with padding/truncation to 512 tokens.
          2. Forward through LLM with output_hidden_states=True.
          3. Mean-normalize each layer's hidden states.
          4. MEAN_POOLING across all 28 layers -> [B, 512, 3584].
          5. Project 3584 -> 1024 via trainable linear layer.
          6. Zero out padding positions.
        """
        # Ensure Reason1 encoder and projection are on the correct device (lazy move on first call)
        te_device = next(self.native_text_encoder.parameters()).device
        if str(te_device) != str(device):
            logger.info("[Reason1] Moving text encoder from %s to %s", te_device, device)
            self.native_text_encoder = self.native_text_encoder.to(device)
        proj_device = next(self.reason1_proj.parameters()).device
        if str(proj_device) != str(device):
            logger.info("[Reason1] Moving reason1_proj from %s to %s", proj_device, device)
            self.reason1_proj = self.reason1_proj.to(device)

        # Tokenize (simple approach, no chat template needed for DiT conditioning)
        tokens = self.native_tokenizer(
            instructions,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        ).to(device)

        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        # Forward through Reason1 LLM only (no vision inputs)
        with torch.no_grad():
            outputs = self.native_text_encoder.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        # outputs.hidden_states: tuple of (num_layers+1) tensors [B, L, 3584]
        # Index 0 is the embedding layer output; indices 1..28 are transformer layers
        all_hidden = outputs.hidden_states[1:]  # skip embedding layer, keep 28 layers

        # Mean-normalize each layer (matching Cosmos Predict2.5 convention)
        normalized = [self._mean_normalize(h) for h in all_hidden]

        # MEAN_POOLING across layers -> [B, 512, 3584]
        text_embeds = torch.stack(normalized, dim=0).mean(dim=0)  # [B, 512, 3584]

        # Project 3584 -> 1024 (trainable)
        text_embeds = self.reason1_proj(text_embeds.to(dtype=torch.bfloat16))  # [B, 512, 1024]

        # Zero out padding positions
        lengths = attention_mask.sum(dim=1)  # [B]
        for i in range(text_embeds.shape[0]):
            text_embeds[i, lengths[i]:] = 0

        return text_embeds.to(dtype=torch.bfloat16)

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
        target_size = self.config.image_size  # default 224

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
                elif isinstance(img, torch.Tensor):
                    pass
                tensors.append(img)
            images = torch.stack(tensors, dim=0)

        # Handle channel-last format
        if images.ndim == 4 and images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)

        # Normalize to [0, 1] if uint8
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        elif images.max() > 1.5:
            # Likely [0, 255] float
            images = images / 255.0

        # Resize to target size
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
    # Encode
    # -----------------------------------------------------------------

    def encode_images(self, pixel_values: torch.Tensor, text_embeds=None) -> torch.Tensor:
        """Encode preprocessed images into visual token features.

        Args:
            pixel_values: [B, 3, H, W] float tensor in [-1, 1].
            text_embeds: [B, L, d_model] text embeddings for DiT cross-attention.
                         If None, uses dummy zero conditioning (legacy mode).

        Returns:
            [B, N, encoder_dim] visual feature tokens.
        """
        # Use no_grad when encoder is frozen to save memory (like vjepa/wan)
        ctx = torch.no_grad() if self.config.freeze_encoder else nullcontext()
        with ctx:
            return self._encode_images_inner(pixel_values, text_embeds)

    def _encode_images_inner(self, pixel_values, text_embeds=None):
        """Inner encoding logic, separated for no_grad context control."""
        # Move to encoder device and dtype
        encoder_device = next(self.dit.parameters()).device
        pixel_values = pixel_values.to(device=encoder_device, dtype=torch.bfloat16)

        B = pixel_values.shape[0]
        device = pixel_values.device
        dtype = torch.bfloat16

        # --- 1. VAE encode: [B, 3, H, W] -> [B, 16, 1, H/8, W/8] ---
        # Reshape image to single-frame video: [B, 3, 1, H, W]
        video = pixel_values.unsqueeze(2)  # [B, 3, 1, H, W]
        latent = self.vae.encode(video.to(dtype))  # [B, 16, 1, H/8, W/8]

        # --- 1b. Pad latent channels to match DiT in_channels (16 -> 18) ---
        # Cosmos Policy DiT expects 2 extra channels (padding mask)
        if latent.shape[1] < 18:
            pad_channels = 18 - latent.shape[1]
            pad = torch.zeros(
                latent.shape[0], pad_channels, *latent.shape[2:],
                device=latent.device, dtype=latent.dtype,
            )
            latent = torch.cat([latent, pad], dim=1)

        # --- 2. Normalize latent with sigma_data ---
        # At sigma_min, the EDM preconditioning essentially passes through the
        # clean latent. We scale by sigma_data for consistency with training.
        latent = latent * _SIGMA_DATA

        # --- 3. Create conditioning ---
        # Timestep: use sigma_min (clean data limit, minimal denoising)
        timesteps = self._sigma_min.expand(B).to(device=device, dtype=dtype)

        # Cross-attention text embedding for DiT
        if text_embeds is not None:
            # Use real text embeddings from native text encoder
            crossattn_emb = text_embeds.to(device=device, dtype=dtype)
        else:
            # Fallback: dummy zero conditioning (legacy / no text encoder)
            crossattn_emb = torch.zeros(
                B, 1, _DIT_2B_CONFIG["crossattn_emb_channels"],
                device=device, dtype=dtype,
            )

        # --- 4. Forward DiT with intermediate feature extraction ---
        latent = latent.to(dtype)
        # Ensure padding channels also in correct dtype
        latent = latent.to(dtype=torch.bfloat16)
        if self._use_intermediate:
            _, intermediate_features = self.dit(
                x_B_C_T_H_W=latent,
                timesteps_B_T=timesteps,
                crossattn_emb=crossattn_emb,
                intermediate_feature_ids=self.intermediate_layer_ids,
            )
            # intermediate_features: list of [B, T*H*W, model_channels]
            # Each element shape: [B, N, 2048] where N = 1 * (H/8/2) * (W/8/2)

            if len(intermediate_features) > 1 and self.feature_proj is not None:
                # Concatenate along channel dim and project
                concat = torch.cat(intermediate_features, dim=-1)  # [B, N, 2048*num_layers]
                visual_tokens = self.feature_proj(concat)  # [B, N, 2048]
            elif len(intermediate_features) == 1:
                visual_tokens = intermediate_features[0]
            else:
                # Fallback: average
                visual_tokens = torch.stack(intermediate_features, dim=0).mean(dim=0)
        else:
            # Use the final DiT output, but still extract last block features
            # via intermediate_feature_ids for richer representation
            _, intermediate_features = self.dit(
                x_B_C_T_H_W=latent,
                timesteps_B_T=timesteps,
                crossattn_emb=crossattn_emb,
                intermediate_feature_ids=[_DIT_2B_CONFIG["num_blocks"] - 1],
            )
            visual_tokens = intermediate_features[0]  # [B, N, 2048]

        return visual_tokens

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    # -----------------------------------------------------------------
    # V2: Video prediction methods
    # -----------------------------------------------------------------

    def encode_to_latent(self, images: torch.Tensor) -> torch.Tensor:
        """Encode preprocessed images to VAE latent space only (no DiT forward).

        Args:
            images: [B, 3, H, W] float tensor in [-1, 1] (output of preprocess()).

        Returns:
            [B, 16, 1, 28, 28] VAE latent tensor (bf16).
        """
        encoder_device = next(self.dit.parameters()).device
        images = images.to(device=encoder_device, dtype=torch.bfloat16)
        # Reshape to single-frame video: [B, 3, 1, H, W]
        video = images.unsqueeze(2)
        with torch.no_grad():
            latent = self.vae.encode(video)  # [B, 16, 1, H/8, W/8]
        return latent

    def compute_video_loss(
        self,
        latent_t: torch.Tensor,
        latent_t1: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Compute video prediction loss: denoise a 2-frame sequence with condition frame.

        Uses the DiT backbone (must be unfrozen) to predict the next frame
        given the current frame as a condition, following Cosmos-Policy's
        RectifiedFlow preconditioning.

        Args:
            latent_t:  [B, 16, 1, 28, 28] current frame latent (clean).
            latent_t1: [B, 16, 1, 28, 28] next frame latent (clean, prediction target).
            text_embeds: [B, L, 1024] text conditioning for DiT cross-attention.

        Returns:
            Scalar video prediction loss (MSE on predicted frame).
        """
        device = latent_t.device
        dtype = torch.bfloat16
        B = latent_t.shape[0]

        # --- 1. Build 2-frame clean latent: [B, 16, 2, 28, 28] ---
        x0 = torch.cat([latent_t, latent_t1], dim=2)  # [B, 16, 2, 28, 28]
        x0 = x0 * _SIGMA_DATA  # match training scaling

        # --- 2. Pad to 18 channels ---
        if x0.shape[1] < 18:
            pad_ch = 18 - x0.shape[1]
            pad = torch.zeros(B, pad_ch, *x0.shape[2:], device=device, dtype=dtype)
            x0 = torch.cat([x0, pad], dim=1)

        # --- 3. Sample sigma and add noise to ALL frames ---
        # Use LogNormal sampling matching cosmos-policy SDE
        log_sigma = torch.randn(B, device=device) * 1.2 - 1.2
        sigma = log_sigma.exp().clamp(min=1e-4, max=80.0).to(dtype)
        sigma_5d = sigma.view(B, 1, 1, 1, 1)
        epsilon = torch.randn_like(x0)
        xt = x0 + sigma_5d * epsilon

        # --- 4. EDM / RectifiedFlow preconditioning ---
        sigma_f32 = sigma.float()
        sigma_5d_f32 = sigma_f32.view(B, 1, 1, 1, 1)
        t = sigma_5d_f32 / (sigma_5d_f32 + 1.0)
        c_skip = 1.0 - t
        c_out = -t
        c_in = 1.0 - t

        # Per-frame c_noise
        T_frames = 2
        c_noise_val = (sigma_f32 / (sigma_f32 + 1.0))  # [B]
        c_noise_per_frame = c_noise_val.unsqueeze(1).expand(B, T_frames).clone()

        # --- 5. Condition frame handling (frame 0 = condition) ---
        net_input = xt.float() * c_in
        # Replace condition frame (index 0) with gt / sigma_data
        gt_cond = x0[:, :, :1, :, :].float() / _SIGMA_DATA
        net_input[:, :, :1, :, :] = gt_cond
        # Condition frame c_noise = 0 (sigma_conditional = 0)
        c_noise_per_frame[:, 0] = 0.0

        # condition_video_input_mask: [B, 1, T, H, W]
        H_lat, W_lat = x0.shape[3], x0.shape[4]
        cond_mask = torch.zeros(B, 1, T_frames, H_lat, W_lat, device=device, dtype=dtype)
        cond_mask[:, :, :1, :, :] = 1.0

        # --- 6. DiT forward ---
        ts_dtype = torch.float32 if self.dit.use_wan_fp32_strategy else torch.bfloat16
        fps = torch.tensor([16.0] * B, device=device, dtype=torch.bfloat16)
        # Move all DiT inputs to the encoder device explicitly
        dit_device = next(self.dit.parameters()).device
        net_output = self.dit(
            x_B_C_T_H_W=net_input.to(device=dit_device, dtype=dtype),
            timesteps_B_T=c_noise_per_frame.to(device=dit_device, dtype=ts_dtype),
            crossattn_emb=text_embeds.to(device=dit_device, dtype=dtype),
            fps=fps.to(dit_device),
        )

        # --- 7. Combine: x0_pred = c_skip * xt[:16] + c_out * net_output ---
        # DiT outputs 16 channels (out_channels=16), xt has 18 (padded input)
        x0_pred = c_skip * xt[:, :16].float() + c_out * net_output.float()

        # --- 8. Loss: weighted MSE on predicted frame only (frame index 1) ---
        pred_frame = x0_pred[:, :, 1:2, :, :]     # [B, 16, 1, 28, 28]
        gt_frame = x0[:, :16, 1:2, :, :].float()  # [B, 16, 1, 28, 28]

        # Loss weight: (1 + sigma)^2 / sigma^2
        weight = ((1.0 + sigma_f32) ** 2 / (sigma_f32 ** 2)).view(B, 1, 1, 1, 1)
        loss = weight * (pred_frame - gt_frame) ** 2
        return loss.mean()

    @property
    def encoder_dim(self) -> int:
        """Native hidden dimension of the Cosmos DiT encoder (2048 for 2B)."""
        return self._model_channels

    def encode_images_with_video_loss(
        self,
        latent_t: torch.Tensor,
        latent_t1: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single DiT forward that simultaneously yields action features and video loss.

        V2 single-forward design: runs the DiT ONCE to get both:
          - Layer 18 intermediate features  -> for action head
          - Final denoised output           -> for video prediction loss

        Args:
            latent_t:    [B, 16, 1, 28, 28] current frame VAE latent (clean).
            latent_t1:   [B, 16, 1, 28, 28] next frame VAE latent (clean, target).
            text_embeds: [B, L, 1024] text conditioning for DiT cross-attention.

        Returns:
            visual_tokens: [B, N, 2048]  layer-18 intermediate features.
            video_loss:    scalar        weighted MSE on predicted future frame.
        """
        device = latent_t.device
        dtype = torch.bfloat16
        B = latent_t.shape[0]

        # --- 1. Build 2-frame clean latent: [B, 16, 2, 28, 28] ---
        x0 = torch.cat([latent_t, latent_t1], dim=2)  # [B, 16, 2, 28, 28]

        # Pad channels 16 -> 18 (Cosmos-Policy DiT expects 18 input channels)
        pad_ch = 18 - x0.shape[1]  # = 2
        x0_padded = F.pad(x0, (0, 0, 0, 0, 0, 0, 0, pad_ch), value=0)  # [B, 18, 2, 28, 28]
        # x0_scaled = x0_padded * _SIGMA_DATA  # sigma_data = 1.0 (explicit; used below)

        # --- 2. Sample sigma (LogNormal, matching compute_video_loss) ---
        # Sync sigma across ranks to ensure identical backward graph topology
        log_sigma = torch.randn(B, device=device) * 1.2 - 1.2
        sigma = log_sigma.exp().clamp(min=1e-4, max=80.0)  # [B], float32

        # --- 3. Add noise to the padded clean latent ---
        noise = torch.randn_like(x0_padded)
        sigma_5d = sigma.view(B, 1, 1, 1, 1)
        xt = x0_padded + sigma_5d * noise  # [B, 18, 2, 28, 28]

        # --- 4. RectifiedFlow preconditioning scalars ---
        t = sigma / (sigma + 1.0)  # [B], in (0, 1)
        c_skip = 1.0 - t           # [B]
        c_out = -t                 # [B]
        c_in = 1.0 - t             # [B]

        # --- 5. Condition frame handling (frame 0 = given clean frame) ---
        gt_cond = x0_padded[:, :, :1, :, :]  # [B, 18, 1, 28, 28] — clean condition

        # Replace noisy frame 0 with clean condition (no noise on condition frame)
        xt[:, :, :1, :, :] = gt_cond

        # Per-frame c_noise timesteps: condition frame -> 0, prediction frame -> t
        c_noise_cond = torch.zeros(B, device=device)   # [B]
        c_noise_pred = t                                # [B]
        timesteps = torch.stack([c_noise_cond, c_noise_pred], dim=1)  # [B, 2]

        # --- 6. Scale net input by c_in, then restore condition frame ---
        net_input = xt.float() * c_in.view(B, 1, 1, 1, 1)
        # Condition frame is always passed as clean: gt / sigma_data
        net_input[:, :, :1, :, :] = gt_cond.float() / _SIGMA_DATA

        # --- 7. DiT forward — extract layer 18 intermediate + final output ---
        dit_device = next(self.dit.parameters()).device
        ts_dtype = torch.float32 if self.dit.use_wan_fp32_strategy else torch.bfloat16
        fps = torch.tensor([16.0] * B, device=device, dtype=torch.bfloat16)

        net_output, intermediate_features = self.dit(
            x_B_C_T_H_W=net_input.to(device=dit_device, dtype=dtype),
            timesteps_B_T=timesteps.to(device=dit_device, dtype=ts_dtype),
            crossattn_emb=text_embeds.to(device=dit_device, dtype=dtype),
            fps=fps.to(dit_device),
            crossattn_mask=None,
            padding_mask=None,
            condition_video_augment_sigma_B=None,
            intermediate_feature_ids=[18],
        )

        # --- 8. Extract layer-18 visual tokens for action head ---
        visual_tokens = intermediate_features[0]  # [B, N, 2048]

        # --- 9. Reconstruct x0_pred from net_output ---
        # DiT outputs 16 channels; xt has 18 (padded)
        x0_pred = (
            c_skip.view(B, 1, 1, 1, 1) * xt[:, :16, :, :, :].float()
            + c_out.view(B, 1, 1, 1, 1) * net_output.float()
        )  # [B, 16, 2, 28, 28]

        # Output replacement: condition frame prediction = GT condition
        x0_pred[:, :, :1, :, :] = x0[:, :16, :1, :, :].float() * _SIGMA_DATA

        # --- 10. Video loss on predicted future frame only (frame index 1) ---
        pred_future = x0_pred[:, :16, 1:2, :, :]                   # [B, 16, 1, 28, 28]
        gt_future = x0[:, :16, 1:2, :, :].float() * _SIGMA_DATA    # [B, 16, 1, 28, 28]

        # Loss weight: (1 + sigma)^2 / sigma^2
        weight = ((1.0 + sigma) ** 2 / (sigma ** 2)).view(B, 1, 1, 1, 1)
        video_loss = (weight * F.mse_loss(pred_future, gt_future, reduction='none')).mean()

        return visual_tokens, video_loss

    def encode_images_all_layers(
        self,
        pixel_values: torch.Tensor,
        text_embeds=None,
    ) -> List[torch.Tensor]:
        """Encode images and return features from ALL 28 DiT layers (PI-style).

        Runs the same VAE + DiT forward as encode_images(), but requests
        intermediate features from every DiT block.  Useful for PI-style
        layer-wise action heads that consume multi-scale representations.

        Args:
            pixel_values: [B, 3, H, W] float tensor in [-1, 1].
            text_embeds:  [B, L, d_model] text embeddings, or None for dummy zeros.

        Returns:
            List of 28 tensors, each [B, N, 2048] — one per DiT block.
        """
        ctx = torch.no_grad() if self.config.freeze_encoder else nullcontext()
        with ctx:
            return self._encode_images_all_layers_inner(pixel_values, text_embeds)

    def _encode_images_all_layers_inner(
        self,
        pixel_values: torch.Tensor,
        text_embeds=None,
    ) -> List[torch.Tensor]:
        """Inner logic for encode_images_all_layers (separated for no_grad control)."""
        encoder_device = next(self.dit.parameters()).device
        pixel_values = pixel_values.to(device=encoder_device, dtype=torch.bfloat16)

        B = pixel_values.shape[0]
        device = pixel_values.device
        dtype = torch.bfloat16

        # --- 1. VAE encode: [B, 3, H, W] -> [B, 16, 1, H/8, W/8] ---
        video = pixel_values.unsqueeze(2)          # [B, 3, 1, H, W]
        latent = self.vae.encode(video.to(dtype))  # [B, 16, 1, H/8, W/8]

        # --- 2. Pad channels 16 -> 18 ---
        if latent.shape[1] < 18:
            pad_channels = 18 - latent.shape[1]
            pad = torch.zeros(
                latent.shape[0], pad_channels, *latent.shape[2:],
                device=latent.device, dtype=latent.dtype,
            )
            latent = torch.cat([latent, pad], dim=1)

        # --- 3. Scale by sigma_data ---
        latent = latent * _SIGMA_DATA

        # --- 4. Timestep at sigma_min (clean data limit) ---
        timesteps = self._sigma_min.expand(B).to(device=device, dtype=dtype)

        # --- 5. Cross-attention conditioning ---
        if text_embeds is not None:
            crossattn_emb = text_embeds.to(device=device, dtype=dtype)
        else:
            crossattn_emb = torch.zeros(
                B, 1, _DIT_2B_CONFIG["crossattn_emb_channels"],
                device=device, dtype=dtype,
            )

        # --- 6. DiT forward with ALL 28 intermediate layers ---
        latent = latent.to(dtype=dtype)
        all_layer_ids = list(range(_DIT_2B_CONFIG["num_blocks"]))  # [0, 1, ..., 27]
        _, intermediate_features = self.dit(
            x_B_C_T_H_W=latent,
            timesteps_B_T=timesteps,
            crossattn_emb=crossattn_emb,
            intermediate_feature_ids=all_layer_ids,
        )
        # intermediate_features: list of 28 tensors, each [B, N, 2048]
        return intermediate_features

    # -----------------------------------------------------------------
    # V2: Future frame denoising (inference)
    # -----------------------------------------------------------------

    @torch.inference_mode()
    def denoise_future_frame(
        self,
        latent_t: torch.Tensor,      # [B, 16, 1, 28, 28] current frame latent
        text_embeds: torch.Tensor,    # [B, L, 1024] text conditioning
        num_steps: int = 5,
        sigma_min: float = 4.0,
        sigma_max: float = 80.0,
    ) -> torch.Tensor:
        """Iterative denoising to generate future frame prediction.

        Uses Euler sampler with RectifiedFlow preconditioning, matching
        CosmosPolicy's inference approach but for V2's 2-frame setup.

        Args:
            latent_t: Current frame VAE latent (clean).
            text_embeds: Cross-attention text conditioning.
            num_steps: Number of denoising steps.
            sigma_min: Minimum sigma for denoising schedule.
            sigma_max: Maximum sigma for denoising schedule.

        Returns:
            future_latent: [B, 16, 1, 28, 28] predicted future frame latent.
        """
        device = latent_t.device
        dtype = torch.bfloat16
        B = latent_t.shape[0]
        dit_device = next(self.dit.parameters()).device

        # Handle None text_embeds (fallback to dummy zero conditioning)
        if text_embeds is None:
            text_embeds = torch.zeros(
                B, 1, _DIT_2B_CONFIG["crossattn_emb_channels"],
                device=dit_device, dtype=dtype,
            )

        # Build sigma schedule (geometric: sigma_max -> sigma_min)
        sigmas = torch.exp(
            torch.linspace(math.log(sigma_max), math.log(sigma_min), num_steps + 1, device=device)
        )

        # Initialize: [condition_frame (clean), prediction_frame (noise)]
        future_noise = torch.randn(B, 16, 1, 28, 28, device=device, dtype=dtype) * sigmas[0]

        # Build 2-frame latent
        # Pad to 18 channels
        gt_cond_16 = latent_t  # [B, 16, 1, 28, 28]
        pad = torch.zeros(B, 2, 1, 28, 28, device=device, dtype=dtype)
        gt_cond_18 = torch.cat([gt_cond_16, pad], dim=1)  # [B, 18, 1, 28, 28]

        xt_future = future_noise  # [B, 16, 1, 28, 28] - starts as pure noise

        for i in range(num_steps):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]

            # RectifiedFlow preconditioning
            t = sigma / (sigma + 1.0)
            c_skip = 1.0 - t
            c_out = -t
            c_in = 1.0 - t

            # Build full 2-frame input: [condition, noisy_prediction]
            # Pad prediction to 18 channels
            pred_pad = torch.zeros(B, 2, 1, 28, 28, device=device, dtype=dtype)
            xt_pred_18 = torch.cat([xt_future.to(dtype), pred_pad], dim=1)  # [B, 18, 1, 28, 28]

            xt = torch.cat([gt_cond_18, xt_pred_18], dim=2)  # [B, 18, 2, 28, 28]

            # Scale input
            net_input = xt.float() * c_in
            # Restore condition frame to clean
            net_input[:, :, :1, :, :] = gt_cond_18.float() / _SIGMA_DATA

            # Per-frame timesteps
            c_noise_cond = torch.zeros(B, device=device)
            c_noise_pred = t.expand(B) if t.dim() == 0 else t
            timesteps = torch.stack([c_noise_cond, c_noise_pred], dim=1)

            # DiT forward
            ts_dtype = torch.float32 if self.dit.use_wan_fp32_strategy else torch.bfloat16
            fps = torch.tensor([16.0] * B, device=device, dtype=torch.bfloat16)

            net_output = self.dit(
                x_B_C_T_H_W=net_input.to(device=dit_device, dtype=dtype),
                timesteps_B_T=timesteps.to(device=dit_device, dtype=ts_dtype),
                crossattn_emb=text_embeds.to(device=dit_device, dtype=dtype),
                fps=fps.to(dit_device),
            )
            # Handle tuple return (if intermediate features requested)
            if isinstance(net_output, tuple):
                net_output = net_output[0]

            # x0 prediction
            x0_pred = c_skip * xt[:, :16].float() + c_out * net_output.float()

            # Extract predicted future frame
            x0_future = x0_pred[:, :, 1:2, :, :]  # [B, 16, 1, 28, 28]

            # Euler step: move from sigma to sigma_next
            # d = (xt_future - x0_future) / sigma
            # xt_future_next = xt_future + d * (sigma_next - sigma)
            d = (xt_future.float() - x0_future) / sigma
            xt_future = (xt_future.float() + d * (sigma_next - sigma)).to(dtype)

        return xt_future  # [B, 16, 1, 28, 28]

    @torch.inference_mode()
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """VAE decode latent to pixel space.

        Args:
            latent: [B, 16, T, H, W] latent tensor
        Returns:
            video: [B, 3, T*4, H*8, W*8] pixel tensor in [-1, 1]
        """
        return self.vae.decode(latent / _SIGMA_DATA)
