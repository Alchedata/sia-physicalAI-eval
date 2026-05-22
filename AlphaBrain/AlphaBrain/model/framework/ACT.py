# Copyright 2025 VLA-Engine. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

"""
ACT — Action Chunking Transformers (standalone implementation)
=============================================================

Reference:
    Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware
    Zhao et al., RSS 2023

Architecture:
    - ResNet18 visual encoder (per camera view)
    - CVAE encoder:  (robot_state, action_chunk) → z  (training only; z=0 at inference)
    - Transformer encoder:  [z_token, img_tokens, state_token]  → memory
    - Transformer decoder:  query_embed  →  action_chunk

Interface (same as QwenOFT / ToyVLA):
    examples: List[dict]
        - "image"  : List[PIL.Image]   (multi-view, any size)
        - "lang"   : str               (ignored during action prediction, kept for API compat)
        - "action" : np.ndarray        shape (T, action_dim)
        - "state"  : np.ndarray        shape (T_state, state_dim) [optional]

forward(examples) → {"action_loss": tensor}
predict_action(examples) → {"normalized_actions": np.ndarray  (B, chunk_len, action_dim)}
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models
from torchvision import transforms as T
from transformers import PretrainedConfig, PreTrainedModel

from AlphaBrain.model.tools import FRAMEWORK_REGISTRY
from AlphaBrain.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

_img_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=IMG_MEAN, std=IMG_STD),
])


def _encode_images(images_batch: List[List[Image.Image]], device, dtype) -> torch.Tensor:
    """
    images_batch: B × n_views list of PIL images
    Returns: (B, n_views, 512) ResNet18 feature vectors  (before projection)
    """
    B = len(images_batch)
    n_views = len(images_batch[0])
    tensors = []
    for views in images_batch:
        view_tensors = []
        for img in views:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img = img.convert("RGB")
            view_tensors.append(_img_transform(img))
        tensors.append(torch.stack(view_tensors, 0))  # (n_views, 3, 224, 224)
    imgs = torch.stack(tensors, 0).to(device)          # (B, n_views, 3, 224, 224)
    return imgs


# ---------------------------------------------------------------------------
# CVAE Encoder  (used only during training)
# ---------------------------------------------------------------------------
class CVAEEncoder(nn.Module):
    """
    Encodes (robot_state, action_chunk) → (mu, log_var).
    Inputs are projected to hidden_dim then fused through a small Transformer encoder.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, latent_dim: int, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.state_proj  = nn.Linear(state_dim,  hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            batch_first=True, dropout=0.1, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mu_head  = nn.Linear(hidden_dim, latent_dim)
        self.var_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, state: torch.Tensor, action_chunk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        state:        (B, state_dim)
        action_chunk: (B, chunk_len, action_dim)
        Returns: mu, log_var  each (B, latent_dim)
        """
        B = state.shape[0]
        state_emb  = self.state_proj(state).unsqueeze(1)       # (B, 1, H)
        action_emb = self.action_proj(action_chunk)             # (B, T, H)
        cls = self.cls_token.expand(B, -1, -1)                 # (B, 1, H)
        seq = torch.cat([cls, state_emb, action_emb], dim=1)   # (B, 2+T, H)
        out = self.encoder(seq)                                 # (B, 2+T, H)
        cls_out = out[:, 0, :]                                  # (B, H)
        return self.mu_head(cls_out), self.var_head(cls_out)


# ---------------------------------------------------------------------------
# ACT Model
# ---------------------------------------------------------------------------
@FRAMEWORK_REGISTRY.register("ACT")
class ACTModel(PreTrainedModel):
    """
    Standalone ACT (Action Chunking Transformers) model.

    Key design choices vs. paper:
      - Use ResNet18 (torchvision) instead of ResNet18 with backbone unfreezing
      - Replace FiLM conditioning with simple token concatenation
      - Use PyTorch native Transformer encoder / decoder
    """

    config_class = PretrainedConfig

    def __init__(self, config=None, **kwargs):
        super().__init__(PretrainedConfig())
        self.act_config = config

        # -------- hyper-parameters from config --------
        am_cfg = config.framework.action_model if hasattr(config, "framework") else None

        self.action_dim      = getattr(am_cfg, "action_dim", 7)         if am_cfg else 7
        self.state_dim       = getattr(am_cfg, "state_dim", 8)          if am_cfg else 8
        self.hidden_dim      = getattr(am_cfg, "hidden_dim", 256)       if am_cfg else 256
        self.latent_dim      = getattr(am_cfg, "latent_dim", 32)        if am_cfg else 32
        self.num_heads       = getattr(am_cfg, "num_heads", 8)          if am_cfg else 8
        self.num_enc_layers  = getattr(am_cfg, "num_enc_layers", 4)     if am_cfg else 4
        self.num_dec_layers  = getattr(am_cfg, "num_dec_layers", 7)     if am_cfg else 7
        self.kl_weight       = getattr(am_cfg, "kl_weight", 10.0)       if am_cfg else 10.0
        self.chunk_len       = (
            getattr(am_cfg, "future_action_window_size", 7) + 1
            if am_cfg else 8
        )
        self.n_views         = getattr(am_cfg, "n_views", 2)            if am_cfg else 2

        H = self.hidden_dim

        # -------- Visual backbone (ResNet18, per view) --------
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove final FC, keep up-to avgpool → output: (B*n_views, 512)
        self.visual_backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.visual_proj     = nn.Linear(512, H)

        # -------- State projection --------
        self.state_proj = nn.Linear(self.state_dim, H)

        # -------- Latent z projection --------
        self.latent_proj = nn.Linear(self.latent_dim, H)

        # -------- CVAE encoder (training only) --------
        self.cvae_encoder = CVAEEncoder(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=H,
            latent_dim=self.latent_dim,
            num_heads=min(self.num_heads, H // 32),
            num_layers=2,
        )

        # -------- Positional encodings --------
        # We have:  1 (z) + n_views (img) + 1 (state) = total context tokens
        max_ctx = 1 + self.n_views + 1
        self.pos_enc = nn.Embedding(max_ctx + self.chunk_len, H)

        # -------- Main Transformer encoder --------
        enc_layer = nn.TransformerEncoderLayer(
            d_model=H, nhead=self.num_heads, dim_feedforward=H * 4,
            batch_first=True, dropout=0.1, activation="gelu", norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=self.num_enc_layers)

        # -------- Action query embeddings --------
        self.query_embed = nn.Embedding(self.chunk_len, H)

        # -------- Transformer decoder --------
        dec_layer = nn.TransformerDecoderLayer(
            d_model=H, nhead=self.num_heads, dim_feedforward=H * 4,
            batch_first=True, dropout=0.1, activation="gelu", norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(dec_layer, num_layers=self.num_dec_layers)

        # -------- Action head --------
        self.action_head = nn.Linear(H, self.action_dim)

        self.l1_loss = nn.L1Loss()

        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        logger.info(
            f"[ACT] Built: action_dim={self.action_dim}, chunk_len={self.chunk_len}, "
            f"hidden_dim={H}, latent_dim={self.latent_dim}, "
            f"enc_layers={self.num_enc_layers}, dec_layers={self.num_dec_layers}, "
            f"n_views={self.n_views}, params={n_params:.1f}M"
        )

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _extract_visual_features(self, imgs_batch: List[List[Image.Image]]) -> torch.Tensor:
        """
        imgs_batch: B × n_views list of PIL images
        Returns: (B, n_views, H)
        """
        device = next(self.parameters()).device
        dtype  = next(self.parameters()).dtype
        imgs   = _encode_images(imgs_batch, device=device, dtype=torch.float32)   # (B, n_views, 3, 224, 224)
        B, V, C, Hh, Ww = imgs.shape
        flat = imgs.view(B * V, C, Hh, Ww)
        feats = self.visual_backbone(flat)          # (B*V, 512, 1, 1)
        feats = feats.flatten(1)                    # (B*V, 512)
        feats = self.visual_proj(feats.to(dtype))   # (B*V, H)
        feats = feats.view(B, V, -1)                # (B, n_views, H)
        return feats

    def _get_state(self, examples: List[dict]) -> torch.Tensor:
        """Extract current state from examples → (B, state_dim)."""
        device = next(self.parameters()).device
        dtype  = next(self.parameters()).dtype
        states = []
        for ex in examples:
            if "state" in ex and ex["state"] is not None:
                s = np.array(ex["state"])
                # state shape may be (T, D); take the most recent step
                if s.ndim == 2:
                    s = s[-1]             # (D,)
                states.append(s[:self.state_dim].astype(np.float32))
            else:
                states.append(np.zeros(self.state_dim, dtype=np.float32))
        return torch.tensor(np.stack(states), device=device, dtype=dtype)  # (B, state_dim)

    def _build_encoder_input(
        self,
        img_feats: torch.Tensor,   # (B, n_views, H)
        state_feat: torch.Tensor,  # (B, H)
        z: torch.Tensor,           # (B, H)
    ) -> torch.Tensor:
        """
        Concatenate [z_token, img_tokens, state_token] and add positional embeddings.
        Returns: (B, 1+n_views+1, H)
        """
        B = img_feats.shape[0]
        device = img_feats.device
        z_tok    = z.unsqueeze(1)               # (B, 1,      H)
        state_tok = state_feat.unsqueeze(1)     # (B, 1,      H)
        ctx = torch.cat([z_tok, img_feats, state_tok], dim=1)   # (B, 1+V+1, H)
        n_ctx = ctx.shape[1]
        pos_idx = torch.arange(n_ctx, device=device)
        ctx = ctx + self.pos_enc(pos_idx).unsqueeze(0)           # broadcast over B
        return ctx

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def forward(self, examples: List[dict], **kwargs) -> dict:
        """
        Returns:
            dict with keys:
                action_loss  (scalar)
                kl_loss      (scalar)
        """
        device = next(self.parameters()).device
        dtype  = next(self.parameters()).dtype
        B = len(examples)

        # 1. Extract ground-truth actions
        actions_np = np.array([np.array(ex["action"], dtype=np.float32) for ex in examples])  # (B, T, D)
        T = actions_np.shape[1]
        # Align chunk: take the first chunk_len steps if T >= chunk_len, else pad with last
        if T >= self.chunk_len:
            actions_np = actions_np[:, :self.chunk_len, :]
        else:
            pad = np.repeat(actions_np[:, -1:, :], self.chunk_len - T, axis=1)
            actions_np = np.concatenate([actions_np, pad], axis=1)
        actions = torch.tensor(actions_np, device=device, dtype=dtype)  # (B, chunk_len, D)

        # 2. Visual features
        imgs_batch = [ex["image"] for ex in examples]
        # Pad or truncate to n_views
        imgs_batch = [views[:self.n_views] + [views[-1]] * max(0, self.n_views - len(views)) for views in imgs_batch]
        img_feats  = self._extract_visual_features(imgs_batch)           # (B, n_views, H)

        # 3. Robot state
        state_raw  = self._get_state(examples)                           # (B, state_dim)
        state_feat = self.state_proj(state_raw)                          # (B, H)

        # 4. CVAE encoder → z
        mu, log_var = self.cvae_encoder(state_raw, actions)              # each (B, latent_dim)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z_raw = mu + eps * std                                           # (B, latent_dim)
        z = self.latent_proj(z_raw)                                      # (B, H)

        # 5. Transformer encoder
        ctx = self._build_encoder_input(img_feats, state_feat, z)        # (B, 1+V+1, H)
        memory = self.transformer_encoder(ctx)                           # (B, 1+V+1, H)

        # 6. Transformer decoder → action predictions
        n_ctx = ctx.shape[1]
        q_idx = torch.arange(self.chunk_len, device=device)
        queries = self.query_embed(q_idx).unsqueeze(0).expand(B, -1, -1)  # (B, chunk_len, H)
        # add positional embeddings to queries
        q_pos_idx = torch.arange(n_ctx, n_ctx + self.chunk_len, device=device)
        queries = queries + self.pos_enc(q_pos_idx).unsqueeze(0)
        decoded = self.transformer_decoder(queries, memory)               # (B, chunk_len, H)
        pred_actions = self.action_head(decoded)                          # (B, chunk_len, D)

        # 7. L1 reconstruction loss
        l1 = self.l1_loss(pred_actions, actions)

        # 8. KL divergence loss
        kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        action_loss = l1 + self.kl_weight * kl

        return {
            "action_loss": action_loss,
            "l1_loss": l1.detach(),
            "kl_loss": kl.detach(),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        # ---- flat format (from websocket client / M1Inference) ----
        batch_images: List[List] = None,   # B × n_views, each element is np.ndarray or PIL
        instructions: List[str] = None,
        states: np.ndarray = None,         # (B, T, state_dim) or (B, state_dim)
        **kwargs,
    ) -> dict:
        """
        Accepts two input formats:

        1. examples format (train / debug):
               examples = [{"image": [PIL,...], "lang": str, "state": np.ndarray}, ...]

        2. Flat format (from websocket server / M1Inference):
               batch_images = [[img0, img1], ...]   (B × n_views, np.ndarray or PIL)
               instructions = ["task description", ...]
               states       = np.ndarray (B, T, state_dim) or (B, state_dim)

        Returns:
            dict:
                normalized_actions: np.ndarray  (B, chunk_len, action_dim)
        """
        # ---- convert flat format → examples ----
        if examples is None:
            assert batch_images is not None, "Either examples or batch_images must be provided"
            B = len(batch_images)
            examples = []
            for i in range(B):
                imgs = []
                for img in batch_images[i]:
                    if isinstance(img, np.ndarray):
                        imgs.append(Image.fromarray(img.astype(np.uint8)))
                    else:
                        imgs.append(img)
                state = None
                if states is not None:
                    s = np.array(states[i])
                    if s.ndim == 2:
                        s = s[-1]   # take the most recent timestep
                    state = s[np.newaxis, :]   # (1, state_dim)
                examples.append({
                    "image":  imgs,
                    "lang":   instructions[i] if instructions else "",
                    "state":  state,
                    "action": np.zeros((self.chunk_len, self.action_dim), dtype=np.float32),
                })

        device = next(self.parameters()).device
        dtype  = next(self.parameters()).dtype
        B = len(examples)

        # visual features
        imgs_batch = [ex["image"] for ex in examples]
        imgs_batch = [views[:self.n_views] + [views[-1]] * max(0, self.n_views - len(views)) for views in imgs_batch]
        img_feats  = self._extract_visual_features(imgs_batch)

        # state
        state_raw  = self._get_state(examples)
        state_feat = self.state_proj(state_raw)

        # z = 0 at inference (mean of prior)
        z_raw = torch.zeros(B, self.latent_dim, device=device, dtype=dtype)
        z     = self.latent_proj(z_raw)

        # encoder
        ctx    = self._build_encoder_input(img_feats, state_feat, z)
        memory = self.transformer_encoder(ctx)

        # decoder
        n_ctx = ctx.shape[1]
        q_idx = torch.arange(self.chunk_len, device=device)
        queries = self.query_embed(q_idx).unsqueeze(0).expand(B, -1, -1)
        q_pos_idx = torch.arange(n_ctx, n_ctx + self.chunk_len, device=device)
        queries = queries + self.pos_enc(q_pos_idx).unsqueeze(0)
        decoded     = self.transformer_decoder(queries, memory)
        pred_actions = self.action_head(decoded)                           # (B, chunk_len, D)

        return {"normalized_actions": pred_actions.cpu().float().numpy()}
