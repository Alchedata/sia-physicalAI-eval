# Copyright 2025 VLA-Engine. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

"""
ToyVLA — 极简 VLA 调试模型
=============================
设计目标:
  - **无 VLM 依赖**, 无需 Qwen / LLM，秒级加载
  - 接口与 QwenOFT 完全一致 (forward / predict_action 接受同样的 examples List[dict])
  - 能在几百步内 overfit 小样本 → 验证训练管线是否正确

验证方法:
  1. 把 N 个固定样本喂进去，train 几百步
  2. 如果 action_loss 接近 0、eval MSE 接近 0 → 管线正确
  3. 否则说明 data → forward → loss → backward 链路有 bug

Interface (与 QwenOFT 相同):
  examples: List[dict]
    - "image"  : List[PIL.Image]  (multi-view, 各尺寸均可)
    - "lang"   : str
    - "action" : np.ndarray  shape (T, action_dim)

forward(examples) → {"action_loss": scalar_tensor}
predict_action(examples) → {"normalized_actions": np.ndarray (B, chunk_len, action_dim)}
"""

import hashlib
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import PretrainedConfig, PreTrainedModel

from AlphaBrain.model.tools import FRAMEWORK_REGISTRY
from AlphaBrain.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


# ---------------------------------------------------------------------------
# Tiny image encoder: PIL → fixed-length feature vector (no GPU needed)
# ---------------------------------------------------------------------------
class TinyImageEncoder(nn.Module):
    """把任意尺寸 PIL Image 压成 (img_feat_dim,) 向量，纯卷积，参数量 ~10K"""

    def __init__(self, img_feat_dim: int = 128):
        super().__init__()
        self.img_feat_dim = img_feat_dim
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 8, stride=8),   # 224→28
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=4),  # 28→7
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),         # →(32,1,1)
        )
        self.proj = nn.Linear(32, img_feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) → (B, img_feat_dim)"""
        x = self.net(x)
        x = x.flatten(1)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Tiny text encoder: hash str → fixed embedding (deterministic, no tokenizer)
# ---------------------------------------------------------------------------
class TinyTextEncoder(nn.Module):
    def __init__(self, vocab_size: int = 256, text_feat_dim: int = 64):
        super().__init__()
        self.text_feat_dim = text_feat_dim
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, text_feat_dim)

    def forward(self, texts: List[str]) -> torch.Tensor:
        """texts: List[str] → (B, text_feat_dim)"""
        # 把文本 hash 成 vocab_size 以内的 index，简单但可分辨
        indices = []
        for t in texts:
            h = int(hashlib.md5(t.encode()).hexdigest(), 16) % self.vocab_size
            indices.append(h)
        idx = torch.tensor(indices, device=self.emb.weight.device)
        return self.emb(idx)  # (B, text_feat_dim)


# ---------------------------------------------------------------------------
# ToyVLA
# ---------------------------------------------------------------------------
@FRAMEWORK_REGISTRY.register("ToyVLA")
class ToyVLA(PreTrainedModel):
    """
    极简 VLA 调试模型。
    - 用 TinyImageEncoder + TinyTextEncoder 代替 Qwen VLM
    - 用小 MLP 做动作回归
    - 整体 < 200K 参数，单卡几秒即可 overfit 小 batch
    """

    config_class = PretrainedConfig

    def __init__(self, config=None, **kwargs):
        super().__init__(PretrainedConfig())
        self.toy_config = config  # OmegaConf

        # 从 config 中读取超参（提供合理默认值）
        am_cfg = config.framework.action_model if hasattr(config, "framework") else None
        self.action_dim      = getattr(am_cfg, "action_dim", 7)           if am_cfg else 7
        self.future_window   = getattr(am_cfg, "future_action_window_size", 15) if am_cfg else 15
        self.past_window     = getattr(am_cfg, "past_action_window_size",  0)   if am_cfg else 0
        self.chunk_len       = self.past_window + 1 + self.future_window

        img_feat_dim  = 128
        text_feat_dim = 64
        fuse_dim      = img_feat_dim + text_feat_dim  # 192

        self.img_encoder  = TinyImageEncoder(img_feat_dim)
        self.text_encoder = TinyTextEncoder(text_feat_dim=text_feat_dim)

        # 动作预测头: fuse_dim → (chunk_len * action_dim)
        self.action_head = nn.Sequential(
            nn.Linear(fuse_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.chunk_len * self.action_dim),
        )
        self.l1_loss = nn.L1Loss()

        logger.info(
            f"[ToyVLA] Built: action_dim={self.action_dim}, chunk_len={self.chunk_len}, "
            f"params={sum(p.numel() for p in self.parameters())/1e3:.1f}K"
        )

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _encode_batch(self, examples: List[dict]) -> torch.Tensor:
        """examples → fused feature (B, fuse_dim)"""
        device = next(self.parameters()).device
        dtype  = next(self.parameters()).dtype

        # ---- 图像编码 ----
        img_feats_list = []
        for ex in examples:
            imgs = ex["image"]  # List[PIL.Image]
            view_feats = []
            for pil_img in imgs:
                pil_img = pil_img.convert("RGB").resize((224, 224))
                t = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float() / 255.0  # (3,224,224)
                view_feats.append(t)
            # 多视角取平均
            view_tensor = torch.stack(view_feats, 0).mean(0).unsqueeze(0)  # (1,3,224,224)
            img_feats_list.append(view_tensor)

        imgs_batch = torch.cat(img_feats_list, 0).to(device)     # (B,3,224,224)
        img_feat   = self.img_encoder(imgs_batch).to(dtype)       # (B, img_feat_dim)

        # ---- 文本编码 ----
        langs      = [ex["lang"] for ex in examples]
        text_feat  = self.text_encoder(langs).to(dtype)           # (B, text_feat_dim)

        # ---- 融合 ----
        fused = torch.cat([img_feat, text_feat], dim=-1)          # (B, fuse_dim)
        return fused

    # ------------------------------------------------------------------
    # 训练前向
    # ------------------------------------------------------------------
    def forward(self, examples: List[dict], **kwargs) -> dict:
        """
        Returns:
            {"action_loss": scalar tensor}
        """
        fused = self._encode_batch(examples)                      # (B, fuse_dim)
        raw   = self.action_head(fused)                           # (B, chunk_len*action_dim)
        pred  = raw.view(-1, self.chunk_len, self.action_dim)     # (B, chunk_len, action_dim)

        # 标签
        actions = np.array([ex["action"] for ex in examples])    # (B, T, D)
        tgt = torch.tensor(actions, device=pred.device, dtype=pred.dtype)
        # 取最后 chunk_len 步
        tgt = tgt[:, -self.chunk_len:, :]                         # (B, chunk_len, action_dim)

        loss = self.l1_loss(pred, tgt)
        return {"action_loss": loss}

    # ------------------------------------------------------------------
    # 推理前向
    # ------------------------------------------------------------------
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        """
        Returns:
            {"normalized_actions": np.ndarray (B, chunk_len, action_dim)}
        """
        with torch.no_grad():
            fused = self._encode_batch(examples)
            raw   = self.action_head(fused)
            pred  = raw.view(-1, self.chunk_len, self.action_dim)
        return {"normalized_actions": pred.cpu().float().numpy()}
