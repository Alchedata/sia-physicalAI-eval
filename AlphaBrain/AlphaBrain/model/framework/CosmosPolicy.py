# Copyright 2025 VLA-Engine. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Cosmos-Policy Framework — Video diffusion model for robot policy learning.
# Ported from nvidia/cosmos-policy, adapted for VLA-Engine-Developer.

"""
CosmosPolicy Framework

A video diffusion model (Cosmos Predict2 2B DiT) fine-tuned for robot policy prediction.
Unlike VLM-based frameworks (QwenOFT, QwenGR00T), this uses latent-space diffusion:
  - WAN 2.1 VAE encodes images to latent space (frozen)
  - MiniTrainDIT backbone denoises latent sequences (trainable)
  - Actions/proprio/value are injected into latent frames
  - T5 text embeddings provide language conditioning (precomputed)

Latent frame layout (LIBERO, state_t=9):
  [blank, curr_proprio, curr_wrist, curr_primary, action,
   future_proprio, future_wrist, future_primary, value]
"""

import os
import pickle
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from AlphaBrain.model.framework.base_framework import BaseFramework
from AlphaBrain.model.tools import FRAMEWORK_REGISTRY
from AlphaBrain.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


@FRAMEWORK_REGISTRY.register("CosmosPolicy")
class CosmosPolicy(BaseFramework):
    """
    Cosmos-Policy: latent-space video diffusion for robot action prediction.

    Training: VAE encode → inject action/proprio → diffusion loss on latent sequence
    Inference: multi-step denoising → extract action from latent frame
    """

    def __init__(self, config=None, **kwargs):
        super().__init__()
        self.config = config
        cp_cfg = config.framework.cosmos_policy

        # --- Action / sequence config ---
        self.action_dim = cp_cfg.action_dim          # 7 (LIBERO)
        self.chunk_size = cp_cfg.chunk_size           # 16 (LIBERO)
        self.proprio_dim = cp_cfg.proprio_dim         # 9 (LIBERO)
        self.state_t = cp_cfg.state_t                 # 9 (LIBERO)

        # Latent frame indices
        self.blank_idx = 0
        self.curr_proprio_idx = 1
        self.curr_wrist_idx = 2
        self.curr_primary_idx = 3
        self.action_idx = 4
        self.future_proprio_idx = 5
        self.future_wrist_idx = 6
        self.future_primary_idx = 7
        self.value_idx = 8

        self.condition_frame_indices = [0, 1, 2, 3]
        self.prediction_frame_indices = [4, 5, 6, 7, 8]

        # Loss config
        loss_cfg = getattr(cp_cfg, 'loss', None)
        self.action_loss_multiplier = getattr(loss_cfg, 'action_loss_multiplier', 1.0) if loss_cfg else 1.0
        self.world_model_loss_weight = getattr(loss_cfg, 'world_model_loss_weight', 1.0) if loss_cfg else 1.0
        self.value_loss_weight = getattr(loss_cfg, 'value_loss_weight', 0.0) if loss_cfg else 0.0
        self.loss_scale = getattr(loss_cfg, 'loss_scale', 10.0) if loss_cfg else 10.0
        self.sigma_data = getattr(loss_cfg, 'sigma_data', 1.0) if loss_cfg else 1.0
        self.adjust_video_noise = getattr(loss_cfg, 'adjust_video_noise', True) if loss_cfg else True

        # --- 1. VAE Tokenizer (frozen) ---
        # Force deterministic cuDNN for cross-process reproducibility
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

        # Use VLA-Engine own WanVAE (loads same .pth weights, no cosmos_policy dependency).
        from AlphaBrain.model.modules.world_model.cosmos.wan_vae import WanVAEWrapper
        pretrained_dir = cp_cfg.checkpoint.pretrained_dir
        if pretrained_dir is None:
            pretrained_dir = os.path.join(
                os.environ.get('PRETRAINED_MODELS_DIR', 'data/pretrained_models'),
                'Cosmos-Predict2-2B-Video2World'
            )
        self.vae = WanVAEWrapper(
            pretrained_dir=pretrained_dir,
            dtype=torch.bfloat16,
            device="cpu",
            temporal_window=16,
        )

        # --- 2. DIT Backbone (trainable) ---
        # Use VLA own MinimalV1LVGDiT with TE backend (no cosmos_policy dependency).
        # It inherits MiniTrainDIT and adds condition_video_input_mask concatenation
        # in forward(), plus timestep_scale support.
        from AlphaBrain.model.modules.world_model.cosmos.official_dit import MinimalV1LVGDiT
        dit_cfg = cp_cfg.dit
        self.dit = MinimalV1LVGDiT(
            max_img_h=getattr(dit_cfg, 'max_img_h', 240),
            max_img_w=getattr(dit_cfg, 'max_img_w', 240),
            max_frames=getattr(dit_cfg, 'max_frames', 128),
            in_channels=16,   # state_ch=16; MinimalV1LVGDiT adds +1 for condition mask
            out_channels=16,
            patch_spatial=getattr(dit_cfg, 'patch_spatial', 2),
            patch_temporal=getattr(dit_cfg, 'patch_temporal', 1),
            concat_padding_mask=True,
            model_channels=getattr(dit_cfg, 'model_channels', 2048),
            num_blocks=getattr(dit_cfg, 'num_blocks', 28),
            num_heads=getattr(dit_cfg, 'num_heads', 16),
            crossattn_emb_channels=getattr(dit_cfg, 'crossattn_emb_channels', 1024),
            use_crossattn_projection=getattr(dit_cfg, 'use_crossattn_projection', False),
            atten_backend="minimal_a2a",  # Official attention backend (local copy, no cosmos_policy dep)
            pos_emb_cls=getattr(dit_cfg, 'pos_emb_cls', "rope3d"),
            mlp_ratio=getattr(dit_cfg, 'mlp_ratio', 4.0),
            use_adaln_lora=True,
            adaln_lora_dim=256,
            use_wan_fp32_strategy=getattr(dit_cfg, 'use_wan_fp32_strategy', False),
            pos_emb_learnable=getattr(dit_cfg, 'pos_emb_learnable', True),
            rope_enable_fps_modulation=getattr(dit_cfg, 'rope_enable_fps_modulation', False),
            rope_h_extrapolation_ratio=getattr(dit_cfg, 'rope_h_extrapolation_ratio', 3.0),
            rope_w_extrapolation_ratio=getattr(dit_cfg, 'rope_w_extrapolation_ratio', 3.0),
        )

        # --- 3. SDE (noise schedule) ---
        from AlphaBrain.model.modules.world_model.cosmos.hybrid_edm_sde import HybridEDMSDE
        sde_cfg = cp_cfg.sde
        self.sde = HybridEDMSDE(
            p_mean=getattr(sde_cfg, 'p_mean', 1.3862943611198906),
            p_std=getattr(sde_cfg, 'p_std', 1.2),
            sigma_max=getattr(sde_cfg, 'sigma_max', 200.0),
            sigma_min=getattr(sde_cfg, 'sigma_min', 0.01),
            hybrid_sigma_distribution=True,
        )

        # --- 4. Sampler (inference) ---
        from AlphaBrain.model.modules.world_model.cosmos.cosmos_sampler import CosmosPolicySampler
        self.sampler = CosmosPolicySampler()

        # Inference config
        inf_cfg = getattr(cp_cfg, 'inference', None)
        self.action_num_steps = getattr(inf_cfg, 'action_num_steps', 5) if inf_cfg else 5
        self.inference_sigma_min = getattr(inf_cfg, 'sigma_min', 4.0) if inf_cfg else 4.0
        self.inference_sigma_max = getattr(inf_cfg, 'sigma_max', 80.0) if inf_cfg else 80.0

        # --- 5. T5 embeddings (precomputed) ---
        self.t5_embeddings = None  # loaded lazily
        self.t5_embeddings_path = getattr(cp_cfg, 't5_embeddings_path', None)

        # --- 6. Conditioning ---
        self.sigma_conditional = getattr(cp_cfg, 'sigma_conditional', 0.0)

        # Load pretrained DIT weights
        ckpt_path = getattr(cp_cfg.checkpoint, 'load_path', None)
        if ckpt_path is None:
            ckpt_path = os.path.join(pretrained_dir, 'model-480p-16fps.pt')
        if ckpt_path and os.path.exists(ckpt_path):
            self._load_dit_checkpoint(ckpt_path)

        # Convert DIT to bf16, matching original's on_train_start():
        #   self.net = self.net.to(**self.tensor_kwargs)
        # where tensor_kwargs = {"device": "cuda", "dtype": bfloat16}.
        # The original always runs DiT in bf16 for both training and inference.
        # We convert dtype here (device placement handled by trainer or server).
        self.dit = self.dit.to(dtype=torch.bfloat16)

        logger.info(f"[CosmosPolicy] Initialized: state_t={self.state_t}, "
                     f"action_dim={self.action_dim}, chunk_size={self.chunk_size}")

    # ----------------------------------------------------------------
    # Rectified Flow Scaling (confirmed at runtime: model.scaling = RectifiedFlowScaling,
    # config.scaling = "rectified_flow", sigma_data = 1.0)
    # ----------------------------------------------------------------
    def _edm_scaling(self, sigma):
        """
        Compute RectifiedFlow preconditioning coefficients.

        Matches cosmos-policy's RectifiedFlowScaling class (denoiser_scaling.py)
        with sigma_data=1.0.

        Args:
            sigma: noise level tensor, any shape (will broadcast)

        Returns:
            c_skip, c_out, c_in, c_noise
        """
        t = sigma / (sigma + 1)
        c_skip = 1.0 - t
        c_out = -t
        c_in = 1.0 - t
        c_noise = t  # t_scaling_factor=1.0
        return c_skip, c_out, c_in, c_noise

    def _denoise(self, xt, sigma, crossattn_emb, fps, padding_mask,
                 gt_frames=None, n_cond_frames=0):
        """
        Denoise with RectifiedFlow preconditioning, matching official cosmos-policy
        (policy_video2world_model.py denoise()):
          1. Scale all frames by c_in
          2. Replace condition frames with gt_frames / sigma_data
          3. Use sigma_conditional c_noise for condition frames
          4. DIT forward
          5. Combine: c_skip * xt + c_out * net_output
          6. Replace condition frames in output with gt_frames

        Args:
            xt: (B, C, T, H, W) noisy latent
            sigma: (B,) noise level per sample
            crossattn_emb: (B, seq_len, dim) T5 embeddings
            fps: (B,) fps tensor
            padding_mask: (B, 1, H_pixel, W_pixel) spatial padding mask (pixel space)
            gt_frames: optional (B, C, T_cond, H, W) clean condition frames
            n_cond_frames: number of condition frames

        Returns:
            x0_pred: (B, C_out, T, H, W) denoised prediction
        """
        B = xt.shape[0]
        T = xt.shape[2]
        device = xt.device

        # EDM preconditioning coefficients — compute in float32 for precision.
        # Original cosmos-policy keeps x_sigma_max in float32, so the sampler passes
        # sigma as float32 to denoise(). We ensure the same by upcasting here.
        sigma_f32 = sigma.float()
        sigma_5d = sigma_f32.view(B, 1, 1, 1, 1)
        c_skip, c_out, c_in, c_noise = self._edm_scaling(sigma_5d)

        # Scale all frames by c_in (float32 precision for preconditioning)
        net_input = xt.float() * c_in

        # Build per-frame c_noise: (B, T)
        c_noise_per_frame = c_noise.view(B, 1).expand(B, T).clone()

        if gt_frames is not None and n_cond_frames > 0:
            # Official pattern: condition frames use gt / sigma_data as input
            # (not gt * c_in — this is the key difference)
            cond_input = gt_frames.float() / self.sigma_data
            net_input[:, :, :n_cond_frames, :, :] = cond_input

            # Condition frames use c_noise derived from sigma_conditional
            # RectifiedFlowScaling: c_noise = t = sigma / (sigma + 1)
            # For sigma_conditional=0.0: c_noise = 0.0 (clean, no NaN)
            sigma_cond = torch.tensor(self.sigma_conditional, device=device, dtype=torch.float32)
            c_noise_cond = sigma_cond / (sigma_cond + 1.0)
            c_noise_per_frame[:, :n_cond_frames] = c_noise_cond

        # Build condition_video_input_mask: (B, 1, T, H, W)
        # MinimalV1LVGDiT.forward will cat this to the input internally
        cond_mask = torch.zeros(B, 1, T, net_input.shape[3], net_input.shape[4],
                                device=device, dtype=net_input.dtype)
        if n_cond_frames > 0:
            cond_mask[:, :, :n_cond_frames, :, :] = 1.0

        # DIT forward — MinimalV1LVGDiT expects 16ch input and cats condition_mask internally.
        # Original runs DIT in bf16 (model.to(tensor_kwargs) converts all params to bf16).
        # Timestep dtype depends on use_wan_fp32_strategy:
        #   False (LIBERO default): timesteps in bf16, matching tensor_kwargs
        #   True: timesteps in float32 for higher precision AdaLN modulation
        net_input_bf16 = net_input.to(torch.bfloat16)
        cond_mask_bf16 = cond_mask.to(torch.bfloat16)
        ts_dtype = torch.float32 if self.dit.use_wan_fp32_strategy else torch.bfloat16
        net_output = self.dit(
            x_B_C_T_H_W=net_input_bf16,
            timesteps_B_T=c_noise_per_frame.to(ts_dtype),
            crossattn_emb=crossattn_emb.to(torch.bfloat16),
            condition_video_input_mask_B_C_T_H_W=cond_mask_bf16,
            fps=fps.to(torch.bfloat16),
            padding_mask=padding_mask.to(torch.bfloat16),
        )

        # Combine: x0_pred = c_skip * xt + c_out * net_output (in float32)
        x0_pred = c_skip * xt.float() + c_out * net_output.float()

        # Replace condition frames in output with clean gt
        if gt_frames is not None and n_cond_frames > 0:
            x0_pred[:, :, :n_cond_frames, :, :] = gt_frames

        return x0_pred

    # ----------------------------------------------------------------
    # Training forward
    # ----------------------------------------------------------------
    def forward(self, examples, **kwargs):
        """
        Training forward pass: diffusion denoising loss on full 9-frame latent sequence.

        Args:
            examples: batched dict from DataLoader (each value is a stacked tensor),
                      or list of dicts (legacy).

        Returns:
            {"action_loss": total_loss}
        """
        from AlphaBrain.model.modules.world_model.cosmos.latent_utils import (
            replace_latent_with_action_chunk,
            replace_latent_with_proprio,
        )

        device = next(self.dit.parameters()).device

        # Handle both batched dict (from DataLoader) and list of dicts (legacy)
        if isinstance(examples, dict):
            batch = examples
        else:
            batch = {k: torch.stack([ex[k] for ex in examples]) for k in examples[0]}

        B = batch["video"].shape[0]

        # --- Step 1: Gather batch data ---
        videos = batch["video"].to(device)                          # (B, 3, 33, 224, 224) uint8
        actions = batch["actions"].to(device)                       # (B, chunk_size, action_dim)
        proprios = batch["proprio"].to(device)                      # (B, proprio_dim)
        future_proprios = batch["future_proprio"].to(device)
        t5_embs = batch["t5_text_embeddings"].to(device)            # (B, 512, 1024)
        values = batch.get("value_function_return", torch.zeros(B)).to(device=device, dtype=torch.float32)

        # Masks for loss computation
        rollout_masks = batch.get("rollout_data_mask", torch.zeros(B)).to(device=device, dtype=torch.float32)
        wm_masks = batch.get("world_model_sample_mask", torch.zeros(B)).to(device=device, dtype=torch.float32)
        vf_masks = batch.get("value_function_sample_mask", torch.zeros(B)).to(device=device, dtype=torch.float32)

        # --- Step 2: VAE encode → latent ---
        with torch.no_grad():
            # Normalize uint8 [0, 255] → float [-1, 1]
            # IMPORTANT: Normalize in bf16 to match original cosmos-policy's
            # _normalize_video_databatch_inplace: video.to(dtype=bf16) / 127.5 - 1.0
            video_norm = videos.to(dtype=torch.bfloat16) / 127.5 - 1.0
            # Match official pipeline memory layout (channels_last_3d)
            video_norm = video_norm.to(memory_format=torch.channels_last_3d)
            x0 = self.vae.encode(video_norm)  # (B, 16, 9, 28, 28)

        # Apply sigma_data scaling (matches official: encode(x) * sigma_data)
        # The VAE already does per-channel mean/std normalization internally;
        # this additional scaling aligns the latent magnitude with EDM preconditioning.
        x0 = x0 * self.sigma_data
        x0 = x0.to(dtype=torch.bfloat16)

        # --- Step 3: Inject action/proprio/value into latent frames ---
        action_indices = torch.full((B,), self.action_idx, device=device, dtype=torch.long)
        curr_proprio_indices = torch.full((B,), self.curr_proprio_idx, device=device, dtype=torch.long)
        future_proprio_indices = torch.full((B,), self.future_proprio_idx, device=device, dtype=torch.long)

        x0 = replace_latent_with_action_chunk(
            x0, actions.to(x0.dtype), action_indices
        )
        x0 = replace_latent_with_proprio(
            x0, proprios.to(x0.dtype), curr_proprio_indices
        )
        x0 = replace_latent_with_proprio(
            x0, future_proprios.to(x0.dtype), future_proprio_indices
        )

        # Value: expand scalar to fill latent volume at value_idx
        value_flat = values.to(x0.dtype).view(B, 1, 1, 1).expand(B, 16, 28, 28)
        x0[:, :, self.value_idx, :, :] = value_flat

        # --- Step 4: Save clean condition frames for denoise ---
        # Official pattern: noise ALL frames, then replace condition frames inside denoise
        n_cond = len(self.condition_frame_indices)
        gt_frames = x0[:, :, :n_cond, :, :].clone()

        # --- Step 5: Sample sigma and noise ---
        sigma = self.sde.sample_t(B, device=device).to(x0.dtype)

        # Video noise multiplier: sigma *= sqrt(state_t) when adjust_video_noise=True
        # Matches original text2world_model.py behavior (state_t=9 → sigma *= 3.0)
        if self.adjust_video_noise:
            sigma = sigma * (self.state_t ** 0.5)

        epsilon = torch.randn_like(x0)

        # Apply noise to ALL frames (official pattern: marginal_prob returns x0, sigma)
        # xt = x0 + sigma * epsilon
        sigma_5d = sigma.view(B, 1, 1, 1, 1)
        xt = x0 + sigma_5d * epsilon

        # --- Step 6: Denoise with EDM preconditioning ---
        crossattn_emb = t5_embs.to(dtype=torch.bfloat16)
        # padding_mask in pixel space (224x224), matching original
        padding_mask = torch.zeros(B, 1, 224, 224, device=device, dtype=xt.dtype)
        fps = torch.tensor([16] * B, device=device, dtype=torch.float32)  # Must match original (always 16)

        # EDM preconditioning with official condition frame handling:
        # - condition frames replaced with gt / sigma_data in network input
        # - condition frames use sigma_conditional c_noise
        # - output condition frames replaced with clean gt
        n_cond = len(self.condition_frame_indices)
        x0_pred = self._denoise(
            xt, sigma, crossattn_emb, fps, padding_mask,
            gt_frames=gt_frames, n_cond_frames=n_cond
        )

        # --- Step 7: Compute loss ---
        # RectifiedFlow loss weight: (1 + sigma)^2 / sigma^2
        loss_weight = (1 + sigma) ** 2 / sigma ** 2
        loss_weight = loss_weight.view(B, 1, 1, 1, 1)

        # MSE between prediction and ground truth
        pred_mse = (x0_pred - x0) ** 2  # (B, C, T, H, W)

        # Official LIBERO: no per-sample loss masking (all mask flags default False).
        # All 9 latent frames contribute equally to loss.
        # Condition frames contribute ~0 since x0_pred == gt (replaced in denoise).
        edm_loss = pred_mse * loss_weight
        total_loss = edm_loss.mean() * self.loss_scale

        # Log per-component losses for monitoring (no gradient)
        with torch.no_grad():
            action_mse = pred_mse[:, :, self.action_idx, :, :].mean()
            cond_mse = pred_mse[:, :, :len(self.condition_frame_indices), :, :].mean()

        return {"action_loss": total_loss, "action_mse": action_mse, "cond_mse": cond_mse}

    # ----------------------------------------------------------------
    # Inference
    # ----------------------------------------------------------------
    @torch.inference_mode()
    def predict_action(self, examples=None, batch_images=None,
                       instructions=None, **kwargs):
        """
        Inference: multi-step denoising to predict action chunk.

        Matches original cosmos-policy get_action() flow:
        1. Build full 33-frame video (with placeholders for prediction frames)
        2. VAE encode → 9 latent frames
        3. Inject normalized proprio into frame 1
        4. Save condition frames (0-3), replace prediction frames (4-8) with noise
        5. Multi-step denoising
        6. Extract action from denoised latent at action_idx

        Args:
            examples: list of dicts with image, wrist_image, lang, proprio
            batch_images: alternative — list of PIL images
            instructions: alternative — list of strings

        Returns:
            {"normalized_actions": np.ndarray of shape (B, chunk_size, action_dim)}
        """
        from AlphaBrain.model.modules.world_model.cosmos.latent_utils import (
            replace_latent_with_proprio,
        )

        device = next(self.dit.parameters()).device

        # Parse inputs
        if examples is not None:
            primary_images = [ex["image"] for ex in examples]
            wrist_images = [ex.get("wrist_image") for ex in examples]
            instructions = [ex["lang"] for ex in examples]
            proprios = [ex["proprio"] for ex in examples]
        else:
            primary_images = batch_images
            wrist_images = kwargs.get("wrist_images", [None] * len(primary_images))
            proprios = kwargs.get("proprios", [np.zeros(self.proprio_dim)] * len(primary_images))

        B = len(primary_images)

        # Preprocess images to match official cosmos-policy inference pipeline:
        # JPEG compression (quality=95) + 90% center crop + resize to 224x224
        primary_images = [self._preprocess_image(img) for img in primary_images]
        wrist_images = [self._preprocess_image(img) if img is not None else None for img in wrist_images]

        # --- Step 1: Build full 33-frame video (matching original get_action) ---
        full_video = self._build_full_video(primary_images, wrist_images)  # (B, 3, 33, 224, 224)
        full_video = full_video.to(device)

        # --- Step 2: VAE encode full video → 9 latent frames ---
        # IMPORTANT: Normalize in bf16 to match original cosmos-policy's
        # _normalize_video_databatch_inplace: video.to(dtype=bf16) / 127.5 - 1.0
        # Normalizing in float32 then converting to bf16 gives different rounding.
        video_norm = full_video.to(dtype=torch.bfloat16) / 127.5 - 1.0
        # Match official pipeline memory layout (channels_last_3d)
        video_norm = video_norm.to(memory_format=torch.channels_last_3d)

        x0 = self.vae.encode(video_norm)  # (B, 16, 9, 28, 28)
        x0 = x0 * self.sigma_data  # Apply sigma_data scaling
        x0 = x0.to(torch.bfloat16)

        # --- Step 3: Inject normalized proprio into latent frame 1 ---
        proprio_array = np.array(proprios)
        # Normalize proprio to [-1, 1] using dataset stats (matches original)
        proprio_array = self._normalize_proprio(proprio_array)
        proprio_tensor = torch.tensor(proprio_array, device=device, dtype=x0.dtype)
        x0 = replace_latent_with_proprio(
            x0, proprio_tensor,
            torch.full((B,), self.curr_proprio_idx, device=device, dtype=torch.long)
        )

        # --- Step 4: Save condition frames, generate full-frame noise ---
        n_cond = len(self.condition_frame_indices)
        gt_frames = x0[:, :, :n_cond, :, :].clone()  # (B, 16, 4, 28, 28)

        B_lat, C, T_full, H_lat, W_lat = x0.shape

        # Generate noise for ALL frames (matching original: arch_invariant_rand over full state_shape)
        # The sampler receives x_sigma_max with noise on all frames;
        # _denoise() replaces condition frames with gt internally at each step.
        # IMPORTANT: Keep x_sigma_max in float32 (NOT bf16) to match original cosmos-policy.
        # The sampler uses x_sigma_max.dtype as in_dtype for sigma precision, and
        # float32 sigma is needed for accurate EDM preconditioning (c_in, c_noise).
        from AlphaBrain.model.modules.world_model.cosmos.noise_utils import arch_invariant_rand
        x_sigma_max = arch_invariant_rand(
            (B_lat, C, T_full, H_lat, W_lat), seed=1
        ).to(device=device, dtype=torch.float32) * self.inference_sigma_max

        # Get T5 embeddings
        crossattn_emb = self._get_t5_embeddings_for_inference(instructions, device)

        # --- Step 5: Multi-step denoising ---
        # padding_mask in pixel space (224x224), matching original cosmos_utils.py
        padding_mask = torch.zeros(B, 1, 224, 224, device=device, dtype=x0.dtype)
        fps = torch.tensor([16] * B, device=device, dtype=torch.float32)

        def x0_fn(xt, sigma):
            return self._denoise(
                xt, sigma, crossattn_emb, fps, padding_mask,
                gt_frames=gt_frames, n_cond_frames=n_cond
            )
        
        denoised = self.sampler.forward(
            x0_fn=x0_fn,
            x_sigma_max=x_sigma_max,  # full 9-frame noise (matching original)
            num_steps=self.action_num_steps,
            sigma_min=self.inference_sigma_min,
            sigma_max=self.inference_sigma_max,
        )
        # --- Step 6: Extract action from denoised latent ---
        action_latent = denoised[:, :, self.action_idx, :, :]  # (B, 16, 28, 28)
        action_chunk = self._extract_action_from_latent(action_latent)  # (B, chunk_size, action_dim)

        return {
            "normalized_actions": action_chunk.float().cpu().numpy(),
            "denoised_latent": denoised,
            "x0_latent": x0,
        }

    # ----------------------------------------------------------------
    # Helper methods
    # ----------------------------------------------------------------
    def _build_full_video(self, primary_images, wrist_images):
        """
        Build full 33-frame video tensor for inference, matching original cosmos-policy.

        Layout (each latent frame = 4 pixel frames, except blank = 1):
          [blank(1), proprio_placeholder(4), wrist(4), primary(4),
           action_blank(4), future_proprio_blank(4), future_wrist(4),
           future_primary(4), value_blank(4)]
        Total: 1 + 4*8 = 33 frames → 9 latent frames after VAE

        NOTE: cosmos-policy's get_action() calls prepare_images_for_model()
        with flip_images=False (the default). Despite cfg.flip_images=True being
        set in PolicyEvalConfig, it is NOT passed to prepare_images_for_model().
        So no image order swap occurs: wrist slot gets wrist pixels, primary slot
        gets primary pixels. No swap needed here.

        Returns: (B, 3, 33, 224, 224) uint8 tensor
        """

        B = len(primary_images)
        H = W = 224
        DUP = 4  # temporal duplication factor

        frames_list = []
        for i in range(B):
            frames = []
            blank = torch.zeros(3, H, W, dtype=torch.uint8)

            # Wrist image tensor
            if wrist_images[i] is not None:
                wrist = self._pil_to_tensor(wrist_images[i], H, W)
            else:
                wrist = blank.clone()

            # Primary image tensor
            primary = self._pil_to_tensor(primary_images[i], H, W)

            # Frame 0: blank (1 copy)
            frames.append(blank.unsqueeze(1))                              # (3,1,H,W)
            # Frame 1: proprio placeholder blank (4 copies)
            frames.append(blank.unsqueeze(1).expand(-1, DUP, -1, -1))     # (3,4,H,W)
            # Frame 2: current wrist image (4 copies)
            frames.append(wrist.unsqueeze(1).expand(-1, DUP, -1, -1))     # (3,4,H,W)
            # Frame 3: current primary image (4 copies)
            frames.append(primary.unsqueeze(1).expand(-1, DUP, -1, -1))   # (3,4,H,W)
            # Frame 4: action placeholder blank (4 copies)
            frames.append(blank.unsqueeze(1).expand(-1, DUP, -1, -1))     # (3,4,H,W)
            # Frame 5: future proprio placeholder blank (4 copies)
            frames.append(blank.unsqueeze(1).expand(-1, DUP, -1, -1))     # (3,4,H,W)
            # Frame 6: future wrist = copy of current wrist (4 copies)
            frames.append(wrist.unsqueeze(1).expand(-1, DUP, -1, -1))     # (3,4,H,W)
            # Frame 7: future primary = copy of current primary (4 copies)
            frames.append(primary.unsqueeze(1).expand(-1, DUP, -1, -1))   # (3,4,H,W)
            # Frame 8: value placeholder blank (4 copies)
            frames.append(blank.unsqueeze(1).expand(-1, DUP, -1, -1))     # (3,4,H,W)

            video = torch.cat(frames, dim=1)  # (3, 33, H, W)
            frames_list.append(video)

        return torch.stack(frames_list)  # (B, 3, 33, H, W)

    def _pil_to_tensor(self, img, H, W):
        """Convert PIL image to (3, H, W) uint8 tensor, resized."""
        from PIL import Image
        if isinstance(img, Image.Image):
            img = img.resize((W, H), Image.BILINEAR)
            return torch.from_numpy(np.array(img)).permute(2, 0, 1)  # (3, H, W)
        elif isinstance(img, torch.Tensor):
            if img.shape[-2:] != (H, W):
                img = torch.nn.functional.interpolate(
                    img.unsqueeze(0).float(), size=(H, W), mode='bilinear'
                ).squeeze(0).to(torch.uint8)
            return img
        elif isinstance(img, np.ndarray):
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(img).resize((W, H), PILImage.BILINEAR)
            return torch.from_numpy(np.array(pil_img)).permute(2, 0, 1)
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

    def _preprocess_image(self, img):
        """
        Preprocess image to match official cosmos-policy inference pipeline.

        Original flow (cosmos_utils.py prepare_images_for_model + apply_image_transforms):
        1. JPEG compression (quality=95)
        2. PIL resize to 224x224 (PIL default resampling)
        3. 90% area center crop on uint8 tensor + resize back to 224x224

        IMPORTANT: The original keeps images as uint8 throughout the center_crop
        and resize operations (torch.from_numpy → uint8 tensor → F.center_crop →
        F.resize → numpy uint8). Using float intermediates (F_tv.to_tensor) causes
        1-pixel rounding differences that compound through VAE + 5 denoising steps.

        Args:
            img: np.ndarray (H, W, 3) uint8, PIL Image, or torch.Tensor

        Returns:
            np.ndarray (H=224, W=224, 3) uint8
        """
        from PIL import Image
        import io
        import torchvision.transforms.functional as F_tv

        # Convert to numpy uint8 if needed
        if isinstance(img, Image.Image):
            img = np.array(img)
        elif isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
            if img.shape[0] == 3:  # (C, H, W) -> (H, W, C)
                img = img.transpose(1, 2, 0)
        img = img.astype(np.uint8)

        # Step 1: JPEG compression (quality=95)
        pil_img = Image.fromarray(img)
        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG', quality=95)
        buf.seek(0)
        pil_img = Image.open(buf)

        # Step 2: PIL resize to target size (matches original resize_images)
        H_target, W_target = 224, 224
        pil_img = pil_img.resize((W_target, H_target))

        # Step 3: Center crop + resize on uint8 tensor (matches original apply_image_transforms)
        # Original: torch.from_numpy(images).permute(0,3,1,2) → uint8 tensor
        img_np = np.array(pil_img)  # (H, W, 3) uint8
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # (3, H, W) uint8
        crop_size = int(H_target * 0.9 ** 0.5)  # sqrt(0.9)*224 ≈ 212
        img_tensor = F_tv.center_crop(img_tensor, crop_size)
        img_tensor = F_tv.resize(img_tensor, [H_target, W_target], antialias=True)

        # Convert back to numpy uint8 (H, W, 3)
        img = img_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
        return img

    def _extract_action_from_latent(self, action_latent):
        """
        Extract action chunk from latent frame, averaging over all chunks
        that fit in the latent volume (matches official cosmos-policy).

        action_latent: (B, C=16, H=28, W=28) → (B, chunk_size, action_dim)
        """
        B = action_latent.shape[0]
        flat = action_latent.reshape(B, -1)
        num_latent_elements = flat.shape[1]  # 16 * 28 * 28 = 12544
        num_action_elements = self.chunk_size * self.action_dim  # e.g. 16 * 7 = 112
        num_chunks = num_latent_elements // num_action_elements

        # Reshape to (B, num_chunks, chunk_size * action_dim), then to (B, num_chunks, chunk_size, action_dim)
        all_chunks = flat[:, :num_chunks * num_action_elements].reshape(
            B, num_chunks, self.chunk_size, self.action_dim
        )
        # Average over all chunks (matches official extract_action_chunk_from_latent_sequence)
        return torch.mean(all_chunks, dim=1)

    def _get_t5_embeddings_for_inference(self, instructions, device):
        """Load precomputed T5 embeddings for given instructions."""
        if self.t5_embeddings is None and self.t5_embeddings_path:
            with open(self.t5_embeddings_path, 'rb') as f:
                self.t5_embeddings = pickle.load(f)
            logger.info(f"[CosmosPolicy] Loaded T5 embeddings: {len(self.t5_embeddings)} entries")

        B = len(instructions)
        embs = []
        for inst in instructions:
            if self.t5_embeddings and inst in self.t5_embeddings:
                emb = self.t5_embeddings[inst]
                if emb.ndim == 3:
                    emb = emb.squeeze(0)  # (512, 1024)
            else:
                # Fallback: zero embedding
                logger.warning(f"[CosmosPolicy] T5 embedding not found for: {inst[:50]}...")
                emb = torch.zeros(512, 1024)
            embs.append(emb)

        return torch.stack(embs).to(device=device, dtype=torch.bfloat16)

    def set_dataset_stats(self, dataset_stats: dict):
        """
        Store dataset statistics for proprio normalization during inference.

        Args:
            dataset_stats: dict with keys 'proprio_min', 'proprio_max' (np.ndarray).
        """
        self._dataset_stats = dataset_stats
        logger.info("[CosmosPolicy] Dataset stats set for proprio normalization")

    def _normalize_proprio(self, proprio: np.ndarray) -> np.ndarray:
        """
        Normalize proprio to [-1, 1] using dataset stats (matches original rescale_proprio).

        Formula: x_norm = 2 * (x - min) / (max - min) - 1

        Args:
            proprio: (B, proprio_dim) or (proprio_dim,) raw proprio values.

        Returns:
            Normalized proprio, same shape.
        """
        stats = getattr(self, '_dataset_stats', None)
        if stats is None or 'proprio_min' not in stats:
            logger.warning("[CosmosPolicy] No dataset stats for proprio normalization, using raw values")
            return proprio

        p_min = stats["proprio_min"]
        p_max = stats["proprio_max"]
        denom = p_max - p_min
        denom = np.where(np.abs(denom) < 1e-8, 1.0, denom)  # avoid division by zero
        return 2.0 * (proprio - p_min) / denom - 1.0

    def _load_dit_checkpoint(self, ckpt_path: str):
        """Load DIT weights from official (.pt with net. prefix) or VLA-trained checkpoints."""
        logger.info(f"[CosmosPolicy] Loading DIT checkpoint from {ckpt_path}")
        if ckpt_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(ckpt_path, device="cpu")
        else:
            state_dict = torch.load(ckpt_path, map_location="cpu")

        # Handle EMA weights (load_ema_to_reg)
        ema_prefix = "model_ema."
        if any(k.startswith(ema_prefix) for k in state_dict.keys()):
            logger.info("[CosmosPolicy] Found EMA weights, loading EMA → regular")
            ema_dict = {}
            for k, v in state_dict.items():
                if k.startswith(ema_prefix):
                    new_key = k[len(ema_prefix):]
                    ema_dict[new_key] = v
            state_dict = ema_dict

        # Filter to DIT keys only (remove tokenizer/conditioner keys).
        # Skip _extra_state (TE metadata) and check shapes to avoid silent mismatches.
        dit_state = self.dit.state_dict()
        filtered = {}
        missing = []
        shape_mismatch = []
        extra_state_skipped = 0
        for k, model_v in dit_state.items():
            # _extra_state buffers are transformer-engine RMSNorm metadata,
            # not real weights -- skip them.
            if "_extra_state" in k:
                extra_state_skipped += 1
                continue

            ckpt_v = None
            if k in state_dict:
                ckpt_v = state_dict[k]
            else:
                # Try with "net." prefix (cosmos-policy wraps DIT in net)
                net_key = "net." + k
                # Try with "dit." prefix (VLA-Engine saves with dit. prefix)
                dit_key = "dit." + k
                if net_key in state_dict:
                    ckpt_v = state_dict[net_key]
                elif dit_key in state_dict:
                    ckpt_v = state_dict[dit_key]

            if ckpt_v is not None:
                if ckpt_v.shape == model_v.shape:
                    filtered[k] = ckpt_v
                else:
                    shape_mismatch.append((k, ckpt_v.shape, model_v.shape))
            else:
                missing.append(k)

        if extra_state_skipped:
            logger.info(
                "[CosmosPolicy] Skipped %d _extra_state (TE metadata) keys",
                extra_state_skipped,
            )

        if shape_mismatch:
            logger.warning(
                "[CosmosPolicy] Shape mismatch for %d real keys (skipped): %s",
                len(shape_mismatch),
                [(k, "ckpt=%s model=%s" % (cs, ms)) for k, cs, ms in shape_mismatch[:5]],
            )

        if missing:
            logger.warning(
                "[CosmosPolicy] %d real keys missing from checkpoint "
                "(expected for policy-specific params): %s...",
                len(missing), missing[:5],
            )

        load_result = self.dit.load_state_dict(filtered, strict=False)
        real_missing = [k for k in load_result.missing_keys if "_extra_state" not in k]
        real_model_keys = [k for k in dit_state if "_extra_state" not in k]
        n_loaded = len(filtered)
        n_total = len(real_model_keys)
        msg = "[CosmosPolicy] DIT checkpoint loaded: %d / %d real params" % (n_loaded, n_total)
        if real_missing:
            msg += ", %d missing" % len(real_missing)
        if load_result.unexpected_keys:
            msg += ", %d unexpected" % len(load_result.unexpected_keys)
        logger.info(msg)
