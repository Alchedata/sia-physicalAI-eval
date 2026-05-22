"""
WAN VAE Tokenizer wrapper for VLA-Engine-Developer.

Wraps diffusers AutoencoderKLWan with the same encode/decode interface
used by cosmos-policy. Falls back to loading from .pth if needed.

Key specs:
  - z_dim = 16
  - Spatial compression: 8x (e.g. 224 -> 28)
  - Temporal compression: 4x, formula: T_latent = 1 + (T_pixel - 1) // 4
  - latents_mean / latents_std: 16-element vectors for normalization
  - Input video: float32 in [-1, 1] range, shape (B, C=3, T, H, W)
"""

import json
import os
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn


# Normalization constants from vae/config.json
_LATENTS_MEAN = [
    -0.7571, -0.7089, -0.9113,  0.1075, -0.1745,  0.9653,
    -0.1517,  1.5508,  0.4134, -0.0715,  0.5517, -0.3632,
    -0.1922, -0.9497,  0.2503, -0.2921,
]
_LATENTS_STD = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708,
    2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579,
    1.6382, 1.1253, 2.8251, 1.9160,
]


class WanVAEWrapper(nn.Module):
    """
    Thin wrapper around diffusers AutoencoderKLWan.

    encode(video) : (B, 3, T, H, W) float in [-1,1] -> (B, 16, T', H', W') normalized latent
    decode(latent): (B, 16, T', H', W') normalized latent -> (B, 3, T, H, W) float in [-1,1]

    T' = 1 + (T - 1) // 4,  H' = H // 8,  W' = W // 8
    """

    def __init__(
        self,
        pretrained_dir: str,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        temporal_window: int = 16,
    ):
        super().__init__()
        self._temporal_window = temporal_window
        self.dtype = dtype
        self.device_str = device
        self.z_dim = 16
        self.spatial_compression_factor = 8
        self.temporal_compression_factor = 4

        # Normalization tensors: create directly in target dtype to match
        # original cosmos-policy CosmosPolicyWanVAE behavior (creates tensors
        # in bf16 directly, computes 1/std in bf16).
        mean = torch.tensor(_LATENTS_MEAN, dtype=dtype).view(1, 16, 1, 1, 1)
        std = torch.tensor(_LATENTS_STD, dtype=dtype).view(1, 16, 1, 1, 1)
        inv_std = 1.0 / std  # reciprocal in target dtype, matching original
        self.register_buffer("latents_mean", mean)
        self.register_buffer("latents_std", std)
        self.register_buffer("_inv_std", inv_std)

        # Load VAE
        self.vae = self._load_vae(pretrained_dir, dtype, device)
        self._is_pth_vae = isinstance(self.vae, self._get_pth_vae_class())
        self.vae.eval().requires_grad_(False)

        # No autocast — match original cosmos-policy's is_amp=False behavior.
        # Autocast promotes F.normalize (used in VAE's RMS_norm layers) to
        # fp32, producing different latents than the original which runs
        # entirely in bf16 without autocast.
        self._amp_ctx = nullcontext()

        # Move all buffers (latents_mean, latents_std, _inv_std) to target device
        if device != "cpu":
            self.to(device)

    def _load_vae(self, pretrained_dir: str, dtype: torch.dtype, device: str):
        """Load VAE: prefer tokenizer.pth (matches original cosmos-policy), fallback to diffusers."""
        # Prefer .pth format — original cosmos-policy uses tokenizer.pth (WanVAE_),
        # which produces different latents than diffusers AutoencoderKLWan!
        pth_path = os.path.join(pretrained_dir, "tokenizer", "tokenizer.pth")
        if os.path.isfile(pth_path):
            return self._load_pth_vae(pth_path, dtype, device)

        vae_dir = os.path.join(pretrained_dir, "vae")
        if os.path.isdir(vae_dir):
            try:
                return self._load_diffusers_vae(vae_dir, dtype, device)
            except Exception as e:
                print(f"[WanVAEWrapper] diffusers load failed ({e})")

        raise FileNotFoundError(
            f"Could not find VAE weights in {pretrained_dir}. "
            "Expected tokenizer/tokenizer.pth or vae/ subfolder (diffusers)."
        )

    @staticmethod
    def _get_pth_vae_class():
        from AlphaBrain.model.modules.world_model.cosmos.wan_vae_arch import WanVAE_
        return WanVAE_

    def _load_diffusers_vae(self, vae_dir: str, dtype: torch.dtype, device: str):
        from diffusers import AutoencoderKLWan
        vae = AutoencoderKLWan.from_pretrained(vae_dir, torch_dtype=dtype)
        vae = vae.to(device)
        return vae

    def _load_pth_vae(self, pth_path: str, dtype: torch.dtype, device: str):
        """Load from cosmos-policy .pth format using the WanVAE_ architecture."""
        from AlphaBrain.model.modules.world_model.cosmos.wan_vae_arch import WanVAE_
        cfg = dict(
            dim=96,
            z_dim=16,
            dim_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            attn_scales=[],
            temperal_downsample=[False, True, True],
            dropout=0.0,
            temporal_window=self._temporal_window,
        )
        with torch.device("meta"):
            model = WanVAE_(**cfg)
        ckpt = torch.load(pth_path, map_location=device)
        model.load_state_dict(ckpt, assign=True)
        model = model.to(dtype=dtype, device=device)
        return model

    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        return 1 + (num_pixel_frames - 1) // self.temporal_compression_factor

    def get_pixel_num_frames(self, num_latent_frames: int) -> int:
        return (num_latent_frames - 1) * self.temporal_compression_factor + 1

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: (B, 3, T, H, W) float in [-1, 1]
        Returns:
            latent: (B, 16, T', H', W') normalized latent
        """
        in_dtype = video.dtype
        video = video.to(self.dtype)
        if self._is_pth_vae:
            # WanVAE_.encode(x, scale) applies normalization internally:
            # mu = (mu - mean) * (1/std) = (mu - mean) / std
            # Use pre-computed scale in target dtype (matches original cosmos-policy)
            scale = [self.latents_mean, self._inv_std]
            latent = self.vae.encode(video, scale)
        else:
            # diffusers AutoencoderKLWan
            posterior = self.vae.encode(video)
            latent = posterior.latent_dist.mean
            # Apply same normalization
            latent = (latent - self.latents_mean) / self.latents_std
        return latent.to(in_dtype)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (B, 16, T', H', W') normalized latent
        Returns:
            video: (B, 3, T, H, W) float in [-1, 1]
        """
        in_dtype = latent.dtype
        z = latent.to(self.dtype)
        if self._is_pth_vae:
            # Denormalize: z = z / (1/std) + mean = z * std + mean
            scale = [self.latents_mean, self._inv_std]
            video = self.vae.decode(z, scale)
        else:
            # diffusers: denormalize then decode
            z = z * self.latents_std + self.latents_mean
            video = self.vae.decode(z).sample
        return video.to(in_dtype)

    def to(self, *args, **kwargs):
        # Keep buffers in sync when moving device/dtype
        result = super().to(*args, **kwargs)
        if self.vae is not None:
            self.vae = self.vae.to(*args, **kwargs)
        return result


def build_wan_vae(
    pretrained_dir: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> WanVAEWrapper:
    """
    Convenience factory.

    Args:
        pretrained_dir: path to Cosmos-Predict2-2B-Video2World directory
                        (must contain vae/ or tokenizer/tokenizer.pth)
        dtype: model dtype, default bfloat16
        device: target device
    Returns:
        WanVAEWrapper instance (frozen, eval mode)
    """
    return WanVAEWrapper(pretrained_dir=pretrained_dir, dtype=dtype, device=device)
