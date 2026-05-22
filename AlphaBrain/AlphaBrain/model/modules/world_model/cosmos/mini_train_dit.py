# Ported from cosmos-policy/cosmos_policy/_src/predict2/networks/minimal_v4_dit.py
# Modifications:
#   - Replaced transformer_engine attention with torch.nn.functional.scaled_dot_product_attention
#   - Replaced te.pytorch.RMSNorm with torch.nn.RMSNorm (PyTorch 2.4+)
#   - Removed megatron / context-parallel logic
#   - Removed natten / sparse attention code
#   - Removed transformer_engine imports
#   - Default atten_backend="torch", concat_padding_mask=False

import math
from collections import namedtuple
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.amp as amp
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn

# ---------------------------------------------------------------------------
# VideoSize helper
# ---------------------------------------------------------------------------
VideoSize = namedtuple("VideoSize", ["T", "H", "W"])


# ---------------------------------------------------------------------------
# apply_rotary_pos_emb  (replaces transformer_engine version)
# ---------------------------------------------------------------------------
def apply_rotary_pos_emb(x: torch.Tensor, rope_emb: torch.Tensor, tensor_format: str = "bshd", fused: bool = False) -> torch.Tensor:
    """Apply rotary position embeddings to x.

    Args:
        x: shape [B, S, H, D] (tensor_format="bshd")
        rope_emb: shape [S, 1, 1, D] produced by VideoRopePosition3DEmb.
                  The last dim D == head_dim, containing cos||sin concatenated
                  (first D/2 = cos values, last D/2 = sin values).
    Returns:
        x with RoPE applied, same shape as input.
    """
    # rope_emb: [S, 1, 1, D] where D = head_dim
    D = rope_emb.shape[-1]
    half = D // 2
    # cos/sin: [S, 1, 1, D/2]
    cos = rope_emb[..., :half]
    sin = rope_emb[..., half:]

    if tensor_format == "bshd":
        # x: [B, S, H, D]  — D == head_dim
        # Reshape cos/sin from [S, 1, 1, D/2] -> [1, S, 1, D/2] for broadcasting
        cos = cos.permute(1, 0, 2, 3)  # [1, S, 1, D/2]
        sin = sin.permute(1, 0, 2, 3)  # [1, S, 1, D/2]
        x1 = x[..., :half]   # [B, S, H, D/2]
        x2 = x[..., half:]   # [B, S, H, D/2]
        rotated = torch.cat([-x2, x1], dim=-1)  # [B, S, H, D]
        return x * torch.cat([cos, cos], dim=-1) + rotated * torch.cat([sin, sin], dim=-1)
    else:
        raise ValueError(f"Unsupported tensor_format: {tensor_format}")


# ---------------------------------------------------------------------------
# Attention helpers
# ---------------------------------------------------------------------------
def torch_attention_op(
    q_B_S_H_D: torch.Tensor,
    k_B_S_H_D: torch.Tensor,
    v_B_S_H_D: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    flatten_heads: bool = True,
) -> torch.Tensor:
    """Scaled dot-product attention matching official cosmos-policy attention().
    Explicit bf16 cast before SDPA. Inputs [B, S, H, D]. Returns [B, S, H*D] or [B, S, H, D]."""
    q = q_B_S_H_D.to(torch.bfloat16)
    k = k_B_S_H_D.to(torch.bfloat16)
    v = v_B_S_H_D.to(torch.bfloat16)
    q = rearrange(q, "b s h d -> b h s d")
    k = rearrange(k, "b s h d -> b h s d")
    v = rearrange(v, "b s h d -> b h s d")
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    out = out.transpose(1, 2).contiguous()
    if flatten_heads:
        return rearrange(out, "b s h d -> b s (h d)")
    return out


# ---------------------------------------------------------------------------
# Feed-Forward Network
# ---------------------------------------------------------------------------
class GPT2FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.activation = nn.GELU()
        self.layer1 = nn.Linear(d_model, d_ff, bias=False)
        self.layer2 = nn.Linear(d_ff, d_model, bias=False)
        self._layer_id = None
        self._dim = d_model
        self._hidden_dim = d_ff
        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self._dim)
        torch.nn.init.trunc_normal_(self.layer1.weight, std=std, a=-3 * std, b=3 * std)
        std = 1.0 / math.sqrt(self._hidden_dim)
        if self._layer_id is not None:
            std = std / math.sqrt(2 * (self._layer_id + 1))
        torch.nn.init.trunc_normal_(self.layer2.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x


# ---------------------------------------------------------------------------
# Attention module
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim=None,
        n_heads=8,
        head_dim=64,
        dropout=0.0,
        qkv_format: str = "bshd",
        backend: str = "torch",
        use_wan_fp32_strategy: bool = False,
    ) -> None:
        super().__init__()
        self.is_selfattn = context_dim is None
        assert backend in ["torch", "transformer_engine"], f"Unsupported backend: {backend}"
        self.backend = backend

        context_dim = query_dim if context_dim is None else context_dim
        inner_dim = head_dim * n_heads

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.qkv_format = qkv_format
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.use_wan_fp32_strategy = use_wan_fp32_strategy

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.v_norm = nn.Identity()
        self.output_proj = nn.Linear(inner_dim, query_dim, bias=False)
        self.output_dropout = nn.Dropout(dropout) if dropout > 1e-4 else nn.Identity()

        # Always use TE RMSNorm for q_norm/k_norm (matches official cosmos-policy
        # which uses te.pytorch.RMSNorm regardless of attention backend).
        # nn.RMSNorm promotes bf16 to float32 internally, causing numerical drift.
        import transformer_engine.pytorch as te
        self.q_norm = te.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = te.RMSNorm(self.head_dim, eps=1e-6)

        if backend == "transformer_engine":
            from transformer_engine.pytorch.attention import DotProductAttention as TEDotProductAttention
            self.attn_op = TEDotProductAttention(
                self.n_heads, self.head_dim, num_gqa_groups=self.n_heads,
                attention_dropout=0, qkv_format=qkv_format, attn_mask_type="no_mask",
            )
        else:
            self.attn_op = torch_attention_op

        self._query_dim = query_dim
        self._context_dim = context_dim
        self._inner_dim = inner_dim

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self._query_dim)
        torch.nn.init.trunc_normal_(self.q_proj.weight, std=std, a=-3 * std, b=3 * std)
        std = 1.0 / math.sqrt(self._context_dim)
        torch.nn.init.trunc_normal_(self.k_proj.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.v_proj.weight, std=std, a=-3 * std, b=3 * std)
        std = 1.0 / math.sqrt(self._inner_dim)
        torch.nn.init.trunc_normal_(self.output_proj.weight, std=std, a=-3 * std, b=3 * std)
        for layer in (self.q_norm, self.k_norm):
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def compute_qkv(self, x, context=None, rope_emb=None):
        q = self.q_proj(x)
        context = x if context is None else context
        k = self.k_proj(context)
        v = self.v_proj(context)
        q, k, v = map(
            lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim),
            (q, k, v),
        )
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)
        if self.is_selfattn and rope_emb is not None:
            if self.use_wan_fp32_strategy:
                q = q.to(torch.float32)
                k = k.to(torch.float32)
            q = apply_rotary_pos_emb(q, rope_emb, tensor_format=self.qkv_format, fused=True)
            k = apply_rotary_pos_emb(k, rope_emb, tensor_format=self.qkv_format, fused=True)
            if self.use_wan_fp32_strategy:
                q = q.to(x.dtype)
                k = k.to(x.dtype)
        return q, k, v

    def compute_attention(self, q, k, v, video_size=None, kv_cache_cfg=None):
        target_dtype = q.dtype
        k = k.to(target_dtype)
        v = v.to(target_dtype)
        result = self.attn_op(q, k, v)
        # TE DotProductAttention returns (B,S,H,D); torch_attention_op returns (B,S,H*D).
        # Flatten heads if needed before output_proj.
        if result.dim() == 4:
            result = rearrange(result, "b s h d -> b s (h d)")
        # Ensure dtype matches output_proj weights (safety for mixed-precision paths)
        result = result.to(self.output_proj.weight.dtype)
        return self.output_dropout(self.output_proj(result))

    def forward(self, x, context=None, rope_emb=None, video_size=None, kv_cache_cfg=None):
        q, k, v = self.compute_qkv(x, context, rope_emb=rope_emb)
        return self.compute_attention(q, k, v, video_size=video_size, kv_cache_cfg=kv_cache_cfg)


# ---------------------------------------------------------------------------
# Positional Embeddings
# ---------------------------------------------------------------------------
class VideoPositionEmb(nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def seq_dim(self):
        return 1

    def forward(self, x_B_T_H_W_C: torch.Tensor, fps=None) -> torch.Tensor:
        B_T_H_W_C = x_B_T_H_W_C.shape
        return self.generate_embeddings(B_T_H_W_C, fps=fps)

    def generate_embeddings(self, B_T_H_W_C: torch.Size, fps=None):
        raise NotImplementedError

    def reset_parameters(self) -> None:
        pass


class VideoRopePosition3DEmb(VideoPositionEmb):
    def __init__(
        self,
        *,
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        base_fps: int = 24,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        enable_fps_modulation: bool = True,
        **kwargs,
    ):
        del kwargs
        super().__init__()
        self.register_buffer("seq", torch.arange(max(len_h, len_w, len_t), dtype=torch.float))
        self.base_fps = base_fps
        self.max_h = len_h
        self.max_w = len_w
        self.max_t = len_t
        self.enable_fps_modulation = enable_fps_modulation
        dim = head_dim
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        assert dim == dim_h + dim_w + dim_t

        self.register_buffer(
            "dim_spatial_range",
            torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h,
            persistent=True,
        )
        self.register_buffer(
            "dim_temporal_range",
            torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t,
            persistent=True,
        )
        self._dim_h = dim_h
        self._dim_t = dim_t

        self.h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.w_ntk_factor = w_extrapolation_ratio ** (dim_w / (dim_w - 2))
        self.t_ntk_factor = t_extrapolation_ratio ** (dim_t / (dim_t - 2))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        dim_h = self._dim_h
        dim_t = self._dim_t
        self.seq = torch.arange(max(self.max_h, self.max_w, self.max_t)).float().to(self.dim_spatial_range.device)
        self.dim_spatial_range = (
            torch.arange(0, dim_h, 2)[: (dim_h // 2)].float().to(self.dim_spatial_range.device) / dim_h
        )
        self.dim_temporal_range = (
            torch.arange(0, dim_t, 2)[: (dim_t // 2)].float().to(self.dim_spatial_range.device) / dim_t
        )

    def generate_embeddings(
        self,
        B_T_H_W_C: torch.Size,
        fps: Optional[torch.Tensor] = None,
        h_ntk_factor: Optional[float] = None,
        w_ntk_factor: Optional[float] = None,
        t_ntk_factor: Optional[float] = None,
    ):
        h_ntk_factor = h_ntk_factor if h_ntk_factor is not None else self.h_ntk_factor
        w_ntk_factor = w_ntk_factor if w_ntk_factor is not None else self.w_ntk_factor
        t_ntk_factor = t_ntk_factor if t_ntk_factor is not None else self.t_ntk_factor

        h_theta = 10000.0 * h_ntk_factor
        w_theta = 10000.0 * w_ntk_factor
        t_theta = 10000.0 * t_ntk_factor

        h_spatial_freqs = 1.0 / (h_theta ** self.dim_spatial_range.float())
        w_spatial_freqs = 1.0 / (w_theta ** self.dim_spatial_range.float())
        temporal_freqs = 1.0 / (t_theta ** self.dim_temporal_range.float())

        B, T, H, W, _ = B_T_H_W_C
        assert H <= self.max_h and W <= self.max_w
        half_emb_h = torch.outer(self.seq[:H], h_spatial_freqs)
        half_emb_w = torch.outer(self.seq[:W], w_spatial_freqs)

        if self.enable_fps_modulation:
            if fps is None:
                assert T == 1
                half_emb_t = torch.outer(self.seq[:T], temporal_freqs)
            else:
                half_emb_t = torch.outer(self.seq[:T] / fps[:1] * self.base_fps, temporal_freqs)
        else:
            half_emb_t = torch.outer(self.seq[:T], temporal_freqs)

        em_T_H_W_D = torch.cat(
            [
                repeat(half_emb_t, "t d -> t h w d", h=H, w=W),
                repeat(half_emb_h, "h d -> t h w d", t=T, w=W),
                repeat(half_emb_w, "w d -> t h w d", t=T, h=H),
            ]
            * 2,
            dim=-1,
        )
        return rearrange(em_T_H_W_D, "t h w d -> (t h w) 1 1 d").float()

    @property
    def seq_dim(self):
        return 0


# ---------------------------------------------------------------------------
# Modulation helper
# ---------------------------------------------------------------------------
def modulate(x, shift, scale):
    return x * (1 + scale) + shift


# ---------------------------------------------------------------------------
# Timestep embeddings
# ---------------------------------------------------------------------------
class Timesteps(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps_B_T):
        assert timesteps_B_T.ndim == 2
        in_dtype = timesteps_B_T.dtype
        timesteps = timesteps_B_T.flatten().float()
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - 0.0)
        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        return rearrange(emb.to(dtype=in_dtype), "(b t) d -> b t d", b=timesteps_B_T.shape[0], t=timesteps_B_T.shape[1])


class TimestepEmbedding(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_adaln_lora: bool = False):
        super().__init__()
        self.in_dim = in_features
        self.out_dim = out_features
        self.linear_1 = nn.Linear(in_features, out_features, bias=not use_adaln_lora)
        self.activation = nn.SiLU()
        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            self.linear_2 = nn.Linear(out_features, 3 * out_features, bias=False)
        else:
            self.linear_2 = nn.Linear(out_features, out_features, bias=False)
        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.in_dim)
        torch.nn.init.trunc_normal_(self.linear_1.weight, std=std, a=-3 * std, b=3 * std)
        std = 1.0 / math.sqrt(self.out_dim)
        torch.nn.init.trunc_normal_(self.linear_2.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, sample: torch.Tensor):
        emb = self.linear_1(sample)
        emb = self.activation(emb)
        emb = self.linear_2(emb)
        if self.use_adaln_lora:
            return sample, emb  # (emb_B_T_D, adaln_lora_B_T_3D)
        return emb, None


# ---------------------------------------------------------------------------
# PatchEmbed
# ---------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    def __init__(self, spatial_patch_size, temporal_patch_size, in_channels=3, out_channels=768):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.proj = nn.Sequential(
            Rearrange(
                "b c (t r) (h m) (w n) -> b t h w (c r m n)",
                r=temporal_patch_size,
                m=spatial_patch_size,
                n=spatial_patch_size,
            ),
            nn.Linear(
                in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size,
                out_channels,
                bias=False,
            ),
        )
        self.dim = in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size
        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.dim)
        torch.nn.init.trunc_normal_(self.proj[1].weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x):
        assert x.dim() == 5
        _, _, T, H, W = x.shape
        assert H % self.spatial_patch_size == 0 and W % self.spatial_patch_size == 0
        assert T % self.temporal_patch_size == 0
        return self.proj(x)


# ---------------------------------------------------------------------------
# FinalLayer
# ---------------------------------------------------------------------------
class FinalLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        spatial_patch_size,
        temporal_patch_size,
        out_channels,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        use_wan_fp32_strategy: bool = False,
    ):
        super().__init__()
        self.use_wan_fp32_strategy = use_wan_fp32_strategy
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size,
            spatial_patch_size * spatial_patch_size * temporal_patch_size * out_channels,
            bias=False,
        )
        self.hidden_size = hidden_size
        self.n_adaln_chunks = 2
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        if use_adaln_lora:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, self.n_adaln_chunks * hidden_size, bias=False),
            )
        else:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, self.n_adaln_chunks * hidden_size, bias=False)
            )
        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.hidden_size)
        torch.nn.init.trunc_normal_(self.linear.weight, std=std, a=-3 * std, b=3 * std)
        if self.use_adaln_lora:
            torch.nn.init.trunc_normal_(self.adaln_modulation[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.adaln_modulation[2].weight)
        else:
            torch.nn.init.zeros_(self.adaln_modulation[1].weight)
        self.layer_norm.reset_parameters()

    def forward(self, x_B_T_H_W_D, emb_B_T_D, adaln_lora_B_T_3D: Optional[torch.Tensor] = None):
        if self.use_wan_fp32_strategy:
            assert emb_B_T_D.dtype == torch.float32
        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if self.use_adaln_lora:
                assert adaln_lora_B_T_3D is not None
                shift_B_T_D, scale_B_T_D = (
                    self.adaln_modulation(emb_B_T_D) + adaln_lora_B_T_3D[:, :, : 2 * self.hidden_size]
                ).chunk(2, dim=-1)
            else:
                shift_B_T_D, scale_B_T_D = self.adaln_modulation(emb_B_T_D).chunk(2, dim=-1)

            shift = rearrange(shift_B_T_D, "b t d -> b t 1 1 d")
            scale = rearrange(scale_B_T_D, "b t d -> b t 1 1 d")
            x_B_T_H_W_D = self.layer_norm(x_B_T_H_W_D) * (1 + scale) + shift
            x_B_T_H_W_O = self.linear(x_B_T_H_W_D)
        return x_B_T_H_W_O


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        backend: str = "torch",
        image_context_dim: Optional[int] = None,
        use_wan_fp32_strategy: bool = False,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.layer_norm_self_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(
            x_dim, None, num_heads, x_dim // num_heads,
            qkv_format="bshd", backend=backend,
            use_wan_fp32_strategy=use_wan_fp32_strategy,
        )
        self.layer_norm_cross_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = Attention(
            x_dim, context_dim, num_heads, x_dim // num_heads,
            qkv_format="bshd", backend=backend,
        )
        self.layer_norm_mlp = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = GPT2FeedForward(x_dim, int(x_dim * mlp_ratio))

        self.use_adaln_lora = use_adaln_lora
        if self.use_adaln_lora:
            self.adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_cross_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
        else:
            self.adaln_modulation_self_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_cross_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_mlp = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))

        self.use_wan_fp32_strategy = use_wan_fp32_strategy

    def reset_parameters(self) -> None:
        self.layer_norm_self_attn.reset_parameters()
        self.layer_norm_cross_attn.reset_parameters()
        self.layer_norm_mlp.reset_parameters()
        if self.use_adaln_lora:
            std = 1.0 / math.sqrt(self.x_dim)
            torch.nn.init.trunc_normal_(self.adaln_modulation_self_attn[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.trunc_normal_(self.adaln_modulation_cross_attn[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.trunc_normal_(self.adaln_modulation_mlp[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.adaln_modulation_self_attn[2].weight)
            torch.nn.init.zeros_(self.adaln_modulation_cross_attn[2].weight)
            torch.nn.init.zeros_(self.adaln_modulation_mlp[2].weight)
        else:
            torch.nn.init.zeros_(self.adaln_modulation_self_attn[1].weight)
            torch.nn.init.zeros_(self.adaln_modulation_cross_attn[1].weight)
            torch.nn.init.zeros_(self.adaln_modulation_mlp[1].weight)

    def init_weights(self) -> None:
        self.reset_parameters()
        self.self_attn.init_weights()
        self.cross_attn.init_weights()
        self.mlp.init_weights()

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
        kv_cache_cfg=None,
    ) -> torch.Tensor:
        if extra_per_block_pos_emb is not None:
            x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb

        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if self.use_adaln_lora:
                shift_sa, scale_sa, gate_sa = (self.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D).chunk(3, dim=-1)
                shift_ca, scale_ca, gate_ca = (self.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora_B_T_3D).chunk(3, dim=-1)
                shift_mlp, scale_mlp, gate_mlp = (self.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D).chunk(3, dim=-1)
            else:
                shift_sa, scale_sa, gate_sa = self.adaln_modulation_self_attn(emb_B_T_D).chunk(3, dim=-1)
                shift_ca, scale_ca, gate_ca = self.adaln_modulation_cross_attn(emb_B_T_D).chunk(3, dim=-1)
                shift_mlp, scale_mlp, gate_mlp = self.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)

        def _reshape(t):
            return rearrange(t, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)

        shift_sa, scale_sa, gate_sa = _reshape(shift_sa), _reshape(scale_sa), _reshape(gate_sa)
        shift_ca, scale_ca, gate_ca = _reshape(shift_ca), _reshape(scale_ca), _reshape(gate_ca)
        shift_mlp, scale_mlp, gate_mlp = _reshape(shift_mlp), _reshape(scale_mlp), _reshape(gate_mlp)

        B, T, H, W, D = x_B_T_H_W_D.shape

        # Self-attention
        norm_x = self.layer_norm_self_attn(x_B_T_H_W_D) * (1 + scale_sa) + shift_sa
        sa_out = rearrange(
            self.self_attn(
                rearrange(norm_x, "b t h w d -> b (t h w) d"),
                None,
                rope_emb=rope_emb_L_1_1_D,
            ),
            "b (t h w) d -> b t h w d", t=T, h=H, w=W,
        )
        x_B_T_H_W_D = x_B_T_H_W_D + gate_sa * sa_out

        # Cross-attention
        norm_x = self.layer_norm_cross_attn(x_B_T_H_W_D) * (1 + scale_ca) + shift_ca
        ca_out = rearrange(
            self.cross_attn(
                rearrange(norm_x, "b t h w d -> b (t h w) d"),
                crossattn_emb,
            ),
            "b (t h w) d -> b t h w d", t=T, h=H, w=W,
        )
        x_B_T_H_W_D = x_B_T_H_W_D + gate_ca * ca_out

        # MLP
        norm_x = self.layer_norm_mlp(x_B_T_H_W_D) * (1 + scale_mlp) + shift_mlp
        x_B_T_H_W_D = x_B_T_H_W_D + gate_mlp * self.mlp(norm_x)

        return x_B_T_H_W_D


# ---------------------------------------------------------------------------
# MiniTrainDIT — main model
# ---------------------------------------------------------------------------
class MiniTrainDIT(nn.Module):
    """
    Minimal DiT for LIBERO training. Ported from cosmos-policy minimal_v4_dit.py.
    Removes megatron/context-parallel, natten/sparse attention, and transformer_engine deps.
    Uses torch SDPA and torch.nn.RMSNorm.

    LIBERO config: in_channels=16, out_channels=16, num_blocks=28, num_heads=16,
                   model_channels=2048, crossattn_emb_channels=1024,
                   patch_spatial=2, patch_temporal=1, pos_emb_cls="rope3d"
    """

    def __init__(
        self,
        max_img_h: int,
        max_img_w: int,
        max_frames: int,
        in_channels: int,
        out_channels: int,
        patch_spatial: int,
        patch_temporal: int,
        concat_padding_mask: bool = False,
        model_channels: int = 768,
        num_blocks: int = 10,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        atten_backend: str = "torch",
        crossattn_emb_channels: int = 1024,
        use_crossattn_projection: bool = False,
        crossattn_proj_in_channels: int = 1024,
        extra_image_context_dim: Optional[int] = None,
        pos_emb_cls: str = "rope3d",
        pos_emb_learnable: bool = False,
        pos_emb_interpolation: str = "crop",
        min_fps: int = 1,
        max_fps: int = 30,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        rope_h_extrapolation_ratio: float = 1.0,
        rope_w_extrapolation_ratio: float = 1.0,
        rope_t_extrapolation_ratio: float = 1.0,
        extra_per_block_abs_pos_emb: bool = False,
        extra_h_extrapolation_ratio: float = 1.0,
        extra_w_extrapolation_ratio: float = 1.0,
        extra_t_extrapolation_ratio: float = 1.0,
        rope_enable_fps_modulation: bool = True,
        use_wan_fp32_strategy: bool = False,
    ) -> None:
        super().__init__()
        self.max_img_h = max_img_h
        self.max_img_w = max_img_w
        self.max_frames = max_frames
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.model_channels = model_channels
        self.concat_padding_mask = concat_padding_mask
        self.atten_backend = atten_backend
        self.pos_emb_cls = pos_emb_cls
        self.pos_emb_learnable = pos_emb_learnable
        self.pos_emb_interpolation = pos_emb_interpolation
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.rope_h_extrapolation_ratio = rope_h_extrapolation_ratio
        self.rope_w_extrapolation_ratio = rope_w_extrapolation_ratio
        self.rope_t_extrapolation_ratio = rope_t_extrapolation_ratio
        self.extra_per_block_abs_pos_emb = extra_per_block_abs_pos_emb
        self.extra_h_extrapolation_ratio = extra_h_extrapolation_ratio
        self.extra_w_extrapolation_ratio = extra_w_extrapolation_ratio
        self.extra_t_extrapolation_ratio = extra_t_extrapolation_ratio
        self.rope_enable_fps_modulation = rope_enable_fps_modulation
        self.extra_image_context_dim = extra_image_context_dim
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        self.use_wan_fp32_strategy = use_wan_fp32_strategy
        self.use_crossattn_projection = use_crossattn_projection
        self.crossattn_proj_in_channels = crossattn_proj_in_channels

        self._build_patch_embed()
        self._build_pos_embed()

        self.t_embedder = nn.Sequential(
            Timesteps(model_channels),
            TimestepEmbedding(model_channels, model_channels, use_adaln_lora=use_adaln_lora),
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    x_dim=model_channels,
                    context_dim=crossattn_emb_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_adaln_lora=use_adaln_lora,
                    adaln_lora_dim=adaln_lora_dim,
                    backend=atten_backend,
                    image_context_dim=None,
                    use_wan_fp32_strategy=use_wan_fp32_strategy,
                )
                for _ in range(num_blocks)
            ]
        )

        self.final_layer = FinalLayer(
            hidden_size=model_channels,
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            out_channels=out_channels,
            use_adaln_lora=use_adaln_lora,
            adaln_lora_dim=adaln_lora_dim,
            use_wan_fp32_strategy=use_wan_fp32_strategy,
        )

        # Always use TE RMSNorm for t_embedding_norm (matches official cosmos-policy)
        import transformer_engine.pytorch as te
        self.t_embedding_norm = te.RMSNorm(model_channels, eps=1e-6)

        if use_crossattn_projection:
            self.crossattn_proj = nn.Sequential(
                nn.Linear(crossattn_proj_in_channels, crossattn_emb_channels, bias=True),
                nn.GELU(),
            )

        self.init_weights()

    def init_weights(self):
        self.x_embedder.init_weights()
        self.pos_embedder.reset_parameters()
        self.t_embedder[1].init_weights()
        for block in self.blocks:
            block.init_weights()
        self.final_layer.init_weights()
        if hasattr(self.t_embedding_norm, "reset_parameters"):
            self.t_embedding_norm.reset_parameters()

    def _build_patch_embed(self):
        in_ch = self.in_channels + 1 if self.concat_padding_mask else self.in_channels
        self.x_embedder = PatchEmbed(
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            in_channels=in_ch,
            out_channels=self.model_channels,
        )

    def _build_pos_embed(self):
        if self.pos_emb_cls == "rope3d":
            cls_type = VideoRopePosition3DEmb
        else:
            raise ValueError(f"Unknown pos_emb_cls {self.pos_emb_cls}")

        kwargs = dict(
            model_channels=self.model_channels,
            len_h=self.max_img_h // self.patch_spatial,
            len_w=self.max_img_w // self.patch_spatial,
            len_t=self.max_frames // self.patch_temporal,
            max_fps=self.max_fps,
            min_fps=self.min_fps,
            is_learnable=self.pos_emb_learnable,
            interpolation=self.pos_emb_interpolation,
            head_dim=self.model_channels // self.num_heads,
            h_extrapolation_ratio=self.rope_h_extrapolation_ratio,
            w_extrapolation_ratio=self.rope_w_extrapolation_ratio,
            t_extrapolation_ratio=self.rope_t_extrapolation_ratio,
            enable_fps_modulation=self.rope_enable_fps_modulation,
        )
        self.pos_embedder = cls_type(**kwargs)

    def prepare_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.concat_padding_mask:
            assert padding_mask is not None
            import torchvision.transforms.functional as TF
            padding_mask = TF.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]),
                interpolation=TF.InterpolationMode.NEAREST,
            )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
            )
        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)

        if "rope" in self.pos_emb_cls.lower():
            return x_B_T_H_W_D, self.pos_embedder(x_B_T_H_W_D, fps=fps), None
        x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D)
        return x_B_T_H_W_D, None, None

    def unpatchify(self, x_B_T_H_W_M):
        return rearrange(
            x_B_T_H_W_M,
            "B T H W (p1 p2 t C) -> B C (T t) (H p1) (W p2)",
            p1=self.patch_spatial,
            p2=self.patch_spatial,
            t=self.patch_temporal,
        )

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        intermediate_feature_ids: Optional[List[int]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            x_B_C_T_H_W: (B, C, T, H, W) latent video tensor
            timesteps_B_T: (B,) or (B, T) sigma/timestep tensor
            crossattn_emb: (B, N, D) text/condition embeddings
            fps: optional (B,) fps tensor
        """
        x_B_T_H_W_D, rope_emb_L_1_1_D, _ = self.prepare_embedded_sequence(
            x_B_C_T_H_W, fps=fps, padding_mask=padding_mask
        )

        if self.use_crossattn_projection:
            crossattn_emb = self.crossattn_proj(crossattn_emb)

        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if timesteps_B_T.ndim == 1:
                timesteps_B_T = timesteps_B_T.unsqueeze(1)
            t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
            t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        intermediate_features_outputs = []
        for i, block in enumerate(self.blocks):
            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                crossattn_emb,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
            )
            if intermediate_feature_ids and i in intermediate_feature_ids:
                intermediate_features_outputs.append(
                    rearrange(x_B_T_H_W_D, "b t h w d -> b (t h w) d")
                )

        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)

        if intermediate_feature_ids:
            return x_B_C_Tt_Hp_Wp, intermediate_features_outputs
        return x_B_C_Tt_Hp_Wp


class MinimalV1LVGDiT(MiniTrainDIT):
    """
    MiniTrainDIT + condition_video_input_mask concatenation + timestep_scale.
    Replaces cosmos_policy._src.predict2.networks.minimal_v1_lvg_dit.MinimalV1LVGDiT
    to eliminate dependency on the cosmos-policy repository.
    """

    def __init__(self, *args, timestep_scale: float = 1.0, **kwargs):
        assert "in_channels" in kwargs, "in_channels must be provided"
        kwargs["in_channels"] += 1  # Add 1 for the condition mask channel
        self.timestep_scale = timestep_scale
        super().__init__(*args, **kwargs)

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Concatenate condition mask to input (matching official MinimalV1LVGDiT)
        if condition_video_input_mask_B_C_T_H_W is not None:
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)],
                dim=1,
            )
        else:
            B, _, T, H, W = x_B_C_T_H_W.shape
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, torch.zeros((B, 1, T, H, W), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)],
                dim=1,
            )
        return super().forward(
            x_B_C_T_H_W=x_B_C_T_H_W,
            timesteps_B_T=timesteps_B_T * self.timestep_scale,
            crossattn_emb=crossattn_emb,
            fps=fps,
            padding_mask=padding_mask,
        )
