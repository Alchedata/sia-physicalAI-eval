"""CrossAttentionFusion for visual-text fusion."""
import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """Fuse visual and text tokens via cross-attention.

    Visual tokens attend to text tokens, producing task-conditioned
    visual representations.
    """

    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        output_dim: int,
        num_layers: int = 2,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        # When visual_dim == output_dim (e.g. Cosmos 2.5 2048 -> 2048),
        # skip the random-initialized Linear to preserve pretrained visual features.
        # Dim mismatch still gets a learned projection.
        if visual_dim == output_dim:
            self.visual_proj = nn.Identity()
        else:
            self.visual_proj = nn.Linear(visual_dim, output_dim)
        # Text projection is kept as a learnable Linear (text_dim typically != output_dim).
        if text_dim == output_dim:
            self.text_proj = nn.Identity()
        else:
            self.text_proj = nn.Linear(text_dim, output_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=output_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            visual_tokens: [B, N_v, D_v]
            text_tokens:   [B, N_t, D_t]

        Returns:
            Fused visual tokens: [B, N_v, output_dim]
        """
        v = self.visual_proj(visual_tokens)   # [B, N_v, output_dim]
        t = self.text_proj(text_tokens)       # [B, N_t, output_dim]
        fused = self.cross_attn(v, t)         # [B, N_v, output_dim]
        fused = self.norm(fused)
        return fused

