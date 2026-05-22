"""WorldModelEncoderConfig dataclass."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class WorldModelEncoderConfig:
    """Configuration for world-model based visual encoders."""

    backend: str = "vjepa2"  # "vjepa2" | "wan2.2" | "cosmos2.5"
    checkpoint_path: str = ""
    hidden_size: int = 1024  # output hidden size after projection

    # Text encoder settings
    text_encoder_type: str = "t5-small"  # "t5-small" | "clip" | "precomputed"
    text_encoder_path: str = ""
    text_hidden_size: int = 512

    # Fusion settings
    fusion_type: str = "cross_attention"  # "cross_attention"
    num_fusion_layers: int = 2

    # Encoder freeze settings
    freeze_encoder: bool = False

    # Image settings
    image_size: int = 384

    # Intermediate feature extraction
    use_intermediate_features: bool = False
    intermediate_layer_ids: Optional[List[int]] = None

    # Single-layer feature extraction (Cosmos/WAN action-feature layer).
    # If None, each encoder falls back to its hard-coded default
    # (Cosmos=18, WAN=14). Exposed to yaml so ablations can override.
    feature_layer_id: Optional[int] = None

    # For models needing VAE from separate dir
    pretrained_dir: str = ""

    # Cosmos Reason1 (Qwen2.5-VL-7B) text encoder path (for Predict2.5)
    reason1_path: str = ""

    # ------------------------------------------------------------------
    # Diffusion-style backbone hyperparams (Cosmos 2.x / WAN 2.2).
    # Keep defaults matching the original scheduler_config.json values; the
    # yaml now writes them out explicitly for documentation + easy tuning.
    # ------------------------------------------------------------------
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    sigma_data: float = 1.0
    sigma_conditional: float = 0.0001

    # ------------------------------------------------------------------
    # V-JEPA ViT geometry (used by VJEPAEncoder._build_encoder).
    # Defaults match the shipped vjepa2.1_vitG_384 checkpoint (num_frames=16
    # so patch_embed_img switches on in 5D input; tubelet_size=2 so temporal
    # patches cover two frames; RoPE enabled with interpolation).
    # ------------------------------------------------------------------
    vjepa_num_frames: int = 16
    vjepa_patch_size: int = 16
    vjepa_tubelet_size: int = 2
    vjepa_use_rope: bool = True
    vjepa_interpolate_rope: bool = True

    # ------------------------------------------------------------------
    # WAN variant override. If None, WanEncoder auto-detects from the
    # checkpoint path substring ("14b"/"t2v" -> t2v-A14B, else ti2v-5B).
    # Set explicitly in yaml to force a particular variant.
    # ------------------------------------------------------------------
    wan_variant: Optional[str] = None

