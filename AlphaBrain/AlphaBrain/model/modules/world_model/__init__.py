"""World Model encoders for VLA-Engine.

This module hosts the canonical implementations of world-model-backbone visual
encoders (Cosmos 2.0 / 2.5, WAN 2.2, V-JEPA 2.1) plus the shared base layer
(Config, Fusion, LightweightTextEncoder, BaseWorldModelEncoder ABC,
WorldModelVLMInterface wrapper).

Legacy imports from AlphaBrain.model.modules.vlm.world_model_* continue to work
via shim re-exports, but all new code should import from here.
"""

# Re-export all base symbols for convenient 
from .base import (
    WorldModelEncoderConfig,
    LightweightTextEncoder,
    CrossAttentionFusion,
    BaseWorldModelEncoder,
    WorldModelVLMInterface,
    WorldModelEncoderInterface,  # backward-compat alias
)


def get_world_model_encoder(config) -> WorldModelVLMInterface:
    """Factory for the world-model-backbone encoder wrapper.

    Dispatches on config.framework.qwenvl.base_vlm / vlm_type string to the
    appropriate encoder (Cosmos 2 / 2.5 diffusers, WAN 2.2, V-JEPA 2.1).

    Currently this mirrors the legacy factory in
    AlphaBrain.model.modules.vlm.__init__.get_vlm_model so that existing yaml /
    ckpt continue to load unchanged.
    """
    return WorldModelVLMInterface(config)


__all__ = [
    "WorldModelEncoderConfig",
    "LightweightTextEncoder",
    "CrossAttentionFusion",
    "BaseWorldModelEncoder",
    "WorldModelVLMInterface",
    "WorldModelEncoderInterface",
    "get_world_model_encoder",
]
