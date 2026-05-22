"""V-JEPA 2.1 encoder for WM pipeline.

Also hosts the underlying V-JEPA modules that used to live at
modules.vjepa2.* (now sunk under modules.world_model.vjepa.*).
"""
from .encoder import VJEPAEncoder

__all__ = ["VJEPAEncoder"]
