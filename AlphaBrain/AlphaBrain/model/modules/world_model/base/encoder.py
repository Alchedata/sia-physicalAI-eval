"""BaseWorldModelEncoder abstract base class."""
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .config import WorldModelEncoderConfig


class BaseWorldModelEncoder(nn.Module, ABC):
    """Abstract base class for world-model visual encoders."""

    def __init__(self, config: WorldModelEncoderConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def _build_encoder(self) -> None:
        """Build the underlying encoder model."""
        ...

    @abstractmethod
    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode pixel values into visual tokens.

        Args:
            pixel_values: preprocessed image tensor.

        Returns:
            Visual tokens of shape [B, N, D].
        """
        ...

    @abstractmethod
    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess raw images into the format expected by the encoder.

        Args:
            images: raw image tensor.

        Returns:
            Preprocessed tensor ready for encode_images().
        """
        ...

    @property
    @abstractmethod
    def encoder_dim(self) -> int:
        """Return the native hidden dimension of the encoder."""
        ...

    @property
    def device(self) -> torch.device:
        """Return the device of the first parameter."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

