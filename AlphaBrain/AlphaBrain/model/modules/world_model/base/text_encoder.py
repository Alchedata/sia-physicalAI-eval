"""LightweightTextEncoder for WM pipeline."""
import logging
import os
from typing import List, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LightweightTextEncoder(nn.Module):
    """Lightweight wrapper around frozen text encoders for instruction conditioning."""

    def __init__(
        self,
        encoder_type: str = "t5-small",
        encoder_path: str = "",
        output_dim: int = 512,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.encoder_path = encoder_path
        self.output_dim = output_dim
        self.tokenizer = None
        self.text_model = None
        self.projection = None

        if encoder_type == "t5-small":
            self._init_t5(encoder_path, output_dim)
            self._try_load_t5small_precomp(encoder_path)
        elif encoder_type == "precomputed":
            logger.info("Using precomputed text embeddings; no text model loaded.")
        else:
            raise ValueError(f"Unsupported text encoder type: {encoder_type}")

    # -- initializers -------------------------------------------------------

    def _init_t5(self, path: str, output_dim: int) -> None:
        from transformers import T5EncoderModel, T5Tokenizer

        model_name = path if path else "t5-small"
        logger.info("Loading T5 text encoder from %s", model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.text_model = T5EncoderModel.from_pretrained(model_name)
        self.text_model.eval()
        for param in self.text_model.parameters():
            param.requires_grad = False
        t5_hidden = self.text_model.config.d_model
        self.projection = nn.Linear(t5_hidden, output_dim)


    def _try_load_t5small_precomp(self, path: str) -> None:
        import pickle
        candidates = [
            os.path.join("data", "pretrained_models", "t5-small", "t5small_text_embeddings.pkl"),
        ]
        if path and os.path.isdir(path):
            candidates.append(os.path.join(path, "t5small_text_embeddings.pkl"))
        for candidate in candidates:
            if os.path.isfile(candidate):
                logger.info(
                    "Loading precomputed T5-small embeddings from %s (skipping live T5 inference)",
                    candidate,
                )
                with open(candidate, "rb") as _f:
                    self._t5small_precomp_cache = pickle.load(_f)
                logger.info(
                    "T5-small precomp cache loaded: %d instructions",
                    len(self._t5small_precomp_cache),
                )
                return
        self._t5small_precomp_cache = None

    # -- forward / encode ---------------------------------------------------

    def encode(
        self,
        texts: Union[List[str], torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """Encode text strings or accept pre-computed embeddings.

        Args:
            texts: list of strings *or* pre-computed tensor [B, L, D].
            device: target device.

        Returns:
            Tensor of shape [B, L, D] where D == self.output_dim.
        """
        if self.encoder_type == "precomputed":
            if not isinstance(texts, torch.Tensor):
                raise ValueError(
                    "encoder_type='precomputed' expects a tensor, "
                    f"got {type(texts)}"
                )
            return texts.to(device)

        # If texts is already a tensor (e.g. from native backbone text encoder),
        # return directly — the fusion layer has its own text_proj for dim adaptation.
        if isinstance(texts, torch.Tensor):
            return texts.to(device)

        # T5-small precomputed cache lookup (used by V-JEPA backbone)
        precomp = getattr(self, "_t5small_precomp_cache", None)
        if precomp is not None and self.encoder_type == "t5-small":
            TEXT_LEN = 77
            B = len(texts)
            out = torch.zeros(B, TEXT_LEN, self.output_dim, dtype=torch.float32, device=device)
            for i, instr in enumerate(texts):
                if instr in precomp:
                    emb = precomp[instr].to(device=device, dtype=torch.float32)  # [77, 512]
                    if self.projection is not None and emb.shape[-1] != self.output_dim:
                        proj_dev = self.projection.weight.device
                        out[i] = self.projection(emb.to(proj_dev)).to(device)
                    else:
                        out[i] = emb
                else:
                    logger.warning("T5-small precomp cache miss for: %r -- using zeros.", instr)
            return out

        if self.tokenizer is None or self.text_model is None:
            raise RuntimeError("Text model not initialized.")

        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(device)

        with torch.no_grad():
            hidden = self.text_model(**tokens).last_hidden_state  # [B, L, H]

        projected = self.projection(hidden)  # [B, L, output_dim]
        return projected

    def forward(
        self,
        texts: Union[List[str], torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        return self.encode(texts, device)

