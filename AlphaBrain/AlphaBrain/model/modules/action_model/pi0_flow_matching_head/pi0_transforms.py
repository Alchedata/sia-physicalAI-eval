"""
Standalone transforms for Pi0/Pi0.5 inference — no openpi dependency.

Replaces:
  - openpi.transforms.Normalize / Unnormalize
  - openpi.transforms.ResizeImages
  - openpi.transforms.TokenizePrompt
  - openpi.transforms.PadStatesAndActions
  - openpi.policies.libero_policy.LiberoInputs
  - openpi.models.model.Observation
"""
import dataclasses
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ── Observation container (replaces openpi.models.model.Observation) ────────

@dataclasses.dataclass
class Observation:
    """Minimal replacement for openpi's Observation.

    Fields match what PI0Pytorch._preprocess_observation / sample_actions expects.
    """
    images: dict[str, "np.ndarray | torch.Tensor"]
    image_masks: dict[str, "np.ndarray | torch.Tensor"]
    state: "np.ndarray | torch.Tensor"
    tokenized_prompt: "np.ndarray | torch.Tensor | None" = None
    tokenized_prompt_mask: "np.ndarray | torch.Tensor | None" = None
    # pi0-fast fields (unused for pi0/pi05, kept for compat)
    token_ar_mask: "np.ndarray | torch.Tensor | None" = None
    token_loss_mask: "np.ndarray | torch.Tensor | None" = None

    @classmethod
    def from_dict(cls, data: dict) -> "Observation":
        """Build from flat dict (same contract as openpi).

        Expects keys: image (dict), image_mask (dict), state,
        and optionally tokenized_prompt / tokenized_prompt_mask.
        """
        import torch

        images = data["image"]
        # Auto-convert uint8 images to [-1, 1] float32
        for key in list(images.keys()):
            img = images[key]
            if isinstance(img, np.ndarray) and img.dtype == np.uint8:
                images[key] = img.astype(np.float32) / 255.0 * 2.0 - 1.0
            elif isinstance(img, torch.Tensor) and img.dtype == torch.uint8:
                images[key] = img.to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0

        return cls(
            images=images,
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
        )


# ── Norm stats ──────────────────────────────────────────────────────────────

@dataclasses.dataclass
class NormStats:
    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray | None = None
    q99: np.ndarray | None = None


def load_norm_stats(path: str | Path) -> dict[str, NormStats]:
    """Load norm_stats.json (openpi format) → dict of NormStats."""
    with open(path) as f:
        raw = json.load(f)
    # Handle nested {"norm_stats": {...}} wrapper
    if "norm_stats" in raw:
        raw = raw["norm_stats"]
    stats = {}
    for key, s in raw.items():
        stats[key] = NormStats(
            mean=np.array(s.get("mean", []), dtype=np.float32),
            std=np.array(s.get("std", [1.0]), dtype=np.float32),
            q01=np.array(s["q01"], dtype=np.float32) if "q01" in s else None,
            q99=np.array(s["q99"], dtype=np.float32) if "q99" in s else None,
        )
    return stats


# ── Normalize / Unnormalize ────────────────────────────────────────────────

def _pad_to_dim(x: np.ndarray, dim: int, value: float = 0.0) -> np.ndarray:
    """Zero-pad last dimension to `dim`."""
    if x.shape[-1] >= dim:
        return x
    pad_width = [(0, 0)] * (x.ndim - 1) + [(0, dim - x.shape[-1])]
    return np.pad(x, pad_width, constant_values=value)


def normalize_quantile(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """Quantile normalization to [-1, 1]."""
    q01 = q01[..., :x.shape[-1]]
    q99 = q99[..., :x.shape[-1]]
    return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def unnormalize_quantile(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """Inverse quantile normalization."""
    dim = q01.shape[-1]
    if dim < x.shape[-1]:
        return np.concatenate([
            (x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01,
            x[..., dim:],
        ], axis=-1)
    q01 = q01[..., :x.shape[-1]]
    q99 = q99[..., :x.shape[-1]]
    return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


# ── Image resize (matching openpi's resize_with_pad) ───────────────────────────────────

def _resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize image preserving aspect ratio with padding.

    Matches openpi's resize_with_pad behavior:
    - Bilinear interpolation
    - Pad with -1.0 for float images, 0 for uint8
    - For square→square (e.g. 256→224), equivalent to plain resize
    """
    import cv2

    is_float = image.dtype in (np.float32, np.float64)
    cur_h, cur_w = image.shape[:2]
    if cur_h == height and cur_w == width:
        return image

    ratio = max(cur_w / width, cur_h / height)
    new_h = int(cur_h / ratio)
    new_w = int(cur_w / ratio)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if is_float:
        resized = np.clip(resized, -1.0, 1.0)
    else:
        resized = np.clip(np.round(resized), 0, 255).astype(np.uint8)

    # Pad to target size
    pad_h0 = (height - new_h) // 2
    pad_h1 = height - new_h - pad_h0
    pad_w0 = (width - new_w) // 2
    pad_w1 = width - new_w - pad_w0

    pad_val = -1.0 if is_float else 0
    if image.ndim == 3:
        padded = np.pad(resized, ((pad_h0, pad_h1), (pad_w0, pad_w1), (0, 0)), constant_values=pad_val)
    else:
        padded = np.pad(resized, ((pad_h0, pad_h1), (pad_w0, pad_w1)), constant_values=pad_val)

    return padded


# ── Tokenizer ──────────────────────────────────────────────────────────────

class PaliGemmaTokenizer:
    """Standalone PaliGemma tokenizer (sentencepiece).

    Compatible with openpi's tokenizer output format.
    """

    # Search paths for the tokenizer model file.
    # Override via env var PALIGEMMA_TOKENIZER_MODEL; otherwise uses openpi default cache.
    import os as _os
    _SEARCH_PATHS = [
        _p for _p in [
            _os.environ.get("PALIGEMMA_TOKENIZER_MODEL"),
            _os.path.expanduser("~/.cache/openpi/big_vision/paligemma_tokenizer.model"),
        ] if _p
    ]

    def __init__(self, max_len: int = 200, model_path: str | None = None):
        import sentencepiece as spm

        self.max_len = max_len

        if model_path and Path(model_path).exists():
            self._sp = spm.SentencePieceProcessor(model_file=model_path)
            return

        for p in self._SEARCH_PATHS:
            if Path(p).exists():
                self._sp = spm.SentencePieceProcessor(model_file=p)
                logger.info(f"Loaded PaliGemma tokenizer from {p}")
                return

        raise FileNotFoundError(
            f"PaliGemma tokenizer not found. Searched: {self._SEARCH_PATHS}. "
            "Download from gs://big_vision/paligemma_tokenizer.model"
        )

    def tokenize(self, prompt: str, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Tokenize prompt, matching openpi format exactly.

        Returns (tokens, mask) each of shape (max_len,).
        """
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")

        if state is not None:
            # Pi0.5 discrete state format
            discretized = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
            state_str = " ".join(map(str, discretized))
            full_prompt = f"Task: {cleaned}, State: {state_str};\nAction: "
            tokens = self._sp.encode(full_prompt, add_bos=True)
        else:
            # Pi0 format: BOS + text + "\n"
            tokens = self._sp.encode(cleaned, add_bos=True) + self._sp.encode("\n")

        n = len(tokens)
        if n < self.max_len:
            mask = [True] * n + [False] * (self.max_len - n)
            tokens = tokens + [False] * (self.max_len - n)
        else:
            if n > self.max_len:
                logger.warning(f"Token length ({n}) exceeds max_len ({self.max_len}), truncating.")
            tokens = tokens[:self.max_len]
            mask = [True] * self.max_len

        return np.asarray(tokens), np.asarray(mask)


# ── Composite transform for LIBERO ────────────────────────────────────────

class LiberoTransform:
    """Complete input transform for LIBERO eval — replaces the entire openpi transform chain.

    Usage:
        transform = LiberoTransform(norm_stats_path="/.../norm_stats.json")
        processed = transform(raw_element)
        # processed is a dict ready for Observation.from_dict()
    """

    def __init__(
        self,
        norm_stats_path: str | Path,
        image_size: tuple[int, int] = (224, 224),
        max_token_len: int = 200,
        model_action_dim: int = 32,
        pi05: bool = True,
        tokenizer_path: str | None = None,
    ):
        self.norm_stats = load_norm_stats(norm_stats_path)
        self.image_size = image_size
        self.model_action_dim = model_action_dim
        self.pi05 = pi05
        self.tokenizer = PaliGemmaTokenizer(max_len=max_token_len, model_path=tokenizer_path)

    def __call__(self, data: dict) -> dict:
        """Transform raw LIBERO observation dict to model-ready dict.

        Input keys:
            observation/image: np.ndarray (H, W, 3) uint8
            observation/wrist_image: np.ndarray (H, W, 3) uint8
            observation/state: np.ndarray (D,)
            prompt: str

        Output keys match Observation.from_dict() contract.
        """
        # ── Step 1: LiberoInputs — restructure keys (keep uint8, matching openpi) ──
        base_image = np.asarray(data["observation/image"])
        wrist_image = np.asarray(data["observation/wrist_image"])

        # If float, convert to uint8 (matching openpi's _parse_image)
        if np.issubdtype(base_image.dtype, np.floating):
            base_image = (255 * base_image).astype(np.uint8)
        if np.issubdtype(wrist_image.dtype, np.floating):
            wrist_image = (255 * wrist_image).astype(np.uint8)
        # CHW → HWC if needed
        if base_image.shape[0] == 3:
            base_image = np.transpose(base_image, (1, 2, 0))
        if wrist_image.shape[0] == 3:
            wrist_image = np.transpose(wrist_image, (1, 2, 0))

        right_wrist = np.zeros_like(base_image)  # padding image (uint8 zeros)

        # ── Step 2: Normalize state (quantile) ──
        state = data["observation/state"].astype(np.float32)
        # Find matching norm_stats key
        for key, ns in self.norm_stats.items():
            if ns.q01 is not None:
                state = normalize_quantile(state, ns.q01[:state.shape[-1]], ns.q99[:state.shape[-1]])
                break

        # ── Step 3: Resize images (resize_with_pad, matching openpi) ──
        base_image = _resize_with_pad(base_image, *self.image_size)
        wrist_image = _resize_with_pad(wrist_image, *self.image_size)
        right_wrist = _resize_with_pad(right_wrist, *self.image_size)

        # ── Step 4: Tokenize prompt ──
        prompt = data.get("prompt", "")
        tokens, token_mask = self.tokenizer.tokenize(prompt)

        # ── Step 5: Pad state to model_action_dim ──
        state = _pad_to_dim(state, self.model_action_dim)

        # ── Build output dict ──
        return {
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,  # pi0.5: mask padding image
            },
            "state": state,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
        }
