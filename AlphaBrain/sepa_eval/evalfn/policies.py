"""
policies.py — pluggable policy_fn factories for real-simulator eval_fn.

A *policy_fn* maps (obs: dict, instruction: str, model_id: str) -> action.
Three flavours are provided:

  * :func:`make_random_policy_fn`   — uniform random actions (fast smoke / SR=0 baseline)
  * :func:`make_qwenoft_policy_fn`  — in-process Qwenvl_OFT model (torch/AlphaBrain imported lazily)
  * :func:`make_ws_policy_fn`       — thin WebSocket JSON client for a remote policy server
  * :func:`resolve_policy_fn`       — parse a CLI spec ("random" | "model[:path]" | "ws:<uri>")

None of the heavyweight dependencies (torch, transformers, websocket-client)
are imported at module import time; factories fail with a clear error at call
time when their dependency is missing.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Callable

logger = logging.getLogger(__name__)

#: policy_fn contract: (obs, instruction, model_id) -> action
PolicyFn = Callable[[dict, str, str], Any]

DEFAULT_ACTION_DIM = 7


# ---------------------------------------------------------------------------
# Random policy
# ---------------------------------------------------------------------------


def make_random_policy_fn(action_dim: int = DEFAULT_ACTION_DIM, seed: int = 0) -> PolicyFn:
    """Uniform random actions in [-1, 1]; last dim is a binary gripper command."""
    rng = random.Random(seed)

    def policy_fn(obs: dict, instruction: str, model_id: str) -> list[float]:
        action = [rng.uniform(-1.0, 1.0) for _ in range(action_dim - 1)]
        action.append(rng.choice([-1.0, 1.0]))
        return action

    return policy_fn


# ---------------------------------------------------------------------------
# In-process QwenOFT model policy
# ---------------------------------------------------------------------------


def make_qwenoft_policy_fn(
    base_vlm: str,
    action_dim: int = DEFAULT_ACTION_DIM,
    chunk_size: int = 8,
    image_key: str = "agentview_image",
    device: str | None = None,
) -> PolicyFn:
    """
    Build an in-process Qwenvl_OFT policy (see docs_analysis/scripts/qwenoft_test.py).

    Notes
    -----
    * ``attn_implementation="eager"`` is mandatory on Apple MPS (sdpa GQA crash).
    * The action head is randomly initialised unless a fine-tuned checkpoint is
      merged into the state dict — degraded-but-real behaviour by design.
    * torch / AlphaBrain are imported lazily; a missing dependency raises a
      clear RuntimeError at factory-call time, not at module import time.
    """
    try:
        import numpy as np
        import torch
        from omegaconf import OmegaConf
        from PIL import Image

        from AlphaBrain.model.framework.QwenOFT import Qwenvl_OFT
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise RuntimeError(
            f"make_qwenoft_policy_fn requires torch/AlphaBrain model deps: {exc}. "
            "Run inside the alphabrain conda env."
        ) from exc

    cfg = OmegaConf.create(
        {
            "framework": {
                "name": "QwenOFT",
                "qwenvl": {"base_vlm": base_vlm, "attn_implementation": "eager"},
                "action_model": {
                    "action_model_type": "L1RegressionActionHead",
                    "action_dim": action_dim,
                    "future_action_window_size": chunk_size - 1,
                    "past_action_window_size": 0,
                },
            },
            "datasets": {"vla_data": {"image_size": [224, 224]}},
        }
    )
    model = Qwenvl_OFT(cfg)
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device).eval()
    logger.info("QwenOFT policy loaded on %s (base_vlm=%s)", device, base_vlm)

    state: dict[str, list] = {"chunk": [], "instruction": None}

    def policy_fn(obs: dict, instruction: str, model_id: str) -> Any:
        if instruction != state["instruction"]:
            state["chunk"] = []
            state["instruction"] = instruction
        if not state["chunk"]:
            img = Image.fromarray(obs[image_key][::-1].copy())
            with torch.no_grad():
                out = model.predict_action(batch_images=[[img]], instructions=[instruction])
            acts = np.clip(out["normalized_actions"][0], -1, 1)
            state["chunk"] = list(acts)
        return np.asarray(state["chunk"].pop(0), dtype=np.float64)

    return policy_fn


# ---------------------------------------------------------------------------
# WebSocket policy client
# ---------------------------------------------------------------------------


def make_ws_policy_fn(
    uri: str,
    image_key: str = "agentview_image",
    timeout_s: float = 30.0,
) -> PolicyFn:
    """
    Thin JSON-over-WebSocket policy client.

    Protocol (intentionally simple; adapt server-side as needed):
      request : {"instruction": str, "model_id": str, "image_b64": str|null,
                 "image_shape": [h, w, c]|null}
      response: {"action": [..]}  or  {"actions": [[..], ..]} (chunk; consumed FIFO)
    """
    try:
        import websocket  # websocket-client
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise RuntimeError(f"make_ws_policy_fn requires the 'websocket-client' package: {exc}") from exc

    import base64
    import json

    conn = websocket.create_connection(uri, timeout=timeout_s)
    chunk: list = []

    def policy_fn(obs: dict, instruction: str, model_id: str) -> Any:
        if chunk:
            return chunk.pop(0)
        image_b64 = None
        image_shape = None
        img = obs.get(image_key) if isinstance(obs, dict) else None
        if img is not None and hasattr(img, "tobytes"):
            image_b64 = base64.b64encode(img.tobytes()).decode("ascii")
            image_shape = list(getattr(img, "shape", []))
        conn.send(
            json.dumps(
                {
                    "instruction": instruction,
                    "model_id": model_id,
                    "image_b64": image_b64,
                    "image_shape": image_shape,
                }
            )
        )
        reply = json.loads(conn.recv())
        if "actions" in reply:
            chunk.extend(reply["actions"])
            return chunk.pop(0)
        return reply["action"]

    return policy_fn


# ---------------------------------------------------------------------------
# CLI spec parsing
# ---------------------------------------------------------------------------


def resolve_policy_fn(spec: str, **kwargs: Any) -> tuple[PolicyFn, str]:
    """
    Parse a CLI policy spec into (policy_fn, default_model_id).

    Specs:
      "random"          -> random policy               (model id "random-policy")
      "model"           -> QwenOFT with $PRETRAINED_MODELS_DIR/Qwen2.5-VL-3B-Instruct
      "model:<path>"    -> QwenOFT with the given base VLM path
      "ws:<uri>"        -> WebSocket client to <uri>   (model id "ws-policy")
    """
    spec = (spec or "random").strip()
    if spec == "random":
        return make_random_policy_fn(**kwargs), "random-policy"
    if spec == "model" or spec.startswith("model:"):
        base_vlm = spec.partition(":")[2]
        if not base_vlm:
            import os

            base_vlm = os.path.join(
                os.environ.get("PRETRAINED_MODELS_DIR", "./pretrained_models"),
                "Qwen2.5-VL-3B-Instruct",
            )
        return make_qwenoft_policy_fn(base_vlm=base_vlm, **kwargs), "qwenoft-inprocess"
    if spec.startswith("ws:"):
        uri = spec[3:]
        if not uri.startswith("ws"):
            uri = "ws:" + uri  # allow ws://host:port passed as "ws:ws://..." or bare "//host"
        return make_ws_policy_fn(uri=uri, **kwargs), "ws-policy"
    raise ValueError(f"Unknown policy spec '{spec}'. Use 'random', 'model[:path]' or 'ws:<uri>'.")
