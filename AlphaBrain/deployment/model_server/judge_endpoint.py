# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

"""
HTTP ``POST /judge`` endpoint for the SEPA-Eval semantic critic (PRD §8.3).

Runs a small stdlib :mod:`http.server` alongside the WebSocket policy server so
the already-loaded VLM backbone can be reused in *generation* mode as a rollout
judge. No heavyweight web framework is required.

Client contract (see ``sepa_eval/critics/semantic_critic.py``):

Request JSON::

    {"frames": ["<base64 PNG>", ...], "instruction": "Pick up the mug",
     "prompt": "<optional custom prompt>"}

(``"images"`` is accepted as an alias for ``"frames"`` per PRD §8.3.)

Response JSON (200)::

    {"text": "<raw model output>", "completion": 0.85, "completion_score": 0.85,
     "object_correct": true, "collateral_damage": false, "explanation": "..."}

Errors return ``{"error": "..."}`` with HTTP 400 (bad request), 501 (the loaded
model does not support chat/vision generation) or 500 (judge call failed), so
``SemanticCritic`` can raise ``CriticError`` and fall back to GPT-4o-mini.
"""
from __future__ import annotations

import base64
import binascii
import io
import json
import logging
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

logger = logging.getLogger(__name__)

MAX_FRAMES = 10

DEFAULT_JUDGE_PROMPT = (
    "You are a robotics task evaluator. The robot was instructed: '{instruction}'. "
    "Looking at the frames from the rollout, evaluate whether the task was completed. "
    "Respond in JSON with keys: 'completion' (float 0-1), 'object_correct' (bool), "
    "'collateral_damage' (bool), 'explanation' (string)."
)


class JudgeNotSupportedError(Exception):
    """Raised when the loaded framework has no usable chat/vision interface."""


# ---------------------------------------------------------------------------
# Model invocation
# ---------------------------------------------------------------------------


def _decode_frames(b64_frames: list[str]) -> list[Any]:
    """Decode base64 PNG strings into PIL Images (or raw bytes if PIL is absent)."""
    raw: list[bytes] = []
    for entry in b64_frames[:MAX_FRAMES]:
        if not isinstance(entry, str):
            raise ValueError("each frame must be a base64-encoded string")
        try:
            raw.append(base64.b64decode(entry, validate=True))
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"invalid base64 frame: {exc}") from exc
    try:
        from PIL import Image  # lazy import
    except ImportError:
        return raw
    images: list[Any] = []
    for buf in raw:
        try:
            images.append(Image.open(io.BytesIO(buf)).convert("RGB"))
        except Exception as exc:
            raise ValueError(f"could not decode frame as an image: {exc}") from exc
    return images


def _generate_judgment(framework: Any, images: list[Any], prompt: str) -> str:
    """
    Call the loaded framework in text-generation mode.

    Duck-typed resolution order:

    1. ``framework.judge(images, prompt)``          -> str
    2. ``framework.generate_text(images, prompt)``  -> str
    3. Qwen-style VLM interface: ``framework.qwen_vl_interface`` exposing
       ``.model.generate`` and ``.processor`` (chat-template pipeline)

    Raises ``JudgeNotSupportedError`` if none is available.
    """
    judge_fn = getattr(framework, "judge", None)
    if callable(judge_fn):
        return str(judge_fn(images, prompt))

    gen_fn = getattr(framework, "generate_text", None)
    if callable(gen_fn):
        return str(gen_fn(images, prompt))

    interface = getattr(framework, "qwen_vl_interface", None)
    model = getattr(interface, "model", None)
    processor = getattr(interface, "processor", None)
    if interface is not None and callable(getattr(model, "generate", None)) and processor is not None:
        return _generate_via_chat_template(model, processor, images, prompt)

    raise JudgeNotSupportedError(
        f"Loaded framework {type(framework).__name__!r} exposes no judge/generate_text/"
        "chat-capable VLM interface; use the fallback judge instead."
    )


def _generate_via_chat_template(model: Any, processor: Any, images: list[Any], prompt: str) -> str:
    """Run a HuggingFace-style chat-template generation pass on a Qwen-like VLM."""
    import torch  # lazy import — only needed on real model servers

    content: list[dict[str, Any]] = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images or None, return_tensors="pt")
    inputs = inputs.to(model.device)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    trimmed = [out[len(inp) :] for inp, out in zip(inputs["input_ids"], generated, strict=False)]
    decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return decoded[0] if decoded else ""


def _parse_model_text(text: str) -> dict[str, Any]:
    """Best-effort extraction of the judge JSON from raw model output."""
    data: dict[str, Any] = {}
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            candidate = json.loads(match.group(0))
            if isinstance(candidate, dict):
                data = candidate
        except json.JSONDecodeError:
            data = {}

    try:
        completion = float(data.get("completion", data.get("completion_score", 0.0)))
    except (TypeError, ValueError):
        completion = 0.0
    completion = min(1.0, max(0.0, completion))
    return {
        "text": text,
        "completion": completion,
        "completion_score": completion,
        "object_correct": bool(data.get("object_correct", False)),
        "collateral_damage": bool(data.get("collateral_damage", False)),
        "explanation": str(data.get("explanation", "")) or text,
    }


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


class _JudgeRequestHandler(BaseHTTPRequestHandler):
    """Handles ``POST /judge``. The framework is attached to the server instance."""

    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.debug("judge http: " + fmt, *args)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": f"unknown path {self.path!r}"})

    def do_POST(self) -> None:
        if self.path != "/judge":
            self._send_json(404, {"error": f"unknown path {self.path!r}"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": f"invalid JSON body: {exc}"})
            return
        if not isinstance(payload, dict):
            self._send_json(400, {"error": "request body must be a JSON object"})
            return

        instruction = payload.get("instruction")
        frames_b64 = payload.get("frames", payload.get("images"))
        if not isinstance(instruction, str) or not instruction.strip():
            self._send_json(400, {"error": "missing required field 'instruction'"})
            return
        if not isinstance(frames_b64, list):
            self._send_json(400, {"error": "missing required field 'frames' (list of base64 PNG strings)"})
            return

        try:
            images = _decode_frames(frames_b64)
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return

        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            prompt = DEFAULT_JUDGE_PROMPT.format(instruction=instruction)

        framework = self.server.framework  # type: ignore[attr-defined]
        try:
            text = _generate_judgment(framework, images, prompt)
        except JudgeNotSupportedError as exc:
            self._send_json(501, {"error": str(exc), "fallback": True})
            return
        except Exception as exc:
            logger.exception("judge generation failed")
            self._send_json(500, {"error": f"judge generation failed: {exc}"})
            return

        self._send_json(200, _parse_model_text(text))


class _JudgeHTTPServerImpl(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], framework: Any) -> None:
        super().__init__(address, _JudgeRequestHandler)
        self.framework = framework


class JudgeServer:
    """Threaded HTTP server exposing ``POST /judge`` for a loaded framework."""

    def __init__(self, framework: Any, host: str = "0.0.0.0", port: int = 10092) -> None:
        self._server = _JudgeHTTPServerImpl((host, port), framework)
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        """Actual bound port (useful when constructed with ``port=0``)."""
        return int(self._server.server_address[1])

    def start(self) -> "JudgeServer":
        """Start serving in a daemon thread. Returns self for chaining."""
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._server.serve_forever, name="sepa-judge-http", daemon=True)
        self._thread.start()
        logger.info("SEPA /judge endpoint listening on port %d", self.port)
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None


def maybe_start_judge_server(framework: Any, judge_port: int | None, host: str = "0.0.0.0") -> JudgeServer | None:
    """
    Start the /judge endpoint if a port is configured.

    ``judge_port`` may come from ``--judge_port`` or the ``SEPA_JUDGE_PORT`` env
    var; ``None``/``0`` (unset) disables the endpoint and preserves existing
    server behavior.
    """
    if not judge_port:
        return None
    server = JudgeServer(framework, host=host, port=int(judge_port))
    return server.start()
