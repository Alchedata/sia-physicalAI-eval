"""
SemanticCritic — VLM-based judge for VLA rollout task completion.

Routes to the local /judge endpoint by default, or to GPT-4o-mini Vision as a
fallback when the evaluating model_id matches circular_bias_patterns (e.g. when
the VLA being scored and the judge share the same backbone, which inflates scores).

Default circular_bias_patterns: ["Qwen*", "qwen*"]  (configurable via critics.yaml)
"""
from __future__ import annotations

import base64
import fnmatch
import json
import typing
from dataclasses import dataclass
from typing import Any


@dataclass
class CriticResult:
    """Structured output from a single judge call."""

    completion: float        # 0.0 - 1.0 task completion score
    object_correct: bool     # did the robot interact with the correct object?
    collateral_damage: bool  # did the robot disturb objects it should not have?
    explanation: str         # free-text reasoning from the judge


class CriticError(Exception):
    """Raised on non-recoverable judge failures (5xx, network, parse errors)."""


class SemanticCritic:
    """
    VLM-based task-completion judge.

    Routes to ``/judge`` on a local policy server (``default_model="local"``)
    or to GPT-4o-mini Vision (``default_model="gpt-4o-mini"``).

    Circular-bias routing: if the *model being evaluated* matches any pattern in
    ``circular_bias_patterns``, the judge falls back to ``fallback_model`` to avoid
    self-serving scores.
    """

    _DEFAULT_CIRCULAR_BIAS_PATTERNS: typing.ClassVar[list[str]] = ["Qwen*", "qwen*"]

    def __init__(
        self,
        default_model: str = "local",
        fallback_model: str = "gpt-4o-mini",
        server_host: str = "127.0.0.1",
        server_port: int = 10092,
        circular_bias_patterns: list[str] | None = None,
    ) -> None:
        self.default_model = default_model
        self.fallback_model = fallback_model
        self.server_host = server_host
        self.server_port = server_port
        self.circular_bias_patterns: list[str] = (
            circular_bias_patterns
            if circular_bias_patterns is not None
            else list(self._DEFAULT_CIRCULAR_BIAS_PATTERNS)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def judge(
        self,
        frames: list[Any],
        instruction: str,
        model_id: str | None = None,
    ) -> CriticResult:
        """
        Judge task completion given a list of frames and a task instruction.

        Parameters
        ----------
        frames:
            List of image arrays (numpy uint8 HxWxC) or raw PNG bytes.
        instruction:
            Natural-language task instruction that the robot was supposed to execute.
        model_id:
            ID of the VLA model that produced the rollout being judged.
            Used to detect circular bias and route to fallback when needed.

        Returns
        -------
        CriticResult

        Raises
        ------
        CriticError
            On HTTP 5xx responses or network failures.
        """
        if self._should_use_fallback(model_id):
            return self._call_gpt4o_mini(frames, instruction)
        if self.default_model == "local":
            return self._call_local_judge(frames, instruction)
        return self._call_gpt4o_mini(frames, instruction)

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------

    def _should_use_fallback(self, model_id: str | None) -> bool:
        """Return True if model_id matches any circular_bias_pattern (fnmatch/glob)."""
        if not model_id:
            return False
        for pattern in self.circular_bias_patterns:
            if fnmatch.fnmatch(model_id, pattern):
                return True
        return False

    # ------------------------------------------------------------------
    # Backend: local /judge endpoint  (PRD §8.3)
    # ------------------------------------------------------------------

    def _call_local_judge(self, frames: list[Any], instruction: str) -> CriticResult:
        """POST to http://{host}:{port}/judge and parse the JSON response."""
        import requests  # lazy import — not needed in all deployments

        url = f"http://{self.server_host}:{self.server_port}/judge"
        payload = {
            "instruction": instruction,
            "frames": [_encode_frame(f) for f in frames],
        }
        try:
            response = requests.post(url, json=payload, timeout=60)
        except requests.exceptions.RequestException as exc:
            raise CriticError(f"Network failure calling local /judge: {exc}") from exc

        if response.status_code >= 500:
            raise CriticError(
                f"Local judge server returned {response.status_code}: {response.text}"
            )
        if response.status_code >= 400:
            raise CriticError(
                f"Local judge request error {response.status_code}: {response.text}"
            )

        return _parse_judge_response(response.json())

    # ------------------------------------------------------------------
    # Backend: OpenAI GPT-4o-mini Vision
    # ------------------------------------------------------------------

    def _call_gpt4o_mini(self, frames: list[Any], instruction: str) -> CriticResult:
        """Call GPT-4o-mini Vision via the OpenAI SDK (lazy import)."""
        import openai  # lazy import

        client = openai.OpenAI()

        # Build content list: instruction text followed by the frames
        content: list[dict] = [
            {
                "type": "text",
                "text": (
                    f"You are a robotics task evaluator. "
                    f"The robot was instructed: '{instruction}'. "
                    "Evaluate whether the task was completed. "
                    "Respond in JSON with keys: "
                    "'completion' (float 0-1), "
                    "'object_correct' (bool), "
                    "'collateral_damage' (bool), "
                    "'explanation' (string)."
                ),
            }
        ]
        for frame in frames:
            b64 = _encode_frame(frame)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"},
                max_tokens=512,
            )
        except openai.OpenAIError as exc:
            raise CriticError(f"GPT-4o-mini Vision call failed: {exc}") from exc

        raw_text = response.choices[0].message.content or "{}"
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise CriticError(f"Could not parse GPT-4o-mini response as JSON: {exc}") from exc

        return _parse_judge_response(data)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _encode_frame(frame: Any) -> str:
    """
    Encode a single frame to a base64 string.

    Accepts:
    - bytes / bytearray : treated as raw PNG bytes
    - numpy array       : encoded as PNG via cv2 or PIL (lazy import)
    """
    if isinstance(frame, (bytes, bytearray)):
        return base64.b64encode(frame).decode("ascii")

    # Assume numpy-like array; try cv2 first, then PIL
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        success, buf = cv2.imencode(".png", np.asarray(frame))
        if not success:
            raise CriticError("cv2.imencode failed for frame")
        return base64.b64encode(buf.tobytes()).decode("ascii")
    except ImportError:
        pass

    try:
        import io

        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore

        img = Image.fromarray(np.asarray(frame))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except ImportError as exc:
        raise CriticError(
            "Cannot encode numpy frame: neither cv2 nor PIL is available."
        ) from exc


def _parse_judge_response(data: dict) -> CriticResult:
    """Parse a judge JSON dict into a CriticResult, with safe defaults."""
    try:
        return CriticResult(
            completion=float(data.get("completion", 0.0)),
            object_correct=bool(data.get("object_correct", False)),
            collateral_damage=bool(data.get("collateral_damage", False)),
            explanation=str(data.get("explanation", "")),
        )
    except (TypeError, ValueError) as exc:
        raise CriticError(f"Malformed judge response fields: {exc}") from exc
