"""
base.py — BenchmarkAdapter protocol and TraceHook base class for SEPA-Eval.
"""
from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from sepa_eval.memory.schema import (
    EpisodeTrace,
    RolloutData,
    SceneConfig,
    TraceIdentity,
    TraceLabels,
    TaskProvenance,
)

if TYPE_CHECKING:
    from sepa_eval.memory import EvalMemory


# ---------------------------------------------------------------------------
# BenchmarkAdapter protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class BenchmarkAdapter(Protocol):
    """
    Thin wrapper protocol that any benchmark env must satisfy.
    Adapters must declare PROTOCOL_VERSION to allow version-pinning checks.
    """

    PROTOCOL_VERSION: str  # e.g. "1.0"

    def reset(self) -> dict:
        """Return the initial observation dict."""
        ...

    def step(self, action: Any) -> tuple:
        """
        Advance the environment by one step.

        Returns
        -------
        (obs: dict, done: bool, info: dict)
        """
        ...

    def get_task_id(self) -> str:
        """Return a stable string identifier for the current task."""
        ...

    def get_scene_config(self) -> dict:
        """Return the current scene/environment configuration dict."""
        ...


# ---------------------------------------------------------------------------
# TraceHook base class
# ---------------------------------------------------------------------------

class TraceHook:
    """
    Context manager that accumulates per-step (obs, action) pairs during an
    episode and builds a complete EpisodeTrace on on_episode_end().

    No file I/O is performed inside the episode loop.  All persistence is
    delegated to memory.record_trace() inside on_episode_end().

    Parameters
    ----------
    memory:
        An EvalMemory instance used to persist the finished trace.
    identity:
        TraceIdentity describing this particular eval run / trace.
    scene:
        SceneConfig capturing the simulator state snapshot.
    store_obs:
        If False (default): only the last 10 observations are stored
        (failure-window mode).  Images are kept raw.
        If True: every observation is kept; images are batch-compressed
        to PNG bytes in on_episode_end().
    """

    def __init__(
        self,
        memory: "EvalMemory",
        identity: TraceIdentity,
        scene: SceneConfig,
        store_obs: bool = False,
    ) -> None:
        self._memory = memory
        self._identity = identity
        self._scene = scene
        self._store_obs = store_obs

        # Per-step accumulators — no I/O during the episode loop.
        self._obs_buffer: list[dict] = []
        self._action_buffer: list[Any] = []
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Episode-loop API
    # ------------------------------------------------------------------

    def on_step(self, obs: dict, action: Any) -> None:
        """
        Record one (obs, action) pair.

        This method must remain I/O-free; all it does is append to
        in-memory lists.
        """
        self._obs_buffer.append(obs)
        self._action_buffer.append(action)
        self._step_count += 1

    def on_episode_end(self, success: bool) -> EpisodeTrace:
        """
        Finalise the trace, persist it via memory.record_trace(), and
        return the completed EpisodeTrace.

        For store_obs=False  → only the last 10 observations are kept.
        For store_obs=True   → all observations; images are compressed
                                to PNG bytes at this point.
        """
        if self._store_obs:
            observations = _compress_images_in_obs_list(self._obs_buffer)
        else:
            # Failure-window: last 10 steps, images kept raw.
            observations = self._obs_buffer[-10:]

        # Determine first failure step (None for successes).
        failure_step: int | None = None
        if not success:
            # If fewer than 10 steps were recorded the failure window starts
            # at step 0; otherwise it starts at step (total - 10).
            failure_step = max(0, self._step_count - len(observations))

        rollout = RolloutData(
            observations=observations,
            actions=list(self._action_buffer),
            episode_length=self._step_count,
            success=success,
            failure_step=failure_step,
        )

        trace = EpisodeTrace(
            identity=self._identity,
            scene=self._scene,
            rollout=rollout,
            labels=TraceLabels(),
            provenance=TaskProvenance(),
        )

        self._memory.record_trace(trace)
        return trace

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "TraceHook":
        return self

    def __exit__(self, *args: Any) -> None:
        # No automatic flush — callers must call on_episode_end() explicitly.
        # This keeps failure-path semantics explicit.
        pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compress_images_in_obs_list(obs_list: list[dict]) -> list[dict]:
    """
    Return a copy of obs_list where every numpy image array under the
    'images' key is replaced with its PNG-encoded bytes.

    Safe to call even when numpy is absent (images are left as-is).
    """
    np = None
    _numpy_available = False
    try:
        import numpy as np  # optional dependency
        _numpy_available = True
    except ImportError:
        pass

    result: list[dict] = []
    for obs in obs_list:
        new_obs = dict(obs)
        if "images" in new_obs and isinstance(new_obs["images"], dict):
            compressed: dict[str, Any] = {}
            for cam_name, img in new_obs["images"].items():
                if _numpy_available and np is not None and isinstance(img, np.ndarray):
                    compressed[cam_name] = _encode_png(img)
                else:
                    # Already bytes or some other type — pass through.
                    compressed[cam_name] = img
            new_obs = dict(new_obs)
            new_obs["images"] = compressed
        result.append(new_obs)
    return result


def _encode_png(img_array: Any) -> bytes:
    """
    Encode a uint8 numpy array (H, W, C) as PNG bytes using only stdlib.

    Falls back to a raw bytes cast if the PIL/Pillow stack is unavailable,
    so the rest of the pipeline is never blocked by an optional dependency.
    """
    try:
        from PIL import Image  # type: ignore[import-untyped]

        buf = io.BytesIO()
        pil_img = Image.fromarray(img_array)
        pil_img.save(buf, format="PNG", optimize=False)
        return buf.getvalue()
    except ImportError:
        # PIL not available — store raw bytes so data is not lost.
        try:
            return img_array.tobytes()
        except AttributeError:
            return bytes(img_array)
