"""
libero_trace_hook.py — BenchmarkAdapter wrapping a LIBERO eval environment.

The actual LIBERO library is not required at import time; LiberoHook accepts
any duck-typed object that exposes the LIBERO env interface.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from sepa_eval.hooks.base import TraceHook
from sepa_eval.memory.schema import (
    EpisodeTrace,
    SceneConfig,
    TraceIdentity,
)

if TYPE_CHECKING:
    from sepa_eval.memory import EvalMemory


# ---------------------------------------------------------------------------
# LiberoHook — BenchmarkAdapter
# ---------------------------------------------------------------------------


class LiberoHook:
    """
    BenchmarkAdapter wrapping a LIBERO eval environment.

    Parameters
    ----------
    env:
        A duck-typed LIBERO env object.  Required interface::

            env.reset() -> obs_dict
            env.step(action) -> (obs_dict, reward, done, info)
            env.task_id     -> str
            env.get_obs()   -> obs_dict   (optional; used as fallback)

    task_id:
        Stable string identifier for the task being evaluated.  If the
        env exposes ``env.task_id`` directly you may pass the same value;
        having an explicit parameter allows override without mutating the
        env object.
    scene_config:
        Optional dict with simulator / scene initialisation parameters.
        When None, an empty dict is stored.
    """

    PROTOCOL_VERSION: str = "1.0"

    def __init__(
        self,
        env: Any,
        task_id: str,
        scene_config: dict | None = None,
        base_init_state: Any = None,
        replay_on_reset: bool = False,
    ) -> None:
        self._env = env
        self._task_id = task_id
        self._scene_config: dict = scene_config if scene_config is not None else {}
        self._base_init_state = base_init_state
        self._replay_on_reset = replay_on_reset
        self._last_replay_info: dict | None = None

    # ------------------------------------------------------------------
    # BenchmarkAdapter protocol
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        """Reset the LIBERO env and return the initial observation dict.

        When ``replay_on_reset`` is enabled and a mutated ``scene_config`` is
        present, the scene config is replayed into the simulator via
        ``sepa_eval.replay.replay_scene_config`` (set_init_state).  Replay
        failures degrade gracefully to a plain reset ("reseed" mode) and are
        recorded in ``self.last_replay_info``.
        """
        obs = self._env.reset()
        # Some LIBERO env implementations return None from reset() and
        # expose get_obs() separately.
        if obs is None and hasattr(self._env, "get_obs"):
            obs = self._env.get_obs()

        if self._replay_on_reset and self._scene_config:
            from sepa_eval.replay import ReplayError, replay_scene_config

            try:
                replay_obs, info = replay_scene_config(
                    self._env,
                    self._scene_config,
                    base_init_state=self._base_init_state,
                )
                self._last_replay_info = info
                if replay_obs is not None:
                    obs = replay_obs
            except ReplayError as exc:
                logging.getLogger(__name__).warning(
                    "Scene-config replay failed for task %s; falling back to reseed reset: %s",
                    self._task_id,
                    exc,
                )
                self._last_replay_info = {"error": str(exc), "mode": "reseed_fallback"}

        return obs if isinstance(obs, dict) else {"obs": obs}

    @property
    def last_replay_info(self) -> dict | None:
        """Bookkeeping from the most recent scene-config replay (or None)."""
        return self._last_replay_info

    def step(self, action: Any) -> tuple:
        """
        Step the LIBERO env.

        Returns
        -------
        (obs: dict, done: bool, info: dict)

        LIBERO envs return (obs, reward, done, info); we drop the reward
        scalar to match the BenchmarkAdapter protocol.
        """
        result = self._env.step(action)

        # Handle both (obs, reward, done, info) and (obs, done, info).
        if len(result) == 4:
            obs, _reward, done, info = result
        elif len(result) == 3:
            obs, done, info = result
        else:
            raise ValueError(f"Unexpected step() return length {len(result)} from LIBERO env.")

        if not isinstance(obs, dict):
            obs = {"obs": obs}
        if not isinstance(info, dict):
            info = {}
        return obs, bool(done), info

    def get_task_id(self) -> str:
        return self._task_id

    def get_scene_config(self) -> dict:
        return dict(self._scene_config)


# ---------------------------------------------------------------------------
# Helper: run one full LIBERO episode with trace collection
# ---------------------------------------------------------------------------


def run_libero_episode_with_trace(
    env: Any,
    policy_fn: Callable[[dict], Any],
    memory: "EvalMemory",
    identity: TraceIdentity,
    store_obs: bool = False,
) -> EpisodeTrace:
    """
    Run one LIBERO episode end-to-end with trace collection.

    Parameters
    ----------
    env:
        A duck-typed LIBERO env (same contract as LiberoHook.__init__).
    policy_fn:
        Callable that maps an observation dict to an action.  Signature::

            action = policy_fn(obs: dict) -> Any

    memory:
        An EvalMemory instance used to persist the finished trace.
    identity:
        TraceIdentity describing this eval run / trace.
    store_obs:
        Passed through to TraceHook.  See TraceHook docstring.

    Returns
    -------
    EpisodeTrace
        The completed trace (already persisted via memory.record_trace()).
    """
    adapter = LiberoHook(
        env=env,
        task_id=identity.task_id,
    )

    # Build a minimal SceneConfig.  Callers wanting a full state snapshot
    # should pass a pre-built SceneConfig via identity; here we collect
    # what we can from the adapter.
    scene_cfg_dict = adapter.get_scene_config()
    scene = SceneConfig(
        scene_config=scene_cfg_dict,
        init_state=b"",  # simulator snapshot not available at this layer
        replay_mode="reseed",
    )

    with TraceHook(memory=memory, identity=identity, scene=scene, store_obs=store_obs) as hook:
        obs = adapter.reset()
        done = False
        info: dict = {}

        while not done:
            action = policy_fn(obs)
            obs, done, info = adapter.step(action)
            hook.on_step(obs, action)

        # Determine success from info dict (LIBERO convention).
        success = bool(info.get("success", False))
        trace = hook.on_episode_end(success=success)

    return trace
