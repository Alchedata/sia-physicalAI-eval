"""
robocasa_trace_hook.py — BenchmarkAdapter wrapping a RoboCasa simulation env.

The actual RoboCasa / robosuite library is not required at import time;
RobocasaHook accepts any duck-typed object that exposes the RoboCasa env
interface.
"""
from __future__ import annotations

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
# RobocasaHook — BenchmarkAdapter
# ---------------------------------------------------------------------------

class RobocasaHook:
    """
    BenchmarkAdapter wrapping a RoboCasa simulation environment.

    Parameters
    ----------
    env:
        A duck-typed RoboCasa / robosuite env object.  Required interface::

            env.reset() -> obs_dict | None
            env.step(action) -> (obs_dict, reward, done, info)

    task_id:
        Stable string identifier for the current task.
    env_name:
        The RoboCasa environment class name (e.g. ``"KitchenTabletop"``).
        Stored in scene_config and used for provenance.
    scene_config:
        Optional dict with extra simulator / scene initialisation parameters.
        When None, a minimal dict containing ``env_name`` is stored.
    """

    PROTOCOL_VERSION: str = "1.0"

    def __init__(
        self,
        env: Any,
        task_id: str,
        env_name: str,
        scene_config: dict | None = None,
    ) -> None:
        self._env = env
        self._task_id = task_id
        self._env_name = env_name

        if scene_config is not None:
            self._scene_config: dict = scene_config
        else:
            self._scene_config = {"env_name": env_name}

    # ------------------------------------------------------------------
    # BenchmarkAdapter protocol
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        """
        Reset the RoboCasa env and return the initial observation dict.

        RoboCasa / robosuite envs return the obs dict directly from reset().
        """
        obs = self._env.reset()
        if obs is None:
            # Some wrappers expose get_obs() as a fallback.
            if hasattr(self._env, "get_obs"):
                obs = self._env.get_obs()
            else:
                obs = {}
        return obs if isinstance(obs, dict) else {"obs": obs}

    def step(self, action: Any) -> tuple:
        """
        Step the RoboCasa env.

        Returns
        -------
        (obs: dict, done: bool, info: dict)

        RoboCasa envs return (obs, reward, done, info); the reward scalar is
        dropped to match the BenchmarkAdapter protocol.
        """
        result = self._env.step(action)

        # Handle (obs, reward, done, info) and (obs, done, info) variants.
        if len(result) == 4:
            obs, _reward, done, info = result
        elif len(result) == 3:
            obs, done, info = result
        else:
            raise ValueError(
                f"Unexpected step() return length {len(result)} from RoboCasa env."
            )

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
# Helper: run one full RoboCasa episode with trace collection
# ---------------------------------------------------------------------------

def run_robocasa_episode_with_trace(
    env: Any,
    env_name: str,
    policy_fn: Callable[[dict], Any],
    memory: "EvalMemory",
    identity: TraceIdentity,
    store_obs: bool = False,
) -> EpisodeTrace:
    """
    Run one RoboCasa episode end-to-end with trace collection.

    Parameters
    ----------
    env:
        A duck-typed RoboCasa env (same contract as RobocasaHook.__init__).
    env_name:
        RoboCasa environment class name; forwarded to RobocasaHook.
    policy_fn:
        Callable that maps an observation dict to an action::

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
    adapter = RobocasaHook(
        env=env,
        task_id=identity.task_id,
        env_name=env_name,
    )

    scene_cfg_dict = adapter.get_scene_config()
    scene = SceneConfig(
        scene_config=scene_cfg_dict,
        init_state=b"",
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

        # RoboCasa convention: success flag lives in info dict.
        success = bool(info.get("success", False))
        trace = hook.on_episode_end(success=success)

    return trace
