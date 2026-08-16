"""
libero_eval_fn.py — real-simulator eval_fn factory for the SEPA-Eval promotion gates.

Gate contract (see ``sepa_eval.promotion.gates``)::

    sr = eval_fn(candidate, model_id, n_trials)   # -> float in [0, 1]

The orchestrator's EVALUATE step additionally calls the same callable as
``eval_fn(model_id=..., n_trials=...)`` (no candidate), so the returned
function accepts ``candidate=None`` and falls back to the base (unmutated)
task in that case.

For each trial the eval_fn:
  1. resets the LIBERO env and restores a deterministic per-trial init state,
  2. replays the candidate's mutated ``scene_config`` via
     ``sepa_eval.replay.replay_scene_config`` (skipped when empty),
  3. rolls out ``policy_fn(obs, instruction, model_id)`` for up to
     ``max_steps`` steps and records success (env ``done`` = task success).

LIBERO is **not** required at import time.  Tests inject an ``env_factory``;
real usage builds envs lazily and raises a clear RuntimeError when the LIBERO
library is missing.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Sequence

from sepa_eval.replay import ReplayError, replay_scene_config
from sepa_eval.replay.libero_replay import _libero_available

logger = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 60
DEFAULT_SETTLE_STEPS = 5
_NOOP_ACTION = [0.0] * 6 + [-1.0]


# ---------------------------------------------------------------------------
# LIBERO task / env resolution (lazy; only touched when env_factory is None)
# ---------------------------------------------------------------------------


def _load_init_states(task: Any) -> Any:
    """Load a task's init states directly (torch 2.6 needs weights_only=False)."""
    import torch
    from libero.libero import get_libero_path

    path = os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file)
    return torch.load(path, weights_only=False)


def _resolve_libero_task(benchmark: str, instruction: str | None) -> tuple[Any, Any]:
    """Return (suite, task) for a benchmark + instruction, matching on task.language."""
    from libero.libero import benchmark as libero_benchmark

    suite_dict = libero_benchmark.get_benchmark_dict()
    if benchmark not in suite_dict:
        raise ReplayError(f"Unknown LIBERO benchmark '{benchmark}'. Available: {sorted(suite_dict)}")
    suite = suite_dict[benchmark]()

    if instruction:
        needle = instruction.strip().lower()
        for i in range(suite.n_tasks):
            task = suite.get_task(i)
            if task.language.strip().lower() == needle:
                return suite, task
        logger.warning("No %s task matches instruction %r; falling back to task 0.", benchmark, instruction)
    return suite, suite.get_task(0)


def _default_env_factory(camera_size: int, env_seed: int) -> Callable[[str, str | None], dict]:
    """Build the real-LIBERO env factory: (benchmark, instruction) -> env bundle dict."""

    def factory(benchmark: str, instruction: str | None) -> dict:
        if not _libero_available():
            raise RuntimeError(
                "LIBERO is not installed in this Python environment; "
                "make_libero_eval_fn cannot build a real simulator env. "
                "Run inside the alphabrain conda env (see docs_analysis/REAL_E2E_SETUP.md) "
                "or inject env_factory."
            )
        os.environ.setdefault("MUJOCO_GL", "glfw")
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        _suite, task = _resolve_libero_task(benchmark, instruction)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=camera_size, camera_widths=camera_size)
        env.seed(env_seed)
        return {
            "env": env,
            "instruction": task.language,
            "init_states": _load_init_states(task),
        }

    return factory


# ---------------------------------------------------------------------------
# eval_fn factory
# ---------------------------------------------------------------------------


def make_libero_eval_fn(
    policy_fn: Callable[[dict, str, str], Any],
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    settle_steps: int = DEFAULT_SETTLE_STEPS,
    camera_size: int = 256,
    env_seed: int = 0,
    default_benchmark: str = "libero_spatial",
    default_instruction: str | None = None,
    env_factory: Callable[[str, str | None], dict] | None = None,
) -> Callable[..., float]:
    """
    Build a gate-compatible ``eval_fn(candidate, model_id, n_trials) -> SR``.

    Parameters
    ----------
    policy_fn:
        ``(obs: dict, instruction: str, model_id: str) -> action``.  See
        :mod:`sepa_eval.evalfn.policies` for random / in-process-model /
        WebSocket implementations.
    env_factory:
        ``(benchmark, instruction) -> {"env", "instruction", "init_states"}``.
        Defaults to a real-LIBERO factory (requires LIBERO installed).  Env
        bundles are cached per (benchmark, instruction) key across calls.
    default_benchmark / default_instruction:
        Used when the eval_fn is invoked without a candidate (orchestrator
        EVALUATE step).
    """
    if env_factory is None:
        env_factory = _default_env_factory(camera_size=camera_size, env_seed=env_seed)

    env_cache: dict[tuple[str, str | None], dict] = {}

    def _get_bundle(benchmark: str, instruction: str | None) -> dict:
        key = (benchmark, instruction)
        if key not in env_cache:
            env_cache[key] = env_factory(benchmark, instruction)
        return env_cache[key]

    def _run_episode(
        env: Any,
        instruction: str,
        model_id: str,
        init_state: Sequence[float] | None,
        scene_config: dict,
    ) -> bool:
        obs = env.reset()
        if init_state is not None and hasattr(env, "set_init_state"):
            obs = env.set_init_state(init_state)
        if scene_config:
            try:
                replay_obs, replay_info = replay_scene_config(
                    env, scene_config, base_init_state=None if init_state is None else list(init_state)
                )
                if replay_obs is not None:
                    obs = replay_obs
                if replay_info.get("degradations"):
                    logger.warning("Scene replay degraded: %s", replay_info["degradations"])
            except ReplayError as exc:
                logger.warning("Scene-config replay failed (%s); evaluating base scene.", exc)
        for _ in range(settle_steps):
            obs = _step(env, list(_NOOP_ACTION))[0] or obs
        if not isinstance(obs, dict):
            obs = {"obs": obs}

        for _t in range(max_steps):
            action = policy_fn(obs, instruction, model_id)
            obs, done, info = _step(env, action)
            if not isinstance(obs, dict):
                obs = {"obs": obs}
            if done:
                return bool(info.get("success", True))
        return False

    def eval_fn(candidate: Any = None, model_id: str = "default", n_trials: int = 1, **_kw: Any) -> float:
        benchmark = getattr(candidate, "benchmark", None) or default_benchmark
        instruction_hint = getattr(candidate, "instruction", None) or default_instruction
        scene_config = dict(getattr(candidate, "scene_config", None) or {})

        bundle = _get_bundle(benchmark, instruction_hint)
        env = bundle["env"]
        instruction = bundle.get("instruction") or instruction_hint or ""
        init_states = bundle.get("init_states")

        n_trials = max(1, int(n_trials))
        successes = 0
        for trial in range(n_trials):
            init_state = None
            if init_states is not None and len(init_states) > 0:
                init_state = init_states[trial % len(init_states)]
            successes += bool(_run_episode(env, instruction, model_id, init_state, scene_config))
        sr = successes / n_trials
        logger.info(
            "eval_fn: task=%r model=%s trials=%d SR=%.3f (mutated=%s)",
            (getattr(candidate, "task_id", None) or f"{benchmark}/base"),
            model_id,
            n_trials,
            sr,
            bool(scene_config),
        )
        return sr

    def close() -> None:
        """Close all cached envs (best effort)."""
        for bundle in env_cache.values():
            env = bundle.get("env")
            if env is not None and hasattr(env, "close"):
                try:
                    env.close()
                except Exception:
                    pass
        env_cache.clear()

    eval_fn.close = close  # type: ignore[attr-defined]
    return eval_fn


def _step(env: Any, action: Any) -> tuple[Any, bool, dict]:
    """Step an env handling both (obs, reward, done, info) and (obs, done, info)."""
    result = env.step(action)
    if len(result) == 4:
        obs, _reward, done, info = result
    elif len(result) == 3:
        obs, done, info = result
    else:
        raise ValueError(f"Unexpected step() return length {len(result)}.")
    return obs, bool(done), dict(info or {})
