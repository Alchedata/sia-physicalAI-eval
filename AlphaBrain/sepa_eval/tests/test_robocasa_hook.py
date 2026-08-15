"""
Tests for RobocasaHook (BenchmarkAdapter) and run_robocasa_episode_with_trace.

All tests use a duck-typed fake env — no actual RoboCasa / robosuite required.
"""
from __future__ import annotations

import sqlite3
import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest

from sepa_eval.hooks.robocasa_trace_hook import RobocasaHook, run_robocasa_episode_with_trace
from sepa_eval.memory.schema import TraceIdentity


# ---------------------------------------------------------------------------
# Fake RoboCasa env
# ---------------------------------------------------------------------------

class _FakeRoboCasaEnv:
    """
    Minimal duck-typed RoboCasa / robosuite env.

    Runs for ``max_steps`` steps, then sets done=True.
    """

    def __init__(self, max_steps: int = 5, success: bool = True, obs_format: str = "dict"):
        self._max_steps = max_steps
        self._success = success
        self._obs_format = obs_format
        self._step_count = 0

    def reset(self):
        self._step_count = 0
        if self._obs_format == "dict":
            return {"robot0_eef_pos": [0.0, 0.0, 0.0]}
        if self._obs_format == "none":
            return None
        return [0.0, 0.0, 0.0]  # non-dict

    def get_obs(self):
        return {"robot0_eef_pos": [0.0, 0.0, 0.0]}

    def step(self, action):
        self._step_count += 1
        done = self._step_count >= self._max_steps
        obs = {"robot0_eef_pos": [float(self._step_count)] * 3}
        reward = 1.0 if (done and self._success) else 0.0
        info = {"success": self._success} if done else {}
        return obs, reward, done, info


class _FakeRoboCasaEnv3Tuple(_FakeRoboCasaEnv):
    """Env that returns (obs, done, info) — the 3-tuple variant."""

    def step(self, action):
        self._step_count += 1
        done = self._step_count >= self._max_steps
        obs = {"robot0_eef_pos": [float(self._step_count)] * 3}
        info = {"success": self._success} if done else {}
        return obs, done, info


class _FakeMemoryForHook:
    """Minimal EvalMemory for hook/episode tests."""

    def __init__(self):
        import sqlite3
        self._conn = sqlite3.connect(":memory:")
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(
            """
            CREATE TABLE traces (
                id INTEGER PRIMARY KEY,
                trace_id TEXT UNIQUE,
                eval_run_id TEXT,
                benchmark TEXT,
                task_id TEXT,
                task_instruction TEXT,
                model_id TEXT,
                model_version TEXT,
                success INTEGER,
                episode_length INTEGER,
                failure_step INTEGER,
                failure_type TEXT,
                critic_scores TEXT,
                failure_attribution TEXT,
                parent_task_id TEXT,
                mutation_type TEXT,
                promotion_evidence TEXT,
                recorded_at TEXT,
                file_path TEXT
            );
            """
        )
        self._conn.commit()

    def record_trace(self, trace):
        from sepa_eval.memory.eval_memory import _pack_trace
        import os, tempfile
        # Write to a temp file then record in DB (simplified)
        self._conn.execute(
            "INSERT OR REPLACE INTO traces "
            "(trace_id, eval_run_id, benchmark, task_id, task_instruction, "
            " model_id, model_version, success, episode_length, failure_step, "
            " failure_type, critic_scores, failure_attribution, parent_task_id, "
            " mutation_type, promotion_evidence, recorded_at, file_path) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'),?)",
            (
                trace.identity.trace_id, trace.identity.eval_run_id,
                trace.identity.benchmark, trace.identity.task_id,
                trace.identity.task_instruction, trace.identity.model_id,
                trace.identity.model_version,
                int(trace.rollout.success), trace.rollout.episode_length,
                trace.rollout.failure_step, trace.labels.failure_type,
                None, None, trace.provenance.parent_task_id,
                trace.provenance.mutation_type, None,
                f"/tmp/{trace.identity.trace_id}.msgpack",
            ),
        )
        self._conn.commit()
        return trace

    def close(self):
        self._conn.close()


def _make_identity(task_id: str = "tabletop_pick_001") -> TraceIdentity:
    return TraceIdentity(
        trace_id=str(uuid.uuid4()),
        eval_run_id="run-rc-001",
        benchmark="robocasa_tabletop",
        task_id=task_id,
        task_instruction="Pick the apple from the counter",
        model_id="QwenOFT-rc",
        model_version="v1.0",
    )


# ---------------------------------------------------------------------------
# RobocasaHook — BenchmarkAdapter protocol
# ---------------------------------------------------------------------------

class TestRobocasaHook:
    def test_protocol_version_is_set(self):
        assert RobocasaHook.PROTOCOL_VERSION == "1.0"

    def test_get_task_id(self):
        env = _FakeRoboCasaEnv()
        hook = RobocasaHook(env=env, task_id="tabletop_pick_001", env_name="KitchenTabletop")
        assert hook.get_task_id() == "tabletop_pick_001"

    def test_get_scene_config_default(self):
        env = _FakeRoboCasaEnv()
        hook = RobocasaHook(env=env, task_id="t1", env_name="KitchenTabletop")
        cfg = hook.get_scene_config()
        assert cfg["env_name"] == "KitchenTabletop"

    def test_get_scene_config_custom(self):
        env = _FakeRoboCasaEnv()
        custom = {"env_name": "CustomEnv", "lighting": "bright"}
        hook = RobocasaHook(env=env, task_id="t1", env_name="CustomEnv", scene_config=custom)
        cfg = hook.get_scene_config()
        assert cfg["lighting"] == "bright"

    def test_get_scene_config_returns_copy(self):
        """Mutating the returned dict must not affect the hook's internal state."""
        env = _FakeRoboCasaEnv()
        hook = RobocasaHook(env=env, task_id="t1", env_name="KitchenTabletop")
        cfg = hook.get_scene_config()
        cfg["injected"] = True
        assert "injected" not in hook.get_scene_config()

    def test_reset_returns_dict(self):
        env = _FakeRoboCasaEnv(obs_format="dict")
        hook = RobocasaHook(env=env, task_id="t1", env_name="KitchenTabletop")
        obs = hook.reset()
        assert isinstance(obs, dict)

    def test_reset_none_env_uses_get_obs(self):
        """When env.reset() returns None the hook falls back to env.get_obs()."""
        env = _FakeRoboCasaEnv(obs_format="none")
        hook = RobocasaHook(env=env, task_id="t1", env_name="KitchenTabletop")
        obs = hook.reset()
        assert isinstance(obs, dict)

    def test_reset_non_dict_obs_is_wrapped(self):
        env = _FakeRoboCasaEnv(obs_format="list")
        hook = RobocasaHook(env=env, task_id="t1", env_name="KitchenTabletop")
        obs = hook.reset()
        assert isinstance(obs, dict)
        assert "obs" in obs

    def test_step_4tuple_strips_reward(self):
        env = _FakeRoboCasaEnv(max_steps=1)
        hook = RobocasaHook(env=env, task_id="t1", env_name="KitchenTabletop")
        hook.reset()
        obs, done, info = hook.step([0.0] * 7)
        assert isinstance(obs, dict)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_step_3tuple_is_handled(self):
        env = _FakeRoboCasaEnv3Tuple(max_steps=1)
        hook = RobocasaHook(env=env, task_id="t1", env_name="KitchenTabletop")
        hook.reset()
        obs, done, info = hook.step([0.0] * 7)
        assert done is True

    def test_step_non_dict_obs_is_wrapped(self):
        class _NonDictObs(_FakeRoboCasaEnv):
            def step(self, action):
                return [0.0, 0.0], 0.0, False, {}

        env = _NonDictObs()
        hook = RobocasaHook(env=env, task_id="t1", env_name="KitchenTabletop")
        hook.reset()
        obs, done, info = hook.step([0.0])
        assert isinstance(obs, dict)

    def test_step_invalid_tuple_length_raises(self):
        class _BadEnv(_FakeRoboCasaEnv):
            def step(self, action):
                return (1, 2)  # length-2 tuple

        env = _BadEnv()
        hook = RobocasaHook(env=env, task_id="t1", env_name="KitchenTabletop")
        hook.reset()
        with pytest.raises(ValueError, match="Unexpected step\\(\\) return length"):
            hook.step([0.0])


# ---------------------------------------------------------------------------
# run_robocasa_episode_with_trace
# ---------------------------------------------------------------------------

class TestRunRobocasaEpisodeWithTrace:
    def test_successful_episode_returns_trace(self):
        env = _FakeRoboCasaEnv(max_steps=4, success=True)
        mem = _FakeMemoryForHook()
        identity = _make_identity()

        trace = run_robocasa_episode_with_trace(
            env=env,
            env_name="KitchenTabletop",
            policy_fn=lambda obs: [0.0] * 7,
            memory=mem,
            identity=identity,
        )
        mem.close()

        assert trace.rollout.success is True
        assert trace.rollout.episode_length == 4

    def test_failed_episode_marks_success_false(self):
        env = _FakeRoboCasaEnv(max_steps=3, success=False)
        mem = _FakeMemoryForHook()
        identity = _make_identity()

        trace = run_robocasa_episode_with_trace(
            env=env,
            env_name="KitchenTabletop",
            policy_fn=lambda obs: [0.0] * 7,
            memory=mem,
            identity=identity,
        )
        mem.close()

        assert trace.rollout.success is False

    def test_trace_identity_fields_are_preserved(self):
        env = _FakeRoboCasaEnv(max_steps=2)
        mem = _FakeMemoryForHook()
        identity = _make_identity(task_id="custom_task_xyz")

        trace = run_robocasa_episode_with_trace(
            env=env,
            env_name="KitchenTabletop",
            policy_fn=lambda obs: [0.0] * 7,
            memory=mem,
            identity=identity,
        )
        mem.close()

        assert trace.identity.task_id == "custom_task_xyz"
        assert trace.identity.benchmark == "robocasa_tabletop"

    def test_episode_runs_until_done(self):
        """The episode loop terminates exactly when env signals done."""
        step_count = []

        class _CountingEnv(_FakeRoboCasaEnv):
            def step(self, action):
                step_count.append(1)
                return super().step(action)

        env = _CountingEnv(max_steps=6)
        mem = _FakeMemoryForHook()

        run_robocasa_episode_with_trace(
            env=env,
            env_name="CountingEnv",
            policy_fn=lambda obs: [0.0] * 7,
            memory=mem,
            identity=_make_identity(),
        )
        mem.close()

        assert len(step_count) == 6
