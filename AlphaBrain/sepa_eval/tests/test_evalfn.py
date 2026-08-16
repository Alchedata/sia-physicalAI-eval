"""Tests for sepa_eval.evalfn — eval_fn contract, policies and CLI wiring (no LIBERO needed)."""

import json

import pytest

from sepa_eval.evalfn import make_libero_eval_fn, make_random_policy_fn, resolve_policy_fn
from sepa_eval.evalfn.libero_eval_fn import _step
from sepa_eval.memory.schema import CandidateTask
from sepa_eval.promotion.gates import GateOutcome, SolvabilityGate

# ---------------------------------------------------------------------------
# Fakes (mirror test_libero_replay.py conventions)
# ---------------------------------------------------------------------------


class FakeModel:
    def __init__(self, free_joints):
        self._free = free_joints
        self.joint_names = [*free_joints, "robot0_joint1"]

    def get_joint_qpos_addr(self, name):
        if name in self._free:
            start = self._free[name]
            return (start, start + 7)
        return 3


class FakeSim:
    def __init__(self, model):
        self.model = model


class FakeEnv:
    """LIBERO-shaped env: success after `succeed_at` steps, else runs forever."""

    def __init__(self, succeed_at=None, free_joints=None, state_len=30):
        self.succeed_at = succeed_at
        self.t = 0
        self._state = [float(i) / 10 for i in range(state_len)]
        self.sim = FakeSim(FakeModel(free_joints or {"cube_joint0": 5}))
        self.set_state_calls = []
        self.actions = []
        self.closed = False

    def reset(self):
        self.t = 0
        return {"agentview_image": "obs0"}

    def get_sim_state(self):
        return list(self._state)

    def set_init_state(self, state):
        self.set_state_calls.append(list(state))
        self._state = list(state)
        return {"agentview_image": "obs_init"}

    def step(self, action):
        self.actions.append(action)
        self.t += 1
        done = self.succeed_at is not None and self.t >= self.succeed_at
        return {"agentview_image": f"obs{self.t}"}, 0.0, done, {"success": done}

    def close(self):
        self.closed = True


def make_env_factory(env):
    calls = []

    def factory(benchmark, instruction):
        calls.append((benchmark, instruction))
        return {"env": env, "instruction": instruction or "base task", "init_states": [list(env._state)]}

    factory.calls = calls
    return factory


def make_candidate(scene_config=None, benchmark="libero_spatial"):
    return CandidateTask.new(
        parent_task_id="seed-1",
        benchmark=benchmark,
        instruction="pick up the cube",
        scene_config=scene_config or {},
        mutation_type="PosePerturbation",
        mutation_params={},
    )


# ---------------------------------------------------------------------------
# eval_fn contract
# ---------------------------------------------------------------------------


def test_eval_fn_returns_success_rate_and_matches_gate_contract():
    env = FakeEnv(succeed_at=3)
    eval_fn = make_libero_eval_fn(
        make_random_policy_fn(), max_steps=10, settle_steps=0, env_factory=make_env_factory(env)
    )
    sr = eval_fn(make_candidate(), "model-a", 4)  # positional, gate-style call
    assert sr == 1.0


def test_eval_fn_zero_sr_when_never_done():
    env = FakeEnv(succeed_at=None)
    eval_fn = make_libero_eval_fn(
        make_random_policy_fn(), max_steps=5, settle_steps=0, env_factory=make_env_factory(env)
    )
    assert eval_fn(make_candidate(), "model-a", 3) == 0.0
    # bounded rollouts: 3 trials x 5 steps
    assert len(env.actions) == 15


def test_eval_fn_orchestrator_keyword_call_without_candidate():
    env = FakeEnv(succeed_at=1)
    factory = make_env_factory(env)
    eval_fn = make_libero_eval_fn(
        make_random_policy_fn(),
        max_steps=5,
        settle_steps=0,
        env_factory=factory,
        default_benchmark="libero_spatial",
    )
    sr = eval_fn(model_id="m", n_trials=2)  # orchestrator EVALUATE-step style
    assert sr == 1.0
    assert factory.calls[0][0] == "libero_spatial"


def test_eval_fn_applies_mutated_scene_config():
    env = FakeEnv(succeed_at=1)
    eval_fn = make_libero_eval_fn(
        make_random_policy_fn(), max_steps=3, settle_steps=0, env_factory=make_env_factory(env)
    )
    eval_fn(make_candidate(scene_config={"cube_pos": [1.0, 2.0, 3.0]}), "m", 1)
    # First set_init_state restores the trial init state; second applies the mutation.
    assert len(env.set_state_calls) == 2
    # cube free-joint qpos starts at 5 (qpos-relative) -> flattened index 6 (time_dim=1).
    assert env.set_state_calls[-1][6:9] == [1.0, 2.0, 3.0]


def test_eval_fn_env_cache_reused_across_calls():
    env = FakeEnv(succeed_at=1)
    factory = make_env_factory(env)
    eval_fn = make_libero_eval_fn(make_random_policy_fn(), max_steps=3, settle_steps=0, env_factory=factory)
    candidate = make_candidate()
    eval_fn(candidate, "m", 1)
    eval_fn(candidate, "m", 1)
    assert len(factory.calls) == 1
    eval_fn.close()
    assert env.closed


def test_eval_fn_raises_clear_error_without_libero(monkeypatch):
    import sepa_eval.evalfn.libero_eval_fn as mod

    monkeypatch.setattr(mod, "_libero_available", lambda: False)
    eval_fn = make_libero_eval_fn(make_random_policy_fn())
    with pytest.raises(RuntimeError, match="LIBERO is not installed"):
        eval_fn(make_candidate(), "m", 1)


def test_eval_fn_feeds_solvability_gate():
    env = FakeEnv(succeed_at=2)
    eval_fn = make_libero_eval_fn(
        make_random_policy_fn(), max_steps=10, settle_steps=0, env_factory=make_env_factory(env)
    )
    gate = SolvabilityGate(n_trials=3, min_sr=0.5)
    result = gate.evaluate(make_candidate(), eval_fn=eval_fn, model_ids=["m1"])
    assert result.outcome == GateOutcome.PASS
    assert result.evidence["best_sr"] == 1.0


def test_step_handles_3_and_4_tuple():
    class ThreeTupleEnv:
        def step(self, action):
            return {"o": 1}, True, {"success": True}

    obs, done, info = _step(ThreeTupleEnv(), [0.0])
    assert done and info["success"]
    obs, done, info = _step(FakeEnv(succeed_at=1), [0.0])
    assert done and info["success"]


# ---------------------------------------------------------------------------
# Policy spec parsing
# ---------------------------------------------------------------------------


def test_random_policy_action_shape_and_bounds():
    policy_fn = make_random_policy_fn(action_dim=7, seed=42)
    action = policy_fn({"agentview_image": None}, "pick", "m")
    assert len(action) == 7
    assert all(-1.0 <= a <= 1.0 for a in action)
    assert action[-1] in (-1.0, 1.0)


def test_resolve_policy_fn_random():
    policy_fn, model_id = resolve_policy_fn("random")
    assert callable(policy_fn)
    assert model_id == "random-policy"


def test_resolve_policy_fn_unknown_spec_raises():
    with pytest.raises(ValueError, match="Unknown policy spec"):
        resolve_policy_fn("teleport")


# ---------------------------------------------------------------------------
# CLI wiring (`promote --real-eval libero --policy random`)
# ---------------------------------------------------------------------------


@pytest.fixture()
def real_memory_dir(tmp_path):
    """Memory dir with one deferred candidate task."""
    from sepa_eval.memory.eval_memory import EvalMemory

    memory = EvalMemory(db_path=str(tmp_path / "eval.db"), memory_dir=str(tmp_path / "traces"))
    candidate = make_candidate(scene_config={"cube_pos": [0.5, 0.5, 1.0]})
    candidate.promotion_status = "deferred"
    memory.record_candidate_task(candidate)
    memory.close()
    return tmp_path, candidate.task_id


def _run_cli(argv):
    from sepa_eval.__main__ import main

    return main(argv)


def test_cli_promote_real_eval_reaches_gates_and_persists_status(real_memory_dir, monkeypatch):
    """--real-eval libero routes a real eval_fn into the gates; SR=0 -> rejected."""
    import sepa_eval.evalfn as evalfn_pkg

    calls = []

    def fake_make_libero_eval_fn(policy_fn, **kwargs):
        def eval_fn(candidate=None, model_id="default", n_trials=1, **_kw):
            calls.append((getattr(candidate, "task_id", None), model_id, n_trials))
            return 0.0  # unsolvable -> SolvabilityGate DISCARD -> rejected

        return eval_fn

    # Patch both the package attribute and the CLI import site.
    monkeypatch.setattr(evalfn_pkg, "make_libero_eval_fn", fake_make_libero_eval_fn)

    tmp_path, task_id = real_memory_dir
    rc = _run_cli(
        [
            "--memory-dir",
            str(tmp_path),
            "promote",
            "--status",
            "candidate,deferred",
            "--real-eval",
            "libero",
            "--policy",
            "random",
            "--gate-trials",
            "2",
        ]
    )
    assert rc == 0
    assert calls, "real eval_fn was never invoked by the gates"
    assert calls[0] == (task_id, "random-policy", 2)

    import sqlite3

    conn = sqlite3.connect(str(tmp_path / "eval.db"))
    status, evidence = conn.execute(
        "SELECT promotion_status, promotion_evidence FROM tasks WHERE task_id = ?", (task_id,)
    ).fetchone()
    conn.close()
    assert status == "rejected"
    assert json.loads(evidence)["SolvabilityGate"]["best_sr"] == 0.0


def test_cli_promote_default_behaviour_unchanged(real_memory_dir):
    """Without --real-eval, deferred candidates stay deferred (no eval_fn injected)."""
    tmp_path, task_id = real_memory_dir
    rc = _run_cli(["--memory-dir", str(tmp_path), "promote", "--status", "deferred"])
    assert rc == 0

    import sqlite3

    conn = sqlite3.connect(str(tmp_path / "eval.db"))
    (status,) = conn.execute("SELECT promotion_status FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
    conn.close()
    assert status == "deferred"


def test_cli_promote_status_filter_default_excludes_deferred(real_memory_dir, capsys):
    tmp_path, _ = real_memory_dir
    rc = _run_cli(["--memory-dir", str(tmp_path), "promote"])
    assert rc == 0
    assert "No candidates pending promotion" in capsys.readouterr().out


def test_pipeline_run_inline_executes_gates_on_caller_thread():
    import threading

    from sepa_eval.promotion.pipeline import PromotionPipeline

    seen = {}

    class ThreadRecordingGate:
        def evaluate(self, candidate, **kwargs):
            from sepa_eval.promotion.gates import GateOutcome, GateResult

            seen["thread"] = threading.current_thread()
            return GateResult(gate_name="ThreadRecordingGate", outcome=GateOutcome.PASS)

    pipeline = PromotionPipeline(gates=[ThreadRecordingGate()], run_inline=True)
    status, evidence = pipeline.run(make_candidate())
    assert status == "promoted"
    assert seen["thread"] is threading.current_thread()
