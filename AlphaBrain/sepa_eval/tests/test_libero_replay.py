"""Tests for sepa_eval.replay.libero_replay — no LIBERO dependency required."""

import math

import pytest

from sepa_eval.hooks.libero_trace_hook import LiberoHook
from sepa_eval.mutation.distractor_add import DistractorAdd
from sepa_eval.mutation.pose_perturbation import PosePerturbation
from sepa_eval.replay import (
    ReplayError,
    apply_scene_config_to_init_state,
    generate_distractor_bddl,
    make_env_with_distractors,
    replay_scene_config,
    resolve_object_addresses,
)
from sepa_eval.replay.libero_replay import euler_to_quat

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeModel:
    """Mimics mujoco model joint API: free joints named <obj>_joint0 with (start, end) addr."""

    def __init__(self, free_joints):
        # free_joints: dict joint_name -> qpos start (7-dof free joint)
        self._free = free_joints
        self.joint_names = [*free_joints, "robot0_joint1", "gripper_hinge_joint2"]

    def get_joint_qpos_addr(self, name):
        if name in self._free:
            start = self._free[name]
            return (start, start + 7)
        return 3  # scalar addr for non-free joints


class FakeSim:
    def __init__(self, model):
        self.model = model


class FakeEnv:
    def __init__(self, state, free_joints):
        self._state = list(state)
        self.sim = FakeSim(FakeModel(free_joints))
        self.set_state_calls = []

    def get_sim_state(self):
        return list(self._state)

    def set_init_state(self, state):
        self.set_state_calls.append(list(state))
        self._state = list(state)
        return {"agentview_image": "fake_obs"}

    def reset(self):
        return {"agentview_image": "reset_obs"}


def make_state(n=30):
    return [float(i) / 10 for i in range(n)]


# ---------------------------------------------------------------------------
# Pure state-vector math
# ---------------------------------------------------------------------------


def test_apply_pos_writes_absolute_position():
    state = make_state()
    # object "cube" free joint qpos starts at flattened index 5
    new_state, info = apply_scene_config_to_init_state(state, {"cube_pos": [1.0, 2.0, 3.0]}, {"cube": 5})
    assert new_state[5:8] == [1.0, 2.0, 3.0]
    assert new_state[:5] == state[:5]
    assert new_state[8:] == state[8:]
    assert info["applied"] == ["cube_pos"]
    assert info["skipped"] == []


def test_apply_quat_is_normalized():
    state = make_state()
    new_state, info = apply_scene_config_to_init_state(state, {"cube_quat": [2.0, 0.0, 0.0, 0.0]}, {"cube": 5})
    assert new_state[8:12] == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert "cube_quat" in info["applied"]


def test_apply_rot_converts_euler_to_quat():
    state = make_state()
    yaw = math.pi / 2
    new_state, _ = apply_scene_config_to_init_state(state, {"cube_rot": [0.0, 0.0, yaw]}, {"cube": 5})
    assert new_state[8:12] == pytest.approx(euler_to_quat([0.0, 0.0, yaw]))
    assert new_state[8] == pytest.approx(math.cos(yaw / 2))
    assert new_state[11] == pytest.approx(math.sin(yaw / 2))


def test_unknown_object_and_non_pose_keys_are_skipped():
    state = make_state()
    cfg = {"ghost_pos": [1, 2, 3], "instruction": "pick", "num_distractors": 2}
    new_state, info = apply_scene_config_to_init_state(state, cfg, {"cube": 5})
    assert new_state == state
    assert info["applied"] == []
    assert info["skipped"] == ["ghost_pos"]


def test_out_of_bounds_address_raises():
    with pytest.raises(ReplayError):
        apply_scene_config_to_init_state(make_state(8), {"cube_pos": [0, 0, 0]}, {"cube": 5})


def test_pose_perturbation_output_replays_end_to_end():
    """PosePerturbation candidate scene_configs apply cleanly to an init state."""
    seed_cfg = {"cube_pos": [0.1, 0.2, 0.3], "cube_quat": [1.0, 0.0, 0.0, 0.0]}
    candidates = PosePerturbation(delta_pos=[0.05], delta_rot_deg=[10.0]).generate(
        seed_cfg, "pick the cube", parent_task_id="t0", benchmark="libero_spatial"
    )
    assert len(candidates) == 1
    mutated = candidates[0].scene_config
    state = make_state()
    new_state, info = apply_scene_config_to_init_state(state, mutated, {"cube": 5})
    assert info["applied"] == ["cube_pos", "cube_quat"] or set(info["applied"]) == {"cube_pos", "cube_quat"}
    assert new_state[5:8] == pytest.approx(mutated["cube_pos"])
    # position stays within the perturbation bound
    for orig, new in zip(seed_cfg["cube_pos"], new_state[5:8], strict=False):
        assert abs(new - orig) <= 0.05 + 1e-9


# ---------------------------------------------------------------------------
# Live-env replay with fakes
# ---------------------------------------------------------------------------


def test_resolve_object_addresses_from_fake_env():
    env = FakeEnv(make_state(), {"cube_joint0": 4, "bowl_joint0": 11})
    addrs = resolve_object_addresses(env)
    # time_dim=1 offset applied; scalar-addr joints excluded
    assert addrs == {"cube": 5, "bowl": 12}


def test_resolve_object_addresses_requires_sim():
    class NoSim:
        pass

    with pytest.raises(ReplayError, match="sim.model"):
        resolve_object_addresses(NoSim())


def test_replay_scene_config_applies_pose_via_set_init_state():
    env = FakeEnv(make_state(), {"cube_joint0": 4})
    obs, info = replay_scene_config(env, {"cube_pos": [9.0, 8.0, 7.0]})
    assert obs == {"agentview_image": "fake_obs"}
    assert len(env.set_state_calls) == 1
    assert env.set_state_calls[0][5:8] == [9.0, 8.0, 7.0]
    assert info["applied"] == ["cube_pos"]
    assert info["distractors_applied"] is None
    assert info["degradations"] == []


def test_replay_scene_config_degrades_on_distractors():
    env = FakeEnv(make_state(), {"cube_joint0": 4})
    seed = {"cube_pos": [0.1, 0.2, 0.3]}
    cand = DistractorAdd(counts=[2]).generate(seed, "pick", parent_task_id="t0", benchmark="b")[0]
    obs, info = replay_scene_config(env, cand.scene_config)
    assert obs is not None
    assert info["distractors_applied"] is False
    assert len(info["degradations"]) == 1
    assert "BDDL" in info["degradations"][0]
    # pose portion still applied
    assert env.set_state_calls[0][5:8] == pytest.approx(cand.scene_config["cube_pos"])


def test_replay_scene_config_requires_state_source():
    class MinimalEnv:
        sim = FakeSim(FakeModel({"cube_joint0": 4}))

    with pytest.raises(ReplayError, match="base_init_state"):
        replay_scene_config(MinimalEnv(), {"cube_pos": [0, 0, 0]})


# ---------------------------------------------------------------------------
# LiberoHook integration
# ---------------------------------------------------------------------------


def test_libero_hook_replays_scene_config_on_reset():
    env = FakeEnv(make_state(), {"cube_joint0": 4})
    hook = LiberoHook(env, task_id="t1", scene_config={"cube_pos": [1.0, 1.0, 1.0]}, replay_on_reset=True)
    obs = hook.reset()
    assert obs == {"agentview_image": "fake_obs"}  # obs from set_init_state, not plain reset
    assert env.set_state_calls[0][5:8] == [1.0, 1.0, 1.0]
    assert hook.last_replay_info["applied"] == ["cube_pos"]


def test_libero_hook_falls_back_to_reseed_on_replay_error():
    class BrokenEnv:
        def reset(self):
            return {"agentview_image": "reset_obs"}

    hook = LiberoHook(BrokenEnv(), task_id="t1", scene_config={"cube_pos": [1, 1, 1]}, replay_on_reset=True)
    obs = hook.reset()
    assert obs == {"agentview_image": "reset_obs"}
    assert hook.last_replay_info["mode"] == "reseed_fallback"


def test_libero_hook_default_behaviour_unchanged():
    env = FakeEnv(make_state(), {"cube_joint0": 4})
    hook = LiberoHook(env, task_id="t1", scene_config={"cube_pos": [1.0, 1.0, 1.0]})
    obs = hook.reset()
    assert obs == {"agentview_image": "reset_obs"}
    assert env.set_state_calls == []


# ---------------------------------------------------------------------------
# DistractorAdd BDDL generation
# ---------------------------------------------------------------------------

SAMPLE_BDDL = """(define (problem LIBERO_Floor_Manipulation)
  (:domain robosuite)
  (:language Pick the pudding and place it in the basket)
    (:regions
      (target_object_region
          (:target floor)
          (:ranges (
              (-0.145 -0.265 -0.095 -0.215)
            )
          )
      )
    )

  (:fixtures
    floor - floor
  )

  (:objects
    chocolate_pudding_1 - chocolate_pudding
    basket_1 - basket
  )

  (:obj_of_interest
    chocolate_pudding_1
    basket_1
  )

  (:init
    (On chocolate_pudding_1 floor_target_object_region)
    (On basket_1 floor_target_object_region)
  )

  (:goal
    (And (In chocolate_pudding_1 basket_1))
  )

)
"""


def test_generate_distractor_bddl_injects_objects_regions_inits():
    distractors = [{"category": "same", "count": 2, "id": 0}, {"category": "same", "count": 2, "id": 1}]
    out = generate_distractor_bddl(SAMPLE_BDDL, distractors)
    assert "chocolate_pudding_distractor_1 - chocolate_pudding" in out
    assert "chocolate_pudding_distractor_2 - chocolate_pudding" in out
    assert "sepa_distractor_region_0" in out
    assert "sepa_distractor_region_1" in out
    assert "(On chocolate_pudding_distractor_1 floor_sepa_distractor_region_0)" in out
    # goal & obj_of_interest untouched
    assert out.count("(:goal") == 1
    assert "(And (In chocolate_pudding_1 basket_1))" in out
    # balanced parens preserved
    assert out.count("(") == out.count(")")


def test_generate_distractor_bddl_rejects_malformed_text():
    with pytest.raises(ReplayError):
        generate_distractor_bddl("(define (problem x))", [{"id": 0}])


def test_make_env_with_distractors_raises_without_libero(tmp_path, monkeypatch):
    import sepa_eval.replay.libero_replay as lr

    monkeypatch.setattr(lr, "_libero_available", lambda: False)
    bddl = tmp_path / "task.bddl"
    bddl.write_text(SAMPLE_BDDL)
    with pytest.raises(RuntimeError, match="LIBERO is not installed"):
        make_env_with_distractors(str(bddl), {"distractors": [{"id": 0}]})
