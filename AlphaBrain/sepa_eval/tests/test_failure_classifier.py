"""
Unit tests for FailureClassifier.

Three test functions:
    test_classify_8types         – each canonical failure type is triggered by a
                                   synthetic trace designed for that detector.
    test_failure_step_per_type   – failure_step is not None for each classified trace.
    test_fallback_unclassifiable – a trace with no signals returns (None, episode_length).
"""
from __future__ import annotations

import pytest

from sepa_eval.memory.schema import (
    EpisodeTrace,
    RolloutData,
    SceneConfig,
    TaskProvenance,
    TraceIdentity,
    TraceLabels,
)
from sepa_eval.mining.failure_classifier import FailureClassifier


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_identity(task_instruction: str = "pick up the cube") -> TraceIdentity:
    return TraceIdentity(
        trace_id="test-trace-001",
        eval_run_id="run-001",
        benchmark="libero_spatial",
        task_id="task-001",
        task_instruction=task_instruction,
        model_id="model-abc",
        model_version="v1",
    )


def _make_scene() -> SceneConfig:
    return SceneConfig(scene_config={}, init_state=b"")


def _make_trace(
    *,
    observations: list[dict],
    actions: list,
    episode_length: int,
    success: bool,
    failure_step: int | None = None,
    task_instruction: str = "pick up the cube",
) -> EpisodeTrace:
    return EpisodeTrace(
        identity=_make_identity(task_instruction),
        scene=_make_scene(),
        rollout=RolloutData(
            observations=observations,
            actions=actions,
            episode_length=episode_length,
            success=success,
            failure_step=failure_step,
        ),
        labels=TraceLabels(),
        provenance=TaskProvenance(),
    )


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------

def _distinct_actions(n: int, dim: int = 7) -> list[list[float]]:
    """
    Generate *n* action vectors of dimension *dim* such that:
      - Every pairwise L2 distance within the final 10 actions > 0.05
        (defeats the recovery detector, threshold = 0.05).
      - Every action's L2 norm < 0.8
        (avoids triggering the out_of_reach detector, threshold = 0.8).

    Strategy:
      - First (n-10) actions: constant [0.1, 0, ..., 0], safe filler.
      - Last 10 actions: evenly spread across the first 2 dimensions on a
        small circle of radius 0.2.  Step = 2π/10 ≈ 0.628 rad gives a chord
        of 2*0.2*sin(π/10) ≈ 0.124 > 0.05 between any adjacent pair, and the
        L2 norm of each vector = 0.2 < 0.8.
    """
    import math
    filler = [[0.1] + [0.0] * (dim - 1)] * max(0, n - 10)
    tail_count = min(10, n)
    tail = [
        [0.2 * math.cos(2 * math.pi * k / 10),
         0.2 * math.sin(2 * math.pi * k / 10)]
        + [0.0] * (dim - 2)
        for k in range(tail_count)
    ]
    return filler + tail


# ---------------------------------------------------------------------------
# Per-type synthetic traces
# ---------------------------------------------------------------------------

def _trace_timeout() -> EpisodeTrace:
    """episode_length == max_steps (500) AND success=False → timeout."""
    return _make_trace(
        observations=[{}] * 500,
        actions=[[0.0] * 7] * 500,
        episode_length=500,
        success=False,
        failure_step=500,
    )


def _trace_out_of_reach() -> EpisodeTrace:
    """Last 10 actions have mean L2 norm > 0.8 → out_of_reach.
    episode_length=499 so timeout (>= 500) does not fire first.
    """
    normal_actions = [[0.1] * 7] * 489
    # L2 of [1.0]*7 ≈ 2.65 >> 0.8
    high_mag = [[1.0] * 7] * 10
    return _make_trace(
        observations=[{}] * 499,
        actions=normal_actions + high_mag,
        episode_length=499,
        success=False,
        failure_step=489,
    )


def _trace_grasp() -> EpisodeTrace:
    """
    Gripper oscillates (0.0 / 1.0 alternation → high variance) → grasp.
    episode_length short so contact_dynamics and timeout don't fire first.
    failure_step set early so pose_estimation check sees it at step 1 (< 0.3*10=3),
    but grasp takes priority over pose_estimation in the ordering, so this is fine.
    """
    obs = [{"gripper_state": float(i % 2)} for i in range(10)]
    return _make_trace(
        observations=obs,
        actions=[[0.1] * 7] * 10,
        episode_length=10,
        success=False,
        # Set failure_step in the middle so contact_dynamics doesn't fire
        # (need > 0.7 * 10 = 7), and pose_estimation doesn't fire (need < 0.3*10=3).
        failure_step=5,
    )


def _trace_contact_dynamics() -> EpisodeTrace:
    """
    failure_step > 0.7 * episode_length AND success=False → contact_dynamics.
    episode_length must be long enough that timeout doesn't fire (< 500).
    No high-magnitude actions so out_of_reach doesn't fire.
    No gripper oscillation so grasp doesn't fire.
    Actions are all distinct (L2 dist > 0.05 between consecutive) so recovery
    detector doesn't fire.
    """
    ep_len = 100
    failure_step = 85  # > 0.7 * 100 = 70
    obs = [{"gripper_state": 0.0}] * ep_len  # constant → variance 0 → no grasp
    actions = _distinct_actions(ep_len)
    return _make_trace(
        observations=obs,
        actions=actions,
        episode_length=ep_len,
        success=False,
        failure_step=failure_step,
    )


def _trace_recovery() -> EpisodeTrace:
    """
    Repeated (near-duplicate) actions in the tail window → recovery.
    episode_length must be short so timeout doesn't fire.
    failure_step in the middle so contact_dynamics and pose_estimation don't fire.
    No gripper oscillation so grasp doesn't fire.
    """
    ep_len = 50
    unique_actions = [[float(i) * 0.01] * 7 for i in range(40)]
    # Last 10: every other action is near-identical → triggers recovery detector.
    repeated = [[0.05] * 7, [0.5] * 7] * 5  # 10 actions; [0.05]*7 repeats 5×
    obs = [{"gripper_state": 0.0}] * ep_len
    return _make_trace(
        observations=obs,
        actions=unique_actions + repeated,
        episode_length=ep_len,
        success=False,
        failure_step=30,  # 30/50=0.6 → not late (< 0.7) → contact_dynamics won't fire
    )


def _trace_pose_estimation() -> EpisodeTrace:
    """
    failure_step < 0.3 * episode_length → pose_estimation.
    episode_length < 500 so timeout doesn't fire.
    No high-mag actions, no gripper oscillation, no late failure.
    Actions are all distinct (pairwise L2 > 0.05 in last-10 window, norm < 0.8)
    so the recovery detector doesn't fire.
    """
    ep_len = 100
    failure_step = 10  # < 0.3 * 100 = 30
    obs = [{"gripper_state": 0.0}] * ep_len
    actions = _distinct_actions(ep_len)
    return _make_trace(
        observations=obs,
        actions=actions,
        episode_length=ep_len,
        success=False,
        failure_step=failure_step,
    )


def _trace_distractor_confusion() -> EpisodeTrace:
    """
    num_distractors > 0 in observations AND success=False → distractor_confusion.
    failure_step in middle so prior checks don't fire.
    No gripper oscillation, not early, not late.
    Actions are all distinct (pairwise L2 > 0.05 in last-10 window, norm < 0.8)
    so the recovery detector doesn't fire.
    """
    ep_len = 100
    obs = [{"gripper_state": 0.0, "num_distractors": 2}] * ep_len
    actions = _distinct_actions(ep_len)
    return _make_trace(
        observations=obs,
        actions=actions,
        episode_length=ep_len,
        success=False,
        failure_step=50,
    )


def _trace_language_grounding() -> EpisodeTrace:
    """
    obs contains 'object_class' not present in task_instruction →
    language_grounding.
    failure_step in middle so earlier checks don't fire.
    No distractors (so distractor_confusion doesn't fire).
    Actions are all distinct (pairwise L2 > 0.05 in last-10 window, norm < 0.8)
    so the recovery detector doesn't fire.
    """
    ep_len = 100
    obs = [{"gripper_state": 0.0, "object_class": "banana"}] * ep_len
    # 'banana' is NOT in "pick up the cube" → mismatch → language_grounding.
    actions = _distinct_actions(ep_len)
    return _make_trace(
        observations=obs,
        actions=actions,
        episode_length=ep_len,
        success=False,
        failure_step=50,
        task_instruction="pick up the cube",
    )


# ---------------------------------------------------------------------------
# Map of expected type → trace factory
# ---------------------------------------------------------------------------

_TYPE_TRACE_MAP: dict[str, EpisodeTrace] = {
    "timeout": _trace_timeout(),
    "out_of_reach": _trace_out_of_reach(),
    "grasp": _trace_grasp(),
    "contact_dynamics": _trace_contact_dynamics(),
    "recovery": _trace_recovery(),
    "pose_estimation": _trace_pose_estimation(),
    "distractor_confusion": _trace_distractor_confusion(),
    "language_grounding": _trace_language_grounding(),
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFailureClassifier:
    """Tests for FailureClassifier."""

    @pytest.fixture(autouse=True)
    def classifier(self):
        self.clf = FailureClassifier(max_episode_steps=500)

    def test_classify_8types(self):
        """Each failure type must be correctly identified from a synthetic trace."""
        for expected_type, trace in _TYPE_TRACE_MAP.items():
            result_type, result_step = self.clf.classify(trace)
            assert result_type == expected_type, (
                f"Expected '{expected_type}', got '{result_type}' "
                f"(failure_step={result_step})"
            )

    def test_failure_step_per_type(self):
        """failure_step must be a non-None integer for every classified trace."""
        for failure_type, trace in _TYPE_TRACE_MAP.items():
            result_type, result_step = self.clf.classify(trace)
            assert result_type is not None, f"Trace for '{failure_type}' was not classified."
            assert isinstance(result_step, int), (
                f"failure_step for '{failure_type}' should be int, got {type(result_step)}"
            )

    def test_fallback_unclassifiable(self):
        """
        A trace with no failure signals should return (None, episode_length).

        Conditions satisfied to defeat all 8 detectors:
          - timeout:              episode_length=200 < 500
          - out_of_reach:         last-10 mean L2 ≈ 0.26 < 0.8
          - grasp:                no gripper_state/qpos in obs
          - contact_dynamics:     failure_step=100, 100 > 0.7*200=140? No
          - recovery:             actions are all distinct (unique per step)
          - pose_estimation:      failure_step=100, 100 < 0.3*200=60? No
          - distractor_confusion: no num_distractors in obs
          - language_grounding:   no object_class in obs
        """
        ep_len = 200
        # Distinct actions (pairwise L2 > 0.05 in last-10 window, norm < 0.8)
        # prevent recovery detector from firing.
        actions = _distinct_actions(ep_len)
        trace = _make_trace(
            observations=[{}] * ep_len,
            actions=actions,
            episode_length=ep_len,
            success=False,
            failure_step=ep_len // 2,  # middle → not early (< 60), not late (> 140)
        )
        result_type, result_step = self.clf.classify(trace)
        assert result_type is None
        assert result_step == ep_len
