"""
Heuristic failure-type classifier for VLA episode traces.

Priority order: timeout → out_of_reach → grasp → contact_dynamics →
                recovery → pose_estimation → distractor_confusion → language_grounding
"""
from __future__ import annotations

import math
from typing import Any

from sepa_eval.memory.schema import EpisodeTrace

# Default maximum episode length used for timeout detection.
_DEFAULT_MAX_STEPS = 500

# Ordered priority list used by classify().
_PRIORITY_ORDER = [
    "timeout",
    "out_of_reach",
    "grasp",
    "contact_dynamics",
    "recovery",
    "pose_estimation",
    "distractor_confusion",
    "language_grounding",
]


# ---------------------------------------------------------------------------
# Per-type step detectors
# ---------------------------------------------------------------------------

class _TimeoutDetector:
    """Triggers when episode length >= max_episode_steps AND success=False."""

    def detect(self, trace: EpisodeTrace, max_steps: int = _DEFAULT_MAX_STEPS) -> int | None:
        rollout = trace.rollout
        if rollout.success:
            return None
        if rollout.episode_length >= max_steps:
            return rollout.episode_length
        return None


class _OutOfReachDetector:
    """
    Triggers when the last 10 actions have a mean L2 norm > 0.8 (normalised
    action space) AND success=False.
    """

    _WINDOW = 10
    _NORM_THRESHOLD = 0.8

    def detect(self, trace: EpisodeTrace) -> int | None:
        if trace.rollout.success:
            return None
        actions = trace.rollout.actions
        if not actions:
            return None
        window = actions[-self._WINDOW:]
        norms = [_l2_norm(a) for a in window]
        if norms and (sum(norms) / len(norms)) > self._NORM_THRESHOLD:
            # failure_step = index of first action in the tail window
            return max(0, len(actions) - self._WINDOW)
        return None


class _GraspDetector:
    """
    Triggers when gripper state oscillates without stable closure AND success=False.
    Looks for 'gripper_state' or falls back to 'qpos' in observations.
    Heuristic: variance of gripper values > 0.1.
    """

    _VAR_THRESHOLD = 0.1

    def detect(self, trace: EpisodeTrace) -> int | None:
        if trace.rollout.success:
            return None
        obs_list = trace.rollout.observations
        if not obs_list:
            return None
        values: list[float] = []
        for obs in obs_list:
            if "gripper_state" in obs:
                v = obs["gripper_state"]
                values.append(float(v) if not hasattr(v, "__iter__") else float(v[0]))
            elif "qpos" in obs:
                qpos = obs["qpos"]
                # Use last element of qpos as proxy for gripper
                values.append(float(qpos[-1]) if hasattr(qpos, "__iter__") else float(qpos))
        if len(values) < 2:
            return None
        var = _variance(values)
        if var > self._VAR_THRESHOLD:
            # Approximate failure step = first step where oscillation becomes notable.
            return 0
        return None


class _ContactDynamicsDetector:
    """
    Triggers when success=False AND failure occurs late in the episode
    (failure_step > 0.7 * episode_length), indicating a post-grasp contact issue.
    """

    _LATE_THRESHOLD = 0.7

    def detect(self, trace: EpisodeTrace) -> int | None:
        rollout = trace.rollout
        if rollout.success:
            return None
        ep_len = rollout.episode_length
        failure_step = rollout.failure_step
        if failure_step is None:
            failure_step = ep_len
        if ep_len > 0 and failure_step > self._LATE_THRESHOLD * ep_len:
            return failure_step
        return None


class _RecoveryDetector:
    """
    Triggers when the action sequence contains repeated cycles (same action
    appears >= 2 times within a sliding window) AND success=False.
    Uses a window of 10 steps; checks for near-duplicate actions (L2 dist < 0.05).
    """

    _WINDOW = 10
    _REPEAT_THRESHOLD = 2
    _DIST_THRESHOLD = 0.05

    def detect(self, trace: EpisodeTrace) -> int | None:
        if trace.rollout.success:
            return None
        actions = trace.rollout.actions
        if len(actions) < self._REPEAT_THRESHOLD:
            return None
        window = actions[-self._WINDOW:]
        # Check all pairs within the window for near-duplicates.
        for i in range(len(window)):
            count = 1
            for j in range(i + 1, len(window)):
                if _l2_dist(window[i], window[j]) < self._DIST_THRESHOLD:
                    count += 1
                if count >= self._REPEAT_THRESHOLD:
                    # failure_step = first position of the cycle in full sequence
                    return max(0, len(actions) - self._WINDOW + i)
        return None


class _PoseEstimationDetector:
    """
    Triggers when failure occurs early (failure_step < 0.3 * episode_length),
    suggesting the model never reached the correct approach pose.
    """

    _EARLY_THRESHOLD = 0.3

    def detect(self, trace: EpisodeTrace) -> int | None:
        rollout = trace.rollout
        if rollout.success:
            return None
        ep_len = rollout.episode_length
        if ep_len == 0:
            return None
        failure_step = rollout.failure_step if rollout.failure_step is not None else ep_len
        if failure_step < self._EARLY_THRESHOLD * ep_len:
            return failure_step
        return None


class _DistractorConfusionDetector:
    """
    Triggers when any observation contains 'num_distractors' > 0 AND success=False.
    """

    def detect(self, trace: EpisodeTrace) -> int | None:
        if trace.rollout.success:
            return None
        for step_idx, obs in enumerate(trace.rollout.observations):
            if obs.get("num_distractors", 0) > 0:
                return step_idx
        return None


class _LanguageGroundingDetector:
    """
    Triggers when obs contains 'object_class' key that differs from the expected
    object encoded in task_instruction, indicating a language-grounding mismatch.
    """

    def detect(self, trace: EpisodeTrace) -> int | None:
        if trace.rollout.success:
            return None
        instruction = trace.identity.task_instruction.lower()
        for step_idx, obs in enumerate(trace.rollout.observations):
            if "object_class" in obs:
                obj_class = str(obs["object_class"]).lower()
                if obj_class and obj_class not in instruction:
                    return step_idx
        return None


# ---------------------------------------------------------------------------
# FailureStepDetector registry  (one instance per type)
# ---------------------------------------------------------------------------

_DETECTORS: dict[str, Any] = {
    "timeout": _TimeoutDetector(),
    "out_of_reach": _OutOfReachDetector(),
    "grasp": _GraspDetector(),
    "contact_dynamics": _ContactDynamicsDetector(),
    "recovery": _RecoveryDetector(),
    "pose_estimation": _PoseEstimationDetector(),
    "distractor_confusion": _DistractorConfusionDetector(),
    "language_grounding": _LanguageGroundingDetector(),
}


# ---------------------------------------------------------------------------
# Public classifier
# ---------------------------------------------------------------------------

class FailureClassifier:
    """Heuristic failure type classifier for VLA episode traces."""

    def __init__(self, max_episode_steps: int = _DEFAULT_MAX_STEPS) -> None:
        self._max_episode_steps = max_episode_steps

    # ------------------------------------------------------------------
    # Internal per-type detectors
    # ------------------------------------------------------------------

    def _detect_timeout(self, trace: EpisodeTrace) -> int | None:
        return _DETECTORS["timeout"].detect(trace, self._max_episode_steps)

    def _detect_out_of_reach(self, trace: EpisodeTrace) -> int | None:
        return _DETECTORS["out_of_reach"].detect(trace)

    def _detect_grasp(self, trace: EpisodeTrace) -> int | None:
        return _DETECTORS["grasp"].detect(trace)

    def _detect_contact_dynamics(self, trace: EpisodeTrace) -> int | None:
        return _DETECTORS["contact_dynamics"].detect(trace)

    def _detect_recovery(self, trace: EpisodeTrace) -> int | None:
        return _DETECTORS["recovery"].detect(trace)

    def _detect_pose_estimation(self, trace: EpisodeTrace) -> int | None:
        return _DETECTORS["pose_estimation"].detect(trace)

    def _detect_distractor_confusion(self, trace: EpisodeTrace) -> int | None:
        return _DETECTORS["distractor_confusion"].detect(trace)

    def _detect_language_grounding(self, trace: EpisodeTrace) -> int | None:
        return _DETECTORS["language_grounding"].detect(trace)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, trace: EpisodeTrace) -> tuple[str | None, int]:
        """
        Returns (failure_type, failure_step).

        failure_type is None if the trace cannot be classified.
        failure_step falls back to episode_length when unclassified.
        """
        detector_map = {
            "timeout": self._detect_timeout,
            "out_of_reach": self._detect_out_of_reach,
            "grasp": self._detect_grasp,
            "contact_dynamics": self._detect_contact_dynamics,
            "recovery": self._detect_recovery,
            "pose_estimation": self._detect_pose_estimation,
            "distractor_confusion": self._detect_distractor_confusion,
            "language_grounding": self._detect_language_grounding,
        }
        for failure_type in _PRIORITY_ORDER:
            step = detector_map[failure_type](trace)
            if step is not None:
                return failure_type, step
        return None, trace.rollout.episode_length

    def classify_and_update(self, trace: EpisodeTrace) -> EpisodeTrace:
        """
        Classify and update trace.labels.failure_type and
        trace.rollout.failure_step in-place.  Returns the same trace.
        """
        failure_type, failure_step = self.classify(trace)
        trace.labels.failure_type = failure_type
        trace.rollout.failure_step = failure_step
        return trace


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _l2_norm(action: Any) -> float:
    """L2 norm of an action vector."""
    if hasattr(action, "__iter__"):
        return math.sqrt(sum(float(x) ** 2 for x in action))
    return abs(float(action))


def _l2_dist(a: Any, b: Any) -> float:
    """L2 distance between two action vectors."""
    if hasattr(a, "__iter__") and hasattr(b, "__iter__"):
        return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))
    return abs(float(a) - float(b))


def _variance(values: list[float]) -> float:
    """Population variance."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / n
