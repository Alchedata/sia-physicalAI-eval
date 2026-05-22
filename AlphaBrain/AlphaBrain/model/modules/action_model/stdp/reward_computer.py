"""
Self-supervised reward computation for Online STDP.

Computes reward signals from environment interaction without ground truth actions:
- State Prediction Error (SPE): predicted vs actual next state
- Action Temporal Smoothness: penalizes jerk in action sequence
- Action Chunk Consistency: agreement between overlapping predictions
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from AlphaBrain.model.modules.action_model.stdp.stdp_learner import RewardBaseline


class RewardComputer:
    """
    Computes self-supervised reward signals for online STDP during evaluation.

    Combined reward:
        r = w_spe * r_spe + w_smooth * r_smooth + w_consist * r_consist

    All component rewards are negative costs (higher = better).
    The combined reward is normalized via RewardBaseline (Welford running stats).
    """

    def __init__(
        self,
        w_spe: float = 0.5,
        w_smooth: float = 0.3,
        w_consist: float = 0.2,
        reward_baseline_momentum: float = 0.95,
        reward_clip: float = 1.0,
    ):
        self.w_spe = w_spe
        self.w_smooth = w_smooth
        self.w_consist = w_consist

        self._baseline = RewardBaseline(
            momentum=reward_baseline_momentum,
            clip_range=reward_clip,
        )

        # State for tracking previous actions and chunks
        self._prev_action: Optional[np.ndarray] = None
        self._prev_chunk: Optional[np.ndarray] = None
        self._step_count: int = 0

    def compute_step_reward(
        self,
        state_prev: np.ndarray,
        state_curr: np.ndarray,
        action_executed: np.ndarray,
        prev_action: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute combined per-step reward from self-supervised signals.

        Args:
            state_prev: Robot state before action [8] (x,y,z,roll,pitch,yaw,gripper_qpos0,gripper_qpos1).
            state_curr: Robot state after action [8].
            action_executed: The delta action that was executed [7] (x,y,z,roll,pitch,yaw,gripper).
            prev_action: Previous step's action [7], or None if first step.

        Returns:
            Normalized combined reward scalar.
        """
        raw_reward = 0.0

        # 1. State Prediction Error (SPE)
        # The action predicts a delta in position/orientation space.
        # Compare predicted next state with actual next state for position dims (0:6).
        r_spe = self._compute_spe(state_prev, state_curr, action_executed)
        raw_reward += self.w_spe * r_spe

        # 2. Action Temporal Smoothness
        if prev_action is None:
            prev_action = self._prev_action
        if prev_action is not None:
            r_smooth = self._compute_smoothness(action_executed, prev_action)
            raw_reward += self.w_smooth * r_smooth

        # Update history
        self._prev_action = action_executed.copy()
        self._step_count += 1

        # Normalize via running baseline
        normalized_reward = self._baseline.normalize(raw_reward)
        return normalized_reward

    def compute_chunk_consistency(
        self,
        prev_chunk: Optional[np.ndarray],
        curr_chunk: np.ndarray,
    ) -> float:
        """
        Compute consistency reward between overlapping action chunks.

        When using action chunking (e.g., 8 actions per inference), consecutive
        predictions should agree on overlapping timesteps.

        Args:
            prev_chunk: Previous chunk of predicted actions [chunk_size, 7], or None.
            curr_chunk: Current chunk of predicted actions [chunk_size, 7].

        Returns:
            Raw consistency reward (negative L2 distance of overlap, 0 if no overlap).
        """
        if prev_chunk is None:
            prev_chunk = self._prev_chunk
        if prev_chunk is None:
            self._prev_chunk = curr_chunk.copy()
            return 0.0

        chunk_size = curr_chunk.shape[0]

        # prev_chunk was predicted at step t, curr_chunk at step t+chunk_size.
        # If we re-predict every chunk_size steps, there's no overlap in the standard
        # chunking scheme. But if we re-predict more frequently (e.g., every step),
        # there is overlap.
        # For standard chunking (re-predict every chunk_size steps), we compare
        # the last prediction of prev_chunk with the first prediction of curr_chunk
        # as a continuity check.
        prev_last = prev_chunk[-1]  # last action of previous chunk
        curr_first = curr_chunk[0]  # first action of current chunk
        diff = np.linalg.norm(prev_last - curr_first)
        r_consist = -diff

        self._prev_chunk = curr_chunk.copy()
        return r_consist

    def _compute_spe(
        self,
        state_prev: np.ndarray,
        state_curr: np.ndarray,
        action_executed: np.ndarray,
    ) -> float:
        """
        State Prediction Error: how well does the action predict the state transition.

        For position dims (0:3), the action IS the delta:
            predicted_pos = state_prev[0:3] + action[0:3]
            error = ||predicted_pos - state_curr[0:3]||

        For orientation dims (3:6), similar delta prediction.

        Returns negative error (higher = better prediction = more reward).
        """
        # Position prediction error
        predicted_pos = state_prev[:3] + action_executed[:3]
        pos_error = np.linalg.norm(predicted_pos - state_curr[:3])

        # Orientation prediction error
        predicted_orient = state_prev[3:6] + action_executed[3:6]
        orient_error = np.linalg.norm(predicted_orient - state_curr[3:6])

        # Combined SPE (position weighted more heavily as it's the primary signal)
        total_error = pos_error + 0.5 * orient_error
        return -total_error

    def _compute_smoothness(
        self,
        action_curr: np.ndarray,
        action_prev: np.ndarray,
    ) -> float:
        """
        Action smoothness: penalize large changes between consecutive actions.

        Smooth robot motion is generally correct; sudden jumps indicate errors.
        Only considers the continuous dims (0:6), not gripper (binary).

        Returns negative jerk (higher = smoother = more reward).
        """
        # Only penalize continuous action dims (position + orientation), not gripper
        diff = np.linalg.norm(action_curr[:6] - action_prev[:6])
        return -diff

    def reset(self):
        """Reset state at episode start."""
        self._prev_action = None
        self._prev_chunk = None
        self._step_count = 0
        # Note: do NOT reset _baseline — reward statistics should persist across episodes
        # for stable normalization.

    @property
    def step_count(self) -> int:
        return self._step_count
