"""
Online STDP Adapter for Test-Time Adaptation (TTA-STDP).

Adapts SNN action head weights during evaluation using local STDP learning rules
and self-supervised reward signals. No backpropagation needed.

Usage:
    model = BaseFramework.from_pretrained(ckpt_path).to("cuda").eval()
    adapter = OnlineSTDPAdapter(model)
    adapter.enable()

    for episode in episodes:
        adapter.on_episode_start()
        for step in episode:
            # Model forward pass (SpikeMonitor hooks record spikes)
            result = model.predict_action(images, instructions, states)
            adapter.on_inference_done()

            # Execute action, observe next state
            action = result["normalized_actions"][0, 0]
            next_state = env.step(action)

            # Update SNN weights via online STDP
            adapter.update_step(state_prev, state_curr, action_executed, prev_action)

        adapter.on_episode_end(success=done)
    adapter.disable()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from AlphaBrain.model.modules.action_model.stdp.spike_monitor import SpikeMonitor
from AlphaBrain.model.modules.action_model.stdp.stdp_learner import STDPLearner, RewardBaseline
from AlphaBrain.model.modules.action_model.stdp.reward_computer import RewardComputer

logger = logging.getLogger(__name__)


@dataclass
class OnlineSTDPConfig:
    """Configuration for online STDP adaptation."""

    # STDP learning parameters (conservative for online)
    A_plus: float = 0.002
    A_minus: float = 0.0024        # A-/A+ = 1.2 for homeostatic balance
    tau_plus: float = 20.0
    tau_minus: float = 20.0
    trace_decay: float = 0.95
    stdp_lr: float = 1e-5          # 10x smaller than offline

    # Reward weights
    w_spe: float = 0.5
    w_smooth: float = 0.3
    w_consist: float = 0.2

    # Stability
    max_deviation: float = 0.1     # max L2 drift from initial weights per param
    weight_clip: float = 0.2
    max_update_norm: float = 0.02  # per-param update norm cap
    rollback_shrink: float = 0.7   # on failure, undo this fraction of episode delta
    lr_decay: float = 0.995        # per-episode lr decay
    warmup_episodes: int = 2       # collect reward stats before updating

    # Reward normalization
    reward_baseline_momentum: float = 0.95
    reward_clip: float = 1.0

    # Binarize pre-synaptic activity (same as offline STDP)
    binarize_pre: bool = True


class OnlineSTDPAdapter:
    """
    Wraps a NeuroVLA model to apply online STDP weight updates to the SNN
    action head during evaluation.

    Key design:
    - Only updates SNN weights (4 Linear layers preceding LIF neurons)
    - Uses local STDP rules (no backpropagation needed)
    - Self-supervised reward from state prediction error + action smoothness
    - Delta tracking for efficient rollback (no per-episode weight snapshots)
    - Max deviation bound prevents catastrophic drift from initial weights
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[OnlineSTDPConfig] = None,
    ):
        """
        Args:
            model: NeuroVLA model instance (already loaded and on device).
            config: Online STDP configuration. Uses defaults if None.
        """
        self.model = model
        self.config = config or OnlineSTDPConfig()

        # Extract the SNN action model (MLPResNet inside L1RegressionActionHead)
        self.snn_model = model.action_model.model
        self.device = next(self.snn_model.parameters()).device

        # Core STDP components (reuse existing implementations)
        self.spike_monitor = SpikeMonitor(self.snn_model, device=self.device)
        self.stdp_learner = STDPLearner(
            A_plus=self.config.A_plus,
            A_minus=self.config.A_minus,
            tau_plus=self.config.tau_plus,
            tau_minus=self.config.tau_minus,
            trace_decay=self.config.trace_decay,
            weight_clip=self.config.weight_clip,
            device=self.device,
            binarize_pre=self.config.binarize_pre,
        )
        self.reward_computer = RewardComputer(
            w_spe=self.config.w_spe,
            w_smooth=self.config.w_smooth,
            w_consist=self.config.w_consist,
            reward_baseline_momentum=self.config.reward_baseline_momentum,
            reward_clip=self.config.reward_clip,
        )

        # Identify STDP-eligible weight parameters (Linear layers before LIF neurons)
        self._stdp_pairs: List[Tuple[str, nn.Linear]] = []
        self._discover_stdp_pairs()

        # Delta tracking: save initial weights once, track per-episode deltas
        self._w_init: Dict[int, torch.Tensor] = {}
        self._episode_delta: Dict[int, torch.Tensor] = {}
        self._save_initial_weights()

        # Current effective learning rate (decays per episode)
        self._effective_lr: float = self.config.stdp_lr

        # Pending STDP eligibility traces from last inference (applied on update_step)
        self._pending_eligibility: Dict[int, torch.Tensor] = {}

        # Episode and global counters
        self._episode_count: int = 0
        self._total_steps: int = 0
        self._episode_steps: int = 0

        # Stats for logging
        self._episode_rewards: List[float] = []
        self._last_stats: Dict[str, float] = {}

        self._enabled = False

    def _discover_stdp_pairs(self):
        """Find (layer_name, linear_module) pairs eligible for STDP updates."""
        self._stdp_pairs = [
            (name, linear)
            for name, linear, lif in self.spike_monitor.lif_linear_pairs
            if linear is not None
        ]
        logger.info(
            f"Online STDP: found {len(self._stdp_pairs)} eligible (Linear->LIF) pairs: "
            f"{[name for name, _ in self._stdp_pairs]}"
        )

    def _save_initial_weights(self):
        """Save initial weights once for delta tracking and deviation bounds."""
        for _, linear in self._stdp_pairs:
            pid = id(linear.weight)
            self._w_init[pid] = linear.weight.data.clone()
            self._episode_delta[pid] = torch.zeros_like(linear.weight.data)

    def enable(self):
        """Enable spike monitoring hooks. Call before starting evaluation."""
        self.spike_monitor.enable()
        self._enabled = True
        logger.info("Online STDP adapter enabled.")

    def disable(self):
        """Disable spike monitoring hooks. Call after evaluation is done."""
        self.spike_monitor.disable()
        self._enabled = False
        logger.info("Online STDP adapter disabled.")

    # ──────────────────────────────────────────────────────────────────────
    # Episode lifecycle
    # ──────────────────────────────────────────────────────────────────────

    def on_episode_start(self):
        """Reset per-episode state. Call at the start of each evaluation episode."""
        # Zero episode delta accumulators
        for pid in self._episode_delta:
            self._episode_delta[pid].zero_()

        # Reset STDP traces and pending eligibility
        self.stdp_learner.reset()
        self._pending_eligibility.clear()

        # Reset reward computer state (but keep baseline stats)
        self.reward_computer.reset()

        self._episode_steps = 0
        self._episode_rewards.clear()

    def on_episode_end(self, success: bool):
        """
        Handle episode completion. Call after episode finishes.

        On failure: partially rollback SNN weights by subtracting a fraction
        of the accumulated episode delta.

        Args:
            success: Whether the episode task was completed successfully.
        """
        if not success and self._episode_count >= self.config.warmup_episodes:
            self._rollback_episode()

        # Enforce global deviation bound
        self._enforce_deviation_bound()

        # Decay learning rate
        self._effective_lr *= self.config.lr_decay

        self._episode_count += 1

        # Log episode stats
        avg_reward = (
            np.mean(self._episode_rewards) if self._episode_rewards else 0.0
        )
        weight_drift = self._compute_total_drift()
        logger.info(
            f"Episode {self._episode_count}: "
            f"success={success}, "
            f"steps={self._episode_steps}, "
            f"avg_reward={avg_reward:.4f}, "
            f"weight_drift={weight_drift:.6f}, "
            f"lr={self._effective_lr:.2e}"
        )

        self._last_stats.update({
            "online_stdp/episode": self._episode_count,
            "online_stdp/episode_success": float(success),
            "online_stdp/episode_avg_reward": avg_reward,
            "online_stdp/weight_drift": weight_drift,
            "online_stdp/effective_lr": self._effective_lr,
        })

    # ──────────────────────────────────────────────────────────────────────
    # Per-step operations
    # ──────────────────────────────────────────────────────────────────────

    def on_inference_done(self):
        """
        Process spike data after model forward pass. Call after predict_action().

        Computes STDP updates from recorded spikes and stores them as pending
        eligibility traces. The actual weight update is deferred to update_step()
        when the reward signal is available.
        """
        if not self._enabled:
            return

        spike_records = self.spike_monitor.get_records()
        linear_records = self.spike_monitor.get_linear_records()
        self._pending_eligibility.clear()

        for layer_name, linear in self._stdp_pairs:
            record = spike_records.get(layer_name)
            if record is None or record.num_timesteps == 0:
                continue

            linear_record = linear_records.get(layer_name)
            if linear_record is None or linear_record.num_timesteps == 0:
                continue

            pre_activity = linear_record.get_input_tensor()  # [T, B, D_in]
            if pre_activity is None:
                continue

            post_spikes = record.get_spike_tensor()  # [T, B, D_out]
            T = min(post_spikes.shape[0], pre_activity.shape[0])

            weight_param = linear.weight

            # Reset per-timestep traces for this layer
            self.stdp_learner.reset_traces()

            for t in range(T):
                dw = self.stdp_learner.compute_trace_stdp(
                    pre_spikes=pre_activity[t],
                    post_spikes=post_spikes[t],
                    layer_name=layer_name,
                    dt=1.0,
                )
                self.stdp_learner.accumulate_eligibility(weight_param, dw)

            trace = self.stdp_learner.get_eligibility_trace(weight_param)
            if trace is not None and T > 0:
                # Normalize by timesteps to prevent over-accumulation
                self._pending_eligibility[id(weight_param)] = trace / T

        # Reset spike monitor for next forward pass
        self.spike_monitor.reset()
        # Reset eligibility in learner (we saved what we need in _pending_eligibility)
        self.stdp_learner.reset_eligibility()

    def update_step(
        self,
        state_prev: np.ndarray,
        state_curr: np.ndarray,
        action_executed: np.ndarray,
        prev_action: Optional[np.ndarray] = None,
    ):
        """
        Apply reward-modulated STDP weight updates. Call after executing an action
        and observing the next state.

        Args:
            state_prev: Robot state before action [8].
            state_curr: Robot state after action [8].
            action_executed: The unnormalized delta action executed [7].
            prev_action: Previous step's action [7], or None.
        """
        if not self._enabled or not self._pending_eligibility:
            return

        # Skip weight updates during warmup (only collect reward stats)
        in_warmup = self._episode_count < self.config.warmup_episodes

        # Compute self-supervised reward
        reward = self.reward_computer.compute_step_reward(
            state_prev=state_prev,
            state_curr=state_curr,
            action_executed=action_executed,
            prev_action=prev_action,
        )
        self._episode_rewards.append(reward)

        if in_warmup:
            self._episode_steps += 1
            self._total_steps += 1
            return

        # Apply reward-modulated STDP updates
        total_update_norm = 0.0
        for layer_name, linear in self._stdp_pairs:
            pid = id(linear.weight)
            if pid not in self._pending_eligibility:
                continue

            eligibility = self._pending_eligibility[pid]

            # R-STDP: dw = reward * lr * eligibility
            update = reward * self._effective_lr * eligibility

            # Clip per-parameter update norm
            update_norm = update.norm().item()
            if update_norm > self.config.max_update_norm:
                update = update * (self.config.max_update_norm / update_norm)
                update_norm = self.config.max_update_norm

            total_update_norm += update_norm

            # Apply update to weight data (bypasses autograd)
            if update.shape == linear.weight.data.shape:
                linear.weight.data.add_(update)
                # Track episode delta for rollback
                self._episode_delta[pid].add_(update)

        self._episode_steps += 1
        self._total_steps += 1

        # Record stats
        self._last_stats.update({
            "online_stdp/step_reward": reward,
            "online_stdp/update_norm": total_update_norm,
            "online_stdp/total_steps": self._total_steps,
        })

    def update_chunk_consistency(
        self,
        prev_chunk: Optional[np.ndarray],
        curr_chunk: np.ndarray,
    ):
        """
        Provide chunk consistency signal (optional, called when a new action
        chunk is predicted).

        Args:
            prev_chunk: Previous predicted action chunk [chunk_size, 7].
            curr_chunk: Current predicted action chunk [chunk_size, 7].
        """
        self.reward_computer.compute_chunk_consistency(prev_chunk, curr_chunk)

    # ──────────────────────────────────────────────────────────────────────
    # Stability mechanisms
    # ──────────────────────────────────────────────────────────────────────

    def _rollback_episode(self):
        """Partially undo this episode's weight changes on failure."""
        shrink = self.config.rollback_shrink
        for _, linear in self._stdp_pairs:
            pid = id(linear.weight)
            if pid in self._episode_delta:
                delta = self._episode_delta[pid]
                if delta.norm().item() > 0:
                    linear.weight.data.sub_(shrink * delta)

        logger.info(
            f"Rolled back {shrink*100:.0f}% of episode {self._episode_count+1} "
            f"weight changes (failure)."
        )

    def _enforce_deviation_bound(self):
        """
        Clip total weight drift from initial checkpoint.
        Projects weights back onto the L2 ball of radius max_deviation
        centered at w_init.
        """
        max_dev = self.config.max_deviation
        for _, linear in self._stdp_pairs:
            pid = id(linear.weight)
            if pid not in self._w_init:
                continue
            drift = linear.weight.data - self._w_init[pid]
            drift_norm = drift.norm().item()
            if drift_norm > max_dev:
                linear.weight.data.copy_(
                    self._w_init[pid] + max_dev * (drift / drift_norm)
                )

    def _compute_total_drift(self) -> float:
        """Compute total L2 drift from initial weights across all STDP layers."""
        total = 0.0
        for _, linear in self._stdp_pairs:
            pid = id(linear.weight)
            if pid in self._w_init:
                drift = (linear.weight.data - self._w_init[pid]).norm().item()
                total += drift
        return total

    # ──────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, float]:
        """Return stats from the last step/episode for logging."""
        # Add spike rates
        spike_rates = self.spike_monitor.get_spike_rates()
        for name, rate in spike_rates.items():
            safe_name = name.replace(".", "_")
            self._last_stats[f"online_stdp/spike_rate_{safe_name}"] = rate

        return self._last_stats.copy()

    def save_adapted_weights(self, path: str):
        """
        Save the adapted model state dict (full model, not just SNN).

        Args:
            path: Output path for the checkpoint file.
        """
        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved adapted model weights to {path}")

    @property
    def episode_count(self) -> int:
        return self._episode_count

    @property
    def total_steps(self) -> int:
        return self._total_steps

    def reset_to_initial(self):
        """
        Reset all STDP-eligible weights back to initial checkpoint values and
        clear all adaptation state. Call between tasks to prevent cross-task
        interference.
        """
        # Restore initial weights
        for _, linear in self._stdp_pairs:
            pid = id(linear.weight)
            if pid in self._w_init:
                linear.weight.data.copy_(self._w_init[pid])
                self._episode_delta[pid].zero_()

        # Reset internal state
        self._effective_lr = self.config.stdp_lr
        self._pending_eligibility.clear()
        self._episode_count = 0
        self._total_steps = 0
        self._episode_steps = 0
        self._episode_rewards.clear()
        self._last_stats.clear()

        # Reset STDP learner traces
        self.stdp_learner.reset()
        self.reward_computer.reset()

        logger.info("Online STDP adapter reset to initial checkpoint weights.")

    @property
    def is_in_warmup(self) -> bool:
        return self._episode_count < self.config.warmup_episodes
