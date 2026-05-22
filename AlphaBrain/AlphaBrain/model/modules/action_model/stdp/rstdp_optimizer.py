"""
Reward-Modulated STDP Optimizer.

Combines standard backpropagation gradients with R-STDP local learning rules
for hybrid training of Spiking Neural Networks. Compatible with PyTorch
training loops and DeepSpeed ZeRO-2.

Usage:
    monitor = SpikeMonitor(snn_model)
    stdp_learner = STDPLearner(...)
    rstdp = RSTDPOptimizer(
        snn_model=snn_model,
        base_optimizer=adam_optimizer,
        spike_monitor=monitor,
        stdp_learner=stdp_learner,
        alpha=0.7, beta=0.3,
    )

    # In training loop:
    monitor.reset()
    output = model(batch)
    loss = compute_loss(output, target)
    loss.backward()                    # backprop gradients computed
    reward = -loss.item()
    rstdp.step(reward=reward)          # hybrid update: alpha*backprop + beta*STDP
    rstdp.zero_grad()
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import snntorch as snn

from AlphaBrain.model.modules.action_model.stdp.spike_monitor import SpikeMonitor
from AlphaBrain.model.modules.action_model.stdp.stdp_learner import STDPLearner, RewardBaseline


class RSTDPOptimizer:
    """
    Hybrid optimizer that blends backpropagation gradients with R-STDP updates.

    Final weight update per step:
        Δw = α · Δw_backprop + β · Δw_rstdp

    where Δw_rstdp = reward_normalized · eligibility_trace, and the eligibility
    trace is accumulated from per-timestep STDP updates during the forward pass.
    """

    def __init__(
        self,
        snn_model: nn.Module,
        base_optimizer: torch.optim.Optimizer,
        spike_monitor: SpikeMonitor,
        stdp_learner: STDPLearner,
        alpha: float = 0.7,
        beta: float = 0.3,
        mode: str = "hybrid",
        reward_baseline_momentum: float = 0.99,
        stdp_lr: float = 1e-4,
        max_update_norm: float = 0.1,
        warmup_steps: int = 500,
        align_with_grad: bool = True,
    ):
        """
        Args:
            snn_model: The SNN action model (e.g., MLPResNet).
            base_optimizer: Standard PyTorch optimizer for backprop updates.
            spike_monitor: SpikeMonitor attached to snn_model.
            stdp_learner: STDPLearner with STDP parameters.
            alpha: Weight for backpropagation gradient (0.0 to 1.0).
            beta: Weight for STDP update (0.0 to 1.0).
            mode: "hybrid" (alpha*bp + beta*stdp) or "pure_stdp" (only STDP).
            reward_baseline_momentum: Momentum for reward baseline.
            stdp_lr: Learning rate for STDP updates (scales eligibility trace).
            max_update_norm: Max per-parameter update norm for STDP.
            warmup_steps: Number of pure-backprop steps before STDP activates.
            align_with_grad: If True, filter STDP updates that oppose gradient descent.
        """
        self.snn_model = snn_model
        self.base_optimizer = base_optimizer
        self.spike_monitor = spike_monitor
        self.stdp_learner = stdp_learner
        self.alpha = alpha
        self.beta = beta
        self.mode = mode
        self.stdp_lr = stdp_lr
        self.max_update_norm = max_update_norm
        self.warmup_steps = warmup_steps
        self.align_with_grad = align_with_grad
        self._current_step = 0
        self.reward_baseline = RewardBaseline(momentum=reward_baseline_momentum)

        # Identify weight parameters eligible for STDP updates
        # These are Linear layers that immediately precede LIF neurons
        self._stdp_pairs: List[Tuple[str, nn.Linear, snn.Leaky]] = []
        self._discover_stdp_pairs()

        # Stats for logging
        self._last_stats: Dict[str, float] = {}

    def _discover_stdp_pairs(self):
        """Find (linear_layer, lif_layer) pairs for STDP application."""
        self._stdp_pairs = self.spike_monitor.lif_linear_pairs

    def compute_stdp_updates(self) -> Dict[int, torch.Tensor]:
        """
        Compute STDP weight updates from recorded spike data.

        For each (Linear → LIF) pair:
        1. Retrieve pre-synaptic (input to Linear) and post-synaptic (LIF output) spikes
        2. Compute per-timestep trace-based STDP
        3. Accumulate into eligibility traces

        Returns:
            Dict mapping weight parameter id → accumulated STDP update tensor.
        """
        spike_records = self.spike_monitor.get_records()
        linear_records = self.spike_monitor.get_linear_records()
        updates = {}

        for layer_name, linear, lif in self._stdp_pairs:
            if linear is None:
                continue

            record = spike_records.get(layer_name)
            if record is None or record.num_timesteps == 0:
                continue

            # Get pre-synaptic activity: input to the Linear layer [T, B, D_in]
            linear_record = linear_records.get(layer_name)
            if linear_record is None or linear_record.num_timesteps == 0:
                continue
            pre_activity = linear_record.get_input_tensor()  # [T, B, D_in]
            if pre_activity is None:
                continue

            weight_param = linear.weight  # [D_out, D_in]

            # Reset per-timestep traces for this layer
            self.stdp_learner.reset_traces()

            # Get post-synaptic spikes from LIF output: [T, B, D_out]
            post_spikes = record.get_spike_tensor()

            T = min(post_spikes.shape[0], pre_activity.shape[0])

            for t in range(T):
                # pre: input to Linear at time t [B, D_in]
                # post: spike output of LIF at time t [B, D_out]
                # STDP update dw will be [D_out, D_in] matching weight shape
                pre_t = pre_activity[t]   # [B, D_in]
                post_t = post_spikes[t]   # [B, D_out]

                dw = self.stdp_learner.compute_trace_stdp(
                    pre_spikes=pre_t,
                    post_spikes=post_t,
                    layer_name=layer_name,
                    dt=1.0,
                )

                # dw is [D_out, D_in] — matches linear.weight shape
                self.stdp_learner.accumulate_eligibility(weight_param, dw)

            trace = self.stdp_learner.get_eligibility_trace(weight_param)
            # Normalize by number of timesteps to prevent over-accumulation
            if trace is not None and T > 0:
                trace = trace / T
            updates[id(weight_param)] = trace

        return updates

    def _get_effective_beta(self) -> float:
        """Get scheduled beta with warmup and cosine ramp-up."""
        if self._current_step < self.warmup_steps:
            return 0.0
        # Cosine ramp-up over warmup_steps duration after warmup period
        ramp_end = self.warmup_steps * 2
        if self._current_step >= ramp_end:
            return self.beta
        t = (self._current_step - self.warmup_steps) / max(self.warmup_steps, 1)
        return self.beta * 0.5 * (1.0 - math.cos(math.pi * t))

    def step(self, reward: Optional[float] = None):
        """
        Execute one optimization step with hybrid backprop + STDP.

        Args:
            reward: Scalar reward signal for R-STDP modulation.
                    If None, uses raw STDP updates without reward modulation.
        """
        effective_beta = self._get_effective_beta()

        # Compute STDP updates from recorded spikes (skip if beta=0)
        if effective_beta > 0:
            stdp_updates = self.compute_stdp_updates()
        else:
            stdp_updates = {}

        # Normalize reward
        if reward is not None:
            reward_normalized = self.reward_baseline.normalize(reward)
        else:
            reward_normalized = 1.0

        if self.mode == "hybrid":
            # Use full gradients during warmup, alpha-scaled once STDP is active
            bp_scale = self.alpha if effective_beta > 0 else 1.0
            self._scale_backprop_grads(bp_scale)

            # Apply backprop step
            self.base_optimizer.step()

            # Apply STDP updates (additive, after optimizer step)
            if effective_beta > 0:
                self._apply_stdp_updates(stdp_updates, reward_normalized, effective_beta)

        elif self.mode == "pure_stdp":
            self._apply_stdp_updates(stdp_updates, reward_normalized, scale=1.0)

        else:
            raise ValueError(f"Unknown mode: {self.mode}. Expected 'hybrid' or 'pure_stdp'.")

        self._current_step += 1

        # Reset eligibility traces for next step
        self.stdp_learner.reset_eligibility()

        # Record stats
        self._record_stats(stdp_updates, reward_normalized, effective_beta)

    def _scale_backprop_grads(self, scale: float):
        """Scale all backprop gradients by a factor."""
        if scale == 1.0:
            return
        for group in self.base_optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.mul_(scale)

    def _apply_stdp_updates(
        self,
        stdp_updates: Dict[int, torch.Tensor],
        reward: float,
        scale: float,
    ):
        """Apply STDP weight updates to model parameters with proper scaling."""
        for layer_name, linear, lif in self._stdp_pairs:
            if linear is None:
                continue
            pid = id(linear.weight)
            if pid not in stdp_updates or stdp_updates[pid] is None:
                continue

            eligibility = stdp_updates[pid]
            # R-STDP: modulate by reward, scale by beta and learning rate
            update = scale * reward * self.stdp_lr * eligibility

            # Clip update norm to prevent explosive weight changes
            update_norm = update.norm()
            if update_norm > self.max_update_norm:
                update = update * (self.max_update_norm / update_norm)

            # Gradient alignment: filter STDP updates that oppose gradient descent
            if self.align_with_grad and linear.weight.grad is not None:
                neg_grad = -linear.weight.grad.data  # descent direction
                dot = (update * neg_grad).sum()
                if dot < 0:
                    # Remove anti-aligned component via projection
                    neg_grad_norm_sq = neg_grad.norm() ** 2 + 1e-8
                    proj = (dot / neg_grad_norm_sq) * neg_grad
                    update = update - proj

            # Update shape [D_out, D_in] should match weight shape
            if update.shape == linear.weight.data.shape:
                linear.weight.data.add_(update)

    def zero_grad(self):
        """Zero gradients in the base optimizer."""
        self.base_optimizer.zero_grad()

    def _record_stats(self, stdp_updates: Dict[int, torch.Tensor], reward: float,
                       effective_beta: Optional[float] = None):
        """Record statistics for logging."""
        total_stdp_norm = 0.0
        num_updates = 0
        for update in stdp_updates.values():
            if update is not None:
                total_stdp_norm += update.norm().item()
                num_updates += 1

        self._last_stats = {
            "stdp/reward_normalized": reward,
            "stdp/reward_baseline": self.reward_baseline.baseline,
            "stdp/update_norm": total_stdp_norm / max(num_updates, 1),
            "stdp/num_updated_layers": num_updates,
            "stdp/effective_beta": effective_beta if effective_beta is not None else self.beta,
            "stdp/current_step": self._current_step,
        }

        # Add spike rates
        spike_rates = self.spike_monitor.get_spike_rates()
        for name, rate in spike_rates.items():
            safe_name = name.replace(".", "_")
            self._last_stats[f"stdp/spike_rate_{safe_name}"] = rate

    def get_stats(self) -> Dict[str, float]:
        """Return stats from the last step for logging."""
        return self._last_stats.copy()

    @property
    def param_groups(self):
        """Expose base optimizer's param_groups for LR scheduling."""
        return self.base_optimizer.param_groups

    def state_dict(self) -> dict:
        """Return optimizer state for checkpointing."""
        return {
            "base_optimizer": self.base_optimizer.state_dict(),
            "alpha": self.alpha,
            "beta": self.beta,
            "mode": self.mode,
            "stdp_lr": self.stdp_lr,
            "max_update_norm": self.max_update_norm,
            "reward_baseline": self.reward_baseline.baseline,
            "reward_var": self.reward_baseline._var,
            "current_step": self._current_step,
        }

    def load_state_dict(self, state_dict: dict):
        """Load optimizer state from checkpoint."""
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])
        self.alpha = state_dict.get("alpha", self.alpha)
        self.beta = state_dict.get("beta", self.beta)
        self.mode = state_dict.get("mode", self.mode)
        self.stdp_lr = state_dict.get("stdp_lr", self.stdp_lr)
        self.max_update_norm = state_dict.get("max_update_norm", self.max_update_norm)
        self._current_step = state_dict.get("current_step", 0)
        if "reward_baseline" in state_dict:
            self.reward_baseline._baseline = state_dict["reward_baseline"]
        if "reward_var" in state_dict:
            self.reward_baseline._var = state_dict["reward_var"]
