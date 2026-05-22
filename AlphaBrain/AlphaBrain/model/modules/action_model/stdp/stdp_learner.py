"""
STDP Learner: Spike-Timing-Dependent Plasticity weight update rules.

Implements:
- Classic STDP with exponential windows (A+/A- and tau+/tau-)
- Eligibility traces for temporal credit assignment
- Reward modulation (R-STDP) for goal-directed learning
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class STDPLearner:
    """
    Implements STDP weight update computation.

    STDP rule:
        Δw(Δt) = A+ · exp(-Δt/τ+)  if Δt > 0  (post fires after pre → LTP)
        Δw(Δt) = -A- · exp(Δt/τ-)  if Δt < 0  (post fires before pre → LTD)

    For trace-based STDP (online approximation):
        pre_trace += pre_spike; pre_trace *= exp(-dt/τ+)
        post_trace += post_spike; post_trace *= exp(-dt/τ-)
        Δw = A+ · post_spike · pre_trace - A- · pre_spike · post_trace

    Reward modulation:
        Δw_rstdp = reward · eligibility_trace
        eligibility_trace = trace_decay · eligibility_trace + Δw_stdp
    """

    def __init__(
        self,
        A_plus: float = 0.01,
        A_minus: float = 0.012,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        trace_decay: float = 0.95,
        weight_clip: float = 1.0,
        device: Optional[torch.device] = None,
        binarize_pre: bool = True,
    ):
        """
        Args:
            A_plus: LTP amplitude (potentiation strength).
            A_minus: LTD amplitude (depression strength). Slightly > A_plus for stability.
            tau_plus: LTP time constant.
            tau_minus: LTD time constant.
            trace_decay: Eligibility trace decay factor per timestep.
            weight_clip: Maximum absolute STDP weight update magnitude.
            device: Torch device.
            binarize_pre: If True, binarize pre-synaptic activity (threshold at 0).
                          Necessary when pre-synaptic inputs are continuous (e.g. after LayerNorm).
        """
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.trace_decay = trace_decay
        self.weight_clip = weight_clip
        self.device = device or torch.device("cpu")
        self.binarize_pre = binarize_pre

        # Eligibility traces: keyed by weight parameter id
        self._eligibility_traces: Dict[int, torch.Tensor] = {}
        # Running pre/post traces: keyed by layer name
        self._pre_traces: Dict[str, torch.Tensor] = {}
        self._post_traces: Dict[str, torch.Tensor] = {}

    def compute_trace_stdp(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        layer_name: str,
        dt: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute trace-based STDP update for a single timestep.

        This is the online version of STDP that maintains running traces
        and computes weight updates incrementally.

        Args:
            pre_spikes: Pre-synaptic spikes [B, D_pre] (binary or soft).
            post_spikes: Post-synaptic spikes [B, D_post] (binary or soft).
            layer_name: Identifier for this layer's traces.
            dt: Time step duration (for trace decay).

        Returns:
            STDP weight update [D_post, D_pre] (same shape as weight matrix).
        """
        B, D_pre = pre_spikes.shape
        _, D_post = post_spikes.shape

        # Binarize pre-synaptic activity when inputs are continuous
        # (e.g. after LayerNorm, the values are not 0/1 spikes)
        if self.binarize_pre:
            pre_spikes = (pre_spikes > 0).float()

        # Initialize traces if needed
        if layer_name not in self._pre_traces:
            self._pre_traces[layer_name] = torch.zeros(B, D_pre, device=pre_spikes.device)
            self._post_traces[layer_name] = torch.zeros(B, D_post, device=post_spikes.device)

        pre_trace = self._pre_traces[layer_name]
        post_trace = self._post_traces[layer_name]

        # Decay traces
        decay_pre = torch.exp(torch.tensor(-dt / self.tau_plus, device=pre_spikes.device))
        decay_post = torch.exp(torch.tensor(-dt / self.tau_minus, device=post_spikes.device))
        pre_trace = pre_trace * decay_pre + pre_spikes
        post_trace = post_trace * decay_post + post_spikes

        # STDP update: LTP when post fires (use pre_trace), LTD when pre fires (use post_trace)
        # dw[j,i] = A+ * post_spike[j] * pre_trace[i] - A- * pre_spike[i] * post_trace[j]
        # Shape: [B, D_post, D_pre] -> mean over batch -> [D_post, D_pre]
        ltp = self.A_plus * torch.bmm(
            post_spikes.unsqueeze(2),  # [B, D_post, 1]
            pre_trace.unsqueeze(1)     # [B, 1, D_pre]
        )  # [B, D_post, D_pre]

        ltd = self.A_minus * torch.bmm(
            post_trace.unsqueeze(2),   # [B, D_post, 1]
            pre_spikes.unsqueeze(1)    # [B, 1, D_pre]
        )  # [B, D_post, D_pre]

        dw = (ltp - ltd).mean(dim=0)  # [D_post, D_pre]

        # Clip update magnitude
        dw = dw.clamp(-self.weight_clip, self.weight_clip)

        # Store updated traces
        self._pre_traces[layer_name] = pre_trace.detach()
        self._post_traces[layer_name] = post_trace.detach()

        return dw

    def accumulate_eligibility(
        self,
        weight_param: nn.Parameter,
        dw_stdp: torch.Tensor,
    ):
        """
        Accumulate STDP update into the eligibility trace for a weight parameter.

        eligibility_trace = trace_decay * eligibility_trace + dw_stdp

        Args:
            weight_param: The nn.Parameter whose weight this update applies to.
            dw_stdp: STDP weight update [D_out, D_in].
        """
        pid = id(weight_param)
        if pid not in self._eligibility_traces:
            self._eligibility_traces[pid] = torch.zeros_like(weight_param.data)

        self._eligibility_traces[pid] = (
            self.trace_decay * self._eligibility_traces[pid] + dw_stdp
        )

    def get_eligibility_trace(self, weight_param: nn.Parameter) -> Optional[torch.Tensor]:
        """Get current eligibility trace for a weight parameter."""
        pid = id(weight_param)
        return self._eligibility_traces.get(pid)

    def compute_reward_modulated_update(
        self,
        weight_param: nn.Parameter,
        reward: float,
    ) -> Optional[torch.Tensor]:
        """
        Compute reward-modulated STDP update.

        Δw_rstdp = reward * eligibility_trace

        Args:
            weight_param: The weight parameter to update.
            reward: Scalar reward signal (positive = reinforce, negative = suppress).

        Returns:
            Weight update tensor, or None if no eligibility trace exists.
        """
        trace = self.get_eligibility_trace(weight_param)
        if trace is None:
            return None
        update = reward * trace
        return update.clamp(-self.weight_clip, self.weight_clip)

    def reset_traces(self):
        """Reset all running traces (call between episodes/batches)."""
        self._pre_traces.clear()
        self._post_traces.clear()

    def reset_eligibility(self):
        """Reset all eligibility traces."""
        self._eligibility_traces.clear()

    def reset(self):
        """Full reset of all internal state."""
        self.reset_traces()
        self.reset_eligibility()


class RewardBaseline:
    """
    Maintains a running average baseline for reward normalization.

    Uses improved Welford-style online variance tracking with an adaptive
    clip range that tightens during early training for stability.

    reward_normalized = clip((reward - mean) / std, -clip_range, clip_range)
    """

    def __init__(self, momentum: float = 0.99, clip_range: float = 1.0):
        self.momentum = momentum
        self.clip_range = clip_range
        self._baseline: Optional[float] = None
        self._var: float = 1.0  # running variance estimate
        self._count: int = 0

    def normalize(self, reward: float) -> float:
        """Normalize reward to roughly unit scale and clip."""
        self._count += 1

        if self._baseline is None:
            self._baseline = reward
            self._var = 1.0
            return 0.0

        # Normalize by running std
        std = max(self._var ** 0.5, 1e-6)
        normalized = (reward - self._baseline) / std

        # Adaptive clip: tighter during early training for stability
        effective_clip = self.clip_range * min(1.0, self._count / 100.0)
        normalized = max(-effective_clip, min(effective_clip, normalized))

        # Update running statistics (Welford-style for stable variance)
        old_baseline = self._baseline
        self._baseline = self.momentum * self._baseline + (1.0 - self.momentum) * reward
        delta_old = reward - old_baseline
        delta_new = reward - self._baseline
        self._var = self.momentum * self._var + (1.0 - self.momentum) * (delta_old * delta_new)
        self._var = max(self._var, 1e-8)  # prevent negative variance

        return normalized

    @property
    def baseline(self) -> float:
        return self._baseline if self._baseline is not None else 0.0

    def reset(self):
        self._baseline = None
        self._var = 1.0
        self._count = 0
