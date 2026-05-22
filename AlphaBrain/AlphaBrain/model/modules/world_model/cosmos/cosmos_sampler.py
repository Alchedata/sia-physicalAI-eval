# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Ported from:
#   cosmos_policy._src.imaginaire.modules.res_sampler  (Sampler, SamplerConfig, get_rev_ts, differential_equation_solver)
#   cosmos_policy._src.imaginaire.functional.runge_kutta  (phi1, phi2, batch_mul, RK steps)
#   cosmos_policy._src.imaginaire.functional.multi_step   (order2_fn / 2ab)
#   cosmos_policy.modules.cosmos_sampler                  (CosmosPolicySampler)
#
# Pure PyTorch, no megatron/imaginaire/hydra/attrs dependencies.
# Uses standard Python dataclasses.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple, Union

import torch


# ---------------------------------------------------------------------------
# Batch broadcast helpers (inlined from batch_ops.py)
# ---------------------------------------------------------------------------

def _common_broadcast(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    ndims1, ndims2 = x.ndim, y.ndim
    common_ndims = min(ndims1, ndims2)
    for axis in range(common_ndims):
        assert x.shape[axis] == y.shape[axis], f"Dimensions not equal at axis {axis}"
    if ndims1 < ndims2:
        x = x.reshape(x.shape + (1,) * (ndims2 - ndims1))
    elif ndims2 < ndims1:
        y = y.reshape(y.shape + (1,) * (ndims1 - ndims2))
    return x, y


def _batch_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = _common_broadcast(x, y)
    return x * y


# ---------------------------------------------------------------------------
# Phi functions for RES solver (inlined from runge_kutta.py)
# ---------------------------------------------------------------------------

def _phi1(t: torch.Tensor) -> torch.Tensor:
    """(exp(t) - 1) / t"""
    input_dtype = t.dtype
    t = t.to(torch.float64)
    return (torch.expm1(t) / t).to(input_dtype)


def _phi2(t: torch.Tensor) -> torch.Tensor:
    """(phi1(t) - 1) / t"""
    input_dtype = t.dtype
    t = t.to(torch.float64)
    return ((_phi1(t) - 1.0) / t).to(input_dtype)


# ---------------------------------------------------------------------------
# Runge-Kutta step functions (inlined from runge_kutta.py)
# ---------------------------------------------------------------------------

def _reg_x0_euler_step(
    x_s: torch.Tensor, s: torch.Tensor, t: torch.Tensor, x0_s: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    coef_x0 = (s - t) / s
    coef_xs = t / s
    return _batch_mul(coef_x0, x0_s) + _batch_mul(coef_xs, x_s), x0_s


def _res_x0_rk2_step(
    x_s: torch.Tensor,
    t: torch.Tensor,
    s: torch.Tensor,
    x0_s: torch.Tensor,
    s1: torch.Tensor,
    x0_s1: torch.Tensor,
) -> torch.Tensor:
    s = -torch.log(s)
    t = -torch.log(t)
    m = -torch.log(s1)

    dt = t - s
    assert not torch.any(torch.isclose(dt, torch.zeros_like(dt), atol=1e-6)), "Step size too small"
    assert not torch.any(torch.isclose(m - s, torch.zeros_like(dt), atol=1e-6)), "Step size too small"

    c2 = (m - s) / dt
    phi1_val, phi2_val = _phi1(-dt), _phi2(-dt)

    b1 = torch.nan_to_num(phi1_val - 1.0 / c2 * phi2_val, nan=0.0)
    b2 = torch.nan_to_num(1.0 / c2 * phi2_val, nan=0.0)

    return _batch_mul(torch.exp(-dt), x_s) + _batch_mul(dt, _batch_mul(b1, x0_s) + _batch_mul(b2, x0_s1))


def _rk1_euler(
    x_s: torch.Tensor, s: torch.Tensor, t: torch.Tensor, x0_fn: Callable
) -> Tuple[torch.Tensor, torch.Tensor]:
    x0_s = x0_fn(x_s, s)
    return _reg_x0_euler_step(x_s, s, t, x0_s)


# ---------------------------------------------------------------------------
# Multi-step (2ab / Adams-Bashforth) step function (inlined from multi_step.py)
# ---------------------------------------------------------------------------

def _order2_fn(
    x_s: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
    x0_s: torch.Tensor,
    x0_preds: Optional[List],
) -> Tuple[torch.Tensor, List]:
    """2nd-order Adams-Bashforth multistep (RES paper)."""
    if x0_preds:
        x0_s1, s1 = x0_preds[0]
        x_t = _res_x0_rk2_step(x_s, t, s, x0_s, s1, x0_s1)
    else:
        x_t = _reg_x0_euler_step(x_s, s, t, x0_s)[0]
    return x_t, [(x0_s, s)]


# ---------------------------------------------------------------------------
# Solver / Sampler config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SolverConfig:
    is_multi: bool = False
    rk: str = "2mid"
    multistep: str = "2ab"
    s_churn: float = 0.0
    s_t_max: float = float("inf")
    s_t_min: float = 0.05
    s_noise: float = 1.0


@dataclass
class SolverTimestampConfig:
    nfe: int = 50
    t_min: float = 0.002
    t_max: float = 80.0
    order: float = 7.0
    is_forward: bool = False


@dataclass
class SamplerConfig:
    solver: SolverConfig = field(default_factory=SolverConfig)
    timestamps: SolverTimestampConfig = field(default_factory=SolverTimestampConfig)
    sample_clean: bool = True


# ---------------------------------------------------------------------------
# Supported solver registries
# ---------------------------------------------------------------------------

_MULTISTEP_FNs = {"2ab": _order2_fn}
_RK_FNs = {"1euler": _rk1_euler}


def _is_multi_step(name: str) -> bool:
    return name in _MULTISTEP_FNs


def _is_runge_kutta(name: str) -> bool:
    return name in _RK_FNs


# ---------------------------------------------------------------------------
# Timestamp generation
# ---------------------------------------------------------------------------

def get_rev_ts(
    t_min: float,
    t_max: float,
    num_steps: int,
    ts_order: Union[int, float],
    is_forward: bool = False,
) -> torch.Tensor:
    """
    Generate reverse (high→low sigma) time steps for ODE sampling.

    Returns tensor of shape (num_steps+1,).
    """
    if t_min >= t_max:
        raise ValueError("t_min must be less than t_max")
    if not isinstance(ts_order, (int, float)):
        raise TypeError("ts_order must be an integer or float")

    step_indices = torch.arange(num_steps + 1, dtype=torch.float64)
    time_steps = (
        t_max ** (1.0 / ts_order)
        + step_indices / num_steps * (t_min ** (1.0 / ts_order) - t_max ** (1.0 / ts_order))
    ) ** ts_order

    if is_forward:
        return time_steps.flip(dims=(0,))
    return time_steps


# ---------------------------------------------------------------------------
# Differential equation solver
# ---------------------------------------------------------------------------

def differential_equation_solver(
    x0_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    sigmas_L: torch.Tensor,
    solver_cfg: SolverConfig,
    callback_fns: Optional[List[Callable]] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Returns a sample_fn(x_T) -> x_0 that runs the ODE solver loop.

    Args:
        x0_fn: Denoiser function (x_noisy, sigma) -> x0_pred.
        sigmas_L: Sigma schedule tensor of shape [L+1].
        solver_cfg: Solver configuration.
        callback_fns: Optional list of per-step callbacks.
    """
    num_step = len(sigmas_L) - 1

    if solver_cfg.is_multi:
        update_step_fn = _MULTISTEP_FNs[solver_cfg.multistep]
    else:
        update_step_fn = _RK_FNs[solver_cfg.rk]

    eta = min(solver_cfg.s_churn / (num_step + 1), math.sqrt(1.2) - 1)

    def sample_fn(input_xT: torch.Tensor) -> torch.Tensor:
        ones_B = torch.ones(input_xT.size(0), device=input_xT.device, dtype=torch.float64)

        def step_fn(
            i_th: int, state: Tuple[torch.Tensor, Optional[List]]
        ) -> Tuple[torch.Tensor, Optional[List]]:
            input_x, x0_preds = state
            sigma_cur = sigmas_L[i_th]
            sigma_next = sigmas_L[i_th + 1]

            # Stochastic churn (EDM algorithm 2, lines 4-6)
            if solver_cfg.s_t_min < sigma_cur < solver_cfg.s_t_max:
                hat_sigma_cur = sigma_cur + eta * sigma_cur
                input_x = input_x + (
                    hat_sigma_cur ** 2 - sigma_cur ** 2
                ).sqrt() * solver_cfg.s_noise * torch.randn_like(input_x)
                sigma_cur = hat_sigma_cur

            if solver_cfg.is_multi:
                x0_pred = x0_fn(input_x, sigma_cur * ones_B)
                output_x, x0_preds = update_step_fn(
                    input_x, sigma_cur * ones_B, sigma_next * ones_B, x0_pred, x0_preds
                )
            else:
                output_x, x0_preds = update_step_fn(
                    input_x, sigma_cur * ones_B, sigma_next * ones_B, x0_fn
                )

            if callback_fns:
                for cb in callback_fns:
                    cb(**locals())

            return output_x, x0_preds

        # Run the loop
        x, _ = _fori_loop(0, num_step, step_fn, [input_xT, None])
        return x

    return sample_fn


def _fori_loop(lower: int, upper: int, body_fun: Callable[[int, Any], Any], init_val: Any) -> Any:
    val = init_val
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val


# ---------------------------------------------------------------------------
# CosmosPolicySampler
# ---------------------------------------------------------------------------

class CosmosPolicySampler(torch.nn.Module):
    """
    Sampler for Cosmos Policy.

    Key behaviors vs base Sampler:
    - When sample_clean=True and num_steps > 1: subtracts 1 from num_steps so total
      NFE (solver steps + clean step) equals the requested num_steps.
    - When num_steps == 1: skips the ODE loop and directly denoises from sigma_max.

    LIBERO defaults: solver="2ab", num_steps=5 (action), 1 (state/value).
    """

    def __init__(self, cfg: Optional[SamplerConfig] = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else SamplerConfig()

    @torch.no_grad()
    def forward(
        self,
        x0_fn: Callable,
        x_sigma_max: torch.Tensor,
        num_steps: int = 5,
        sigma_min: float = 0.01,
        sigma_max: float = 200.0,
        rho: float = 7.0,
        S_churn: float = 0.0,
        S_min: float = 0.0,
        S_max: float = float("inf"),
        S_noise: float = 1.0,
        solver_option: str = "2ab",
    ) -> torch.Tensor:
        in_dtype = x_sigma_max.dtype

        def float64_x0_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return x0_fn(x.to(in_dtype), t.to(in_dtype)).to(torch.float64)

        is_multistep = _is_multi_step(solver_option)
        is_rk = _is_runge_kutta(solver_option)
        assert is_multistep or is_rk, f"Unsupported solver: {solver_option}"

        solver_cfg = SolverConfig(
            s_churn=S_churn,
            s_t_max=S_max,
            s_t_min=S_min,
            s_noise=S_noise,
            is_multi=is_multistep,
            rk=solver_option,
            multistep=solver_option,
        )

        sample_clean = True
        # Official subtracts 1 so total NFE (solver + clean) = num_steps
        if sample_clean and num_steps > 1:
            effective_steps = num_steps - 1
        else:
            effective_steps = num_steps

        timestamps_cfg = SolverTimestampConfig(
            nfe=effective_steps, t_min=sigma_min, t_max=sigma_max, order=rho
        )
        sampler_cfg = SamplerConfig(
            solver=solver_cfg, timestamps=timestamps_cfg, sample_clean=sample_clean
        )

        return self._forward_impl(
            float64_x0_fn, x_sigma_max, sampler_cfg, num_steps=num_steps
        ).to(in_dtype)

    @torch.no_grad()
    def _forward_impl(
        self,
        denoiser_fn: Callable,
        noisy_input: torch.Tensor,
        sampler_cfg: Optional[SamplerConfig] = None,
        callback_fns: Optional[List[Callable]] = None,
        num_steps: int = 5,
    ) -> torch.Tensor:
        sampler_cfg = self.cfg if sampler_cfg is None else sampler_cfg
        solver_order = 1 if sampler_cfg.solver.is_multi else int(sampler_cfg.solver.rk[0])
        num_timestamps = sampler_cfg.timestamps.nfe // solver_order

        sigmas_L = get_rev_ts(
            sampler_cfg.timestamps.t_min,
            sampler_cfg.timestamps.t_max,
            num_timestamps,
            sampler_cfg.timestamps.order,
        ).to(noisy_input.device)

        if num_steps > 1:
            denoised = differential_equation_solver(
                denoiser_fn, sigmas_L, sampler_cfg.solver, callback_fns=callback_fns
            )(noisy_input)

            if sampler_cfg.sample_clean:
                ones = torch.ones(denoised.size(0), device=denoised.device, dtype=denoised.dtype)
                denoised = denoiser_fn(denoised, sigmas_L[-1] * ones)
        else:
            # num_steps == 1: single direct denoising from sigma_max
            denoised = noisy_input
            ones = torch.ones(denoised.size(0), device=denoised.device, dtype=denoised.dtype)
            denoised = denoiser_fn(denoised, sigmas_L[0] * ones)

        return denoised
