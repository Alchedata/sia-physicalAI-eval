"""
Episode rollout for GRPO.

Key design:
  - Each episode stores trajectory as tensors: (traj_len, ...) for batched forward
  - finish_step tracks actual episode length for masking
  - Multiprocess env workers (one process per env)
  - Gaussian policy: a ~ N(μ, σ²I), log_prob = -||a-μ||²/(2σ²)
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from AlphaBrain.training.reinforcement_learning.envs.libero_env import LiberoEnv


def _save_video(frames: List[np.ndarray], path: str, fps: int = 10) -> str:
    # Ensure all frames are numpy arrays
    clean_frames = []
    for f in frames:
        if isinstance(f, Image.Image):
            clean_frames.append(np.array(f))
        elif isinstance(f, np.ndarray):
            clean_frames.append(f)
        else:
            clean_frames.append(np.asarray(f))
    try:
        import imageio
        imageio.mimwrite(path, clean_frames, fps=fps, codec="libx264")
    except Exception:
        gif_path = path.replace(".mp4", ".gif")
        pil_frames = [Image.fromarray(f) for f in clean_frames]
        pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:],
                           loop=0, duration=int(1000 / fps))
        return gif_path
    return path


DUMMY_ACTION = np.array([0.0] * 6 + [-1.0], dtype=np.float32)


@dataclass
class StepRecord:
    """One inference step = one action-chunk prediction."""
    primary_image: np.ndarray        # (H, W, 3) uint8
    wrist_image:   np.ndarray        # (H, W, 3) uint8
    instruction:   str
    norm_action:   np.ndarray        # (chunk_len, 7) normalized, sampled
    old_log_prob:  float             # scalar
    value:         float = 0.0       # V(s) from critic (used by PPO+GAE)
    action_token_ids: np.ndarray = None  # (n_action_tokens,) discrete token IDs


@dataclass
class Episode:
    step_records: List[StepRecord] = field(default_factory=list)
    reward:       float = 0.0
    task_id:      int   = 0
    success:      bool  = False
    finish_step:  int   = 0          # actual number of steps taken
    video_path:   Optional[str] = None
    state_idx:    int   = -1         # initial state index used for this episode


# ------------------------------------------------------------------
# Single-episode rollout (runs in a thread)
# ------------------------------------------------------------------

def _rollout_one(
    env: LiberoEnv,
    model_fn,           # (images, instructions) -> mean_norm np.ndarray (1, chunk_len, 7)
    suite_name: str,
    task_id: int,
    state_idx: int,
    action_norm_stats: dict,
    max_steps: int,
    chunk_len: int,
    explore_std: float,
    num_steps_wait: int,
    seed: int,
    record_video: bool,
    episode_idx: int,
    group_idx: int,
    video_dir: Optional[str],
    model_lock,
    value_fn=None,      # optional: (images, instructions) -> float V(s)
) -> Episode:
    episode = Episode(task_id=task_id, state_idx=state_idx)
    frames: List[np.ndarray] = [] if record_video else None
    rng = np.random.RandomState(seed)  # per-episode isolated RNG

    obs = env.reset(
        suite_name=suite_name,
        task_id=task_id,
        initial_state_idx=state_idx,
        seed=seed,
    )
    task_description = env.task_description

    env_step = 0
    action_cache: Optional[np.ndarray] = None
    cache_idx = 0

    while env_step < max_steps + num_steps_wait:
        if env_step < num_steps_wait:
            obs, _, done = env.step(DUMMY_ACTION)
            env_step += 1
            continue

        if action_cache is None or cache_idx >= chunk_len:
            images = [[obs["primary_image"], obs["wrist_image"]]]
            with model_lock:
                pred = model_fn(images, [task_description])

            # Support both continuous (dict with only normalized_actions)
            # and discrete (dict with action_token_ids + log_probs)
            if isinstance(pred, dict) and "action_token_ids" in pred:
                # Discrete token model (QwenOFT_Discrete)
                sampled_norm = pred["normalized_actions"]  # (1, chunk_len, 7)
                log_prob = float(pred["log_probs"][0])
                action_token_ids = pred["action_token_ids"][0]  # (56,)
            else:
                # Continuous model (QwenOFT) — Gaussian exploration
                mean_norm = pred if isinstance(pred, np.ndarray) else pred["normalized_actions"]
                # Use per-dim learned std if available, else fallback to fixed explore_std
                if isinstance(pred, dict) and "std" in pred:
                    std_vec = pred["std"]  # (action_dim,) from LearnedLogStd
                else:
                    std_vec = np.full(mean_norm.shape[-1], explore_std, dtype=np.float32)
                noise = rng.randn(*mean_norm.shape).astype(np.float32) * std_vec
                sampled_norm = mean_norm + noise
                # Full Gaussian log prob: -log(σ) - 0.5*log(2π) - 0.5*((x-μ)/σ)²
                var = std_vec ** 2
                per_dim_lp = -np.log(std_vec) - 0.5 * np.log(2 * np.pi) - 0.5 * (noise ** 2) / var
                log_prob = float(per_dim_lp.sum())
                action_token_ids = None

            # V(s): prefer inline value from model_fn (single forward), fallback to separate value_fn
            step_value = 0.0
            if isinstance(pred, dict) and "value" in pred:
                step_value = float(pred["value"])
            elif value_fn is not None:
                with model_lock:
                    step_value = value_fn(images, [task_description])

            episode.step_records.append(StepRecord(
                primary_image=obs["primary_image"].copy(),
                wrist_image=obs["wrist_image"].copy(),
                instruction=task_description,
                norm_action=sampled_norm[0].copy() if sampled_norm.ndim == 3 else sampled_norm.copy(),
                old_log_prob=log_prob,
                value=step_value,
                action_token_ids=action_token_ids,
            ))

            action_cache = _unnormalize(sampled_norm[0], action_norm_stats)
            cache_idx = 0

        env_action = _postprocess_action(action_cache[cache_idx])
        cache_idx += 1
        obs, reward, done = env.step(env_action)
        env_step += 1

        if frames is not None:
            frames.append(obs["primary_image"].copy())

        if done:
            episode.success = bool(reward > 0.5)
            episode.reward = 1.0 if episode.success else 0.0
            break

    episode.finish_step = len(episode.step_records)

    if frames and video_dir is not None:
        os.makedirs(video_dir, exist_ok=True)
        status = "success" if episode.success else "fail"
        vpath = os.path.join(video_dir, f"group{group_idx:02d}_task{task_id:02d}_ep{episode_idx:02d}_{status}.mp4")
        episode.video_path = _save_video(frames, vpath)

    return episode


# ------------------------------------------------------------------
# Collect G episodes in parallel (ThreadPoolExecutor)
# ------------------------------------------------------------------

@torch.no_grad()
def collect_group(
    model,
    suite_name: str,
    task_id: int,
    n_initial_states: int,
    action_norm_stats: dict,
    max_steps: int,
    chunk_len: int,
    G: int = 8,
    libero_python: Optional[str] = None,
    seed: int = 42,
    explore_std: float = 0.1,
    num_steps_wait: int = 10,
    device: str = "cuda",
    video_dir: Optional[str] = None,
    num_envs: int = 4,
    group_idx: int = 0,
    value_fn=None,      # optional: (images, instructions) -> float V(s)
    custom_model_fn=None,  # optional: external model_fn (e.g. combined action+value)
) -> List[Episode]:
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    model.eval()
    model_lock = threading.Lock()

    if custom_model_fn is not None:
        model_fn = custom_model_fn
    else:
        # Default: separate model_fn
        _is_discrete = hasattr(model, 'action_tokenizer')

        def model_fn(images, instructions):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if _is_discrete:
                    temperature = explore_std if explore_std > 0 else 0.0
                    pred = model.predict_action(
                        batch_images=images,
                        instructions=instructions,
                        temperature=max(temperature, 1.0) if temperature > 0 else 0.0,
                    )
                    return pred
                else:
                    pred = model.predict_action(
                        batch_images=images,
                        instructions=instructions,
                    )
                    return pred["normalized_actions"]

    n_workers = min(G, num_envs)
    envs = [LiberoEnv(libero_python=libero_python) for _ in range(n_workers)]

    episodes = [None] * G
    try:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {}
            for g in range(G):
                # Use (group_idx * G + g) so different groups get different initial states
                global_episode_idx = group_idx * G + g
                fut = pool.submit(
                    _rollout_one,
                    env=envs[g % n_workers],
                    model_fn=model_fn,
                    suite_name=suite_name,
                    task_id=task_id,
                    state_idx=global_episode_idx % n_initial_states,
                    action_norm_stats=action_norm_stats,
                    max_steps=max_steps,
                    chunk_len=chunk_len,
                    explore_std=explore_std,
                    num_steps_wait=num_steps_wait,
                    seed=seed + g,
                    record_video=(video_dir is not None),
                    episode_idx=g,
                    group_idx=group_idx,
                    video_dir=video_dir,
                    model_lock=model_lock,
                    value_fn=value_fn,
                )
                futures[fut] = g
            done_count = 0
            n_success = 0
            for fut in as_completed(futures):
                ep = fut.result()
                episodes[futures[fut]] = ep
                done_count += 1
                n_success += int(ep.success)
                if G > 4 and done_count % max(1, G // 10) == 0:
                    print(f"  [rollout] {done_count}/{G} episodes done "
                          f"(success so far: {n_success}/{done_count})", flush=True)
            if G > 4:
                print(f"  [rollout] All {G} episodes complete. "
                      f"SR={n_success}/{G}={n_success/G:.1%}", flush=True)
    finally:
        for env in envs:
            env.close()

    return episodes


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _unnormalize(norm_actions: np.ndarray, stats: dict) -> np.ndarray:
    norm_mode = stats.get("norm_mode", "q99")
    hi_key, lo_key = ("max", "min") if norm_mode == "min_max" else ("q99", "q01")
    mask = stats.get("mask", np.ones(norm_actions.shape[-1], dtype=bool))
    action_hi = np.array(stats[hi_key])
    action_lo = np.array(stats[lo_key])
    clipped = np.clip(norm_actions, -1.0, 1.0)
    # Binarize gripper (dim 6) BEFORE unnormalization — must match M1Inference
    # In normalized space: < 0.5 → 0 (close), >= 0.5 → 1 (open)
    clipped[..., 6] = np.where(clipped[..., 6] < 0.5, 0.0, 1.0)
    return np.where(mask, 0.5 * (clipped + 1.0) * (action_hi - action_lo) + action_lo, clipped)


def _postprocess_action(action_7d: np.ndarray) -> np.ndarray:
    action = action_7d.copy()
    # Match eval pipeline: 1.0 - 2.0 * (v > 0.5) → open(<=0.5) = 1.0, close(>0.5) = -1.0
    action[6] = 1.0 - 2.0 * (action[6] > 0.5)
    return action
