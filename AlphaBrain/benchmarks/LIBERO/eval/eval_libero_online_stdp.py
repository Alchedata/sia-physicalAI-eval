"""
LIBERO evaluation with Online STDP test-time adaptation.

Loads the NeuroVLA model directly (no WebSocket server) and applies online STDP
weight updates to the SNN action head during evaluation. The model adapts across
episodes using self-supervised reward signals from environment interaction.

Usage:
    python benchmarks/LIBERO/eval/eval_libero_online_stdp.py \
        --pretrained_path /path/to/checkpoint \
        --task_suite_name libero_goal \
        --num_trials_per_task 10
"""

import sys
import os

# Override via VLA_EXTRA_SYSPATH env var (colon-separated), e.g. "/path/to/AlphaBrain:/path/to/LIBERO"
for _p in [p for p in os.environ.get("VLA_EXTRA_SYSPATH", "").split(":") if p]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dataclasses
import json
import logging
import math
import pathlib
from pathlib import Path
import time

import imageio
import numpy as np
import torch
import tqdm
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from AlphaBrain.model.framework.base_framework import BaseFramework
from AlphaBrain.model.framework.config_utils import read_mode_config
from AlphaBrain.model.modules.action_model.stdp.online_stdp import (
    OnlineSTDPAdapter,
    OnlineSTDPConfig,
)
from typing import Union


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


def _binarize_gripper_open(open_val: Union[np.ndarray, float]) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


@dataclasses.dataclass
class Args:
    # Model
    pretrained_path: str = ""
    host: str = "127.0.0.1"  # unused, kept for CLI compat
    port: int = 10093         # unused, kept for CLI compat
    resize_size: tuple = (224, 224)
    use_bf16: bool = False

    # LIBERO environment
    task_suite_name: str = "libero_goal"
    num_steps_wait: int = 10
    num_trials_per_task: int = 10

    # Output
    video_out_path: str = "experiments/libero/logs_online_stdp"
    seed: int = 7
    post_process_action: bool = True
    job_name: str = "online_stdp"
    norm_mode: str = "q99"

    # Online STDP hyperparameters
    stdp_lr: float = 1e-5
    stdp_A_plus: float = 0.002
    stdp_A_minus: float = 0.0024
    stdp_max_deviation: float = 0.1
    stdp_max_update_norm: float = 0.02
    stdp_rollback_shrink: float = 0.7
    stdp_warmup_episodes: int = 2
    stdp_lr_decay: float = 0.995
    stdp_w_spe: float = 0.5
    stdp_w_smooth: float = 0.3
    stdp_w_consist: float = 0.2

    # Baseline mode: skip STDP entirely for fair comparison
    no_stdp: bool = False

    # Reset adapter weights at each new task (prevent cross-task interference)
    stdp_reset_per_task: bool = False

    # Save adapted model
    save_adapted: bool = True


def _build_stdp_config(args: Args) -> OnlineSTDPConfig:
    """Build OnlineSTDPConfig from CLI args."""
    return OnlineSTDPConfig(
        A_plus=args.stdp_A_plus,
        A_minus=args.stdp_A_minus,
        stdp_lr=args.stdp_lr,
        max_deviation=args.stdp_max_deviation,
        max_update_norm=args.stdp_max_update_norm,
        rollback_shrink=args.stdp_rollback_shrink,
        warmup_episodes=args.stdp_warmup_episodes,
        lr_decay=args.stdp_lr_decay,
        w_spe=args.stdp_w_spe,
        w_smooth=args.stdp_w_smooth,
        w_consist=args.stdp_w_consist,
    )


def _load_model(args: Args):
    """Load NeuroVLA model directly (no server)."""
    logging.info(f"Loading model from {args.pretrained_path}")
    model = BaseFramework.from_pretrained(args.pretrained_path)
    if args.use_bf16:
        model = model.to(torch.bfloat16)
    model = model.to("cuda").eval()
    logging.info("Model loaded successfully.")
    return model


def _get_action_norm_stats(pretrained_path: str, norm_mode: str = "q99"):
    """Load action normalization stats from checkpoint."""
    ckpt_path = Path(pretrained_path)
    _, norm_stats = read_mode_config(ckpt_path)
    # Auto-detect the dataset key
    if len(norm_stats) == 1:
        unnorm_key = next(iter(norm_stats.keys()))
    else:
        unnorm_key = next(iter(norm_stats.keys()))
        logging.warning(
            f"Multiple dataset keys found: {list(norm_stats.keys())}. "
            f"Using first: {unnorm_key}"
        )
    return norm_stats[unnorm_key]["action"]


def _unnormalize_actions(
    normalized_actions: np.ndarray,
    action_norm_stats: dict,
) -> np.ndarray:
    """Unnormalize actions from [-1, 1] to original scale."""
    norm_mode = action_norm_stats.get("norm_mode", "q99")
    if norm_mode == "min_max":
        ref_key_high, ref_key_low = "max", "min"
    else:
        ref_key_high, ref_key_low = "q99", "q01"
    mask = action_norm_stats.get(
        "mask", np.ones_like(action_norm_stats[ref_key_low], dtype=bool)
    )
    action_high = np.array(action_norm_stats[ref_key_high])
    action_low = np.array(action_norm_stats[ref_key_low])
    normalized_actions = np.clip(normalized_actions, -1, 1)
    normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1)
    actions = np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )
    return actions


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _get_max_steps(task_suite_name: str) -> int:
    limits = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    if task_suite_name not in limits:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    return limits[task_suite_name]


def eval_libero_online_stdp(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")

    np.random.seed(args.seed)

    # Load model directly
    model = _load_model(args)
    action_norm_stats = _get_action_norm_stats(args.pretrained_path, args.norm_mode)

    # Create online STDP adapter (unless baseline mode)
    use_stdp = not args.no_stdp
    adapter = None
    stdp_config = None
    if use_stdp:
        stdp_config = _build_stdp_config(args)
        adapter = OnlineSTDPAdapter(model, config=stdp_config)
        adapter.enable()

    # Initialize LIBERO
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    max_steps = _get_max_steps(args.task_suite_name)

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    logging.info(f"Task suite: {args.task_suite_name}")
    if use_stdp:
        logging.info(f"Online STDP enabled: lr={stdp_config.stdp_lr}, "
                     f"warmup={stdp_config.warmup_episodes} episodes, "
                     f"reset_per_task={args.stdp_reset_per_task}")
    else:
        logging.info("Online STDP DISABLED (baseline mode)")

    # Evaluation loop
    total_episodes, total_successes = 0, 0
    # Track success rate over time for adaptation analysis
    success_history = []

    for task_id in tqdm.tqdm(range(num_tasks_in_suite), desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0

        for episode_idx in tqdm.tqdm(
            range(args.num_trials_per_task), desc=f"Task {task_id}", leave=False
        ):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx % len(initial_states)])

            # Notify adapter of new episode
            if use_stdp:
                adapter.on_episode_start()

            # Initialize state history
            n = 16
            state = np.concatenate((
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            ))
            states = np.tile(state, (n, 1))

            t = 0
            replay_images = []
            full_actions = []
            prev_action = None
            prev_state = state.copy()

            # Action chunking state
            cached_actions = None
            chunk_step = 0
            prev_chunk = None

            logging.info(f"Starting episode {task_episodes + 1}...")

            while t < max_steps + args.num_steps_wait:
                # Wait for objects to stabilize
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # Prepare observation
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(
                    obs["robot0_eye_in_hand_image"][::-1, ::-1]
                )
                replay_images.append(img)

                state = np.concatenate((
                    obs["robot0_eef_pos"],
                    _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                ))
                states = np.vstack([states[1:], state])

                # Model inference (with action chunking)
                did_inference = False
                if cached_actions is None or chunk_step >= len(cached_actions):
                    did_inference = True
                    # Run model forward pass
                    batch_images = [[
                        Image.fromarray(img),
                        Image.fromarray(wrist_img),
                    ]]
                    states_input = np.expand_dims(states, axis=0).astype(np.float32)

                    result = model.predict_action(
                        batch_images=batch_images,
                        instructions=[task_description],
                        states=states_input,
                    )

                    # Notify adapter that inference is done (computes STDP eligibility)
                    if use_stdp:
                        adapter.on_inference_done()

                    # Unnormalize actions
                    normalized_actions = result["normalized_actions"][0]  # [chunk, 7]
                    cached_actions = _unnormalize_actions(
                        normalized_actions, action_norm_stats
                    )
                    chunk_step = 0

                    # Chunk consistency signal
                    if use_stdp:
                        adapter.update_chunk_consistency(prev_chunk, normalized_actions)
                    prev_chunk = normalized_actions.copy()

                # Get current action from chunk
                raw_actions = cached_actions[chunk_step]
                chunk_step += 1

                world_vector_delta = raw_actions[:3].astype(np.float32)
                rotation_delta = raw_actions[3:6].astype(np.float32)
                open_gripper = raw_actions[6:7].astype(np.float32)
                gripper = _binarize_gripper_open(open_gripper)

                delta_action = np.concatenate(
                    [world_vector_delta, rotation_delta, gripper], axis=0
                )
                full_actions.append(delta_action)

                # Execute action
                obs, reward, done, info = env.step(delta_action.tolist())

                # Get new state after action
                new_state = np.concatenate((
                    obs["robot0_eef_pos"],
                    _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                ))

                # Online STDP update: only on the inference step (temporal alignment)
                # Cached chunk replay steps are not temporally aligned with eligibility
                if use_stdp and did_inference:
                    adapter.update_step(
                        state_prev=prev_state,
                        state_curr=new_state,
                        action_executed=delta_action,
                        prev_action=prev_action,
                    )

                prev_action = delta_action.copy()
                prev_state = new_state.copy()

                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1

            # Notify adapter of episode end
            if use_stdp:
                adapter.on_episode_end(success=done)

            task_episodes += 1
            total_episodes += 1
            success_history.append(float(done))

            # Save replay video
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            if replay_images:
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path)
                    / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

            # Log progress
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed: {total_episodes}")
            logging.info(
                f"# successes: {total_successes} "
                f"({total_successes / total_episodes * 100:.1f}%)"
            )

            # Log STDP stats
            if use_stdp:
                stdp_stats = adapter.get_stats()
                if stdp_stats:
                    logging.info(
                        f"STDP stats: drift={stdp_stats.get('online_stdp/weight_drift', 0):.6f}, "
                        f"lr={stdp_stats.get('online_stdp/effective_lr', 0):.2e}"
                    )

        logging.info(
            f"Task {task_id} success rate: "
            f"{float(task_successes) / float(task_episodes):.3f}"
        )
        env.close()

        # Reset adapter weights between tasks to prevent cross-task interference
        if use_stdp and args.stdp_reset_per_task:
            adapter.reset_to_initial()
            logging.info(f"STDP adapter reset after task {task_id}")

    # Final results
    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes):.3f}")
    logging.info(f"Total episodes: {total_episodes}")

    # Log adaptation trend (windowed success rate)
    if len(success_history) >= 10:
        early = np.mean(success_history[:10])
        late = np.mean(success_history[-10:])
        logging.info(
            f"Adaptation trend: first 10 episodes={early:.2f}, "
            f"last 10 episodes={late:.2f}, delta={late - early:+.2f}"
        )

    # Save adapted model
    if use_stdp and args.save_adapted:
        adapted_path = pathlib.Path(args.video_out_path) / "adapted_model.pt"
        adapter.save_adapted_weights(str(adapted_path))

    # Save evaluation results
    results = {
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "success_rate": float(total_successes) / float(total_episodes),
        "success_history": success_history,
        "stdp_enabled": use_stdp,
    }
    if use_stdp:
        results["stdp_config"] = dataclasses.asdict(stdp_config) if dataclasses.is_dataclass(stdp_config) else vars(stdp_config)
        results["final_weight_drift"] = adapter.get_stats().get("online_stdp/weight_drift", 0)
        results["stdp_reset_per_task"] = args.stdp_reset_per_task
    results_path = pathlib.Path(args.video_out_path) / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logging.info(f"Results saved to {results_path}")

    if use_stdp:
        adapter.disable()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )
    args = tyro.cli(Args)
    eval_libero_online_stdp(args)
