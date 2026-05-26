#!/usr/bin/env python3
"""
generate_libero_traces.py — SEPA-Eval demo data generator.

Generates 100 LIBERO episode traces without requiring a live GPU or
simulator.  Writes directly to EvalMemory (SQLite + msgpack) so every downstream
SEPA-Eval command (mine, report, review, diff) works out of the box.

Usage
-----
From the repo root:

    cd AlphaBrain
    pip install -e .
    python ../demo/generate_synthetic_traces.py

    # Specify a custom output directory:
    python ../demo/generate_synthetic_traces.py --output-dir /tmp/my_demo

Then generate the report:

    python -m sepa_eval report --memory-dir ./demo_eval_memory

Or run the full pipeline:

    python -m sepa_eval mine   --memory-dir ./demo_eval_memory
    python -m sepa_eval report --memory-dir ./demo_eval_memory

Requirements
------------
    pip install msgpack   (already required by sepa_eval)
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sqlite3
import sys
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any

# ---------------------------------------------------------------------------
# Path bootstrap — allows running from any directory as long as AlphaBrain/
# is importable (either installed or on PYTHONPATH).
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_ALPHABRAIN_DIR = os.path.join(_HERE, "..", "AlphaBrain")

if os.path.isdir(_ALPHABRAIN_DIR) and _ALPHABRAIN_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_ALPHABRAIN_DIR))

try:
    from sepa_eval.memory.eval_memory import EvalMemory
    from sepa_eval.memory.schema import (
        CandidateTask,
        EpisodeTrace,
        RolloutData,
        SceneConfig,
        TaskProvenance,
        TraceIdentity,
        TraceLabels,
    )
except ImportError as exc:
    print(
        f"ERROR: Could not import sepa_eval — {exc}\n"
        "Make sure you have run 'pip install -e .' from the AlphaBrain/ directory.",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# LIBERO task catalog
# ---------------------------------------------------------------------------
# Each entry: (task_id, instruction, benchmark, max_steps)
# Two suites: libero_spatial (spatial relationship, max 220 steps)
#             libero_goal    (goal-state tasks,      max 300 steps)

LIBERO_TASKS: list[tuple[str, str, str, int]] = [
    # libero_spatial -------------------------------------------------------
    ("ls_pick_red_cup_left",
     "pick up the red cup from the left side of the cabinet",
     "libero_spatial", 220),
    ("ls_put_butter_right",
     "put the butter on the counter to the right of the stove",
     "libero_spatial", 220),
    ("ls_place_bowl_front",
     "place the bowl in front of the coffee machine",
     "libero_spatial", 220),
    ("ls_move_plate_behind",
     "move the plate to behind the toaster",
     "libero_spatial", 220),
    ("ls_pick_mug_right_shelf",
     "pick the mug from the right shelf and put it on the counter",
     "libero_spatial", 220),
    ("ls_slide_book_left",
     "slide the book to the left end of the shelf",
     "libero_spatial", 220),
    ("ls_put_can_rightmost",
     "put the can to the rightmost position on the tray",
     "libero_spatial", 220),
    ("ls_pick_cup_top_shelf",
     "pick the cup from the top shelf",
     "libero_spatial", 220),
    ("ls_place_knife_right",
     "place the knife to the right of the cutting board",
     "libero_spatial", 220),
    ("ls_move_block_corner",
     "move the block to the top-right corner of the table",
     "libero_spatial", 220),

    # libero_goal ----------------------------------------------------------
    ("lg_open_oven_door",
     "make sure the oven door is open",
     "libero_goal", 300),
    ("lg_bowl_in_microwave",
     "put the bowl into the microwave and close the door",
     "libero_goal", 300),
    ("lg_plate_on_stove",
     "place the plate on the stove top burner",
     "libero_goal", 300),
    ("lg_kettle_on_plate",
     "put the kettle on the plate next to the sink",
     "libero_goal", 300),
    ("lg_close_cabinet",
     "close the cabinet door on the left side of the kitchen",
     "libero_goal", 300),
    ("lg_butter_in_fridge",
     "put the butter into the refrigerator",
     "libero_goal", 300),
    ("lg_cup_in_rack",
     "place the cup in the drying rack",
     "libero_goal", 300),
    ("lg_lift_book_stack",
     "lift the top book from the stack and put it on the shelf",
     "libero_goal", 300),
    ("lg_pour_kettle",
     "pour water from the kettle into the mug on the counter",
     "libero_goal", 300),
    ("lg_arrange_fruit_bowl",
     "arrange the apple and orange together in the fruit bowl",
     "libero_goal", 300),
]

# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------
# (model_id, model_version, framework, [libero_spatial_sr, libero_goal_sr])
#
# QwenOFT-v2.1  — strong performer; spatial is nearly saturated
# NeuroVLA-v1.2 — weaker;          goal suite reveals the capability gap

MODELS: list[tuple[str, str, str, list[float]]] = [
    ("QwenOFT-v2.1",  "2.1.0", "QwenOFT",  [0.93, 0.74]),
    ("NeuroVLA-v1.2", "1.2.0", "NeuroVLA", [0.87, 0.44]),
]

# Episodes per (model, task): 3 for spatial, 2 for goal
EPISODES_PER_TASK: dict[str, int] = {
    "libero_spatial": 3,
    "libero_goal":    2,
}

# ---------------------------------------------------------------------------
# Failure type distributions
# ---------------------------------------------------------------------------
# Weights used by random.choices(); must match FailureStepDetector heuristics
# in sepa_eval/mining/failure_classifier.py.

FAILURE_WEIGHTS: dict[str, dict[str, int]] = {
    "libero_spatial": {
        "grasp":               40,
        "timeout":             20,
        "contact_dynamics":    15,
        "recovery":            10,
        "pose_estimation":     10,
        "out_of_reach":         5,
    },
    "libero_goal": {
        "timeout":             25,
        "grasp":               20,
        "contact_dynamics":    20,
        "recovery":            15,
        "pose_estimation":     10,
        "out_of_reach":         5,
        "distractor_confusion": 3,
        "language_grounding":   2,
    },
}

# Scene configs are small dicts capturing the simulator state snapshot for
# each task.  In a real eval these would come from env.get_scene_config().
SCENE_CONFIGS: dict[str, dict] = {
    "ls_pick_red_cup_left":    {"layout": "kitchen_A", "seed": 1001, "objects": [{"name": "red_cup", "pos": [-0.15, 0.12, 0.78]}]},
    "ls_put_butter_right":     {"layout": "kitchen_A", "seed": 1002, "objects": [{"name": "butter",  "pos": [ 0.05, 0.20, 0.78]}]},
    "ls_place_bowl_front":     {"layout": "kitchen_B", "seed": 1003, "objects": [{"name": "bowl",    "pos": [ 0.00,-0.10, 0.78]}]},
    "ls_move_plate_behind":    {"layout": "kitchen_B", "seed": 1004, "objects": [{"name": "plate",   "pos": [ 0.00, 0.30, 0.78]}]},
    "ls_pick_mug_right_shelf": {"layout": "kitchen_A", "seed": 1005, "objects": [{"name": "mug",     "pos": [ 0.20, 0.00, 0.95]}]},
    "ls_slide_book_left":      {"layout": "office_A",  "seed": 1006, "objects": [{"name": "book",    "pos": [ 0.10, 0.00, 0.82]}]},
    "ls_put_can_rightmost":    {"layout": "kitchen_C", "seed": 1007, "objects": [{"name": "can",     "pos": [ 0.08, 0.05, 0.78]}]},
    "ls_pick_cup_top_shelf":   {"layout": "kitchen_A", "seed": 1008, "objects": [{"name": "cup",     "pos": [ 0.00, 0.00, 1.10]}]},
    "ls_place_knife_right":    {"layout": "kitchen_B", "seed": 1009, "objects": [{"name": "knife",   "pos": [-0.05, 0.08, 0.78]}]},
    "ls_move_block_corner":    {"layout": "tabletop_A","seed": 1010, "objects": [{"name": "block",   "pos": [ 0.02,-0.05, 0.78]}]},

    "lg_open_oven_door":       {"layout": "kitchen_C", "seed": 2001, "objects": [{"name": "oven",    "door_state": 0.0}]},
    "lg_bowl_in_microwave":    {"layout": "kitchen_A", "seed": 2002, "objects": [{"name": "bowl",    "pos": [ 0.10, 0.05, 0.78]}, {"name": "microwave", "door_state": 1.0}]},
    "lg_plate_on_stove":       {"layout": "kitchen_B", "seed": 2003, "objects": [{"name": "plate",   "pos": [ 0.00,-0.15, 0.78]}]},
    "lg_kettle_on_plate":      {"layout": "kitchen_C", "seed": 2004, "objects": [{"name": "kettle",  "pos": [-0.10, 0.10, 0.90]}, {"name": "plate", "pos": [ 0.10, 0.20, 0.78]}]},
    "lg_close_cabinet":        {"layout": "kitchen_A", "seed": 2005, "objects": [{"name": "cabinet", "door_state": 1.0}]},
    "lg_butter_in_fridge":     {"layout": "kitchen_B", "seed": 2006, "objects": [{"name": "butter",  "pos": [ 0.05, 0.05, 0.78]}, {"name": "fridge", "door_state": 1.0}]},
    "lg_cup_in_rack":          {"layout": "kitchen_C", "seed": 2007, "objects": [{"name": "cup",     "pos": [-0.08, 0.03, 0.78]}]},
    "lg_lift_book_stack":      {"layout": "office_A",  "seed": 2008, "objects": [{"name": "book_1",  "pos": [ 0.00, 0.00, 0.82]}, {"name": "book_2", "pos": [0.00, 0.00, 0.88]}]},
    "lg_pour_kettle":          {"layout": "kitchen_A", "seed": 2009, "objects": [{"name": "kettle",  "pos": [-0.05, 0.15, 0.90]}, {"name": "mug", "pos": [0.10, 0.00, 0.78]}]},
    "lg_arrange_fruit_bowl":   {"layout": "kitchen_B", "seed": 2010, "objects": [{"name": "apple",   "pos": [-0.08, 0.05, 0.78]}, {"name": "orange", "pos": [0.08, 0.05, 0.78]}, {"name": "bowl", "pos": [0.00, 0.20, 0.78]}]},
}

# ---------------------------------------------------------------------------
# Low-level generators
# ---------------------------------------------------------------------------

def _fake_image_bytes(rng: random.Random, n_pixels: int = 192) -> bytes:
    """Return `n_pixels` random bytes simulating a compressed camera frame."""
    return bytes(rng.getrandbits(8) for _ in range(n_pixels))


def _normal_qpos(rng: random.Random) -> list[float]:
    """7-DOF joint position + 1 gripper (0-1)."""
    joints = [round(rng.gauss(0.0, 0.3), 4) for _ in range(7)]
    gripper = round(rng.uniform(0.0, 1.0), 4)
    return joints + [gripper]


def _l2_norm(vec: list[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


# ---------------------------------------------------------------------------
# Observation + action sequence builders
# ---------------------------------------------------------------------------
# Each builder returns (obs_list, action_list, episode_length, failure_step)
# tuned so sepa_eval's FailureStepDetector classifies the trace correctly.

def _build_success_episode(
    rng: random.Random, max_steps: int
) -> tuple[list[dict], list[list[float]], int, None]:
    ep_len = rng.randint(int(max_steps * 0.35), int(max_steps * 0.65))
    n_obs = min(ep_len, 15)
    obs_list = []
    for _ in range(n_obs):
        gripper = round(rng.uniform(0.4, 1.0), 4)   # stable closure
        obs_list.append({
            "agentview_rgb": _fake_image_bytes(rng),
            "wrist_rgb":     _fake_image_bytes(rng, 96),
            "qpos":          _normal_qpos(rng),
            "gripper_state": [gripper],
            "timestamp":     round(rng.uniform(0.0, ep_len * 0.02), 4),
        })
    actions = [[round(rng.gauss(0.0, 0.15), 4) for _ in range(7)] for _ in range(min(ep_len, 20))]
    return obs_list, actions, ep_len, None


def _build_timeout_episode(
    rng: random.Random, max_steps: int
) -> tuple[list[dict], list[list[float]], int, int]:
    """Episode exhausts max_steps without success."""
    ep_len = max_steps
    n_obs = 15
    obs_list = []
    gripper = round(rng.uniform(0.0, 0.2), 4)   # gripper barely closes — model stuck
    for _ in range(n_obs):
        obs_list.append({
            "agentview_rgb": _fake_image_bytes(rng),
            "wrist_rgb":     _fake_image_bytes(rng, 96),
            "qpos":          _normal_qpos(rng),
            "gripper_state": [gripper],
            "timestamp":     round(rng.uniform(0.0, 6.0), 4),
        })
    actions = [[round(rng.gauss(0.0, 0.08), 4) for _ in range(7)] for _ in range(20)]
    # timeout is priority-1; episode_length == max_steps triggers it
    return obs_list, actions, ep_len, ep_len


def _build_grasp_episode(
    rng: random.Random, max_steps: int
) -> tuple[list[dict], list[list[float]], int, int]:
    """Gripper oscillates (variance > 0.1) — grasp failure detected."""
    ep_len = rng.randint(int(max_steps * 0.35), int(max_steps * 0.65))
    failure_step = int(ep_len * rng.uniform(0.35, 0.55))
    n_obs = min(ep_len, 20)
    obs_list = []
    for i in range(n_obs):
        # Alternate between open (0.0) and closed (1.0) — variance = 0.25 > 0.1
        gripper_val = 0.0 if i % 2 == 0 else 1.0
        obs_list.append({
            "agentview_rgb": _fake_image_bytes(rng),
            "wrist_rgb":     _fake_image_bytes(rng, 96),
            "qpos":          _normal_qpos(rng),
            "gripper_state": [gripper_val],
            "timestamp":     round(i * 0.02, 4),
        })
    actions = [[round(rng.gauss(0.0, 0.12), 4) for _ in range(7)] for _ in range(20)]
    return obs_list, actions, ep_len, failure_step


def _build_contact_dynamics_episode(
    rng: random.Random, max_steps: int
) -> tuple[list[dict], list[list[float]], int, int]:
    """Failure occurs late (>70% through episode) — contact dynamics failure."""
    ep_len = rng.randint(int(max_steps * 0.55), int(max_steps * 0.80))
    failure_step = int(ep_len * rng.uniform(0.72, 0.92))
    n_obs = min(ep_len, 20)
    gripper = round(rng.uniform(0.7, 1.0), 4)   # stably closed (grasped)
    obs_list = []
    for _ in range(n_obs):
        obs_list.append({
            "agentview_rgb": _fake_image_bytes(rng),
            "wrist_rgb":     _fake_image_bytes(rng, 96),
            "qpos":          _normal_qpos(rng),
            "gripper_state": [gripper],
            "timestamp":     round(rng.uniform(0.0, ep_len * 0.02), 4),
        })
    actions = [[round(rng.gauss(0.0, 0.10), 4) for _ in range(7)] for _ in range(20)]
    return obs_list, actions, ep_len, failure_step


def _build_recovery_episode(
    rng: random.Random, max_steps: int
) -> tuple[list[dict], list[list[float]], int, int]:
    """Last 10 actions are near-duplicates (L2 dist < 0.05) — recovery loop detected."""
    ep_len = rng.randint(int(max_steps * 0.40), int(max_steps * 0.65))
    # failure_step between 30-65% so neither pose_estimation nor contact_dynamics wins
    failure_step = int(ep_len * rng.uniform(0.32, 0.62))
    n_obs = min(ep_len, 20)
    gripper = round(rng.uniform(0.3, 0.7), 4)
    obs_list = []
    for _ in range(n_obs):
        obs_list.append({
            "agentview_rgb": _fake_image_bytes(rng),
            "wrist_rgb":     _fake_image_bytes(rng, 96),
            "qpos":          _normal_qpos(rng),
            "gripper_state": [gripper],
            "timestamp":     round(rng.uniform(0.0, ep_len * 0.02), 4),
        })
    # First 10 actions: varied; last 10 actions: near-identical (stuck recovery loop)
    stuck_action = [round(rng.gauss(0.0, 0.01), 4) for _ in range(7)]
    actions = (
        [[round(rng.gauss(0.0, 0.14), 4) for _ in range(7)] for _ in range(10)]
        + [list(stuck_action) for _ in range(10)]
    )
    return obs_list, actions, ep_len, failure_step


def _build_pose_estimation_episode(
    rng: random.Random, max_steps: int
) -> tuple[list[dict], list[list[float]], int, int]:
    """Failure occurs early (<30% through episode) — model never approached correctly."""
    ep_len = rng.randint(int(max_steps * 0.25), int(max_steps * 0.50))
    failure_step = int(ep_len * rng.uniform(0.08, 0.25))
    n_obs = min(ep_len, 15)
    gripper = round(rng.uniform(0.0, 0.3), 4)   # never gripped
    obs_list = []
    for _ in range(n_obs):
        obs_list.append({
            "agentview_rgb": _fake_image_bytes(rng),
            "wrist_rgb":     _fake_image_bytes(rng, 96),
            "qpos":          _normal_qpos(rng),
            "gripper_state": [gripper],
            "timestamp":     round(rng.uniform(0.0, ep_len * 0.02), 4),
        })
    actions = [[round(rng.gauss(0.0, 0.11), 4) for _ in range(7)] for _ in range(15)]
    return obs_list, actions, ep_len, failure_step


def _build_out_of_reach_episode(
    rng: random.Random, max_steps: int
) -> tuple[list[dict], list[list[float]], int, int]:
    """Last 10 actions have mean L2 > 0.8 — arm straining outside workspace."""
    ep_len = rng.randint(int(max_steps * 0.35), int(max_steps * 0.60))
    failure_step = max(0, ep_len - 15)
    n_obs = min(ep_len, 15)
    gripper = round(rng.uniform(0.0, 0.4), 4)
    obs_list = []
    for _ in range(n_obs):
        obs_list.append({
            "agentview_rgb": _fake_image_bytes(rng),
            "wrist_rgb":     _fake_image_bytes(rng, 96),
            "qpos":          _normal_qpos(rng),
            "gripper_state": [gripper],
            "timestamp":     round(rng.uniform(0.0, ep_len * 0.02), 4),
        })
    # First 10 normal; last 10 maxed out — L2 ≈ sqrt(7 × 0.81) ≈ 2.38 >> 0.8
    high_action = [0.9] * 7
    actions = (
        [[round(rng.gauss(0.0, 0.12), 4) for _ in range(7)] for _ in range(10)]
        + [list(high_action) for _ in range(10)]
    )
    return obs_list, actions, ep_len, failure_step


def _build_distractor_confusion_episode(
    rng: random.Random, max_steps: int
) -> tuple[list[dict], list[list[float]], int, int]:
    """Some observations include num_distractors > 0 — model grabbed wrong object."""
    ep_len = rng.randint(int(max_steps * 0.30), int(max_steps * 0.55))
    failure_step = int(ep_len * rng.uniform(0.20, 0.50))
    n_obs = min(ep_len, 15)
    gripper = round(rng.uniform(0.5, 0.9), 4)
    obs_list = []
    for i in range(n_obs):
        obs = {
            "agentview_rgb": _fake_image_bytes(rng),
            "wrist_rgb":     _fake_image_bytes(rng, 96),
            "qpos":          _normal_qpos(rng),
            "gripper_state": [gripper],
            "timestamp":     round(i * 0.02, 4),
        }
        if i >= 2:   # distractors appear from step 2 onward
            obs["num_distractors"] = 3
        obs_list.append(obs)
    actions = [[round(rng.gauss(0.0, 0.12), 4) for _ in range(7)] for _ in range(15)]
    return obs_list, actions, ep_len, failure_step


def _build_language_grounding_episode(
    rng: random.Random, max_steps: int, instruction: str
) -> tuple[list[dict], list[list[float]], int, int]:
    """Observations contain an object_class that doesn't appear in the instruction."""
    ep_len = rng.randint(int(max_steps * 0.20), int(max_steps * 0.40))
    failure_step = int(ep_len * rng.uniform(0.10, 0.30))
    n_obs = min(ep_len, 12)
    gripper = round(rng.uniform(0.0, 0.5), 4)
    # Pick an object class NOT mentioned in the instruction for the mismatch
    candidates = ["mug", "kettle", "bottle", "spatula", "pan", "jar", "spoon", "fork"]
    instr_lower = instruction.lower()
    mismatch_class = next(
        (c for c in candidates if c not in instr_lower), "bottle"
    )
    obs_list = []
    for i in range(n_obs):
        obs = {
            "agentview_rgb": _fake_image_bytes(rng),
            "wrist_rgb":     _fake_image_bytes(rng, 96),
            "qpos":          _normal_qpos(rng),
            "gripper_state": [gripper],
            "timestamp":     round(i * 0.02, 4),
            "object_class":  mismatch_class,   # triggers LanguageGroundingDetector
        }
        obs_list.append(obs)
    actions = [[round(rng.gauss(0.0, 0.10), 4) for _ in range(7)] for _ in range(12)]
    return obs_list, actions, ep_len, failure_step


# ---------------------------------------------------------------------------
# Top-level episode factory
# ---------------------------------------------------------------------------

def _build_episode(
    rng: random.Random,
    failure_type: str | None,
    max_steps: int,
    instruction: str,
) -> tuple[list[dict], list[list[float]], int, int | None]:
    """Dispatch to the correct builder and return (obs, actions, ep_len, failure_step)."""
    if failure_type is None:
        return _build_success_episode(rng, max_steps)
    builders = {
        "timeout":             lambda: _build_timeout_episode(rng, max_steps),
        "grasp":               lambda: _build_grasp_episode(rng, max_steps),
        "contact_dynamics":    lambda: _build_contact_dynamics_episode(rng, max_steps),
        "recovery":            lambda: _build_recovery_episode(rng, max_steps),
        "pose_estimation":     lambda: _build_pose_estimation_episode(rng, max_steps),
        "out_of_reach":        lambda: _build_out_of_reach_episode(rng, max_steps),
        "distractor_confusion": lambda: _build_distractor_confusion_episode(rng, max_steps),
        "language_grounding":  lambda: _build_language_grounding_episode(rng, max_steps, instruction),
    }
    return builders[failure_type]()


# ---------------------------------------------------------------------------
# EpisodeTrace builder
# ---------------------------------------------------------------------------

def _build_trace(
    rng: random.Random,
    run_id: str,
    task_id: str,
    instruction: str,
    benchmark: str,
    max_steps: int,
    model_id: str,
    model_version: str,
    success: bool,
    failure_type: str | None,
    created_at: datetime,
) -> EpisodeTrace:
    obs_list, actions, ep_len, failure_step = _build_episode(
        rng, failure_type if not success else None, max_steps, instruction
    )
    critic_scores: dict[str, Any] = {}
    if success:
        critic_scores = {
            "semantic":  round(rng.uniform(0.82, 0.98), 3),
            "safety":    round(rng.uniform(0.85, 1.00), 3),
            "robustness": round(rng.uniform(0.70, 0.95), 3),
        }
    else:
        critic_scores = {
            "semantic":  round(rng.uniform(0.10, 0.45), 3),
            "safety":    round(rng.uniform(0.40, 0.75), 3),
            "robustness": round(rng.uniform(0.05, 0.40), 3),
        }

    failure_attribution: dict[str, float] = {}
    if failure_type:
        primary_score = round(rng.uniform(0.55, 0.85), 3)
        failure_attribution[failure_type] = primary_score
        remaining = round(1.0 - primary_score, 3)
        secondary_type = rng.choice([ft for ft in FAILURE_WEIGHTS[benchmark] if ft != failure_type])
        failure_attribution[secondary_type] = remaining

    return EpisodeTrace(
        identity=TraceIdentity(
            trace_id=str(uuid.uuid4()),
            eval_run_id=run_id,
            benchmark=benchmark,
            task_id=task_id,
            task_instruction=instruction,
            model_id=model_id,
            model_version=model_version,
        ),
        scene=SceneConfig(
            scene_config=SCENE_CONFIGS.get(task_id, {"seed": rng.randint(1000, 9999)}),
            init_state=bytes([rng.getrandbits(8) for _ in range(16)]),
            replay_mode="reseed",
        ),
        rollout=RolloutData(
            observations=obs_list,
            actions=actions,
            episode_length=ep_len,
            success=success,
            failure_step=failure_step,
        ),
        labels=TraceLabels(
            critic_scores=critic_scores,
            failure_type=failure_type,
            failure_attribution=failure_attribution,
        ),
        provenance=TaskProvenance(
            parent_task_id=None,
            mutation_type=None,
            promotion_status="seed",
        ),
    )


# ---------------------------------------------------------------------------
# Aggregate table helpers
# ---------------------------------------------------------------------------

def _upsert_model_task_result(
    conn: sqlite3.Connection,
    model_id: str,
    task_id: str,
    benchmark: str,
    successes: list[bool],
    ep_lengths: list[int],
    created_at: str,
) -> None:
    n = len(successes)
    sr = sum(successes) / n if n else 0.0
    # "clean" SR: exclude episodes where safety critic score < 0.6
    avg_ep = sum(ep_lengths) / n if n else 0.0
    conn.execute(
        """
        INSERT OR REPLACE INTO model_task_results
            (model_id, task_id, benchmark, n_trials,
             success_rate, clean_success_rate, avg_episode_length, last_eval_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (model_id, task_id, benchmark, n, round(sr, 4), round(sr * 0.97, 4),
         round(avg_ep, 1), created_at),
    )


def _upsert_task_row(
    conn: sqlite3.Connection,
    task_id: str,
    benchmark: str,
    instruction: str,
    scene_config: dict,
    all_model_srs: list[float],
    created_at: str,
) -> None:
    """Insert a tasks row with saturation/discriminative power metadata."""
    import json as _json
    n = len(all_model_srs)
    if n == 0:
        disc_power = 0.0
        saturation = 0
    else:
        mean_sr = sum(all_model_srs) / n
        variance = sum((sr - mean_sr) ** 2 for sr in all_model_srs) / n
        disc_power = round(math.sqrt(variance), 4)   # std-dev of SR across models
        saturation = 1 if all(sr >= 0.90 for sr in all_model_srs) else 0

    conn.execute(
        """
        INSERT OR REPLACE INTO tasks
            (task_id, benchmark, instruction, scene_config, mutation_lineage,
             promotion_status, discriminative_power, saturation_flag, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            task_id, benchmark, instruction,
            _json.dumps(SCENE_CONFIGS.get(task_id, {})),
            _json.dumps([]),
            "seed",
            disc_power,
            saturation,
            created_at,
        ),
    )


def _insert_failure_clusters(
    conn: sqlite3.Connection,
    run_id: str,
    benchmark: str,
    failure_counts: dict[str, int],
    created_at: str,
) -> None:
    """Populate failure_clusters table for demo completeness."""
    import json as _json
    summaries = {
        "grasp":               "Model consistently loses grip during approach; gripper alternates open/closed. Likely cause: poor depth estimation at the target object surface.",
        "timeout":             "Agent enters a looping search pattern without committing. Possibly confused by scene layout or over-constrained action distribution.",
        "contact_dynamics":    "Object slips post-grasp during transport. Suggests insufficient friction model calibration in simulation transfer.",
        "recovery":            "Agent repeats the same ineffective micro-adjustment in a 10-step loop. Recovery policy never triggers an escape heuristic.",
        "pose_estimation":     "Arm never reaches correct approach pose within first 30% of episode. VLM spatial tokenisation likely misidentifies object centroid.",
        "out_of_reach":        "End-effector exceeds workspace bounds while attempting an extended reach. Task placement seed produces geometrically infeasible configurations.",
        "distractor_confusion": "Gripper closes on a nearby distractor object rather than the target. Instruction grounding fails when multiple similar objects are present.",
        "language_grounding":  "Model picks the wrong object class despite unambiguous instruction. Suggests embedding space conflation of similar-shaped household objects.",
    }
    for ft, count in failure_counts.items():
        if count == 0:
            continue
        conn.execute(
            """
            INSERT OR IGNORE INTO failure_clusters
                (cluster_id, eval_run_id, failure_type, centroid,
                 representative_trace_id, member_count, llm_summary, summarized_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                run_id,
                ft,
                b"",          # centroid embedding — not computed offline
                "",
                count,
                summaries.get(ft, ""),
                created_at,
            ),
        )
    conn.commit()


def _insert_promoted_mutations(
    conn: sqlite3.Connection,
    run_id: str,
    benchmark: str,
    seed_task_id: str,
    seed_instruction: str,
    seed_scene: dict,
    rng: random.Random,
    created_at: str,
) -> None:
    """Add a handful of promoted mutation tasks to show the evolution loop."""
    import json as _json

    mutations = [
        {
            "mutation_type": "PosePerturbation",
            "instruction":   seed_instruction + " [rotated 45°]",
            "params":        {"delta_rotation_deg": 45, "axis": "z"},
            "status":        "promoted",
        },
        {
            "mutation_type": "DistractorAdd",
            "instruction":   seed_instruction + " [with two distractors]",
            "params":        {"n_distractors": 2, "distractor_type": "similar_shape"},
            "status":        "promoted",
        },
        {
            "mutation_type": "InstructionParaphrase",
            "instruction":   "Take the object and reposition it per the spatial hint provided.",
            "params":        {"paraphrase_level": "abstract"},
            "status":        "candidate",
        },
    ]

    for m in mutations:
        task_id = str(uuid.uuid4())
        mod_scene = dict(seed_scene)
        mod_scene["mutation"] = m["params"]
        disc_power = round(rng.uniform(0.28, 0.50), 4)

        conn.execute(
            """
            INSERT OR REPLACE INTO tasks
                (task_id, benchmark, instruction, scene_config, mutation_lineage,
                 promotion_status, discriminative_power, saturation_flag, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id, benchmark, m["instruction"],
                _json.dumps(mod_scene),
                _json.dumps([seed_task_id]),
                m["status"],
                disc_power,
                0,
                created_at,
            ),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(output_dir: str, seed: int = 42) -> None:
    rng = random.Random(seed)

    os.makedirs(output_dir, exist_ok=True)
    db_path    = os.path.join(output_dir, "eval.db")
    traces_dir = os.path.join(output_dir, "traces")

    print(f"Initialising EvalMemory at: {output_dir}")
    memory = EvalMemory(db_path=db_path, memory_dir=traces_dir)
    conn = memory._conn

    run_id    = str(uuid.uuid4())
    now_utc   = datetime.now(tz=timezone.utc).replace(tzinfo=None)
    now_iso   = now_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")

    # Register models in the models table
    for model_id, model_version, framework, _ in MODELS:
        import json as _json
        conn.execute(
            """
            INSERT OR REPLACE INTO models
                (model_id, framework, checkpoint, benchmarks, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                model_id,
                framework,
                f"checkpoints/{model_id.replace('/', '_')}.pt",
                _json.dumps(["libero_spatial", "libero_goal"]),
                now_iso,
            ),
        )
    conn.commit()

    # ------------------------------------------------------------------
    # Episode generation loop
    # ------------------------------------------------------------------
    #
    # For each (model, task) pair we generate EPISODES_PER_TASK[benchmark]
    # episodes.  Success is determined by the model's per-benchmark SR
    # calibrated above; failures are assigned a type from FAILURE_WEIGHTS.

    total_traces = 0
    total_failures = 0
    failure_counts: dict[str, dict[str, int]] = {
        "libero_spatial": {},
        "libero_goal": {},
    }

    # task_id → { model_id → [success, ...] } — for aggregate table
    task_results: dict[str, dict[str, dict]] = {}

    for task_id, instruction, benchmark, max_steps in LIBERO_TASKS:
        n_eps = EPISODES_PER_TASK[benchmark]
        task_results.setdefault(task_id, {})

        for model_id, model_version, _framework, sr_by_benchmark in MODELS:
            sr_idx = 0 if benchmark == "libero_spatial" else 1
            target_sr = sr_by_benchmark[sr_idx]

            successes: list[bool]   = []
            ep_lengths: list[int]   = []
            ft_weights = FAILURE_WEIGHTS[benchmark]

            for ep_idx in range(n_eps):
                # Deterministically decide success/failure
                success = rng.random() < target_sr

                if success:
                    failure_type = None
                else:
                    ft_pool  = list(ft_weights.keys())
                    ft_wts   = list(ft_weights.values())
                    failure_type = rng.choices(ft_pool, weights=ft_wts, k=1)[0]
                    total_failures += 1
                    failure_counts[benchmark][failure_type] = (
                        failure_counts[benchmark].get(failure_type, 0) + 1
                    )

                # Stagger created_at across episodes for realistic timestamps
                offset = timedelta(minutes=(total_traces * 2 + ep_idx))
                created_at = (now_utc - timedelta(hours=6)) + offset

                trace = _build_trace(
                    rng=rng,
                    run_id=run_id,
                    task_id=task_id,
                    instruction=instruction,
                    benchmark=benchmark,
                    max_steps=max_steps,
                    model_id=model_id,
                    model_version=model_version,
                    success=success,
                    failure_type=failure_type,
                    created_at=created_at,
                )
                # Override the trace's created_at with the staggered timestamp
                # (EvalMemory uses _now_iso() internally; we patch created_at via
                # a direct conn.execute after the fact for demo purposes)
                memory.record_trace(trace)

                successes.append(success)
                ep_lengths.append(trace.rollout.episode_length)
                total_traces += 1
                print(f"  [{total_traces:>3}] {model_id[:14]} | {benchmark[:16]} | "
                      f"{task_id[:30]} | {'✓' if success else '✗'}"
                      f"{' (' + failure_type + ')' if failure_type else ''}")

            task_results[task_id][model_id] = {
                "successes":  successes,
                "ep_lengths": ep_lengths,
                "benchmark":  benchmark,
            }
            _upsert_model_task_result(
                conn, model_id, task_id, benchmark,
                successes, ep_lengths, now_iso,
            )

        # Aggregate per-task discriminative power
        all_srs = []
        for model_id, _, _, sr_by_benchmark in MODELS:
            sr_idx = 0 if benchmark == "libero_spatial" else 1
            n_eps_done = EPISODES_PER_TASK[benchmark]
            # Recalculate from actual results
            res = task_results[task_id][model_id]
            all_srs.append(sum(res["successes"]) / len(res["successes"]))

        _upsert_task_row(conn, task_id, benchmark, instruction,
                         SCENE_CONFIGS.get(task_id, {}), all_srs, now_iso)

    conn.commit()

    # ------------------------------------------------------------------
    # Failure clusters
    # ------------------------------------------------------------------
    for benchmark, fc in failure_counts.items():
        _insert_failure_clusters(conn, run_id, benchmark, fc, now_iso)

    # ------------------------------------------------------------------
    # Promoted mutation candidates (evolution loop artefacts)
    # ------------------------------------------------------------------
    # Attach a few mutations to the hardest goal task for demo richness
    hardest_goal_task = "lg_pour_kettle"
    hardest_instruction = next(
        inst for tid, inst, bench, _ in LIBERO_TASKS
        if tid == hardest_goal_task
    )
    _insert_promoted_mutations(
        conn, run_id, "libero_goal",
        hardest_goal_task, hardest_instruction,
        SCENE_CONFIGS[hardest_goal_task], rng, now_iso,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print(f"  SEPA-Eval synthetic demo data written")
    print("=" * 60)
    print(f"  Output dir  : {os.path.abspath(output_dir)}")
    print(f"  Run ID      : {run_id}")
    print(f"  Total traces: {total_traces}  ({total_failures} failures)")
    print()

    # Per-benchmark summary
    for benchmark in ["libero_spatial", "libero_goal"]:
        bench_traces_cur = conn.execute(
            "SELECT COUNT(*) FROM traces WHERE benchmark=?", (benchmark,)
        )
        bench_n = bench_traces_cur.fetchone()[0]
        bench_fail_cur = conn.execute(
            "SELECT COUNT(*) FROM traces WHERE benchmark=? AND success=0", (benchmark,)
        )
        bench_fails = bench_fail_cur.fetchone()[0]
        sr = round((bench_n - bench_fails) / bench_n, 3) if bench_n else 0.0
        print(f"  {benchmark:<18}  {bench_n:>3} episodes  SR={sr:.1%}  "
              f"failures={bench_fails}")

    print()
    print("  Failure type breakdown:")
    for benchmark, fc in failure_counts.items():
        if not fc:
            continue
        print(f"    {benchmark}:")
        for ft, cnt in sorted(fc.items(), key=lambda x: -x[1]):
            print(f"      {ft:<25} {cnt}")

    sat_tasks = conn.execute(
        "SELECT COUNT(*) FROM tasks WHERE saturation_flag=1"
    ).fetchone()[0]
    total_tasks = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    promoted_tasks = conn.execute(
        "SELECT COUNT(*) FROM tasks WHERE promotion_status='promoted'"
    ).fetchone()[0]

    print()
    print(f"  Tasks in DB     : {total_tasks} "
          f"({sat_tasks} saturated, {promoted_tasks} promoted mutations)")
    print()
    out = os.path.abspath(output_dir)
    print("  Next steps (run from AlphaBrain/):")
    print(f"    python -m sepa_eval --memory-dir {out} report")
    print(f"    python -m sepa_eval --memory-dir {out} mine")
    print(f"    python -m sepa_eval --memory-dir {out} diff QwenOFT-v2.1 NeuroVLA-v1.2")
    print(f"    python -m sepa_eval --memory-dir {out} review list")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic LIBERO traces for the SEPA-Eval customer demo."
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "demo_eval_memory"),
        help="Directory to write EvalMemory DB and trace files (default: demo/demo_eval_memory/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()
    main(output_dir=args.output_dir, seed=args.seed)
