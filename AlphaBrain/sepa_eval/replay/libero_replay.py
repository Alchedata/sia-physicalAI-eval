"""
libero_replay.py — turn SEPA-Eval mutated ``scene_config`` dicts into LIBERO init states.

Background
----------
LIBERO evaluation restores episodes with ``env.set_init_state(state)`` where
``state`` is a *flattened MuJoCo sim state*: ``[time(1), qpos(nq), qvel(nv)]``
(see ``LIBERO/libero/libero/envs/env_wrapper.py`` — ``get_sim_state`` /
``regenerate_obs_from_state``).  Every movable LIBERO object owns a free joint
whose qpos slice is 7 numbers: ``(x, y, z, qw, qx, qy, qz)``.

SEPA-Eval mutation operators (``PosePerturbation``) emit *absolute* perturbed
values under keys named ``<object>_pos`` / ``<object>_pose`` (len-3 position),
``<object>_quat`` (len-4 quaternion) and ``<object>_rot`` (len-3 euler, radians).
This module writes those values into the flattened init-state vector so the
mutated scene can be replayed deterministically.

DistractorAdd caveat
--------------------
A compiled MuJoCo model cannot grow new bodies at ``set_init_state`` time, so
``DistractorAdd`` entries **cannot** be realised via the init-state vector.
Instead we generate a mutated BDDL problem file (textual injection of extra
same-category objects + placement regions) and the caller must rebuild the env
from the new BDDL file.  When BDDL regeneration is not possible the replay
*degrades gracefully*: the pose portion is still applied, the distractor
portion is skipped, and the degradation is recorded in the returned info dict
and via ``logging`` so downstream reporting can flag partially-replayed scenes.

LIBERO itself is **not** required at import time; only the env-building helper
``make_env_with_distractors`` needs it and raises a clear ``RuntimeError``
when the library is missing.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import re
from typing import Any, Sequence

logger = logging.getLogger(__name__)

#: number of leading elements in a flattened MuJoCo state before qpos (time).
MUJOCO_TIME_DIM = 1
#: free-joint qpos layout: 3 position + 4 quaternion (w, x, y, z).
FREE_JOINT_QPOS_DIM = 7

_POSE_SUFFIXES = ("_pos", "_pose")
_QUAT_SUFFIX = "_quat"
_ROT_SUFFIX = "_rot"


class ReplayError(RuntimeError):
    """Raised when a scene config cannot be replayed into a LIBERO env."""


def _libero_available() -> bool:
    return importlib.util.find_spec("libero") is not None


# ---------------------------------------------------------------------------
# Pure state-vector math (no LIBERO / MuJoCo dependency)
# ---------------------------------------------------------------------------


def euler_to_quat(euler: Sequence[float]) -> list[float]:
    """Convert XYZ-order euler angles (radians) to a (w, x, y, z) quaternion."""
    roll, pitch, yaw = float(euler[0]), float(euler[1]), float(euler[2])
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


def _strip_pose_key(key: str) -> tuple[str, str] | None:
    """Return (object_name, kind) for a recognised pose key, else None."""
    for suffix in _POSE_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)], "pos"
    if key.endswith(_QUAT_SUFFIX):
        return key[: -len(_QUAT_SUFFIX)], "quat"
    if key.endswith(_ROT_SUFFIX):
        return key[: -len(_ROT_SUFFIX)], "rot"
    return None


def apply_scene_config_to_init_state(
    init_state: Sequence[float],
    scene_config: dict,
    object_addr: dict[str, int],
    time_dim: int = MUJOCO_TIME_DIM,
) -> tuple[list[float], dict]:
    """Write pose values from ``scene_config`` into a flattened init-state vector.

    Parameters
    ----------
    init_state:
        Flattened MuJoCo state ``[time..., qpos..., qvel...]`` (list or ndarray).
    scene_config:
        Mutated scene config.  Keys ``<obj>_pos``/``<obj>_pose`` (len 3),
        ``<obj>_quat`` (len 4, wxyz) and ``<obj>_rot`` (len 3 euler radians)
        are applied; all other keys are ignored.
    object_addr:
        Maps object name -> index *within the flattened state vector* where the
        object's 7-dim free-joint qpos slice starts (i.e. already offset by
        ``time_dim``).  Use :func:`resolve_object_addresses` to build this from
        a live env, or supply it directly in tests.
    time_dim:
        Kept for callers that pass qpos-relative addresses; when an address is
        smaller than ``time_dim`` it is treated as qpos-relative and shifted.

    Returns
    -------
    (new_state, info)
        ``new_state`` is a plain ``list[float]`` copy with poses applied.
        ``info`` records ``applied`` and ``skipped`` key lists.
    """
    state = [float(v) for v in init_state]
    applied: list[str] = []
    skipped: list[str] = []

    for key, value in scene_config.items():
        parsed = _strip_pose_key(key)
        if parsed is None:
            continue
        obj_name, kind = parsed
        if obj_name not in object_addr:
            skipped.append(key)
            continue
        if not isinstance(value, (list, tuple)) or not all(isinstance(v, (int, float)) for v in value):
            skipped.append(key)
            continue

        addr = object_addr[obj_name]
        if addr + FREE_JOINT_QPOS_DIM > len(state):
            raise ReplayError(f"Object '{obj_name}' qpos address {addr} out of bounds for state of length {len(state)}.")

        if kind == "pos":
            if len(value) < 3:
                skipped.append(key)
                continue
            state[addr : addr + 3] = [float(v) for v in value[:3]]
        elif kind == "quat":
            if len(value) != 4:
                skipped.append(key)
                continue
            state[addr + 3 : addr + 7] = _normalize_quat(value)
        else:  # rot (euler radians)
            if len(value) != 3:
                skipped.append(key)
                continue
            state[addr + 3 : addr + 7] = euler_to_quat(value)
        applied.append(key)

    return state, {"applied": applied, "skipped": skipped}


def _normalize_quat(quat: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(float(v) * float(v) for v in quat))
    if norm < 1e-12:
        return [1.0, 0.0, 0.0, 0.0]
    return [float(v) / norm for v in quat]


# ---------------------------------------------------------------------------
# Live-env helpers (duck-typed; require a mujoco-backed LIBERO env)
# ---------------------------------------------------------------------------


def resolve_object_addresses(env: Any, time_dim: int = MUJOCO_TIME_DIM) -> dict[str, int]:
    """Map object names to their free-joint qpos start index in the *flattened* state.

    Works with any duck-typed LIBERO ``ControlEnv``/``OffScreenRenderEnv``-like
    object exposing ``env.sim.model`` with ``joint_names``,
    ``get_joint_qpos_addr(name)`` and free-joint naming ``<object_name>_joint0``
    (LIBERO convention).  Raises ``ReplayError`` when the env does not expose a
    MuJoCo sim (e.g. LIBERO not installed / dummy env).
    """
    sim = getattr(env, "sim", None)
    model = getattr(sim, "model", None)
    if model is None:
        raise ReplayError(
            "Cannot resolve object addresses: env does not expose `sim.model`. "
            "Is LIBERO installed and is this a real simulator env?"
        )

    addresses: dict[str, int] = {}
    for joint_name in getattr(model, "joint_names", []):
        match = re.match(r"^(.+?)_joint\d*$", joint_name)
        if match is None:
            continue
        obj_name = match.group(1)
        addr = model.get_joint_qpos_addr(joint_name)
        if isinstance(addr, tuple):  # (start, end) for multi-dof joints
            start, end = addr
            if end - start != FREE_JOINT_QPOS_DIM:
                continue  # not a free joint
            addresses[obj_name] = int(start) + time_dim
        else:
            continue  # scalar addr => hinge/slide joint, not an object root
    return addresses


def replay_scene_config(
    env: Any,
    scene_config: dict,
    base_init_state: Sequence[float] | None = None,
    object_addr: dict[str, int] | None = None,
) -> tuple[Any, dict]:
    """Apply a mutated scene config to a live LIBERO env via ``set_init_state``.

    Parameters
    ----------
    env:
        Duck-typed LIBERO env with ``set_init_state(state)`` and (optionally)
        ``get_sim_state()`` / ``sim.model`` for address resolution.
    scene_config:
        Mutated scene config from a SEPA-Eval mutation operator.
    base_init_state:
        Flattened state to start from.  Defaults to ``env.get_sim_state()``.
    object_addr:
        Optional precomputed address map (see
        :func:`apply_scene_config_to_init_state`).  Resolved from the env when
        omitted.

    Returns
    -------
    (obs, info)
        ``obs`` is whatever ``env.set_init_state`` returns.  ``info`` contains
        ``applied``, ``skipped`` and ``distractors_applied`` /
        ``degradations`` bookkeeping.
    """
    if base_init_state is None:
        if not hasattr(env, "get_sim_state"):
            raise ReplayError("replay_scene_config needs `base_init_state` or an env exposing `get_sim_state()`.")
        base_init_state = env.get_sim_state()

    if object_addr is None:
        object_addr = resolve_object_addresses(env)

    new_state, info = apply_scene_config_to_init_state(base_init_state, scene_config, object_addr)

    degradations: list[str] = []
    distractors = scene_config.get("distractors") or []
    if distractors:
        # A compiled MuJoCo model cannot grow bodies; distractors need a BDDL
        # rebuild (see generate_distractor_bddl / make_env_with_distractors).
        msg = (
            f"DistractorAdd: {len(distractors)} distractor(s) cannot be injected via set_init_state; "
            "rebuild the env from a mutated BDDL (make_env_with_distractors) to realise them. "
            "Replaying pose changes only."
        )
        logger.warning(msg)
        degradations.append(msg)

    info["distractors_applied"] = False if distractors else None
    info["degradations"] = degradations

    if not hasattr(env, "set_init_state"):
        raise ReplayError("env does not expose `set_init_state`; cannot replay scene config.")
    obs = env.set_init_state(new_state)
    return obs, info


# ---------------------------------------------------------------------------
# DistractorAdd: BDDL regeneration (pure-text; LIBERO only needed to build env)
# ---------------------------------------------------------------------------


def generate_distractor_bddl(
    bddl_text: str,
    distractors: Sequence[dict],
    region_origin: tuple[float, float] = (0.20, 0.20),
    region_half: float = 0.025,
    spacing: float = 0.07,
) -> str:
    """Return BDDL text with same-category distractor objects injected.

    Each distractor duplicates the category of the first entry in
    ``(:objects ...)`` (matching DistractorAdd's ``{"category": "same"}``
    contract), adds a small floor placement region per distractor, declares the
    new object and an ``(On ...)`` init predicate.  Goal and obj_of_interest
    are left untouched so task semantics are preserved.

    Raises
    ------
    ReplayError
        If the BDDL text cannot be parsed well enough to inject objects.
    """
    obj_match = re.search(r"\(:objects\s*\n(.*?)\n\s*\)", bddl_text, re.DOTALL)
    if obj_match is None:
        raise ReplayError("generate_distractor_bddl: no (:objects ...) block found in BDDL text.")
    obj_lines = [ln.strip() for ln in obj_match.group(1).splitlines() if ln.strip()]
    first = re.match(r"^(\S+)\s+-\s+(\S+)$", obj_lines[0]) if obj_lines else None
    if first is None:
        raise ReplayError("generate_distractor_bddl: could not parse first object declaration.")
    category = first.group(2)

    fixtures_match = re.search(r"\(:fixtures\s*\n\s*(\S+)\s+-\s+\S+", bddl_text)
    surface = fixtures_match.group(1) if fixtures_match else "floor"

    existing_names = {re.match(r"^(\S+)", ln).group(1) for ln in obj_lines}
    new_objects, new_regions, new_inits = [], [], []
    idx = 0
    for i, _d in enumerate(distractors):
        while True:
            name = f"{category}_distractor_{idx + 1}"
            idx += 1
            if name not in existing_names:
                break
        x = region_origin[0] + spacing * i
        y = region_origin[1]
        region = f"sepa_distractor_region_{i}"
        new_objects.append(f"    {name} - {category}")
        new_regions.append(
            f"      ({region}\n"
            f"          (:target {surface})\n"
            f"          (:ranges (\n"
            f"              ({x - region_half} {y - region_half} {x + region_half} {y + region_half})\n"
            f"            )\n"
            f"          )\n"
            f"      )"
        )
        new_inits.append(f"    (On {name} {surface}_{region})")

    text = bddl_text
    # objects: append before the closing paren of the :objects block.
    text = text.replace(obj_match.group(0), obj_match.group(0)[:-1].rstrip() + "\n" + "\n".join(new_objects) + "\n  )")

    regions_match = re.search(r"\(:regions\s*\n", text)
    if regions_match is None:
        raise ReplayError("generate_distractor_bddl: no (:regions ...) block found in BDDL text.")
    insert_at = regions_match.end()
    text = text[:insert_at] + "\n".join(new_regions) + "\n" + text[insert_at:]

    init_match = re.search(r"\(:init\s*\n(.*?)\n(\s*\))", text, re.DOTALL)
    if init_match is None:
        raise ReplayError("generate_distractor_bddl: no (:init ...) block found in BDDL text.")
    text = text[: init_match.end(1)] + "\n" + "\n".join(new_inits) + text[init_match.end(1) :]
    return text


def make_env_with_distractors(
    bddl_file: str,
    scene_config: dict,
    out_bddl_file: str | None = None,
    env_kwargs: dict | None = None,
) -> Any:
    """Build an ``OffScreenRenderEnv`` from a distractor-augmented BDDL file.

    Requires the LIBERO library.  Raises ``RuntimeError`` with a clear message
    when LIBERO is not installed so callers can degrade to pose-only replay.
    """
    if not _libero_available():
        raise RuntimeError(
            "LIBERO is not installed: cannot build a distractor-augmented env. "
            "Install LIBERO (pip install -e LIBERO/) or fall back to pose-only "
            "replay via replay_scene_config()."
        )

    with open(bddl_file, encoding="utf-8") as f:
        bddl_text = f.read()
    mutated = generate_distractor_bddl(bddl_text, scene_config.get("distractors") or [])

    if out_bddl_file is None:
        out_bddl_file = bddl_file.replace(".bddl", "_sepa_distractors.bddl")
    with open(out_bddl_file, "w", encoding="utf-8") as f:
        f.write(mutated)

    from libero.libero.envs import OffScreenRenderEnv

    kwargs = {"bddl_file_name": out_bddl_file}
    kwargs.update(env_kwargs or {})
    return OffScreenRenderEnv(**kwargs)
