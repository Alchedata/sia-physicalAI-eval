#!/usr/bin/env python3
"""
LIBERO environment worker — runs inside the LIBERO Python environment.

Launched as a subprocess by libero_env.py (VLA Python env).
Communicates via stdin/stdout using length-prefixed msgpack messages.

Protocol (both directions):
  [4-byte little-endian uint32 length][msgpack payload]

Commands received (from parent):
  {"cmd": "reset", "task_suite": str, "task_id": int, "initial_state_idx": int, "seed": int}
  {"cmd": "step",  "action": [7 floats]}
  {"cmd": "close"}

Responses sent (to parent):
  {"status": "ok", "obs": {"primary": <bytes PNG>, "wrist": <bytes PNG>, "state": [8 floats]},
   "reward": float, "done": bool}
  {"status": "error", "message": str}
"""

import io
import struct
import sys
import traceback

import msgpack
import numpy as np
from PIL import Image

# ── LIBERO imports ─────────────────────────────────────────────────────────────
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

LIBERO_ENV_RESOLUTION = 256

MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object":  280,
    "libero_goal":    320,
    "libero_10":      520,
    "libero_90":      400,
}


# ── Wire protocol helpers ───────────────────────────────────────────────────────

def _read_msg(stream) -> dict:
    raw_len = stream.read(4)
    if len(raw_len) < 4:
        raise EOFError("Parent closed stdin")
    length = struct.unpack("<I", raw_len)[0]
    data = stream.read(length)
    return msgpack.unpackb(data, raw=False)


def _write_msg(stream, obj: dict):
    data = msgpack.packb(obj, use_bin_type=True)
    stream.write(struct.pack("<I", len(data)))
    stream.write(data)
    stream.flush()


def _img_to_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    denom = np.sqrt(1.0 - quat[3] ** 2)
    if denom < 1e-8:
        return np.zeros(3)
    return (quat[:3] / denom) * 2.0 * np.arcsin(denom)


def _parse_obs(obs: dict) -> dict:
    primary = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist   = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    state = np.concatenate([
        obs["robot0_eef_pos"],
        _quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
    ]).astype(np.float32)
    return {
        "primary": _img_to_bytes(primary),
        "wrist":   _img_to_bytes(wrist),
        "state":   state.tolist(),
    }


# ── Main loop ───────────────────────────────────────────────────────────────────

def main():
    stdin  = sys.stdin.buffer
    stdout = sys.stdout.buffer

    env = None
    task_suite_cache = {}  # suite_name -> task_suite object

    def get_suite(name: str):
        if name not in task_suite_cache:
            benchmark_dict = benchmark.get_benchmark_dict()
            task_suite_cache[name] = benchmark_dict[name]()
        return task_suite_cache[name]

    while True:
        try:
            msg = _read_msg(stdin)
        except EOFError:
            break

        cmd = msg.get("cmd")
        try:
            if cmd == "reset":
                suite_name = msg["task_suite"]
                task_id    = msg["task_id"]
                state_idx  = msg.get("initial_state_idx", 0)
                seed       = msg.get("seed", 42)

                task_suite = get_suite(suite_name)
                task = task_suite.get_task(task_id)
                initial_states = task_suite.get_task_init_states(task_id)

                if env is not None:
                    env.close()

                env_args = {
                    "bddl_file_name": task_suite.get_task_bddl_file_path(task_id),
                    "camera_heights": LIBERO_ENV_RESOLUTION,
                    "camera_widths":  LIBERO_ENV_RESOLUTION,
                }
                env = OffScreenRenderEnv(**env_args)
                env.seed(seed)
                obs = env.reset()
                obs = env.set_init_state(initial_states[state_idx])

                _write_msg(stdout, {
                    "status": "ok",
                    "obs":    _parse_obs(obs),
                    "reward": 0.0,
                    "done":   False,
                    "task_description": task.language,
                    "max_steps": MAX_STEPS[suite_name],
                })

            elif cmd == "step":
                action = msg["action"]  # list[7]
                obs, reward, done, info = env.step(action)
                _write_msg(stdout, {
                    "status": "ok",
                    "obs":    _parse_obs(obs),
                    "reward": float(reward),  # paper scheme: success=1.0, failure=0.0
                    "done":   bool(done),
                })

            elif cmd == "close":
                if env is not None:
                    env.close()
                    env = None
                break

            else:
                _write_msg(stdout, {"status": "error", "message": f"Unknown cmd: {cmd}"})

        except Exception:
            _write_msg(stdout, {"status": "error", "message": traceback.format_exc()})


if __name__ == "__main__":
    main()
