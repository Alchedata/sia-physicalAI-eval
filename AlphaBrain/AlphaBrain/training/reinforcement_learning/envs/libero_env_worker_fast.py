#!/usr/bin/env python3
"""
Fast LIBERO environment worker — socket pair IPC, MuJoCo env reuse.

Launched by persistent_env_pool.py. Receives socket FD as argv[1].
Communicates via socket (not stdin/stdout) — eliminates pipe buffer deadlock.

Key optimization: same task → only reset() + set_init_state() (no env recreation).
"""

import os
import socket
import struct
import sys
import time
import traceback

import msgpack
import numpy as np

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


# ── Socket IPC helpers ──

def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Parent closed socket")
        buf.extend(chunk)
    return bytes(buf)


def _read_msg_sock(sock: socket.socket) -> dict:
    raw_len = _recv_exact(sock, 4)
    length = struct.unpack("<I", raw_len)[0]
    data = _recv_exact(sock, length)
    return msgpack.unpackb(data, raw=False)


def _write_msg_sock(sock: socket.socket, obj: dict):
    data = msgpack.packb(obj, use_bin_type=True)
    header = struct.pack("<I", len(data))
    sock.sendall(header + data)


# ── Obs helpers ──

def _img_to_bytes(arr: np.ndarray) -> bytes:
    return arr.tobytes()


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


# ── Main loop ──

def main():
    # Get socket FD from command line
    sock_fd = int(sys.argv[1])
    sock = socket.fromfd(sock_fd, socket.AF_UNIX, socket.SOCK_STREAM)
    os.close(sock_fd)  # fromfd dup'd it

    env = None
    current_task_key = None
    task_suite_cache = {}

    def get_suite(name: str):
        if name not in task_suite_cache:
            benchmark_dict = benchmark.get_benchmark_dict()
            task_suite_cache[name] = benchmark_dict[name]()
        return task_suite_cache[name]

    while True:
        try:
            msg = _read_msg_sock(sock)
        except (ConnectionError, EOFError):
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

                new_task_key = (suite_name, task_id)

                if new_task_key != current_task_key:
                    if env is not None:
                        try:
                            env.close()
                        except Exception:
                            pass
                    env_args = {
                        "bddl_file_name": task_suite.get_task_bddl_file_path(task_id),
                        "camera_heights": LIBERO_ENV_RESOLUTION,
                        "camera_widths":  LIBERO_ENV_RESOLUTION,
                    }
                    for attempt in range(3):
                        try:
                            env = OffScreenRenderEnv(**env_args)
                            env.seed(seed)
                            break
                        except Exception as e:
                            print(f"[WORKER] OffScreenRenderEnv failed (attempt {attempt+1}): {e}",
                                  file=sys.stderr, flush=True)
                            time.sleep(1.0 + attempt)
                            if attempt == 2:
                                raise
                    current_task_key = new_task_key

                obs = env.reset()
                obs = env.set_init_state(initial_states[state_idx])

                _write_msg_sock(sock, {
                    "status": "ok",
                    "obs":    _parse_obs(obs),
                    "reward": 0.0,
                    "done":   False,
                    "task_description": task.language,
                    "max_steps": MAX_STEPS.get(suite_name, 320),
                })

            elif cmd == "step":
                action = msg["action"]
                obs, reward, done, info = env.step(action)
                _write_msg_sock(sock, {
                    "status": "ok",
                    "obs":    _parse_obs(obs),
                    "reward": float(reward),
                    "done":   bool(done),
                })

            elif cmd == "step_chunk":
                actions = msg["actions"]
                final_reward = 0.0
                final_done = False
                steps_taken = 0
                for a in actions:
                    obs, reward, done, info = env.step(a)
                    steps_taken += 1
                    if done:
                        final_done = True
                        final_reward = float(reward)
                        break
                if not final_done:
                    final_reward = float(reward)
                _write_msg_sock(sock, {
                    "status": "ok",
                    "obs": _parse_obs(obs),
                    "reward": final_reward,
                    "done": final_done,
                    "steps_taken": steps_taken,
                })

            elif cmd == "close":
                if env is not None:
                    env.close()
                break

            else:
                _write_msg_sock(sock, {"status": "error", "message": f"Unknown cmd: {cmd}"})

        except Exception:
            tb = traceback.format_exc()
            try:
                _write_msg_sock(sock, {"status": "error", "message": tb})
            except Exception:
                pass

    sock.close()


if __name__ == "__main__":
    main()
