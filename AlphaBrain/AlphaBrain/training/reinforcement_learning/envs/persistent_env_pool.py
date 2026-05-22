"""
Persistent LiberoEnv pool — keeps subprocess envs alive across rollout iterations.

IPC: socket pair (bidirectional, with settimeout) instead of stdin/stdout pipes.
This eliminates the pipe buffer deadlock that occurs with high-concurrency pipe I/O.

Two-layer optimization:
  1. Subprocess pool: LiberoEnv subprocesses created once, reused across iterations
  2. Fast worker: libero_env_worker_fast.py reuses MuJoCo env for same task
"""

import io
import logging
import os
import socket
import struct
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import msgpack
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_FAST_WORKER_SCRIPT = str(Path(__file__).parent / "libero_env_worker_fast.py")


# ── Socket-based IPC (replaces pipe-based _write_msg/_read_msg) ──

def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes from socket."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Worker closed connection")
        buf.extend(chunk)
    return bytes(buf)


def _write_msg_sock(sock: socket.socket, obj: dict):
    data = msgpack.packb(obj, use_bin_type=True)
    header = struct.pack("<I", len(data))
    sock.sendall(header + data)


def _read_msg_sock(sock: socket.socket, timeout: float = 60) -> dict:
    sock.settimeout(timeout)
    try:
        raw_len = _recv_exact(sock, 4)
    except socket.timeout:
        raise TimeoutError(f"Worker timed out ({timeout}s)")
    except ConnectionError:
        raise RuntimeError("Worker closed connection unexpectedly")
    length = struct.unpack("<I", raw_len)[0]
    data = _recv_exact(sock, length)
    return msgpack.unpackb(data, raw=False)


def _parse_obs(resp_obs: dict) -> dict:
    """Decode raw numpy bytes (fast path) or PNG (fallback)."""
    raw_p = resp_obs["primary"]
    raw_w = resp_obs["wrist"]
    if len(raw_p) == 256 * 256 * 3:
        primary = np.frombuffer(raw_p, dtype=np.uint8).reshape(256, 256, 3).copy()
        wrist = np.frombuffer(raw_w, dtype=np.uint8).reshape(256, 256, 3).copy()
    else:
        primary = np.array(Image.open(io.BytesIO(raw_p)))
        wrist = np.array(Image.open(io.BytesIO(raw_w)))
    state = np.array(resp_obs["state"], dtype=np.float32)
    return {"primary_image": primary, "wrist_image": wrist, "state": state}


class _FastLiberoEnv:
    """Lightweight env proxy using socket pair IPC (no pipe deadlock)."""

    def __init__(self, libero_python: Optional[str] = None, egl_gpu_id: Optional[int] = None):
        self._python_bin = libero_python or os.environ.get("LIBERO_PYTHON", "python")
        self._worker_env = os.environ.copy()
        libero_home = os.environ.get("LIBERO_HOME", "")
        if libero_home:
            existing = self._worker_env.get("PYTHONPATH", "")
            self._worker_env["PYTHONPATH"] = f"{libero_home}:{existing}" if existing else libero_home
        # Route MuJoCo EGL rendering to specific GPU
        if egl_gpu_id is not None:
            self._worker_env["MUJOCO_EGL_DEVICE_ID"] = str(egl_gpu_id)

        self._sock: Optional[socket.socket] = None
        self._proc: Optional[subprocess.Popen] = None
        self.task_description: str = ""
        self.max_steps: int = 300
        self._closed = False

        self._start_worker()

    def _start_worker(self):
        """Launch worker subprocess with socket pair for IPC."""
        parent_sock, child_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        child_fd = child_sock.fileno()
        os.set_inheritable(child_fd, True)

        self._sock = parent_sock
        self._sock.settimeout(120)  # default timeout

        self._proc = subprocess.Popen(
            [self._python_bin, _FAST_WORKER_SCRIPT, str(child_fd)],
            close_fds=False,  # inherit the socket FD
            env=self._worker_env,
        )
        child_sock.close()  # parent doesn't need child's end

    def _restart_worker(self):
        """Kill and restart the worker subprocess."""
        print(f"  [WARNING] Restarting hung LIBERO worker...", flush=True)
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
        if self._proc:
            try:
                self._proc.kill()
                self._proc.wait(timeout=5)
            except Exception:
                pass
        self._start_worker()

    def reset(self, suite_name: str, task_id: int, initial_state_idx: int = 0, seed: int = 42) -> dict:
        _write_msg_sock(self._sock, {
            "cmd": "reset",
            "task_suite": suite_name,
            "task_id": task_id,
            "initial_state_idx": initial_state_idx,
            "seed": seed,
        })
        try:
            resp = _read_msg_sock(self._sock, timeout=120)
        except (TimeoutError, RuntimeError, ConnectionError, socket.timeout):
            self._restart_worker()
            _write_msg_sock(self._sock, {
                "cmd": "reset", "task_suite": suite_name,
                "task_id": task_id, "initial_state_idx": initial_state_idx, "seed": seed,
            })
            resp = _read_msg_sock(self._sock, timeout=120)
        if resp.get("status") != "ok":
            raise RuntimeError(f"Worker error: {resp.get('message', 'unknown')}")
        self.task_description = resp["task_description"]
        self.max_steps = resp["max_steps"]
        return _parse_obs(resp["obs"])

    def step(self, action_7d: np.ndarray) -> Tuple[dict, float, bool]:
        try:
            _write_msg_sock(self._sock, {"cmd": "step", "action": action_7d.tolist()})
            resp = _read_msg_sock(self._sock, timeout=60)
        except (TimeoutError, RuntimeError, ConnectionError, socket.timeout):
            self._restart_worker()
            raise RuntimeError("Worker timed out during step, restarted")
        if resp.get("status") != "ok":
            raise RuntimeError(f"Worker error: {resp.get('message', 'unknown')}")
        return _parse_obs(resp["obs"]), resp["reward"], resp["done"]

    def step_chunk(self, actions: list) -> Tuple[dict, float, bool, int]:
        """Execute multiple actions in one round-trip. Returns (obs, reward, done, steps_taken)."""
        try:
            _write_msg_sock(self._sock, {"cmd": "step_chunk", "actions": [a.tolist() for a in actions]})
            resp = _read_msg_sock(self._sock, timeout=60)
        except (TimeoutError, RuntimeError, ConnectionError, socket.timeout):
            self._restart_worker()
            raise RuntimeError("Worker timed out during step_chunk, restarted")
        if resp.get("status") != "ok":
            raise RuntimeError(f"Worker error: {resp.get('message', 'unknown')}")
        return _parse_obs(resp["obs"]), resp["reward"], resp["done"], resp["steps_taken"]

    def close(self):
        if not self._closed:
            try:
                _write_msg_sock(self._sock, {"cmd": "close"})
            except Exception:
                pass
            if self._sock:
                try:
                    self._sock.close()
                except Exception:
                    pass
            if self._proc:
                try:
                    self._proc.wait(timeout=5)
                except Exception:
                    try:
                        self._proc.kill()
                    except Exception:
                        pass
            self._closed = True

    def __del__(self):
        self.close()


class PersistentEnvPool:
    """
    Pool of persistent fast LiberoEnv subprocess workers.

    Each worker subprocess stays alive for the entire training run.
    On reset with same task: only reset() + set_init_state() in worker (fast).
    On reset with different task: worker recreates MuJoCo env (slower, but rare).
    """

    def __init__(
        self,
        num_envs: int,
        libero_python: Optional[str] = None,
        egl_gpu_id: Optional[int] = None,
    ):
        self.num_envs = num_envs
        self.libero_python = libero_python
        self.envs: List[_FastLiberoEnv] = []

        gpu_label = f", EGL GPU={egl_gpu_id}" if egl_gpu_id is not None else ""
        print(f"Creating {num_envs} persistent fast LiberoEnv workers{gpu_label}...", flush=True)
        for i in range(num_envs):
            env = _FastLiberoEnv(libero_python=libero_python, egl_gpu_id=egl_gpu_id)
            self.envs.append(env)
            if (i + 1) % 10 == 0 or i == num_envs - 1:
                print(f"  env workers: {i+1}/{num_envs}", flush=True)
        print(f"PersistentEnvPool ready: {num_envs} workers", flush=True)

    def reset_env(
        self,
        env_idx: int,
        suite_name: str,
        task_id: int,
        state_idx: int,
        seed: int,
    ) -> dict:
        """Reset a single env to given task/state. Returns obs dict."""
        return self.envs[env_idx].reset(
            suite_name=suite_name,
            task_id=task_id,
            initial_state_idx=state_idx,
            seed=seed,
        )

    def step_env(self, env_idx: int, action: np.ndarray):
        """Step a single env. Returns (obs, reward, done)."""
        return self.envs[env_idx].step(action)

    @property
    def task_descriptions(self) -> List[str]:
        return [env.task_description for env in self.envs]

    def close(self):
        """Close all subprocess workers."""
        for env in self.envs:
            try:
                env.close()
            except Exception:
                pass
        self.envs.clear()
        print("PersistentEnvPool closed", flush=True)

    def __len__(self):
        return self.num_envs

    def __del__(self):
        self.close()
