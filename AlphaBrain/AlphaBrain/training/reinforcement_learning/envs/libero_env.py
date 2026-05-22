"""
LIBERO environment proxy — runs in the VLA Python environment.

Spawns `libero_env_worker.py` as a subprocess using `LIBERO_PYTHON`
(the separate conda env that has `libero` installed), then communicates
via stdin/stdout with length-prefixed msgpack messages.

Usage matches the original direct API:
    env = LiberoEnv(suite_name, task_id, seed)
    obs = env.reset(initial_state_idx=0)
    obs, reward, done = env.step(action_7d)
    env.close()
"""

import io
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import msgpack
import numpy as np
from PIL import Image


# ── Wire protocol helpers ───────────────────────────────────────────────────────

def _write_msg(proc: subprocess.Popen, obj: dict):
    data = msgpack.packb(obj, use_bin_type=True)
    proc.stdin.write(struct.pack("<I", len(data)))
    proc.stdin.write(data)
    proc.stdin.flush()


def _read_msg(proc: subprocess.Popen) -> dict:
    raw_len = proc.stdout.read(4)
    if not raw_len:
        stderr = proc.stderr.read().decode(errors="replace")
        raise RuntimeError(f"LIBERO worker exited unexpectedly.\nWorker stderr:\n{stderr}")
    length = struct.unpack("<I", raw_len)[0]
    data = proc.stdout.read(length)
    return msgpack.unpackb(data, raw=False)


def _bytes_to_pil(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b))


# ── Worker path ─────────────────────────────────────────────────────────────────

_WORKER_SCRIPT = str(Path(__file__).parent / "libero_env_worker.py")


# ── LiberoEnv ───────────────────────────────────────────────────────────────────

class LiberoEnv:
    """
    Proxy to a LIBERO environment running in a separate Python process.

    The worker process is started once per LiberoEnv instance and reused
    across reset() calls (different tasks can be loaded with reset).
    """

    def __init__(
        self,
        libero_python: Optional[str] = None,
    ):
        """
        Args:
            libero_python: Path to the LIBERO conda env Python binary.
                           Defaults to LIBERO_PYTHON env var, then 'python'.
        """
        python_bin = (
            libero_python
            or os.environ.get("LIBERO_PYTHON", "python")
        )

        # Inherit the current env so LIBERO can find its own packages.
        # Inject LIBERO_HOME into PYTHONPATH so the editable install is not required.
        worker_env = os.environ.copy()
        libero_home = os.environ.get("LIBERO_HOME", "")
        if libero_home:
            existing = worker_env.get("PYTHONPATH", "")
            worker_env["PYTHONPATH"] = f"{libero_home}:{existing}" if existing else libero_home

        self._proc = subprocess.Popen(
            [python_bin, _WORKER_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=worker_env,
        )

        self.task_description: str = ""
        self.max_steps: int = 300
        self._closed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(
        self,
        suite_name: str,
        task_id: int,
        initial_state_idx: int = 0,
        seed: int = 42,
    ) -> dict:
        """
        Reset the environment to a specific task and initial state.

        Returns obs dict:
          - "primary_image"  : PIL.Image
          - "wrist_image"    : PIL.Image
          - "state"          : np.ndarray (8,)
        """
        _write_msg(self._proc, {
            "cmd":               "reset",
            "task_suite":        suite_name,
            "task_id":           task_id,
            "initial_state_idx": initial_state_idx,
            "seed":              seed,
        })
        resp = _read_msg(self._proc)
        _check_resp(resp)

        self.task_description = resp["task_description"]
        self.max_steps = resp["max_steps"]
        return _parse_obs(resp["obs"])

    def step(self, action_7d: np.ndarray) -> Tuple[dict, float, bool]:
        """
        Execute one env step.

        Returns:
            obs_dict  : parsed observation
            reward    : 0.0 / 1.0
            done      : episode termination flag
        """
        _write_msg(self._proc, {"cmd": "step", "action": action_7d.tolist()})
        resp = _read_msg(self._proc)
        _check_resp(resp)
        return _parse_obs(resp["obs"]), float(resp["reward"]), bool(resp["done"])

    def close(self):
        if not self._closed:
            try:
                _write_msg(self._proc, {"cmd": "close"})
            except Exception:
                pass
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            self._closed = True

    def __del__(self):
        self.close()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _check_resp(resp: dict):
    if resp.get("status") != "ok":
        raise RuntimeError(f"LIBERO worker error: {resp.get('message', resp)}")


def _parse_obs(obs_raw: dict) -> dict:
    return {
        "primary_image": _bytes_to_pil(obs_raw["primary"]),
        "wrist_image":   _bytes_to_pil(obs_raw["wrist"]),
        "state":         np.array(obs_raw["state"], dtype=np.float32),
    }


# ------------------------------------------------------------------
# Suite info helper (no env needed — just metadata)
# ------------------------------------------------------------------

MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object":  280,
    "libero_goal":    320,
    "libero_10":      520,
    "libero_90":      400,
}


def get_suite_info(suite_name: str, libero_python: Optional[str] = None) -> dict:
    """
    Query task count and task names from the LIBERO worker without
    opening an environment.

    Returns: {"n_tasks": int, "task_names": [str, ...]}
    """
    python_bin = libero_python or os.environ.get("LIBERO_PYTHON", "python")
    script = (
        "import sys; _real_stdout = sys.stdout; sys.stdout = sys.stderr; "
        "from libero.libero import benchmark; "
        f"s = benchmark.get_benchmark_dict()['{suite_name}'](); "
        "sys.stdout = _real_stdout; "
        "import json; "
        "json.dump({'n_tasks': s.n_tasks, "
        "'task_names': [s.get_task(i).language for i in range(s.n_tasks)]}, sys.stdout)"
    )
    run_env = os.environ.copy()
    libero_home = os.environ.get("LIBERO_HOME", "")
    if libero_home:
        existing = run_env.get("PYTHONPATH", "")
        run_env["PYTHONPATH"] = f"{libero_home}:{existing}" if existing else libero_home
    result = subprocess.run(
        [python_bin, "-c", script],
        capture_output=True, text=True, timeout=30,
        env=run_env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"get_suite_info failed:\n{result.stderr}")
    import json
    return json.loads(result.stdout)
