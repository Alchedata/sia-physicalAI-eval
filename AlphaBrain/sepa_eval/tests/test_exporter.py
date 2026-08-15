"""Tests for HardCaseExporter."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sepa_eval.exporter.hard_case_exporter import HardCaseExporter
from sepa_eval.memory.schema import (
    EpisodeTrace,
    TraceIdentity,
    SceneConfig,
    RolloutData,
    TraceLabels,
    TaskProvenance,
)


def _make_trace(trace_id: str, success: bool) -> EpisodeTrace:
    return EpisodeTrace(
        identity=TraceIdentity(
            trace_id=trace_id,
            eval_run_id="run-001",
            benchmark="libero_spatial",
            task_id="task-pick",
            task_instruction="Pick up the cube",
            model_id="QwenOFT-test",
            model_version="v1.0",
        ),
        scene=SceneConfig(scene_config={}, init_state=b""),
        rollout=RolloutData(
            observations=[{"gripper_state": [0.0]}] * 5,
            actions=[[0.0] * 7] * 5,
            episode_length=5,
            success=success,
            failure_step=None if success else 4,
        ),
        labels=TraceLabels(),
        provenance=TaskProvenance(),
    )


def _make_mock_memory(n_failures: int) -> MagicMock:
    """Build a mock EvalMemory returning n_failures failed trace rows."""
    memory = MagicMock()
    trace_rows = [
        {
            "trace_id": f"trace-{i:03d}",
            "eval_run_id": "run-001",
            "benchmark": "libero_spatial",
            "task_id": "task-pick",
            "task_instruction": "Pick up the cube",
            "model_id": "QwenOFT-test",
            "success": 0,
            "episode_length": 5,
            "trace_path": f"/fake/traces/run-001/trace-{i:03d}.msgpack",
        }
        for i in range(n_failures)
    ]
    memory.get_failures.return_value = trace_rows

    def _load_trace(trace_path: str) -> EpisodeTrace:
        idx = int(trace_path.split("trace-")[-1].split(".")[0])
        return _make_trace(f"trace-{idx:03d}", success=False)

    memory.load_trace_file.side_effect = _load_trace
    return memory


def test_hard_case_exporter_lerobot_format(tmp_path: Path) -> None:
    """export() with 3 failed traces produces exported files."""
    memory = _make_mock_memory(3)
    exporter = HardCaseExporter(memory=memory, output_format="lerobot")
    result = exporter.export(output_dir=str(tmp_path / "out"), max_episodes=100)

    assert result["episodes_exported"] == 3
    assert result["format"] == "lerobot"

    # At least the meta/info.json should exist
    meta_path = tmp_path / "out" / "meta" / "info.json"
    assert meta_path.exists(), f"Expected {meta_path} to exist"
    with meta_path.open() as f:
        info = json.load(f)
    assert info["episodes"] == 3


def test_exporter_zero_failures(tmp_path: Path) -> None:
    """export() with 0 failed traces returns episodes_exported=0, no crash."""
    memory = _make_mock_memory(0)
    exporter = HardCaseExporter(memory=memory, output_format="jsonl")
    result = exporter.export(output_dir=str(tmp_path / "out"))

    assert result["episodes_exported"] == 0
    assert result["format"] == "jsonl"
