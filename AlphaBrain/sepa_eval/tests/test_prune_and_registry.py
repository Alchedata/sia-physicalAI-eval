"""
Tests for EvalMemory.prune(), ModelsRegistry, and the new CLI commands
(diff, sync-models, prune, eval --ci-mode).
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sepa_eval.memory.eval_memory import EvalMemory, _ISO_FMT
from sepa_eval.memory.schema import (
    EpisodeTrace,
    RolloutData,
    SceneConfig,
    TaskProvenance,
    TraceIdentity,
    TraceLabels,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_memory(tmp_path: Path) -> EvalMemory:
    return EvalMemory(
        db_path=str(tmp_path / "eval.db"),
        memory_dir=str(tmp_path / "traces"),
    )


def _make_trace(success: bool = False) -> EpisodeTrace:
    return EpisodeTrace(
        identity=TraceIdentity(
            trace_id=str(uuid.uuid4()),
            eval_run_id="run-prune-test",
            benchmark="libero_spatial",
            task_id="task_001",
            task_instruction="Pick the red cup",
            model_id="model_x",
            model_version="0.1.0",
        ),
        scene=SceneConfig(
            scene_config={"env": "libero"},
            init_state=b"\x00",
            replay_mode="exact",
        ),
        rollout=RolloutData(
            observations=[],
            actions=[],
            episode_length=10,
            success=success,
            failure_step=None if success else 5,
        ),
        labels=TraceLabels(failure_type=None if success else "grasp"),
        provenance=TaskProvenance(promotion_status="seed"),
    )


# ---------------------------------------------------------------------------
# Test: EvalMemory.prune() removes files older than retention_days
# ---------------------------------------------------------------------------

def test_prune_removes_old_files(tmp_path: Path):
    mem = _make_memory(tmp_path)
    trace = _make_trace()
    mem.record_trace(trace)

    # Back-date the DB row to 100 days ago
    old_date = (
        datetime.now(tz=timezone.utc).replace(tzinfo=None) - timedelta(days=100)
    ).strftime(_ISO_FMT)
    mem._conn.execute(
        "UPDATE traces SET created_at = ? WHERE trace_id = ?",
        (old_date, trace.identity.trace_id),
    )
    mem._conn.commit()

    # Confirm file exists before prune
    cur = mem._conn.execute(
        "SELECT trace_path FROM traces WHERE trace_id = ?",
        (trace.identity.trace_id,),
    )
    trace_path = cur.fetchone()["trace_path"]
    assert os.path.isfile(trace_path), "File should exist before prune"

    deleted = mem.prune(retention_days=90)
    assert deleted == 1, f"Expected 1 file deleted, got {deleted}"
    assert not os.path.isfile(trace_path), "File should be removed after prune"

    # DB row must still exist
    row = mem._conn.execute(
        "SELECT 1 FROM traces WHERE trace_id = ?",
        (trace.identity.trace_id,),
    ).fetchone()
    assert row is not None, "DB row must be kept after prune"

    mem.close()


def test_prune_keeps_recent_files(tmp_path: Path):
    """Files within retention window must not be deleted."""
    mem = _make_memory(tmp_path)
    trace = _make_trace()
    mem.record_trace(trace)

    deleted = mem.prune(retention_days=90)
    assert deleted == 0, "Recent file should not be pruned"
    mem.close()


# ---------------------------------------------------------------------------
# Test: ModelsRegistry loads and syncs to DB
# ---------------------------------------------------------------------------

def test_models_registry_sync(tmp_path: Path):
    yaml_content = """
models:
  - model_id: test_model_a
    framework: alphabrain
    checkpoint: checkpoints/test_a.pth
    benchmarks:
      - libero_spatial
  - model_id: test_model_b
    framework: qwen_oft
    checkpoint: checkpoints/test_b.pth
    benchmarks:
      - libero_object
"""
    yaml_path = tmp_path / "models.yaml"
    yaml_path.write_text(yaml_content)

    mem = _make_memory(tmp_path)
    from sepa_eval.registry.models_registry import ModelsRegistry

    registry = ModelsRegistry(yaml_path=str(yaml_path), memory=mem)
    count = registry.sync_to_db()

    assert count == 2, f"Expected 2 models synced, got {count}"

    rows = mem._conn.execute(
        "SELECT model_id FROM models ORDER BY model_id"
    ).fetchall()
    model_ids = [r[0] for r in rows]
    assert "test_model_a" in model_ids
    assert "test_model_b" in model_ids

    mem.close()


def test_models_registry_missing_yaml(tmp_path: Path):
    """A missing YAML file should return [] without raising."""
    from sepa_eval.registry.models_registry import ModelsRegistry
    registry = ModelsRegistry(yaml_path=str(tmp_path / "nonexistent.yaml"))
    result = registry.load()
    assert result == [], "Expected [] for missing yaml file"


# ---------------------------------------------------------------------------
# Test: CLI diff command
# ---------------------------------------------------------------------------

def test_cli_diff_no_data(tmp_path: Path):
    """diff with no model_task_results rows prints a friendly message."""
    from sepa_eval.__main__ import main

    mem_dir = str(tmp_path / "mem")
    os.makedirs(mem_dir, exist_ok=True)
    # Pre-create DB
    EvalMemory(
        db_path=os.path.join(mem_dir, "eval.db"),
        memory_dir=os.path.join(mem_dir, "traces"),
    ).close()

    rc = main(["--memory-dir", mem_dir, "diff", "model_a", "model_b"])
    assert rc == 0


# ---------------------------------------------------------------------------
# Test: CLI prune command
# ---------------------------------------------------------------------------

def test_cli_prune_command(tmp_path: Path):
    """prune command runs without error and returns 0."""
    from sepa_eval.__main__ import main

    mem_dir = str(tmp_path / "mem")
    rc = main(["--memory-dir", mem_dir, "prune", "--retention-days", "90"])
    assert rc == 0


# ---------------------------------------------------------------------------
# Test: CLI sync-models command
# ---------------------------------------------------------------------------

def test_cli_sync_models_command(tmp_path: Path):
    """sync-models with a valid YAML file returns 0 and registers models."""
    yaml_content = "models:\n  - model_id: cli_model\n    framework: act\n    checkpoint: ckpt.pth\n    benchmarks: [libero_spatial]\n"
    yaml_path = tmp_path / "models.yaml"
    yaml_path.write_text(yaml_content)

    mem_dir = str(tmp_path / "mem")
    from sepa_eval.__main__ import main

    rc = main(["--memory-dir", mem_dir, "sync-models", "--yaml", str(yaml_path)])
    assert rc == 0

    mem = EvalMemory(
        db_path=os.path.join(mem_dir, "eval.db"),
        memory_dir=os.path.join(mem_dir, "traces"),
    )
    row = mem._conn.execute(
        "SELECT model_id FROM models WHERE model_id = 'cli_model'"
    ).fetchone()
    assert row is not None, "cli_model should be in DB after sync-models"
    mem.close()
