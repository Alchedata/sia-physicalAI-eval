"""
Tests for EvalMemory.

All 7 tests are self-contained; they use pytest's tmp_path fixture for
isolation and unittest.mock where injection is required.
"""
from __future__ import annotations

import os
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from sepa_eval.memory.schema import (
    CandidateTask,
    EpisodeTrace,
    RolloutData,
    SceneConfig,
    TaskProvenance,
    TraceIdentity,
    TraceLabels,
)
from sepa_eval.memory.eval_memory import EvalMemory


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_trace(
    trace_id: str | None = None,
    eval_run_id: str | None = None,
    success: bool = False,
    failure_type: str | None = "grasp",
) -> EpisodeTrace:
    """Return a minimal EpisodeTrace suitable for tests."""
    return EpisodeTrace(
        identity=TraceIdentity(
            trace_id=trace_id or str(uuid.uuid4()),
            eval_run_id=eval_run_id or str(uuid.uuid4()),
            benchmark="libero_spatial",
            task_id="task_001",
            task_instruction="Pick the red cup",
            model_id="alphabrain_v1",
            model_version="0.1.0",
        ),
        scene=SceneConfig(
            scene_config={"env": "libero", "seed": 42},
            init_state=b"\x00\x01\x02",
            replay_mode="exact",
        ),
        rollout=RolloutData(
            observations=[{"cam": b"\xff\xfe", "proprioception": [0.1, 0.2]}],
            actions=[[0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 1.0]],
            episode_length=1,
            success=success,
            failure_step=0 if not success else None,
        ),
        labels=TraceLabels(failure_type=failure_type if not success else None),
        provenance=TaskProvenance(promotion_status="seed"),
    )


def _make_memory(tmp_path) -> EvalMemory:
    db_path = str(tmp_path / "eval.db")
    memory_dir = str(tmp_path / "traces")
    return EvalMemory(db_path=db_path, memory_dir=memory_dir)


# ---------------------------------------------------------------------------
# Test 1: happy path — .tmp written, renamed, DB row committed
# ---------------------------------------------------------------------------

def test_record_trace_happy_path(tmp_path):
    mem = _make_memory(tmp_path)
    trace = _make_trace()

    mem.record_trace(trace)

    # DB row must exist
    conn = sqlite3.connect(str(tmp_path / "eval.db"))
    row = conn.execute(
        "SELECT * FROM traces WHERE trace_id = ?", (trace.identity.trace_id,)
    ).fetchone()
    conn.close()
    assert row is not None, "DB row not committed"

    # Final file must exist
    trace_path = row[15]  # trace_path column index
    assert os.path.isfile(trace_path), "Final msgpack file missing"

    # .tmp must NOT exist
    assert not os.path.isfile(trace_path + ".tmp"), "Orphan .tmp file left behind"

    mem.close()


# ---------------------------------------------------------------------------
# Test 2: file write fails → no DB row, .tmp cleaned up
# ---------------------------------------------------------------------------

def test_record_trace_file_write_fails(tmp_path):
    mem = _make_memory(tmp_path)
    trace = _make_trace()

    # Patch built-in open so the .tmp write raises an IOError
    real_open = open

    def _bad_open(path, mode="r", **kwargs):
        if isinstance(path, str) and path.endswith(".tmp"):
            raise IOError("Simulated disk full")
        return real_open(path, mode, **kwargs)

    with patch("builtins.open", side_effect=_bad_open):
        with pytest.raises(IOError, match="Simulated disk full"):
            mem.record_trace(trace)

    # No DB row
    conn = sqlite3.connect(str(tmp_path / "eval.db"))
    row = conn.execute(
        "SELECT 1 FROM traces WHERE trace_id = ?", (trace.identity.trace_id,)
    ).fetchone()
    conn.close()
    assert row is None, "DB row should not exist after file write failure"

    # No .tmp orphan
    orphans = list((tmp_path / "traces").rglob("*.tmp")) if (tmp_path / "traces").exists() else []
    assert len(orphans) == 0, f"Orphan .tmp files found: {orphans}"

    mem.close()


# ---------------------------------------------------------------------------
# Test 3: DB commit fails → file deleted (compensating write), no orphan
# ---------------------------------------------------------------------------

def test_record_trace_db_commit_fails(tmp_path):
    mem = _make_memory(tmp_path)
    trace = _make_trace()

    # Python 3.12 sqlite3.Connection attributes are read-only C extensions; mock
    # the Python-level _insert_trace_row wrapper instead.
    with patch.object(
        mem,
        "_insert_trace_row",
        side_effect=sqlite3.OperationalError("Simulated DB failure"),
    ):
        with pytest.raises(sqlite3.OperationalError, match="Simulated DB failure"):
            mem.record_trace(trace)

    # The final msgpack file must have been removed (compensating write)
    run_dir = tmp_path / "traces" / trace.identity.eval_run_id
    leftover_files = list(run_dir.glob("*.msgpack")) if run_dir.exists() else []
    assert len(leftover_files) == 0, f"Compensating delete failed; files left: {leftover_files}"

    # No .tmp either
    leftover_tmp = list(run_dir.glob("*.tmp")) if run_dir.exists() else []
    assert len(leftover_tmp) == 0

    mem.close()


# ---------------------------------------------------------------------------
# Test 4: empty DB returns [] not an exception
# ---------------------------------------------------------------------------

def test_get_failures_empty_db(tmp_path):
    mem = _make_memory(tmp_path)
    result = mem.get_failures()
    assert result == []
    mem.close()


# ---------------------------------------------------------------------------
# Test 5: update_critic_score with unknown trace_id raises ValueError
# ---------------------------------------------------------------------------

def test_update_critic_score_unknown_id(tmp_path):
    mem = _make_memory(tmp_path)
    with pytest.raises(ValueError, match="trace_id not found"):
        mem.update_critic_score(
            trace_id="nonexistent-id",
            critic_name="task_completion",
            score=0.8,
        )
    mem.close()


# ---------------------------------------------------------------------------
# Test 6: 4 concurrent writes via ThreadPoolExecutor, all committed, no loss
# ---------------------------------------------------------------------------

def test_concurrent_writes_wal_mode(tmp_path):
    db_path = str(tmp_path / "eval.db")
    memory_dir = str(tmp_path / "traces")

    # Pre-create the DB schema in the main thread to avoid concurrent DDL contention.
    EvalMemory(db_path=db_path, memory_dir=memory_dir).close()

    # Each thread creates its own EvalMemory instance sharing the same DB
    N = 4
    traces = [_make_trace(eval_run_id="run_concurrent") for _ in range(N)]
    trace_ids = {t.identity.trace_id for t in traces}

    errors: list[Exception] = []

    def _write(trace: EpisodeTrace) -> None:
        mem = EvalMemory(db_path=db_path, memory_dir=memory_dir)
        try:
            mem.record_trace(trace)
        except Exception as exc:
            errors.append(exc)
        finally:
            mem.close()

    with ThreadPoolExecutor(max_workers=N) as pool:
        futures = [pool.submit(_write, t) for t in traces]
        for f in as_completed(futures):
            f.result()  # re-raise any exception

    assert errors == [], f"Concurrent write errors: {errors}"

    # Verify all rows committed
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        f"SELECT trace_id FROM traces WHERE eval_run_id = 'run_concurrent'"
    ).fetchall()
    conn.close()
    committed_ids = {r[0] for r in rows}
    assert committed_ids == trace_ids, (
        f"Missing rows: {trace_ids - committed_ids}"
    )


# ---------------------------------------------------------------------------
# Test 7: fsck detects and removes orphan .tmp files
# ---------------------------------------------------------------------------

def test_eval_memory_fsck(tmp_path):
    mem = _make_memory(tmp_path)

    # Plant two orphan .tmp files in different sub-directories
    run_dir_a = tmp_path / "traces" / "run_a"
    run_dir_b = tmp_path / "traces" / "run_b"
    run_dir_a.mkdir(parents=True, exist_ok=True)
    run_dir_b.mkdir(parents=True, exist_ok=True)

    orphan_a = run_dir_a / "some_trace.msgpack.tmp"
    orphan_b = run_dir_b / "another_trace.msgpack.tmp"
    orphan_a.write_bytes(b"partial data")
    orphan_b.write_bytes(b"partial data")

    removed = mem.fsck()

    assert removed == 2, f"Expected 2 orphans removed, got {removed}"
    assert not orphan_a.exists(), ".tmp file A not removed"
    assert not orphan_b.exists(), ".tmp file B not removed"

    mem.close()
