"""
Tests for P2 storage hardening: schema migrations (PRAGMA user_version),
portable trace_relpath, secondary indexes + critic upsert, extended fsck,
threaded access on a shared instance, and the read-only query API.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from sepa_eval.memory import migrations
from sepa_eval.memory.eval_memory import EvalMemory, FsckResult
from sepa_eval.memory.schema import (
    EpisodeTrace,
    RolloutData,
    SceneConfig,
    TaskProvenance,
    TraceIdentity,
    TraceLabels,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LEGACY_V1_DDL = """
CREATE TABLE traces (
    trace_id          TEXT PRIMARY KEY,
    eval_run_id       TEXT,
    benchmark         TEXT,
    task_id           TEXT,
    task_instruction  TEXT,
    model_id          TEXT,
    model_version     TEXT,
    success           INTEGER,
    failure_step      INTEGER,
    episode_length    INTEGER,
    failure_type      TEXT,
    promotion_status  TEXT,
    parent_task_id    TEXT,
    mutation_type     TEXT,
    created_at        TIMESTAMP,
    trace_path        TEXT
);
CREATE TABLE tasks (
    task_id              TEXT PRIMARY KEY,
    benchmark            TEXT,
    instruction          TEXT,
    scene_config         TEXT,
    mutation_lineage     TEXT,
    promotion_status     TEXT,
    discriminative_power REAL,
    saturation_flag      INTEGER,
    promotion_evidence   TEXT,
    created_at           TIMESTAMP
);
CREATE TABLE critic_scores (
    trace_id    TEXT,
    critic_name TEXT,
    score       REAL,
    explanation TEXT,
    confidence  REAL,
    scored_at   TIMESTAMP
);
CREATE TABLE model_task_results (
    model_id           TEXT,
    task_id            TEXT,
    benchmark          TEXT,
    n_trials           INTEGER,
    success_rate       REAL,
    clean_success_rate REAL,
    avg_episode_length REAL,
    last_eval_at       TIMESTAMP,
    PRIMARY KEY (model_id, task_id)
);
CREATE TABLE failure_clusters (
    cluster_id               TEXT PRIMARY KEY,
    eval_run_id              TEXT,
    failure_type             TEXT,
    centroid                 BLOB,
    representative_trace_id  TEXT,
    member_count             INTEGER,
    llm_summary              TEXT,
    summarized_at            TIMESTAMP
);
CREATE TABLE models (
    model_id    TEXT PRIMARY KEY,
    framework   TEXT,
    checkpoint  TEXT,
    benchmarks  TEXT,
    created_at  TIMESTAMP
);
"""


def _make_trace(trace_id=None, run_id="run-hard", success=False):
    return EpisodeTrace(
        identity=TraceIdentity(
            trace_id=trace_id or str(uuid.uuid4()),
            eval_run_id=run_id,
            benchmark="libero_spatial",
            task_id="task_001",
            task_instruction="Pick the red cup",
            model_id="alphabrain_v1",
            model_version="0.1.0",
        ),
        scene=SceneConfig(scene_config={"seed": 1}, init_state=b"\x00"),
        rollout=RolloutData(
            observations=[{"proprioception": [0.1]}],
            actions=[[0.0] * 7],
            episode_length=1,
            success=success,
            failure_step=None if success else 0,
        ),
        labels=TraceLabels(failure_type=None if success else "grasp"),
        provenance=TaskProvenance(promotion_status="seed"),
    )


def _make_memory(tmp_path) -> EvalMemory:
    return EvalMemory(db_path=str(tmp_path / "eval.db"), memory_dir=str(tmp_path / "traces"))


def _build_legacy_v1_db(tmp_path):
    """Create a v1-schema DB + msgpack store with absolute trace_paths."""
    db_path = str(tmp_path / "eval.db")
    memory_dir = str(tmp_path / "traces")
    run_dir = os.path.join(memory_dir, "run-legacy")
    os.makedirs(run_dir, exist_ok=True)
    trace_file = os.path.join(run_dir, "trace-legacy.msgpack")
    with open(trace_file, "wb") as fh:
        fh.write(b"\x81\xa1a\x01")  # arbitrary msgpack-ish bytes

    conn = sqlite3.connect(db_path)
    conn.executescript(_LEGACY_V1_DDL)
    conn.execute(
        "INSERT INTO traces (trace_id, eval_run_id, success, created_at, trace_path)"
        " VALUES ('trace-legacy', 'run-legacy', 0, '2024-01-01T00:00:00.000000', ?)",
        (trace_file,),
    )
    # Duplicate critic scores for the same (trace_id, critic_name)
    conn.execute(
        "INSERT INTO critic_scores VALUES ('trace-legacy', 'semantic', 0.2, 'old', 1.0, '2024-01-01T00:00:00.000000')"
    )
    conn.execute(
        "INSERT INTO critic_scores VALUES ('trace-legacy', 'semantic', 0.9, 'new', 1.0, '2024-01-02T00:00:00.000000')"
    )
    conn.commit()
    conn.close()
    return db_path, memory_dir, trace_file


# ---------------------------------------------------------------------------
# P2.3 — migration machinery
# ---------------------------------------------------------------------------


class TestMigrations:
    def test_fresh_db_gets_latest_version_and_indexes(self, tmp_path):
        mem = _make_memory(tmp_path)
        assert mem.schema_version == migrations.SCHEMA_VERSION
        rows = mem.query("PRAGMA user_version")
        assert next(iter(rows[0].values())) == migrations.SCHEMA_VERSION

        index_names = {r["name"] for r in mem.query("SELECT name FROM sqlite_master WHERE type='index'")}
        assert "idx_traces_task_id" in index_names
        assert "idx_traces_model_success" in index_names
        assert "idx_mtr_benchmark" in index_names
        assert "idx_critic_scores_unique" in index_names
        mem.close()

    def test_legacy_v1_db_is_migrated(self, tmp_path):
        db_path, memory_dir, trace_file = _build_legacy_v1_db(tmp_path)

        mem = EvalMemory(db_path=db_path, memory_dir=memory_dir)
        assert mem.schema_version == migrations.SCHEMA_VERSION

        row = mem.query("SELECT trace_path, trace_relpath FROM traces WHERE trace_id='trace-legacy'")[0]
        # Legacy absolute path preserved; portable relpath backfilled
        assert row["trace_path"] == trace_file
        assert row["trace_relpath"] == "run-legacy/trace-legacy.msgpack"

        # Duplicate critic rows deduped, latest kept
        scores = mem.query("SELECT * FROM critic_scores WHERE trace_id='trace-legacy'")
        assert len(scores) == 1
        assert scores[0]["score"] == 0.9
        mem.close()

    def test_migration_is_idempotent(self, tmp_path):
        db_path, memory_dir, _ = _build_legacy_v1_db(tmp_path)
        EvalMemory(db_path=db_path, memory_dir=memory_dir).close()
        mem = EvalMemory(db_path=db_path, memory_dir=memory_dir)  # reopen: no-op migrate
        assert mem.schema_version == migrations.SCHEMA_VERSION
        mem.close()

    def test_newer_schema_version_refused(self, tmp_path):
        db_path = str(tmp_path / "eval.db")
        conn = sqlite3.connect(db_path)
        conn.executescript(_LEGACY_V1_DDL)
        conn.execute("PRAGMA user_version = 99")
        conn.commit()
        conn.close()
        with pytest.raises(RuntimeError, match="newer than supported"):
            EvalMemory(db_path=db_path, memory_dir=str(tmp_path / "traces"))


# ---------------------------------------------------------------------------
# P2.1 — portable relative trace paths
# ---------------------------------------------------------------------------


class TestRelativeTracePaths:
    def test_record_trace_stores_relpath(self, tmp_path):
        mem = _make_memory(tmp_path)
        trace = _make_trace()
        mem.record_trace(trace)
        row = mem.query("SELECT trace_path, trace_relpath FROM traces WHERE trace_id=?", (trace.identity.trace_id,))[0]
        assert row["trace_relpath"] == f"{trace.identity.eval_run_id}/{trace.identity.trace_id}.msgpack"
        assert os.path.isabs(row["trace_path"])  # legacy column kept absolute
        mem.close()

    def test_load_trace_file_accepts_relative_path(self, tmp_path):
        mem = _make_memory(tmp_path)
        trace = _make_trace()
        mem.record_trace(trace)
        relpath = f"{trace.identity.eval_run_id}/{trace.identity.trace_id}.msgpack"
        loaded = mem.load_trace_file(relpath)
        assert loaded.identity.trace_id == trace.identity.trace_id
        mem.close()

    def test_store_survives_relocation_via_relpath(self, tmp_path):
        """Moving memory_dir keeps traces loadable through trace_relpath."""
        src = tmp_path / "site_a"
        src.mkdir()
        mem = EvalMemory(db_path=str(src / "eval.db"), memory_dir=str(src / "traces"))
        trace = _make_trace()
        mem.record_trace(trace)
        mem.close()

        dst = tmp_path / "site_b"
        shutil.move(str(src), str(dst))

        mem2 = EvalMemory(db_path=str(dst / "eval.db"), memory_dir=str(dst / "traces"))
        row = mem2.query("SELECT * FROM traces WHERE trace_id=?", (trace.identity.trace_id,))[0]
        resolved = mem2.resolve_trace_path(row)
        assert resolved is not None and os.path.isfile(resolved)
        assert resolved.startswith(str(dst))
        loaded = mem2.load_trace_file(row["trace_relpath"])
        assert loaded.identity.trace_id == trace.identity.trace_id
        # fsck must not flag the relocated trace as broken
        report = mem2.fsck()
        assert report.broken_links == []
        mem2.close()

    def test_prune_resolves_relpath(self, tmp_path):
        mem = _make_memory(tmp_path)
        trace = _make_trace()
        mem.record_trace(trace)
        # Age the row and blank the legacy absolute path: prune must use relpath
        with mem._lock:
            mem._conn.execute(
                "UPDATE traces SET created_at='2000-01-01T00:00:00.000000', trace_path=NULL WHERE trace_id=?",
                (trace.identity.trace_id,),
            )
            mem._conn.commit()
        assert mem.prune(retention_days=90) == 1
        mem.close()


# ---------------------------------------------------------------------------
# P2.2 — critic score upsert + indexes on query paths
# ---------------------------------------------------------------------------


class TestCriticUpsert:
    def test_update_critic_score_is_true_upsert(self, tmp_path):
        mem = _make_memory(tmp_path)
        trace = _make_trace()
        mem.record_trace(trace)
        tid = trace.identity.trace_id
        mem.update_critic_score(tid, "semantic", 0.3, "first", 0.5)
        mem.update_critic_score(tid, "semantic", 0.8, "second", 0.9)
        rows = mem.query("SELECT * FROM critic_scores WHERE trace_id=? AND critic_name='semantic'", (tid,))
        assert len(rows) == 1
        assert rows[0]["score"] == 0.8
        assert rows[0]["explanation"] == "second"
        # A different critic gets its own row
        mem.update_critic_score(tid, "safety", 1.0)
        assert len(mem.query("SELECT * FROM critic_scores WHERE trace_id=?", (tid,))) == 2
        mem.close()

    def test_failure_query_uses_model_success_index(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.record_trace(_make_trace())
        plan = mem.query("EXPLAIN QUERY PLAN SELECT * FROM traces WHERE model_id=? AND success=0", ("alphabrain_v1",))
        plan_text = " ".join(str(v) for row in plan for v in row.values())
        assert "idx_traces_model_success" in plan_text
        mem.close()


# ---------------------------------------------------------------------------
# P2.4 — fsck deep scan, threaded shared instance, read-only query API
# ---------------------------------------------------------------------------


class TestFsck:
    def test_fsck_reports_broken_links_and_orphans(self, tmp_path):
        mem = _make_memory(tmp_path)
        kept = _make_trace()
        broken = _make_trace()
        mem.record_trace(kept)
        mem.record_trace(broken)

        # Break one link: remove its file
        broken_row = mem.query("SELECT * FROM traces WHERE trace_id=?", (broken.identity.trace_id,))[0]
        os.remove(mem.resolve_trace_path(broken_row))

        # Plant an orphan .msgpack and an orphan .tmp
        orphan_file = tmp_path / "traces" / "run-x" / "orphan.msgpack"
        orphan_file.parent.mkdir(parents=True, exist_ok=True)
        orphan_file.write_bytes(b"junk")
        tmp_file = tmp_path / "traces" / "run-x" / "partial.msgpack.tmp"
        tmp_file.write_bytes(b"junk")

        report = mem.fsck()
        assert isinstance(report, FsckResult)
        assert int(report) == 1  # legacy int contract: .tmp files removed
        assert not tmp_file.exists()
        assert report.broken_links == [broken.identity.trace_id]
        assert report.orphan_files == [str(orphan_file)]
        assert report.orphans_removed == 0
        assert orphan_file.exists(), "default fsck must only report orphans"

        # repair=True actually deletes orphan .msgpack files
        report2 = mem.fsck(repair=True)
        assert report2.orphans_removed == 1
        assert not orphan_file.exists()
        mem.close()


class TestConcurrencyAndQueryAPI:
    def test_shared_instance_threaded_writes(self, tmp_path):
        """A single EvalMemory shared across threads is serialized by its lock."""
        mem = _make_memory(tmp_path)
        traces = [_make_trace(run_id="run-shared") for _ in range(8)]

        def _work(trace):
            mem.record_trace(trace)
            mem.update_critic_score(trace.identity.trace_id, "semantic", 0.5)
            return mem.get_failures(eval_run_id="run-shared")

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(_work, traces))

        rows = mem.query("SELECT COUNT(*) AS n FROM traces WHERE eval_run_id='run-shared'")
        assert rows[0]["n"] == 8
        mem.close()

    def test_query_rejects_writes(self, tmp_path):
        mem = _make_memory(tmp_path)
        with pytest.raises(ValueError, match="read-only"):
            mem.query("DELETE FROM traces")
        with pytest.raises(ValueError, match="read-only"):
            mem.query("INSERT INTO models VALUES ('x','y','z','[]','now')")
        assert mem.query("SELECT 1 AS one") == [{"one": 1}]
        assert mem.query_scalar("SELECT COUNT(*) FROM traces") == 0
        mem.close()

    def test_report_generator_uses_query_api_not_conn(self, tmp_path):
        """ReportGenerator works against a memory exposing only query()."""
        from sepa_eval.reporting.report_generator import ReportGenerator

        mem = _make_memory(tmp_path)
        mem.record_trace(_make_trace())

        class QueryOnly:
            def __init__(self, inner):
                self._inner = inner

            def query(self, sql, params=()):
                return self._inner.query(sql, params)

            def __getattr__(self, name):  # deny _conn access
                raise AttributeError(name)

        gen = ReportGenerator(memory=QueryOnly(mem))
        out = tmp_path / "report.md"
        content = gen.generate(str(out))
        assert out.is_file()
        assert "Failure Taxonomy" in content or len(content) > 0
        mem.close()
