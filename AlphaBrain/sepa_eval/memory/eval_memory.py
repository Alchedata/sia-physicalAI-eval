"""
EvalMemory: SQLite (WAL) + local msgpack file store for SEPA-Eval episode traces.
"""

from __future__ import annotations

import glob
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any

try:
    import msgpack
except ImportError as exc:  # pragma: no cover
    raise ImportError("msgpack is required for EvalMemory.  Install it with: pip install msgpack") from exc

from . import migrations
from .schema import (
    CandidateTask,
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

_ISO_FMT = "%Y-%m-%dT%H:%M:%S.%f"


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(tzinfo=None).strftime(_ISO_FMT)


def _dt_to_iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return dt.strftime(_ISO_FMT)


# ---------- msgpack serialisation of EpisodeTrace ---------------------------


def _pack_trace(trace: EpisodeTrace) -> bytes:
    """Serialise an EpisodeTrace to msgpack bytes.

    numpy arrays are converted to plain Python lists so msgpack can handle
    them without a numpy dependency at deserialization time.
    """

    def _coerce(obj: Any) -> Any:
        # numpy ndarray → list (handles nested containers recursively)
        type_name = type(obj).__name__
        if type_name == "ndarray":
            return obj.tolist()
        if isinstance(obj, bytes):
            return obj  # msgpack native bytes
        if isinstance(obj, dict):
            return {k: _coerce(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_coerce(v) for v in obj]
        return obj

    payload = {
        "identity": {
            "trace_id": trace.identity.trace_id,
            "eval_run_id": trace.identity.eval_run_id,
            "benchmark": trace.identity.benchmark,
            "task_id": trace.identity.task_id,
            "task_instruction": trace.identity.task_instruction,
            "model_id": trace.identity.model_id,
            "model_version": trace.identity.model_version,
        },
        "scene": {
            "scene_config": _coerce(trace.scene.scene_config),
            "init_state": trace.scene.init_state,
            "replay_mode": trace.scene.replay_mode,
        },
        "rollout": {
            "observations": _coerce(trace.rollout.observations),
            "actions": _coerce(trace.rollout.actions),
            "episode_length": trace.rollout.episode_length,
            "success": trace.rollout.success,
            "failure_step": trace.rollout.failure_step,
        },
        "labels": {
            "critic_scores": _coerce(trace.labels.critic_scores),
            "failure_type": trace.labels.failure_type,
            "failure_attribution": _coerce(trace.labels.failure_attribution),
        },
        "provenance": {
            "parent_task_id": trace.provenance.parent_task_id,
            "mutation_type": trace.provenance.mutation_type,
            "promotion_status": trace.provenance.promotion_status,
        },
    }
    result = msgpack.packb(payload, use_bin_type=True)
    assert result is not None
    return result


def _unpack_trace(data: bytes) -> EpisodeTrace:
    """Deserialise msgpack bytes back to an EpisodeTrace.

    Lists are kept as lists (numpy arrays are not reconstructed).
    """
    payload = msgpack.unpackb(data, raw=False)

    identity = TraceIdentity(**payload["identity"])
    scene_raw = payload["scene"]
    scene = SceneConfig(
        scene_config=scene_raw["scene_config"],
        init_state=scene_raw["init_state"],
        replay_mode=scene_raw.get("replay_mode", "exact"),
    )
    roll_raw = payload["rollout"]
    rollout = RolloutData(
        observations=roll_raw["observations"],
        actions=roll_raw["actions"],
        episode_length=roll_raw["episode_length"],
        success=roll_raw["success"],
        failure_step=roll_raw.get("failure_step"),
    )
    lab_raw = payload["labels"]
    labels = TraceLabels(
        critic_scores=lab_raw.get("critic_scores", {}),
        failure_type=lab_raw.get("failure_type"),
        failure_attribution=lab_raw.get("failure_attribution", {}),
    )
    prov_raw = payload["provenance"]
    provenance = TaskProvenance(
        parent_task_id=prov_raw.get("parent_task_id"),
        mutation_type=prov_raw.get("mutation_type"),
        promotion_status=prov_raw.get("promotion_status", "seed"),
    )
    return EpisodeTrace(
        identity=identity,
        scene=scene,
        rollout=rollout,
        labels=labels,
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------
# The table schema and migration machinery live in ``migrations.py``
# (PRAGMA user_version based).  Only connection-level PRAGMAs remain here.

_PRAGMAS = """
PRAGMA journal_mode=WAL;
PRAGMA busy_timeout=5000;
"""


class FsckResult(int):
    """Integer-compatible fsck result.

    ``int(result)`` equals the number of orphan ``.tmp`` files removed (legacy
    contract).  Extra attributes carry the deep-scan report:

    * ``tmp_removed``     — orphan ``.tmp`` files deleted
    * ``broken_links``    — trace_ids whose DB row points at a missing file
    * ``orphan_files``    — ``.msgpack`` files with no DB row
    * ``orphans_removed`` — orphan ``.msgpack`` files deleted (repair=True only)
    """

    tmp_removed: int
    broken_links: list[str]
    orphan_files: list[str]
    orphans_removed: int

    def __new__(
        cls,
        tmp_removed: int,
        broken_links: list[str] | None = None,
        orphan_files: list[str] | None = None,
        orphans_removed: int = 0,
    ) -> "FsckResult":
        obj = super().__new__(cls, tmp_removed)
        obj.tmp_removed = tmp_removed
        obj.broken_links = list(broken_links or [])
        obj.orphan_files = list(orphan_files or [])
        obj.orphans_removed = orphans_removed
        return obj


# ---------------------------------------------------------------------------
# EvalMemory
# ---------------------------------------------------------------------------


class EvalMemory:
    """Persistent storage for SEPA-Eval: SQLite (WAL) + msgpack file store."""

    def __init__(self, db_path: str, memory_dir: str | None = None) -> None:
        self._db_path = db_path

        # memory_dir resolution order: explicit arg → env var → default
        if memory_dir is None:
            memory_dir = os.environ.get(
                "SEPA_MEMORY_DIR",
                os.path.join(os.path.dirname(os.path.abspath(db_path)), "traces"),
            )
        self._memory_dir = memory_dir
        os.makedirs(self._memory_dir, exist_ok=True)

        # Persistent connection (WAL is set at PRAGMA level, not per-statement).
        # WAL allows concurrent readers alongside one writer across *separate*
        # connections, but a single shared ``sqlite3.Connection`` is not safe
        # for concurrent statement execution — so all access to ``self._conn``
        # from this instance is serialized through ``self._lock``.
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._apply_pragmas()
        with self._lock:
            self.schema_version = migrations.migrate(self._conn, self._memory_dir)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_pragmas(self) -> None:
        cur = self._conn.cursor()
        for stmt in _PRAGMAS.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                cur.execute(stmt)
        self._conn.commit()

    @staticmethod
    def _trace_relpath(run_id: str, trace_id: str) -> str:
        """Portable trace path relative to memory_dir (always '/'-separated)."""
        return f"{run_id}/{trace_id}.msgpack"

    def _trace_path(self, run_id: str, trace_id: str) -> str:
        run_dir = os.path.join(self._memory_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        return os.path.join(run_dir, f"{trace_id}.msgpack")

    def resolve_trace_path(self, row: dict) -> str | None:
        """Resolve the on-disk trace file for a traces row dict.

        Preference order:
          1. ``trace_relpath`` joined onto the current ``memory_dir`` (portable)
          2. legacy absolute ``trace_path``, if that file still exists
        Returns None if neither resolves to an existing file, falling back to
        whichever non-empty path was stored (so callers can report it).
        """
        relpath = row.get("trace_relpath")
        abspath = row.get("trace_path")
        if relpath:
            candidate = os.path.join(self._memory_dir, *relpath.split("/"))
            if os.path.isfile(candidate):
                return candidate
        if abspath and os.path.isfile(abspath):
            return abspath
        if relpath:
            return os.path.join(self._memory_dir, *relpath.split("/"))
        return abspath or None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_trace(self, trace: EpisodeTrace) -> None:
        """Persist a full EpisodeTrace (file-first, then DB).

        Crash-safe protocol:
          1. Write packed bytes to <path>.tmp
          2. Atomic rename to final path
          3. Insert DB row
          4. On DB failure: delete the file (compensating write)
        """
        run_id = trace.identity.eval_run_id
        trace_id = trace.identity.trace_id
        final_path = self._trace_path(run_id, trace_id)
        relpath = self._trace_relpath(run_id, trace_id)
        tmp_path = final_path + ".tmp"

        # -- Step 1: write to .tmp
        packed = _pack_trace(trace)
        with open(tmp_path, "wb") as fh:
            fh.write(packed)

        # -- Step 2: atomic rename
        os.replace(tmp_path, final_path)

        # -- Step 3 + 4: DB insert with compensating delete on failure
        params = (
            trace_id,
            run_id,
            trace.identity.benchmark,
            trace.identity.task_id,
            trace.identity.task_instruction,
            trace.identity.model_id,
            trace.identity.model_version,
            int(trace.rollout.success),
            trace.rollout.failure_step,
            trace.rollout.episode_length,
            trace.labels.failure_type,
            trace.provenance.promotion_status,
            trace.provenance.parent_task_id,
            trace.provenance.mutation_type,
            _now_iso(),
            final_path,
            relpath,
        )
        try:
            self._insert_trace_row(params)
        except Exception:
            # Compensating write: remove the file so we don't have orphans
            try:
                os.remove(final_path)
            except OSError:
                pass
            raise

    def _insert_trace_row(self, params: tuple) -> None:
        """Execute the traces INSERT and commit. Extracted for testability."""
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO traces (
                    trace_id, eval_run_id, benchmark, task_id, task_instruction,
                    model_id, model_version, success, failure_step, episode_length,
                    failure_type, promotion_status, parent_task_id, mutation_type,
                    created_at, trace_path, trace_relpath
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                params,
            )
            self._conn.commit()

    def get_failures(
        self,
        benchmark: str | None = None,
        model_id: str | None = None,
        failure_type: str | None = None,
        eval_run_id: str | None = None,
        limit: int = 1000,
    ) -> list[dict]:
        """Return trace row dicts for failed episodes matching the filters."""
        clauses = ["success = 0"]
        params: list[Any] = []

        if benchmark is not None:
            clauses.append("benchmark = ?")
            params.append(benchmark)
        if model_id is not None:
            clauses.append("model_id = ?")
            params.append(model_id)
        if failure_type is not None:
            clauses.append("failure_type = ?")
            params.append(failure_type)
        if eval_run_id is not None:
            clauses.append("eval_run_id = ?")
            params.append(eval_run_id)

        where = " AND ".join(clauses)
        params.append(limit)

        with self._lock:
            cur = self._conn.execute(
                f"SELECT * FROM traces WHERE {where} ORDER BY created_at DESC LIMIT ?",
                params,
            )
            return [dict(row) for row in cur.fetchall()]

    def load_trace_file(self, trace_path: str) -> EpisodeTrace:
        """Deserialise a msgpack file back to an EpisodeTrace.

        ``trace_path`` may be absolute (legacy) or relative to ``memory_dir``
        (portable ``trace_relpath`` form).
        """
        if not os.path.isabs(trace_path):
            trace_path = os.path.join(self._memory_dir, *trace_path.replace(os.sep, "/").split("/"))
        with open(trace_path, "rb") as fh:
            data = fh.read()
        return _unpack_trace(data)

    def get_saturated_tasks(self, threshold: float = 0.95) -> list[dict]:
        """Return tasks where all models succeed ≥ threshold (or saturation_flag set)."""
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT * FROM tasks
                WHERE saturation_flag = 1
                   OR discriminative_power <= ?
                ORDER BY discriminative_power ASC
                """,
                (1.0 - threshold,),
            )
            return [dict(row) for row in cur.fetchall()]

    def get_discriminative_tasks(self, min_spread: float = 0.2) -> list[dict]:
        """Return tasks with discriminative_power >= min_spread."""
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT * FROM tasks
                WHERE discriminative_power >= ?
                ORDER BY discriminative_power DESC
                """,
                (min_spread,),
            )
            return [dict(row) for row in cur.fetchall()]

    def update_critic_score(
        self,
        trace_id: str,
        critic_name: str,
        score: float,
        explanation: str = "",
        confidence: float = 1.0,
    ) -> None:
        """Upsert a critic score row (unique per (trace_id, critic_name)).

        Raises ValueError if trace_id is unknown.
        """
        with self._lock:
            row = self._conn.execute("SELECT 1 FROM traces WHERE trace_id = ?", (trace_id,)).fetchone()
            if row is None:
                raise ValueError(f"trace_id not found: {trace_id!r}")

            self._conn.execute(
                """
                INSERT INTO critic_scores (trace_id, critic_name, score, explanation, confidence, scored_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(trace_id, critic_name) DO UPDATE SET
                    score = excluded.score,
                    explanation = excluded.explanation,
                    confidence = excluded.confidence,
                    scored_at = excluded.scored_at
                """,
                (trace_id, critic_name, score, explanation, confidence, _now_iso()),
            )
            self._conn.commit()

    def record_candidate_task(self, task: CandidateTask) -> None:
        """Insert or replace a CandidateTask into the tasks table."""
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO tasks (
                    task_id, benchmark, instruction, scene_config,
                    mutation_lineage, promotion_status, discriminative_power,
                    saturation_flag, promotion_evidence, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    task.task_id,
                    task.benchmark,
                    task.instruction,
                    json.dumps(task.scene_config),
                    task.mutation_type,
                    task.promotion_status,
                    None,  # discriminative_power not yet computed
                    0,  # saturation_flag default false
                    json.dumps(task.promotion_evidence) if task.promotion_evidence else None,
                    _dt_to_iso(task.created_at),
                ),
            )
            self._conn.commit()

    def update_task_promotion_status(
        self,
        task_id: str,
        status: str,
        evidence: dict | None = None,
    ) -> None:
        """Update the promotion_status (and optionally evidence) for a task."""
        with self._lock:
            self._conn.execute(
                """
                UPDATE tasks
                SET promotion_status = ?,
                    promotion_evidence = COALESCE(?, promotion_evidence)
                WHERE task_id = ?
                """,
                (
                    status,
                    json.dumps(evidence) if evidence is not None else None,
                    task_id,
                ),
            )
            self._conn.commit()

    def register_model(
        self,
        model_id: str,
        framework: str,
        checkpoint: str,
        benchmarks: list[str],
    ) -> None:
        """Insert or replace a model registration record."""
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO models (model_id, framework, checkpoint, benchmarks, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (model_id, framework, checkpoint, json.dumps(benchmarks), _now_iso()),
            )
            self._conn.commit()

    def get_failures_by_cluster_window(self, last_n_runs: int = 5) -> list[dict]:
        """Return failed traces from the most recent `last_n_runs` distinct eval_run_ids."""
        # Identify the N most recent distinct eval_run_ids
        with self._lock:
            recent_cur = self._conn.execute(
                """
                SELECT DISTINCT eval_run_id
                FROM traces
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (last_n_runs,),
            )
            recent_run_ids = [row[0] for row in recent_cur.fetchall()]
            if not recent_run_ids:
                return []

            placeholders = ",".join("?" * len(recent_run_ids))
            cur = self._conn.execute(
                f"""
                SELECT * FROM traces
                WHERE success = 0
                  AND eval_run_id IN ({placeholders})
                ORDER BY created_at DESC
                """,
                recent_run_ids,
            )
            return [dict(row) for row in cur.fetchall()]

    def prune(self, retention_days: int = 90) -> int:
        """Delete trace files older than retention_days. DB rows are kept.

        Returns the number of files deleted.
        """
        from datetime import timedelta

        cutoff = datetime.now(tz=timezone.utc).replace(tzinfo=None) - timedelta(days=retention_days)
        cutoff_str = cutoff.strftime(_ISO_FMT)

        with self._lock:
            cur = self._conn.execute(
                "SELECT trace_id, trace_path, trace_relpath FROM traces WHERE created_at < ?",
                (cutoff_str,),
            )
            rows = [dict(row) for row in cur.fetchall()]

        deleted = 0
        for row in rows:
            trace_path = self.resolve_trace_path(row)
            if trace_path and os.path.isfile(trace_path):
                try:
                    os.remove(trace_path)
                    deleted += 1
                except OSError:
                    pass

        return deleted

    def fsck(self, repair: bool = False) -> FsckResult:
        """Consistency check between the DB and the msgpack file store.

        Always removes orphan ``*.tmp`` files (interrupted writes).  Also
        reports:

        * broken links — DB rows whose trace file is missing on disk
        * orphan files — committed ``.msgpack`` files with no DB row

        Orphan ``.msgpack`` files are only *deleted* when ``repair=True``;
        by default they are just reported.  Broken links are never auto-fixed.

        Returns an :class:`FsckResult`; ``int(result)`` equals the number of
        ``.tmp`` files removed (backward-compatible with the old contract).
        """
        # -- 1. orphan .tmp files (always removed)
        pattern = os.path.join(self._memory_dir, "**", "*.tmp")
        orphans = glob.glob(pattern, recursive=True)
        # Also check top-level (glob ** may not match root in all Python versions)
        top_level = glob.glob(os.path.join(self._memory_dir, "*.tmp"))
        all_tmp = list(set(orphans) | set(top_level))
        for path in all_tmp:
            try:
                os.remove(path)
            except OSError:
                pass

        # -- 2. broken links: DB rows whose file is missing
        with self._lock:
            cur = self._conn.execute("SELECT trace_id, trace_path, trace_relpath FROM traces")
            rows = [dict(row) for row in cur.fetchall()]

        broken_links: list[str] = []
        referenced: set[str] = set()
        for row in rows:
            resolved = self.resolve_trace_path(row)
            if resolved:
                referenced.add(os.path.abspath(resolved))
            if not resolved or not os.path.isfile(resolved):
                broken_links.append(row["trace_id"])

        # -- 3. orphan .msgpack files: on disk but not referenced by any row
        disk_files = set(glob.glob(os.path.join(self._memory_dir, "**", "*.msgpack"), recursive=True)) | set(
            glob.glob(os.path.join(self._memory_dir, "*.msgpack"))
        )
        orphan_files = sorted(path for path in disk_files if os.path.abspath(path) not in referenced)

        orphans_removed = 0
        if repair:
            for path in orphan_files:
                try:
                    os.remove(path)
                    orphans_removed += 1
                except OSError:
                    pass

        return FsckResult(
            tmp_removed=len(all_tmp),
            broken_links=broken_links,
            orphan_files=orphan_files,
            orphans_removed=orphans_removed,
        )

    # ------------------------------------------------------------------
    # Read-only query API (used by reporting/CLI — do not reach into _conn)
    # ------------------------------------------------------------------

    _READONLY_PREFIXES = ("select", "with", "pragma", "explain")

    def query(self, sql: str, params: tuple | list = ()) -> list[dict]:
        """Execute a read-only SQL statement and return rows as dicts.

        Only SELECT/WITH/PRAGMA/EXPLAIN statements are allowed; anything else
        raises ValueError. Serialized through the internal lock.
        """
        head = sql.lstrip().split(None, 1)[0].lower() if sql.strip() else ""
        if head not in self._READONLY_PREFIXES:
            raise ValueError(f"query() only accepts read-only statements, got: {head!r}")
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            return [dict(row) for row in cur.fetchall()]

    def query_scalar(self, sql: str, params: tuple | list = (), default=None):
        """Read-only helper returning the first column of the first row."""
        rows = self.query(sql, params)
        if not rows:
            return default
        first = rows[0]
        return next(iter(first.values()), default)

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._conn.close()
