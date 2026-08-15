"""
Lightweight schema migration machinery for EvalMemory.

Versioning uses ``PRAGMA user_version``:
  * v0  — empty / fresh database (no tables yet)
  * v1  — original SEPA-Eval schema (pre-migration-machinery databases)
  * v2  — storage hardening: portable ``trace_relpath`` column, secondary
          indexes, and a UNIQUE(trace_id, critic_name) index on critic_scores.

A fresh database is created directly at the latest schema and stamped with
``SCHEMA_VERSION``.  Existing databases are migrated in order, one version at a
time, inside a transaction per step.

This module also exposes :func:`relativize_trace_paths` as a standalone repair
helper (e.g. for fixing demo databases with broken absolute paths) — it is
never run automatically against data outside the opened database.
"""

from __future__ import annotations

import logging
import os
import sqlite3

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2

# ---------------------------------------------------------------------------
# Latest schema DDL (v2)
# ---------------------------------------------------------------------------
# NOTE: column order in ``traces`` is load-bearing — ``trace_path`` must remain
# the 16th column (index 15) for backward compatibility with existing readers;
# ``trace_relpath`` is appended after it.

LATEST_DDL = """
CREATE TABLE IF NOT EXISTS traces (
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
    trace_path        TEXT,
    trace_relpath     TEXT
);

CREATE TABLE IF NOT EXISTS tasks (
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

CREATE TABLE IF NOT EXISTS critic_scores (
    trace_id    TEXT,
    critic_name TEXT,
    score       REAL,
    explanation TEXT,
    confidence  REAL,
    scored_at   TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_task_results (
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

CREATE TABLE IF NOT EXISTS failure_clusters (
    cluster_id               TEXT PRIMARY KEY,
    eval_run_id              TEXT,
    failure_type             TEXT,
    centroid                 BLOB,
    representative_trace_id  TEXT,
    member_count             INTEGER,
    llm_summary              TEXT,
    summarized_at            TIMESTAMP
);

CREATE TABLE IF NOT EXISTS models (
    model_id    TEXT PRIMARY KEY,
    framework   TEXT,
    checkpoint  TEXT,
    benchmarks  TEXT,
    created_at  TIMESTAMP
);
"""

# Secondary indexes for the hot query paths (v2).
INDEX_DDL = """
CREATE INDEX IF NOT EXISTS idx_traces_task_id ON traces(task_id);
CREATE INDEX IF NOT EXISTS idx_traces_model_success ON traces(model_id, success);
CREATE INDEX IF NOT EXISTS idx_traces_run_created ON traces(eval_run_id, created_at);
CREATE INDEX IF NOT EXISTS idx_mtr_benchmark ON model_task_results(benchmark);
CREATE UNIQUE INDEX IF NOT EXISTS idx_critic_scores_unique ON critic_scores(trace_id, critic_name);
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _exec_multi(conn: sqlite3.Connection, ddl: str) -> None:
    for stmt in ddl.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()
    return row is not None


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    return column in cols


def get_user_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def set_user_version(conn: sqlite3.Connection, version: int) -> None:
    conn.execute(f"PRAGMA user_version = {int(version)}")


def relativize_trace_paths(conn: sqlite3.Connection, memory_dir: str) -> int:
    """Backfill ``traces.trace_relpath`` from absolute ``trace_path`` values.

    Rows whose absolute path lives under ``memory_dir`` get a portable
    ``{run_id}/{trace_id}.msgpack``-style relative path.  For other rows we
    fall back to the last two path components when that file actually exists
    under ``memory_dir`` (repairs traces whose store was moved).  Returns the
    number of rows updated.  Never touches files on disk.
    """
    memory_dir = os.path.abspath(memory_dir)
    rows = conn.execute(
        "SELECT trace_id, trace_path FROM traces "
        "WHERE trace_path IS NOT NULL AND (trace_relpath IS NULL OR trace_relpath = '')"
    ).fetchall()
    updated = 0
    for trace_id, trace_path in rows:
        relpath: str | None = None
        abspath = os.path.abspath(trace_path)
        if abspath.startswith(memory_dir + os.sep):
            relpath = os.path.relpath(abspath, memory_dir).replace(os.sep, "/")
        else:
            # Store may have moved: try <run_id>/<file> tail under memory_dir.
            parts = trace_path.replace(os.sep, "/").split("/")
            if len(parts) >= 2:
                tail = "/".join(parts[-2:])
                if os.path.isfile(os.path.join(memory_dir, *tail.split("/"))):
                    relpath = tail
        if relpath is not None:
            conn.execute(
                "UPDATE traces SET trace_relpath = ? WHERE trace_id = ?",
                (relpath, trace_id),
            )
            updated += 1
    return updated


# ---------------------------------------------------------------------------
# Migration steps
# ---------------------------------------------------------------------------


def _migrate_v1_to_v2(conn: sqlite3.Connection, memory_dir: str) -> None:
    """v1 → v2: trace_relpath column, secondary indexes, critic upsert key."""
    if not _column_exists(conn, "traces", "trace_relpath"):
        conn.execute("ALTER TABLE traces ADD COLUMN trace_relpath TEXT")
    relativize_trace_paths(conn, memory_dir)

    # Deduplicate critic_scores (keep the most recent row per key) so the
    # UNIQUE index can be created.
    conn.execute(
        """
        DELETE FROM critic_scores
        WHERE rowid NOT IN (
            SELECT MAX(rowid) FROM critic_scores GROUP BY trace_id, critic_name
        )
        """
    )
    _exec_multi(conn, INDEX_DDL)


# Ordered mapping: MIGRATIONS[v] upgrades a database from version v to v + 1.
MIGRATIONS: dict[int, callable] = {
    1: _migrate_v1_to_v2,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def migrate(conn: sqlite3.Connection, memory_dir: str) -> int:
    """Bring the database at ``conn`` up to ``SCHEMA_VERSION``.

    Returns the resulting schema version.
    """
    version = get_user_version(conn)

    if version == 0:
        if not _table_exists(conn, "traces"):
            # Fresh database: create the latest schema directly.
            _exec_multi(conn, LATEST_DDL)
            _exec_multi(conn, INDEX_DDL)
            set_user_version(conn, SCHEMA_VERSION)
            conn.commit()
            return SCHEMA_VERSION
        # Legacy pre-versioning database → treat as v1.
        version = 1
        set_user_version(conn, 1)

    if version > SCHEMA_VERSION:
        raise RuntimeError(
            f"Database schema version {version} is newer than supported version {SCHEMA_VERSION}; "
            "upgrade sepa_eval to open this database."
        )

    while version < SCHEMA_VERSION:
        step = MIGRATIONS[version]
        logger.info("Migrating EvalMemory schema v%d → v%d", version, version + 1)
        step(conn, memory_dir)
        version += 1
        set_user_version(conn, version)
        conn.commit()

    # Idempotent safety net: make sure base tables exist even for stamped DBs.
    _exec_multi(conn, LATEST_DDL)
    conn.commit()
    return version
