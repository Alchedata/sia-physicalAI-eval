"""
Unit tests for FailureClusterer.

Two test functions:
    test_dbscan_empty_guard   – 0–4 trace rows → returns [], no crash.
    test_cluster_window       – filter_by_window keeps only the N most recent
                                distinct eval_run_ids.
"""
from __future__ import annotations

import pytest

from sepa_eval.mining.failure_cluster import FailureClusterer


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_row(
    trace_id: str,
    eval_run_id: str,
    failure_type: str | None = "grasp",
    failure_step: int = 50,
    episode_length: int = 100,
) -> dict:
    return {
        "trace_id": trace_id,
        "eval_run_id": eval_run_id,
        "benchmark": "libero_spatial",
        "task_id": "task-001",
        "task_instruction": "pick up the cube",
        "failure_type": failure_type,
        "failure_step": failure_step,
        "episode_length": episode_length,
        "scene_config": {},
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFailureClusterer:
    """Tests for FailureClusterer."""

    @pytest.fixture(autouse=True)
    def clusterer(self):
        self.clusterer = FailureClusterer(eps=0.5, min_samples=5, last_n_runs=5)

    # ------------------------------------------------------------------
    # test_dbscan_empty_guard
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("n_rows", [0, 1, 2, 3, 4])
    def test_dbscan_empty_guard(self, n_rows: int):
        """
        When fewer than min_samples (5) trace rows are provided,
        cluster() must return an empty list without raising.
        """
        rows = [
            _make_row(f"trace-{i}", f"run-{i}")
            for i in range(n_rows)
        ]
        result = self.clusterer.cluster(rows)
        assert result == [], (
            f"Expected [] for {n_rows} rows, got {result}"
        )

    # ------------------------------------------------------------------
    # test_cluster_window
    # ------------------------------------------------------------------

    def test_cluster_window(self):
        """
        Given 15 trace rows spread across 6 distinct eval_run_ids,
        filter_by_window(last_n_runs=5) must keep exactly the 5 most recent
        run IDs and drop the oldest one.
        """
        # Build rows: 3 rows per run_id, 6 run_ids, insertion order = run_0..run_5.
        rows: list[dict] = []
        for run_idx in range(6):
            run_id = f"run-{run_idx}"
            for row_idx in range(3):
                rows.append(
                    _make_row(
                        trace_id=f"trace-run{run_idx}-{row_idx}",
                        eval_run_id=run_id,
                    )
                )

        # The 6 distinct run IDs in insertion order: run-0 … run-5.
        # The 5 most recent are run-1 … run-5; run-0 should be excluded.
        filtered = self.clusterer.filter_by_window(rows, last_n_runs=5)

        filtered_run_ids = {r["eval_run_id"] for r in filtered}
        assert "run-0" not in filtered_run_ids, (
            "Oldest run_id 'run-0' should have been filtered out."
        )
        expected_run_ids = {f"run-{i}" for i in range(1, 6)}
        assert filtered_run_ids == expected_run_ids, (
            f"Expected run IDs {expected_run_ids}, got {filtered_run_ids}"
        )
        # 6 run IDs total, exclude the oldest (run-0) → keep 5 × 3 rows = 15.
        assert len(filtered) == 15, (
            f"Expected 15 filtered rows, got {len(filtered)}"
        )
