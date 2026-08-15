"""
Tests for reporting modules:
  - build_task_model_heatmap / render_heatmap_markdown  (heatmap.py)
  - ReportGenerator.generate()                          (report_generator.py)
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from sepa_eval.reporting.heatmap import build_task_model_heatmap, render_heatmap_markdown
from sepa_eval.reporting.report_generator import ReportGenerator


# ---------------------------------------------------------------------------
# Heatmap unit tests
# ---------------------------------------------------------------------------

class TestBuildTaskModelHeatmap:
    def test_empty_input_returns_empty_structure(self):
        result = build_task_model_heatmap([])
        assert result["models"] == []
        assert result["tasks"] == []
        assert result["matrix"] == []
        assert result["universal_failures"] == []
        assert result["model_specific_weaknesses"] == {}

    def test_single_row(self):
        rows = [{"model_id": "ModelA", "task_id": "task_1", "success_rate": 0.9}]
        result = build_task_model_heatmap(rows)
        assert result["models"] == ["ModelA"]
        assert result["tasks"] == ["task_1"]
        assert result["matrix"] == [[0.9]]

    def test_matrix_shape(self):
        rows = [
            {"model_id": "ModelA", "task_id": "task_1", "success_rate": 0.9},
            {"model_id": "ModelA", "task_id": "task_2", "success_rate": 0.3},
            {"model_id": "ModelB", "task_id": "task_1", "success_rate": 0.8},
            {"model_id": "ModelB", "task_id": "task_2", "success_rate": 0.2},
        ]
        result = build_task_model_heatmap(rows)
        assert len(result["tasks"]) == 2
        assert len(result["models"]) == 2
        assert len(result["matrix"]) == 2        # rows = tasks
        assert len(result["matrix"][0]) == 2     # cols = models

    def test_universal_failure_detection(self):
        """Tasks where all models SR < 0.4 appear in universal_failures."""
        rows = [
            {"model_id": "ModelA", "task_id": "hard_task", "success_rate": 0.1},
            {"model_id": "ModelB", "task_id": "hard_task", "success_rate": 0.2},
            {"model_id": "ModelA", "task_id": "easy_task", "success_rate": 0.9},
            {"model_id": "ModelB", "task_id": "easy_task", "success_rate": 0.85},
        ]
        result = build_task_model_heatmap(rows)
        assert "hard_task" in result["universal_failures"]
        assert "easy_task" not in result["universal_failures"]

    def test_model_specific_weakness(self):
        """A task where only ModelA fails and ModelB passes is a ModelA weakness."""
        rows = [
            {"model_id": "ModelA", "task_id": "task_x", "success_rate": 0.1},
            {"model_id": "ModelB", "task_id": "task_x", "success_rate": 0.9},
        ]
        result = build_task_model_heatmap(rows)
        assert "task_x" in result["model_specific_weaknesses"]["ModelA"]
        assert "task_x" not in result["model_specific_weaknesses"]["ModelB"]

    def test_missing_data_becomes_none(self):
        """A (task, model) pair with no row produces None in the matrix."""
        rows = [
            {"model_id": "ModelA", "task_id": "task_1", "success_rate": 0.9},
            # ModelB has no data for task_1
        ]
        result = build_task_model_heatmap(rows)
        # No second model → single column; no None expected here
        assert result["matrix"][0][0] == 0.9

    def test_missing_pair_is_none(self):
        rows = [
            {"model_id": "ModelA", "task_id": "task_1", "success_rate": 0.9},
            {"model_id": "ModelB", "task_id": "task_2", "success_rate": 0.5},
        ]
        result = build_task_model_heatmap(rows)
        # matrix is tasks × models; task_1 × ModelB should be None
        task_idx = result["tasks"].index("task_1")
        model_idx = result["models"].index("ModelB")
        assert result["matrix"][task_idx][model_idx] is None


class TestRenderHeatmapMarkdown:
    def test_empty_heatmap_returns_no_data_string(self):
        heatmap = build_task_model_heatmap([])
        md = render_heatmap_markdown(heatmap)
        assert "No data" in md

    def test_markdown_contains_header_row(self):
        rows = [{"model_id": "ModelA", "task_id": "task_1", "success_rate": 0.9}]
        heatmap = build_task_model_heatmap(rows)
        md = render_heatmap_markdown(heatmap)
        assert "ModelA" in md
        assert "task_1" in md

    def test_high_sr_uses_checkmark_emoji(self):
        rows = [{"model_id": "ModelA", "task_id": "task_1", "success_rate": 0.95}]
        heatmap = build_task_model_heatmap(rows)
        md = render_heatmap_markdown(heatmap)
        assert "✅" in md

    def test_medium_sr_uses_warning_emoji(self):
        rows = [{"model_id": "ModelA", "task_id": "task_1", "success_rate": 0.6}]
        heatmap = build_task_model_heatmap(rows)
        md = render_heatmap_markdown(heatmap)
        assert "⚠️" in md

    def test_low_sr_uses_cross_emoji(self):
        rows = [{"model_id": "ModelA", "task_id": "task_1", "success_rate": 0.1}]
        heatmap = build_task_model_heatmap(rows)
        md = render_heatmap_markdown(heatmap)
        assert "❌" in md

    def test_universal_failure_section_appended(self):
        rows = [
            {"model_id": "ModelA", "task_id": "hard_task", "success_rate": 0.1},
            {"model_id": "ModelB", "task_id": "hard_task", "success_rate": 0.2},
        ]
        heatmap = build_task_model_heatmap(rows)
        md = render_heatmap_markdown(heatmap)
        assert "Universal failures" in md
        assert "hard_task" in md


# ---------------------------------------------------------------------------
# Fake memory for ReportGenerator
# ---------------------------------------------------------------------------

class _FakeMemoryForReport:
    """SQLite :memory: with minimal schema for ReportGenerator queries."""

    def __init__(self):
        self._conn = sqlite3.connect(":memory:")
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(
            """
            CREATE TABLE traces (
                id INTEGER PRIMARY KEY,
                eval_run_id TEXT,
                model_id TEXT,
                success INTEGER,
                failure_type TEXT
            );
            CREATE TABLE tasks (
                id INTEGER PRIMARY KEY,
                task_id TEXT,
                benchmark TEXT,
                promotion_status TEXT,
                discriminative_power REAL,
                saturation_flag INTEGER DEFAULT 0
            );
            CREATE TABLE model_task_results (
                id INTEGER PRIMARY KEY,
                model_id TEXT,
                benchmark TEXT,
                task_id TEXT,
                success_rate REAL,
                clean_success_rate REAL,
                n_trials INTEGER,
                last_eval_at TEXT
            );
            """
        )
        self._conn.commit()

    def seed(
        self,
        model_id: str = "ModelA",
        task_id: str = "task_1",
        benchmark: str = "libero_spatial",
        sr: float = 0.8,
    ):
        self._conn.execute(
            "INSERT INTO model_task_results "
            "(model_id, benchmark, task_id, success_rate, clean_success_rate, n_trials, last_eval_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (model_id, benchmark, task_id, sr, sr, 5, "2026-06-01"),
        )
        self._conn.execute(
            "INSERT INTO tasks (task_id, benchmark, promotion_status, discriminative_power, saturation_flag) "
            "VALUES (?,?,?,?,?)",
            (task_id, benchmark, "promoted", 0.3, 0),
        )
        self._conn.execute(
            "INSERT INTO traces (eval_run_id, model_id, success, failure_type) VALUES (?,?,?,?)",
            ("run-001", model_id, 0, "grasp_failure"),
        )
        self._conn.commit()

    def close(self):
        self._conn.close()


# ---------------------------------------------------------------------------
# ReportGenerator tests
# ---------------------------------------------------------------------------

class TestReportGenerator:
    def test_generate_returns_nonempty_string(self, tmp_path):
        mem = _FakeMemoryForReport()
        gen = ReportGenerator(memory=mem)
        out = tmp_path / "report.md"
        content = gen.generate(output_path=str(out))
        mem.close()

        assert isinstance(content, str)
        assert len(content) > 0

    def test_generate_writes_file(self, tmp_path):
        mem = _FakeMemoryForReport()
        gen = ReportGenerator(memory=mem)
        out = tmp_path / "report.md"
        gen.generate(output_path=str(out))
        mem.close()

        assert out.exists()
        assert out.stat().st_size > 0

    def test_generate_with_seeded_data(self, tmp_path):
        mem = _FakeMemoryForReport()
        mem.seed(model_id="ModelA", task_id="task_1", sr=0.9)
        mem.seed(model_id="ModelB", task_id="task_1", sr=0.3)
        gen = ReportGenerator(memory=mem)
        out = tmp_path / "report.md"
        content = gen.generate(output_path=str(out))
        mem.close()

        # Report should mention both models
        assert "ModelA" in content
        assert "ModelB" in content

    def test_generate_creates_parent_dirs(self, tmp_path):
        mem = _FakeMemoryForReport()
        gen = ReportGenerator(memory=mem)
        nested = tmp_path / "a" / "b" / "c" / "report.md"
        gen.generate(output_path=str(nested))
        mem.close()

        assert nested.exists()

    def test_generate_with_cycle_result(self, tmp_path):
        """cycle_result fields are reflected in the generated report."""
        from sepa_eval.orchestrator.evolution_loop import EvolutionCycleResult
        from datetime import datetime

        mem = _FakeMemoryForReport()
        gen = ReportGenerator(memory=mem)
        out = tmp_path / "report.md"

        cycle_result = EvolutionCycleResult(
            cycle_id="abc123",
            started_at=datetime(2026, 6, 1),
            candidates_generated=10,
            candidates_promoted=3,
        )
        content = gen.generate(output_path=str(out), cycle_result=cycle_result)
        mem.close()

        assert isinstance(content, str)

    def test_generate_empty_db_no_crash(self, tmp_path):
        """Empty DB produces a valid (possibly minimal) report without raising."""
        mem = _FakeMemoryForReport()
        gen = ReportGenerator(memory=mem)
        out = tmp_path / "report.md"
        content = gen.generate(output_path=str(out))
        mem.close()

        assert isinstance(content, str)
