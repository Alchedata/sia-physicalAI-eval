"""
Tests for EvolutionLoopOrchestrator.

All external collaborators are replaced with lightweight fakes so the tests
run without a real simulator, GPU, or OpenAI key.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sepa_eval.orchestrator.evolution_loop import EvolutionCycleResult, EvolutionLoopOrchestrator


# ---------------------------------------------------------------------------
# Fake collaborators
# ---------------------------------------------------------------------------

class _FakeMemory:
    """Minimal EvalMemory stand-in."""

    def __init__(self):
        import sqlite3
        self._conn = sqlite3.connect(":memory:")
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS traces "
            "(id INTEGER PRIMARY KEY, eval_run_id TEXT, model_id TEXT)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS tasks "
            "(id INTEGER PRIMARY KEY, task_id TEXT, promotion_status TEXT)"
        )
        self._conn.commit()
        self._candidates: list = []

    def get_failures_by_cluster_window(self, last_n_runs: int = 5) -> list[dict]:
        return []

    def get_saturated_tasks(self, threshold: float = 0.95) -> list[dict]:
        return []

    def record_candidate_task(self, candidate) -> None:
        self._candidates.append(candidate)

    def close(self):
        self._conn.close()


class _FakeClusterer:
    def cluster(self, trace_rows: list[dict]) -> list:
        return []


class _FakeSeedExtractor:
    def extract(self, cluster, trace_rows):
        seed = MagicMock()
        seed.scene_config = {}
        seed.task_instruction = "Pick up the cube"
        seed.trace_id = "trace-001"
        seed.benchmark = "libero_spatial"
        return seed


class _FakeOperator:
    name = "FakeOperator"

    def generate(self, seed_scene_config, seed_instruction, parent_task_id, benchmark):
        candidate = MagicMock()
        candidate.task_id = "candidate-001"
        return [candidate]


class _FakePromotionPipeline:
    def run(self, candidate, **gate_kwargs):
        return "promoted", {"FakeGate": {"passed": True}}


class _FakeReportGenerator:
    def __init__(self):
        self.last_output_path = None

    def generate(self, output_path: str, cycle_result=None):
        self.last_output_path = output_path
        Path(output_path).write_text("# Report\n", encoding="utf-8")
        return "# Report\n"


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def orchestrator(tmp_path):
    mem = _FakeMemory()
    orch = EvolutionLoopOrchestrator(
        memory=mem,
        clusterer=_FakeClusterer(),
        seed_extractor=_FakeSeedExtractor(),
        mutation_engine=[],
        promotion_pipeline=None,
        report_generator=None,
        config={},
        log_path=str(tmp_path / "evolution_loop_log.jsonl"),
        metrics_path=str(tmp_path / "sepa_eval_metrics.json"),
    )
    yield orch
    mem.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_run_cycle_returns_result(orchestrator):
    """run_cycle() with empty DB returns a valid EvolutionCycleResult."""
    result = orchestrator.run_cycle()

    assert isinstance(result, EvolutionCycleResult)
    assert result.error is None
    assert result.finished_at is not None
    assert "evaluate" in result.steps_completed
    assert "diagnose" in result.steps_completed
    assert "generate" in result.steps_completed
    assert "validate_promote" in result.steps_completed
    assert "monitor" in result.steps_completed
    assert "report" in result.steps_completed


def test_run_cycle_writes_metrics(tmp_path):
    """run_cycle() writes sepa_eval_metrics.json with required keys."""
    mem = _FakeMemory()
    metrics_path = tmp_path / "sepa_eval_metrics.json"
    orch = EvolutionLoopOrchestrator(
        memory=mem,
        clusterer=_FakeClusterer(),
        seed_extractor=_FakeSeedExtractor(),
        mutation_engine=[],
        promotion_pipeline=None,
        report_generator=None,
        config={},
        log_path=str(tmp_path / "loop.jsonl"),
        metrics_path=str(metrics_path),
    )
    orch.run_cycle()
    mem.close()

    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text())
    assert "traces_written" in metrics
    assert "candidates_generated" in metrics
    assert "promotion_yield" in metrics
    assert "gate_timeout_count" in metrics


def test_run_cycle_writes_jsonl_log(tmp_path):
    """run_cycle() appends JSONL records for each step."""
    mem = _FakeMemory()
    log_path = tmp_path / "loop.jsonl"
    orch = EvolutionLoopOrchestrator(
        memory=mem,
        clusterer=_FakeClusterer(),
        seed_extractor=_FakeSeedExtractor(),
        mutation_engine=[],
        promotion_pipeline=None,
        report_generator=None,
        config={},
        log_path=str(log_path),
        metrics_path=str(tmp_path / "metrics.json"),
    )
    orch.run_cycle()
    mem.close()

    assert log_path.exists()
    lines = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    steps_logged = {l["step"] for l in lines}
    assert "evaluate" in steps_logged
    assert "diagnose" in steps_logged
    assert "generate" in steps_logged


def test_run_cycle_with_mutation_and_promotion(tmp_path):
    """Mutation operators and promotion pipeline are invoked when clusters exist."""

    class _ClustererWithData:
        def cluster(self, trace_rows):
            cluster = MagicMock()
            cluster.label = 0
            return [cluster]

    mem = _FakeMemory()
    report_gen = _FakeReportGenerator()
    orch = EvolutionLoopOrchestrator(
        memory=mem,
        clusterer=_ClustererWithData(),
        seed_extractor=_FakeSeedExtractor(),
        mutation_engine=[_FakeOperator()],
        promotion_pipeline=_FakePromotionPipeline(),
        report_generator=report_gen,
        config={"orchestrator": {"max_candidates_per_cycle": 5}},
        log_path=str(tmp_path / "loop.jsonl"),
        metrics_path=str(tmp_path / "metrics.json"),
    )

    # Give the memory fake trace rows for the window call
    mem.get_failures_by_cluster_window = lambda **kw: [{"trace_id": "t1"}]

    result = orch.run_cycle()
    mem.close()

    assert result.candidates_generated >= 1
    assert result.candidates_promoted >= 1
    assert report_gen.last_output_path is not None


def test_run_cycle_eval_fn_called(tmp_path):
    """When eval_fn and model_ids are supplied, eval_fn is called per model."""
    call_log = []

    def fake_eval(model_id, n_trials):
        call_log.append(model_id)
        return 0.75

    mem = _FakeMemory()
    orch = EvolutionLoopOrchestrator(
        memory=mem,
        clusterer=_FakeClusterer(),
        seed_extractor=_FakeSeedExtractor(),
        mutation_engine=[],
        promotion_pipeline=None,
        report_generator=None,
        config={},
        log_path=str(tmp_path / "loop.jsonl"),
        metrics_path=str(tmp_path / "metrics.json"),
    )
    orch.run_cycle(eval_fn=fake_eval, model_ids=["model_a", "model_b"])
    mem.close()

    assert set(call_log) == {"model_a", "model_b"}


def test_run_cycle_eval_fn_exception_does_not_crash(tmp_path):
    """An eval_fn that raises for one model should not abort the cycle."""

    def bad_eval(model_id, n_trials):
        raise RuntimeError("simulator crash")

    mem = _FakeMemory()
    orch = EvolutionLoopOrchestrator(
        memory=mem,
        clusterer=_FakeClusterer(),
        seed_extractor=_FakeSeedExtractor(),
        mutation_engine=[],
        promotion_pipeline=None,
        report_generator=None,
        config={},
        log_path=str(tmp_path / "loop.jsonl"),
        metrics_path=str(tmp_path / "metrics.json"),
    )
    result = orch.run_cycle(eval_fn=bad_eval, model_ids=["model_a"])
    mem.close()

    # Cycle must complete even when eval_fn raises.
    assert result.error is None
    assert "evaluate" in result.steps_completed


def test_check_saturation_returns_zero_on_empty(orchestrator):
    """_check_saturation with no promoted tasks returns 0."""
    count = orchestrator._check_saturation(threshold=0.95)
    assert count == 0


def test_log_step_creates_file(tmp_path):
    """_log_step creates the JSONL file and writes a valid JSON record."""
    mem = _FakeMemory()
    log_path = tmp_path / "subdir" / "loop.jsonl"
    orch = EvolutionLoopOrchestrator(
        memory=mem,
        clusterer=_FakeClusterer(),
        seed_extractor=_FakeSeedExtractor(),
        mutation_engine=[],
        promotion_pipeline=None,
        report_generator=None,
        config={},
        log_path=str(log_path),
        metrics_path=str(tmp_path / "metrics.json"),
    )
    orch._log_step("cycle-abc", "test_step", "started", foo="bar")
    mem.close()

    assert log_path.exists()
    record = json.loads(log_path.read_text().strip())
    assert record["step"] == "test_step"
    assert record["status"] == "started"
    assert record["foo"] == "bar"
