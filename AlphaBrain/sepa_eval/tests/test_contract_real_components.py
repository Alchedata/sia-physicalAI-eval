"""
Contract tests: assemble REAL SEPA-Eval components end-to-end (only external
I/O — simulator rollouts and LLM calls — is stubbed via eval_fn).

These tests exist because earlier unit tests used Fake objects whose call
signatures drifted from the real implementations (e.g. PromotionPipeline.run
batch-vs-single mismatch).  Any interface drift between orchestrator, mining,
mutation, promotion, and memory must fail here.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sepa_eval.critics.robustness_critic import RobustnessCritic
from sepa_eval.critics.safety_critic import SafetyCritic
from sepa_eval.memory.eval_memory import EvalMemory
from sepa_eval.memory.schema import (
    EpisodeTrace,
    RolloutData,
    SceneConfig,
    TaskProvenance,
    TraceIdentity,
    TraceLabels,
)
from sepa_eval.mining.failure_cluster import FailureClusterer
from sepa_eval.mining.seed_extractor import SeedExtractor
from sepa_eval.mutation.pose_perturbation import PosePerturbation
from sepa_eval.orchestrator.evolution_loop import EvolutionLoopOrchestrator
from sepa_eval.promotion.gates import (
    DiscriminativePowerGate,
    HumanReviewGate,
    RedundancyGate,
    ReproducibilityGate,
    SolvabilityGate,
)
from sepa_eval.promotion.pipeline import PromotionPipeline
from sepa_eval.reporting.report_generator import ReportGenerator


def _make_trace(i: int) -> EpisodeTrace:
    return EpisodeTrace(
        identity=TraceIdentity(
            trace_id=f"trace-{i:03d}",
            eval_run_id="run-001",
            benchmark="libero_spatial",
            task_id="task-pick-cube",
            task_instruction="Pick up the cube",
            model_id="QwenOFT-test",
            model_version="v1.0",
        ),
        scene=SceneConfig(
            scene_config={"objects": [{"name": "cube", "position": [0.1, 0.2, 0.3]}]},
            init_state=b"",
        ),
        rollout=RolloutData(
            observations=[{"gripper_state": [0.0], "qpos": [0.0] * 7}] * 10,
            actions=[[0.0] * 7] * 10,
            episode_length=10,
            success=False,
            failure_step=8,
        ),
        labels=TraceLabels(failure_type="grasp_failure"),
        provenance=TaskProvenance(),
    )


@pytest.fixture
def memory(tmp_path) -> EvalMemory:
    mem = EvalMemory(
        db_path=str(tmp_path / "eval.db"),
        memory_dir=str(tmp_path / "traces"),
    )
    for i in range(5):
        mem.record_trace(_make_trace(i))
    yield mem
    mem.close()


def _eval_fn(candidate, model_id, n_trials=1) -> float:
    """Stubbed rollout: mid-range SR so Solvability passes and Discriminative
    power sees variance across models."""
    return 0.5 if model_id.endswith("a") else 0.2


def _build_orchestrator(memory, tmp_path) -> EvolutionLoopOrchestrator:
    gates = [
        SolvabilityGate(),
        ReproducibilityGate(),
        RedundancyGate(),
        DiscriminativePowerGate(),
        HumanReviewGate(queue_path=str(tmp_path / "review_queue.jsonl")),
    ]
    return EvolutionLoopOrchestrator(
        memory=memory,
        clusterer=FailureClusterer(),
        seed_extractor=SeedExtractor(memory=memory),
        mutation_engine=[PosePerturbation()],
        promotion_pipeline=PromotionPipeline(gates=gates, gate_timeout_minutes=0.5),
        report_generator=ReportGenerator(memory=memory),
        config={"orchestrator": {"max_candidates_per_cycle": 4}},
        log_path=str(tmp_path / "loop_log.jsonl"),
        metrics_path=str(tmp_path / "metrics.json"),
        critics={"safety": SafetyCritic(), "robustness": RobustnessCritic()},
    )


class TestRealComponentCycle:
    def test_full_cycle_with_real_components(self, memory, tmp_path):
        orch = _build_orchestrator(memory, tmp_path)
        result = orch.run_cycle(
            eval_fn=_eval_fn,
            model_ids=["model-a", "model-b"],
        )

        assert result.error is None
        assert set(result.steps_completed) >= {
            "evaluate", "diagnose", "generate", "validate_promote", "monitor", "report",
        }
        assert result.candidates_generated > 0

    def test_promotion_status_persisted(self, memory, tmp_path):
        orch = _build_orchestrator(memory, tmp_path)
        orch.run_cycle(eval_fn=_eval_fn, model_ids=["model-a", "model-b"])

        rows = memory._conn.execute(
            "SELECT promotion_status, promotion_evidence FROM tasks"
        ).fetchall()
        assert rows, "candidates must be recorded in tasks table"
        statuses = {r[0] for r in rows}
        # every candidate ended in a terminal pipeline state (not left dangling)
        assert statuses <= {"promoted", "rejected", "archived", "deferred"}
        # evidence audit trail persisted for at least one candidate
        assert any(r[1] for r in rows)

    def test_critic_scores_written(self, memory, tmp_path):
        orch = _build_orchestrator(memory, tmp_path)
        orch.run_cycle(eval_fn=_eval_fn, model_ids=["model-a", "model-b"])

        n = memory._conn.execute("SELECT COUNT(*) FROM critic_scores").fetchone()[0]
        assert n > 0, "safety critic scores must be persisted"

    def test_metrics_report_real_critic_latency(self, memory, tmp_path):
        orch = _build_orchestrator(memory, tmp_path)
        orch.run_cycle(eval_fn=_eval_fn, model_ids=["model-a", "model-b"])

        metrics = json.loads(Path(tmp_path / "metrics.json").read_text())
        assert metrics["critic_latency_ms"] > 0.0

    def test_seed_extractor_recovers_scene_config(self, memory):
        rows = memory.get_failures_by_cluster_window(last_n_runs=5)
        clusters = FailureClusterer().cluster(rows)
        assert clusters
        seed = SeedExtractor(memory=memory).extract(clusters[0], rows)
        assert seed.scene_config, "scene_config must be recovered from msgpack trace"
        assert seed.scene_config.get("objects")
