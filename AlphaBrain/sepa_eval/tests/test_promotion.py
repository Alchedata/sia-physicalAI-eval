"""
Tests for the SEPA-Eval promotion pipeline.

Two tests:
  1. test_promotion_gates_mock      — all gates pass → status "promoted"
  2. test_gate_timeout_deferred     — first gate times out → status "deferred"
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from sepa_eval.promotion.gates import (
    DiscriminativePowerGate,
    GateOutcome,
    GateResult,
    HumanReviewGate,
    RedundancyGate,
    ReproducibilityGate,
    SolvabilityGate,
)
from sepa_eval.promotion.pipeline import PromotionPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candidate(task_id: str = "task_test_001") -> MagicMock:
    """Return a minimal CandidateTask-like mock."""
    candidate = MagicMock()
    candidate.task_id = task_id
    candidate.instruction = "Pick the blue mug from the shelf"
    candidate.mutation_type = "PosePerturbation"
    candidate.embedding = None  # no embedding → RedundancyGate skips similarity
    return candidate


def _eval_fn_high_sr(candidate, model_id: str, n_trials: int) -> float:
    """Always return SR=0.8 — satisfies both SolvabilityGate and ReproducibilityGate."""
    return 0.8


# ---------------------------------------------------------------------------
# Test 1: all gates pass → "promoted"
# ---------------------------------------------------------------------------

def test_promotion_gates_mock():
    """
    With eval_fn returning SR=0.8, promoted_embeddings=[], and two model_ids,
    all gates should pass and the pipeline should return status="promoted".
    """
    gates = [
        SolvabilityGate(n_trials=10, min_sr=0.5),
        ReproducibilityGate(n_trials=20, min_failure_rate=0.1),  # failure_rate = 0.2 >= 0.1
        RedundancyGate(similarity_threshold=0.85),
        DiscriminativePowerGate(n_trials=20, min_spread=0.0),    # spread = 0.0 >= 0.0
        HumanReviewGate(queue_path="/tmp/sepa_test_review_queue.jsonl"),
    ]
    pipeline = PromotionPipeline(gates=gates, max_parallel_eval_workers=2)
    candidate = _make_candidate()

    status, evidence = pipeline.run(
        candidate,
        eval_fn=_eval_fn_high_sr,
        model_ids=["model_a", "model_b"],
        promoted_embeddings=[],
    )

    assert status == "promoted", f"Expected 'promoted', got '{status}'. Evidence: {evidence}"
    # All five gates must appear in the evidence dict.
    gate_names = set(evidence.keys())
    assert "SolvabilityGate" in gate_names
    assert "ReproducibilityGate" in gate_names
    assert "RedundancyGate" in gate_names
    assert "DiscriminativePowerGate" in gate_names
    assert "HumanReviewGate" in gate_names


# ---------------------------------------------------------------------------
# Test 2: first gate times out → "deferred"
# ---------------------------------------------------------------------------

def test_gate_timeout_deferred():
    """
    When SolvabilityGate.evaluate sleeps beyond the pipeline timeout,
    the pipeline should return status="deferred".
    """
    # Build a gate whose evaluate blocks indefinitely until the timeout fires.
    def _slow_evaluate(candidate, **kwargs):
        # Sleep much longer than the configured timeout.
        time.sleep(10)
        return GateResult(
            gate_name="SolvabilityGate",
            outcome=GateOutcome.PASS,
            evidence={},
        )

    slow_gate = MagicMock(spec=SolvabilityGate)
    slow_gate.evaluate = _slow_evaluate

    pipeline = PromotionPipeline(
        gates=[slow_gate],
        max_parallel_eval_workers=2,
        # 0.001 minutes = 0.06 seconds — much shorter than the 10-second sleep.
        gate_timeout_minutes=0.001,
    )
    candidate = _make_candidate()

    status, evidence = pipeline.run(candidate, eval_fn=_eval_fn_high_sr, model_ids=["m"])

    assert status == "deferred", f"Expected 'deferred', got '{status}'. Evidence: {evidence}"
    # The evidence dict should contain a timeout marker for the slow gate.
    gate_key = list(evidence.keys())[0]
    assert evidence[gate_key].get("timeout") is True
