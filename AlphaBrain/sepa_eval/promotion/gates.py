"""
Promotion gates for the SEPA-Eval candidate task pipeline.

Five gates in order:
  1. SolvabilityGate         — at least one model can solve the task
  2. ReproducibilityGate     — the failure recurs reliably
  3. RedundancyGate          — the task is not a near-duplicate of an existing one
  4. DiscriminativePowerGate — the task differentiates model quality
  5. HumanReviewGate         — async queue for human oversight (always PASS immediately)
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from sepa_eval.promotion.human_review_queue import HumanReviewQueue

# ---------------------------------------------------------------------------
# Shared result types
# ---------------------------------------------------------------------------

class GateOutcome(Enum):
    PASS = "pass"
    FAIL = "fail"
    DEFER = "defer"
    ARCHIVE = "archive"
    DISCARD = "discard"


@dataclass
class GateResult:
    """Result produced by a single promotion gate."""

    gate_name: str
    outcome: GateOutcome
    evidence: dict = field(default_factory=dict)
    message: str = ""


# ---------------------------------------------------------------------------
# Gate 1 — Solvability
# ---------------------------------------------------------------------------

class SolvabilityGate:
    """
    Pass if at least one model achieves SR >= min_sr over n_trials trials.
    Discard (hard reject) if *no* model reaches the threshold.
    """

    def __init__(self, n_trials: int = 10, min_sr: float = 0.5) -> None:
        self.n_trials = n_trials
        self.min_sr = min_sr

    def evaluate(
        self,
        candidate: Any,
        eval_fn: Callable[[Any, str, int], float],
        model_ids: list[str] | None = None,
        **kwargs: Any,
    ) -> GateResult:
        """
        Parameters
        ----------
        candidate:
            CandidateTask instance.
        eval_fn:
            Callable(candidate, model_id, n_trials) -> float SR.
        model_ids:
            List of model IDs to evaluate.  Defaults to a single sentinel.
        """
        model_ids = model_ids or ["default"]
        results: dict[str, float] = {}
        best_sr = 0.0

        for model_id in model_ids:
            sr = eval_fn(candidate, model_id, self.n_trials)
            results[model_id] = sr
            best_sr = max(best_sr, sr)

        if best_sr >= self.min_sr:
            return GateResult(
                gate_name="SolvabilityGate",
                outcome=GateOutcome.PASS,
                evidence={"model_sr": results, "best_sr": best_sr},
                message=f"Best SR {best_sr:.3f} >= threshold {self.min_sr}",
            )
        return GateResult(
            gate_name="SolvabilityGate",
            outcome=GateOutcome.DISCARD,
            evidence={"model_sr": results, "best_sr": best_sr},
            message=(
                f"No model reached SR >= {self.min_sr}. "
                f"Best SR was {best_sr:.3f}. Discarding candidate."
            ),
        )


# ---------------------------------------------------------------------------
# Gate 2 — Reproducibility
# ---------------------------------------------------------------------------

class ReproducibilityGate:
    """
    Pass if at least one model fails the task in >= min_failure_rate of trials,
    confirming the failure is a reliable, reproducible signal and not noise.
    """

    def __init__(
        self,
        n_trials: int = 20,
        min_failure_rate: float = 0.6,
    ) -> None:
        self.n_trials = n_trials
        self.min_failure_rate = min_failure_rate

    def evaluate(
        self,
        candidate: Any,
        eval_fn: Callable[[Any, str, int], float],
        model_ids: list[str] | None = None,
        **kwargs: Any,
    ) -> GateResult:
        model_ids = model_ids or ["default"]
        results: dict[str, float] = {}
        max_failure_rate = 0.0

        for model_id in model_ids:
            sr = eval_fn(candidate, model_id, self.n_trials)
            failure_rate = 1.0 - sr
            results[model_id] = failure_rate
            max_failure_rate = max(max_failure_rate, failure_rate)

        if max_failure_rate >= self.min_failure_rate:
            return GateResult(
                gate_name="ReproducibilityGate",
                outcome=GateOutcome.PASS,
                evidence={"model_failure_rate": results, "max_failure_rate": max_failure_rate},
                message=(
                    f"Max failure rate {max_failure_rate:.3f} >= "
                    f"threshold {self.min_failure_rate}"
                ),
            )
        return GateResult(
            gate_name="ReproducibilityGate",
            outcome=GateOutcome.FAIL,
            evidence={"model_failure_rate": results, "max_failure_rate": max_failure_rate},
            message=(
                f"Failure is not reliably reproducible. "
                f"Max failure rate {max_failure_rate:.3f} < {self.min_failure_rate}."
            ),
        )


# ---------------------------------------------------------------------------
# Gate 3 — Redundancy
# ---------------------------------------------------------------------------

class RedundancyGate:
    """
    Pass if the candidate's instruction embedding has cosine similarity < threshold
    vs. every already-promoted task embedding.

    Archive (soft reject) if the candidate is too similar to an existing task.
    """

    def __init__(self, similarity_threshold: float = 0.85) -> None:
        self.similarity_threshold = similarity_threshold

    def evaluate(
        self,
        candidate: Any,
        promoted_embeddings: list[bytes],
        **kwargs: Any,
    ) -> GateResult:
        """
        Parameters
        ----------
        candidate:
            CandidateTask with a .embedding attribute (bytes, float32 vector).
        promoted_embeddings:
            List of float32 embedding bytes for all promoted tasks.
        """
        if not promoted_embeddings:
            return GateResult(
                gate_name="RedundancyGate",
                outcome=GateOutcome.PASS,
                evidence={"max_similarity": 0.0, "n_promoted": 0},
                message="No promoted tasks yet; no redundancy check needed.",
            )

        candidate_vec = _bytes_to_vec(candidate.embedding or b"")
        if candidate_vec is None:
            return GateResult(
                gate_name="RedundancyGate",
                outcome=GateOutcome.PASS,
                evidence={"max_similarity": None, "n_promoted": len(promoted_embeddings)},
                message="Candidate has no embedding; skipping redundancy check.",
            )

        max_sim = 0.0
        most_similar_idx = -1
        for idx, emb_bytes in enumerate(promoted_embeddings):
            promoted_vec = _bytes_to_vec(emb_bytes)
            if promoted_vec is None:
                continue
            sim = _cosine_similarity(candidate_vec, promoted_vec)
            if sim > max_sim:
                max_sim = sim
                most_similar_idx = idx

        if max_sim >= self.similarity_threshold:
            return GateResult(
                gate_name="RedundancyGate",
                outcome=GateOutcome.ARCHIVE,
                evidence={
                    "max_similarity": max_sim,
                    "most_similar_idx": most_similar_idx,
                    "n_promoted": len(promoted_embeddings),
                },
                message=(
                    f"Candidate is too similar to an existing promoted task "
                    f"(similarity {max_sim:.3f} >= {self.similarity_threshold}). Archiving."
                ),
            )
        return GateResult(
            gate_name="RedundancyGate",
            outcome=GateOutcome.PASS,
            evidence={
                "max_similarity": max_sim,
                "most_similar_idx": most_similar_idx,
                "n_promoted": len(promoted_embeddings),
            },
            message=f"Max similarity {max_sim:.3f} < threshold {self.similarity_threshold}.",
        )


# ---------------------------------------------------------------------------
# Gate 4 — Discriminative Power
# ---------------------------------------------------------------------------

class DiscriminativePowerGate:
    """
    Pass if the best model's SR and worst model's SR differ by >= min_spread.
    Defer (do not fail) if spread is insufficient — perhaps more data is needed.
    """

    def __init__(self, n_trials: int = 20, min_spread: float = 0.2) -> None:
        self.n_trials = n_trials
        self.min_spread = min_spread

    def evaluate(
        self,
        candidate: Any,
        eval_fn: Callable[[Any, str, int], float],
        model_ids: list[str],
        **kwargs: Any,
    ) -> GateResult:
        if len(model_ids) < 2:
            # Cannot compute spread with fewer than 2 models; defer.
            return GateResult(
                gate_name="DiscriminativePowerGate",
                outcome=GateOutcome.DEFER,
                evidence={"model_sr": {}, "spread": None, "model_ids": model_ids},
                message="Need at least 2 models to evaluate discriminative power.",
            )

        results: dict[str, float] = {}
        for model_id in model_ids:
            results[model_id] = eval_fn(candidate, model_id, self.n_trials)

        best_sr = max(results.values())
        worst_sr = min(results.values())
        spread = best_sr - worst_sr

        if spread >= self.min_spread:
            return GateResult(
                gate_name="DiscriminativePowerGate",
                outcome=GateOutcome.PASS,
                evidence={"model_sr": results, "spread": spread, "best_sr": best_sr, "worst_sr": worst_sr},
                message=f"SR spread {spread:.3f} >= threshold {self.min_spread}.",
            )
        return GateResult(
            gate_name="DiscriminativePowerGate",
            outcome=GateOutcome.DEFER,
            evidence={"model_sr": results, "spread": spread, "best_sr": best_sr, "worst_sr": worst_sr},
            message=(
                f"SR spread {spread:.3f} < {self.min_spread}. "
                "Deferring for additional evaluation."
            ),
        )


# ---------------------------------------------------------------------------
# Gate 5 — Human Review (async, non-blocking)
# ---------------------------------------------------------------------------

class HumanReviewGate:
    """
    Async human review gate.  Enqueues the candidate and immediately returns
    PASS so it does not block the automated pipeline.
    """

    def __init__(
        self,
        queue_path: str = "./eval_memory/human_review_queue.jsonl",
    ) -> None:
        self._queue = HumanReviewQueue(queue_path=queue_path)

    def enqueue(self, candidate: Any) -> None:
        """Delegate directly to HumanReviewQueue.enqueue."""
        self._queue.enqueue(candidate)

    def evaluate(self, candidate: Any, **kwargs: Any) -> GateResult:
        """Enqueue and return PASS immediately (non-blocking)."""
        self._queue.enqueue(candidate)
        return GateResult(
            gate_name="HumanReviewGate",
            outcome=GateOutcome.PASS,
            evidence={"queued": True, "task_id": getattr(candidate, "task_id", None)},
            message="Candidate queued for async human review.",
        )


# ---------------------------------------------------------------------------
# Vector math helpers
# ---------------------------------------------------------------------------

def _bytes_to_vec(data: bytes) -> list[float] | None:
    """Decode a bytes blob as a packed float32 vector."""
    if not data:
        return None
    n_floats = len(data) // 4
    if n_floats == 0:
        return None
    try:
        return list(struct.unpack_from(f"{n_floats}f", data))
    except struct.error:
        return None


def _dot(a: list[float], b: list[float]) -> float:
    """Dot product of two equal-length lists."""
    return sum(x * y for x, y in zip(a, b, strict=False))


def _norm(v: list[float]) -> float:
    """L2 norm."""
    import math
    return math.sqrt(sum(x * x for x in v))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [0, 1]; returns 0.0 for zero vectors."""
    na = _norm(a)
    nb = _norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    # Clamp to handle floating-point rounding past ±1.
    sim = _dot(a, b) / (na * nb)
    return max(0.0, min(1.0, sim))
