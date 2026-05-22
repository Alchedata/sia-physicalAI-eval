"""
Promotion pipeline package for SEPA-Eval.

Exports the five promotion gates and the orchestrating PromotionPipeline,
plus the standalone HumanReviewQueue for audit / async human oversight.
"""
from __future__ import annotations

from sepa_eval.promotion.gates import (
    GateOutcome,
    GateResult,
    SolvabilityGate,
    ReproducibilityGate,
    RedundancyGate,
    DiscriminativePowerGate,
    HumanReviewGate,
)
from sepa_eval.promotion.pipeline import PromotionPipeline
from sepa_eval.promotion.human_review_queue import HumanReviewQueue

__all__ = [
    "GateOutcome",
    "GateResult",
    "SolvabilityGate",
    "ReproducibilityGate",
    "RedundancyGate",
    "DiscriminativePowerGate",
    "HumanReviewGate",
    "PromotionPipeline",
    "HumanReviewQueue",
]
