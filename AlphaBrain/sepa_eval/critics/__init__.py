"""
Critics package for SEPA-Eval.

Provides three critic implementations:
  - SemanticCritic  : VLM-based task-completion judge
  - SafetyCritic    : Heuristic safety analysis over rollout traces
  - RobustnessCritic: Aggregates success rates across variant families
"""
from __future__ import annotations

from sepa_eval.critics.semantic_critic import CriticError, CriticResult, SemanticCritic
from sepa_eval.critics.safety_critic import SafetyCritic, SafetyCriticResult
from sepa_eval.critics.robustness_critic import RobustnessCritic, RobustnessCriticResult

__all__ = [
    "CriticResult",
    "CriticError",
    "SemanticCritic",
    "SafetyCriticResult",
    "SafetyCritic",
    "RobustnessCriticResult",
    "RobustnessCritic",
]
