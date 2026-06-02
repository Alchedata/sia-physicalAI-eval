"""
RobustnessCritic — aggregates success rates across a variant family.

Given a list of trace rows for all variants of a single task family, the critic:
  1. Groups traces by mutation_type.
  2. Computes per-group success rate (SR).
  3. Identifies the baseline SR (mutation_type == None or "seed").
  4. Flags mutation types where SR dropped below (baseline_sr - sensitivity_threshold).
  5. Reports an overall robustness_score = mean SR across all variant groups.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RobustnessCriticResult:
    """Aggregated robustness assessment across a task family."""

    robustness_score: float                # mean SR across all variant groups
    sensitive_to: list[str]                # mutation types with large SR drops
    variant_breakdown: dict                # {mutation_type: {"sr": float, "n": int}}


class RobustnessCritic:
    """
    Aggregates success rates across a variant family of trace rows.

    Parameters
    ----------
    sensitivity_threshold:
        Minimum SR drop (relative to baseline) that marks a mutation type as
        one the model is "sensitive to".  Default: 0.15.
    """

    # Canonical names for the baseline group (no mutation applied).
    _BASELINE_MUTATION_TYPES: frozenset[str] = frozenset({"seed", "none"})

    def __init__(self, sensitivity_threshold: float = 0.15) -> None:
        self.sensitivity_threshold = sensitivity_threshold

    def evaluate(self, trace_rows: list[dict]) -> RobustnessCriticResult:
        """
        Evaluate robustness across a family of trace rows.

        Parameters
        ----------
        trace_rows:
            List of dicts. Each dict must have at minimum:
              - ``success``       : bool
              - ``mutation_type`` : str | None

        Returns
        -------
        RobustnessCriticResult
        """
        if not trace_rows:
            return RobustnessCriticResult(
                robustness_score=0.0,
                sensitive_to=[],
                variant_breakdown={},
            )

        # ---------------------------------------------------------------
        # 1. Group by mutation_type (normalise None → "none")
        # ---------------------------------------------------------------
        groups: dict[str, list[bool]] = {}
        for row in trace_rows:
            raw_type = row.get("mutation_type")
            mutation_type: str = (
                raw_type.lower() if isinstance(raw_type, str) and raw_type else "none"
            )
            groups.setdefault(mutation_type, []).append(bool(row.get("success", False)))

        # ---------------------------------------------------------------
        # 2. Compute SR per group
        # ---------------------------------------------------------------
        variant_breakdown: dict[str, dict] = {}
        for mutation_type, successes in groups.items():
            n = len(successes)
            sr = sum(successes) / n if n > 0 else 0.0
            variant_breakdown[mutation_type] = {"sr": sr, "n": n}

        # ---------------------------------------------------------------
        # 3. Baseline SR
        # ---------------------------------------------------------------
        baseline_sr = self._compute_baseline_sr(variant_breakdown)

        # ---------------------------------------------------------------
        # 4. sensitive_to: mutation types where SR < baseline - threshold
        # ---------------------------------------------------------------
        drop_threshold = baseline_sr - self.sensitivity_threshold
        sensitive_to: list[str] = [
            mt
            for mt, stats in variant_breakdown.items()
            if mt not in self._BASELINE_MUTATION_TYPES
            and stats["sr"] < drop_threshold
        ]
        # Sort for deterministic output.
        sensitive_to.sort()

        # ---------------------------------------------------------------
        # 5. robustness_score = mean SR across ALL groups
        # ---------------------------------------------------------------
        all_srs = [stats["sr"] for stats in variant_breakdown.values()]
        robustness_score = sum(all_srs) / len(all_srs) if all_srs else 0.0

        return RobustnessCriticResult(
            robustness_score=robustness_score,
            sensitive_to=sensitive_to,
            variant_breakdown=variant_breakdown,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_baseline_sr(self, variant_breakdown: dict[str, dict]) -> float:
        """
        Compute the baseline success rate.

        Prefers the group with mutation_type == "seed"; falls back to "none".
        If neither exists, returns the mean SR of all groups as a proxy.
        """
        for name in ("seed", "none"):
            if name in variant_breakdown:
                return variant_breakdown[name]["sr"]

        # No explicit baseline group; use global mean.
        srs = [v["sr"] for v in variant_breakdown.values()]
        return sum(srs) / len(srs) if srs else 0.0
