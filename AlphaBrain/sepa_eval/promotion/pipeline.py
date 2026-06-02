"""
PromotionPipeline — orchestrates the five SEPA-Eval promotion gates.

Gate execution order:
  1. SolvabilityGate
  2. ReproducibilityGate
  3. RedundancyGate
  4. DiscriminativePowerGate
  5. HumanReviewGate

Each gate runs in a ThreadPoolExecutor with a configurable timeout.
The pipeline stops early on DISCARD, ARCHIVE, or timeout (deferred).
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any

from sepa_eval.promotion.gates import GateOutcome, GateResult

logger = logging.getLogger(__name__)


class PromotionPipeline:
    """
    Runs promotion gates sequentially (each submitted to a thread pool for
    timeout enforcement) and collects evidence.

    Parameters
    ----------
    gates:
        Ordered list of gate objects.  Each must expose
        ``evaluate(candidate, **gate_kwargs) -> GateResult``.
    max_parallel_eval_workers:
        Maximum worker threads for the internal ThreadPoolExecutor.
    gate_timeout_minutes:
        Per-gate wall-clock timeout in minutes.  Fractional values are accepted
        (e.g. 0.001 for unit tests).
    """

    def __init__(
        self,
        gates: list[Any],
        max_parallel_eval_workers: int = 4,
        gate_timeout_minutes: float = 120,
    ) -> None:
        self.gates = gates
        self.max_parallel_eval_workers = max_parallel_eval_workers
        self.gate_timeout_minutes = gate_timeout_minutes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        candidate: Any,
        **gate_kwargs: Any,
    ) -> tuple[str, dict]:
        """
        Run all gates in order for *candidate*.

        Parameters
        ----------
        candidate:
            A CandidateTask (or any object with ``task_id``, ``instruction``, etc.)
        **gate_kwargs:
            Passed through to every gate's ``evaluate`` call.  Common keys:
              - eval_fn        (Callable)
              - model_ids      (list[str])
              - promoted_embeddings (list[bytes])

        Returns
        -------
        (final_status, evidence_dict)

        final_status one of:
          "promoted"  — all gates passed
          "rejected"  — a DISCARD or FAIL outcome was returned by any gate
          "archived"  — a ARCHIVE outcome was returned by any gate
          "deferred"  — a gate timed out or returned DEFER
        evidence_dict:
          {gate_name: gate_evidence_dict}  — full audit trail
        """
        evidence: dict[str, Any] = {}
        timeout_seconds = self.gate_timeout_minutes * 60

        with ThreadPoolExecutor(max_workers=self.max_parallel_eval_workers) as executor:
            for gate in self.gates:
                gate_name = type(gate).__name__

                future = executor.submit(gate.evaluate, candidate, **gate_kwargs)
                try:
                    result: GateResult = future.result(timeout=timeout_seconds)
                except FuturesTimeout:
                    logger.warning(
                        "Gate %s timed out after %.2f minutes for task %s.",
                        gate_name,
                        self.gate_timeout_minutes,
                        getattr(candidate, "task_id", "?"),
                    )
                    evidence[gate_name] = {"timeout": True}
                    return "deferred", evidence
                except Exception as exc:
                    logger.error(
                        "Gate %s raised an unexpected exception: %s", gate_name, exc
                    )
                    evidence[gate_name] = {"error": str(exc)}
                    return "deferred", evidence

                evidence[gate_name] = result.evidence

                if result.outcome == GateOutcome.DISCARD:
                    logger.info(
                        "Gate %s DISCARDED task %s: %s",
                        gate_name,
                        getattr(candidate, "task_id", "?"),
                        result.message,
                    )
                    return "rejected", evidence

                if result.outcome == GateOutcome.ARCHIVE:
                    logger.info(
                        "Gate %s ARCHIVED task %s: %s",
                        gate_name,
                        getattr(candidate, "task_id", "?"),
                        result.message,
                    )
                    return "archived", evidence

                if result.outcome == GateOutcome.DEFER:
                    logger.info(
                        "Gate %s DEFERRED task %s: %s",
                        gate_name,
                        getattr(candidate, "task_id", "?"),
                        result.message,
                    )
                    return "deferred", evidence

                if result.outcome == GateOutcome.FAIL:
                    logger.info(
                        "Gate %s FAILED task %s: %s",
                        gate_name,
                        getattr(candidate, "task_id", "?"),
                        result.message,
                    )
                    return "rejected", evidence

                # GateOutcome.PASS — continue to next gate.
                logger.debug(
                    "Gate %s PASSED for task %s.",
                    gate_name,
                    getattr(candidate, "task_id", "?"),
                )

        return "promoted", evidence
