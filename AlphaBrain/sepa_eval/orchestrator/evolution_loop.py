"""
SEPA-Eval 6-step evolution loop orchestrator.

Evaluate → Diagnose → Generate → Validate → Monitor → Report
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(tzinfo=None).isoformat()


def _extract_frames(trace: Any) -> list:
    """Pull per-step camera images out of an EpisodeTrace (best effort)."""
    frames: list = []
    rollout = getattr(trace, "rollout", None)
    for obs in getattr(rollout, "observations", None) or []:
        images = obs.get("images") if isinstance(obs, dict) else None
        if images:
            # take the first camera view per step
            frames.append(next(iter(images.values())))
    return frames


@dataclass
class EvolutionCycleResult:
    cycle_id: str
    started_at: datetime
    finished_at: datetime | None = None
    steps_completed: list[str] = field(default_factory=list)
    candidates_generated: int = 0
    candidates_promoted: int = 0
    tasks_saturated: int = 0
    error: str | None = None


class EvolutionLoopOrchestrator:
    """
    Runs the full Evaluate → Diagnose → Generate → Validate → Monitor → Report
    evolution cycle for SEPA-Eval.
    """

    def __init__(
        self,
        memory,                         # EvalMemory
        clusterer,                      # FailureClusterer
        seed_extractor,                 # SeedExtractor
        mutation_engine,                # list[MutationOperator]
        promotion_pipeline,             # PromotionPipeline
        report_generator,               # ReportGenerator
        config: dict | None = None,
        log_path: str = "./eval_memory/evolution_loop_log.jsonl",
        metrics_path: str = "./eval_memory/sepa_eval_metrics.json",
        critics: dict | None = None,    # {"safety": SafetyCritic, "robustness": RobustnessCritic, "semantic": ...}
    ) -> None:
        self._memory = memory
        self._critics = critics or {}
        self._last_critic_latency_ms: float = 0.0
        self._clusterer = clusterer
        self._seed_extractor = seed_extractor
        self._mutation_engine = mutation_engine if mutation_engine is not None else []
        self._promotion_pipeline = promotion_pipeline
        self._report_generator = report_generator
        self._config = config or {}
        self._log_path = Path(log_path)
        self._metrics_path = Path(metrics_path)

        # Ensure log directories exist
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._metrics_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_cycle(
        self,
        eval_fn: Callable | None = None,
        model_ids: list[str] | None = None,
        max_candidates: int | None = None,
    ) -> EvolutionCycleResult:
        """
        Execute one full SEPA-Eval evolution cycle.

        Step 1 EVALUATE: optionally invoke eval_fn; traces may already be in memory.
        Step 2 DIAGNOSE: cluster recent failure traces via FailureClusterer.
        Step 3 GENERATE: extract seeds and run mutation operators to produce candidates.
        Step 4 VALIDATE+PROMOTE: run PromotionPipeline on each candidate.
        Step 5 MONITOR: check saturation across promoted tasks.
        Step 6 REPORT: generate the eval report.

        Returns EvolutionCycleResult with per-step telemetry.
        """
        cycle_id = str(uuid.uuid4())[:8]
        started_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        result = EvolutionCycleResult(cycle_id=cycle_id, started_at=started_at)

        cfg_orch = self._config.get("orchestrator", {})
        saturation_threshold = cfg_orch.get("saturation_threshold", 0.95)
        if max_candidates is None:
            max_candidates = int(cfg_orch.get("max_candidates_per_cycle", 50) or 50)

        logger.info("SEPA-Eval evolution cycle %s started", cycle_id)

        try:
            # ----------------------------------------------------------
            # Step 1: EVALUATE
            # ----------------------------------------------------------
            self._log_step(cycle_id, "evaluate", "started")
            traces_written = 0

            if eval_fn is not None and model_ids:
                # Caller provides an eval function; invoke it for each model.
                for model_id in model_ids:
                    try:
                        n_trials = cfg_orch.get("n_trials_per_task", 5)
                        score = eval_fn(model_id=model_id, n_trials=n_trials)
                        traces_written += 1
                        logger.debug("eval_fn returned %.3f for model %s", score, model_id)
                    except Exception as exc:
                        logger.warning("eval_fn raised for model %s: %s", model_id, exc)

            self._log_step(cycle_id, "evaluate", "completed", traces_written=traces_written)
            result.steps_completed.append("evaluate")

            # ----------------------------------------------------------
            # Step 2: DIAGNOSE
            # ----------------------------------------------------------
            self._log_step(cycle_id, "diagnose", "started")
            cluster_window = self._config.get("failure_mining", {}).get(
                "cluster_window", {}
            )
            last_n_runs = cluster_window.get("last_n_runs", 5)

            trace_rows = self._memory.get_failures_by_cluster_window(
                last_n_runs=last_n_runs
            )
            clusters = []
            if trace_rows:
                try:
                    clusters = self._clusterer.cluster(trace_rows)
                except Exception as exc:
                    logger.warning("FailureClusterer.cluster() failed: %s", exc)

            self._log_step(
                cycle_id,
                "diagnose",
                "completed",
                trace_rows=len(trace_rows),
                clusters_found=len(clusters),
            )
            result.steps_completed.append("diagnose")

            # ----------------------------------------------------------
            # Step 3: GENERATE
            # ----------------------------------------------------------
            self._log_step(cycle_id, "generate", "started")
            all_candidates = []

            for cluster in clusters:
                seed = self._seed_extractor.extract(cluster, trace_rows)

                for operator in self._mutation_engine:
                    if len(all_candidates) >= max_candidates:
                        break
                    try:
                        new_candidates = operator.generate(
                            seed_scene_config=seed.scene_config,
                            seed_instruction=seed.task_instruction,
                            parent_task_id=seed.trace_id,
                            benchmark=seed.benchmark,
                        )
                        for candidate in new_candidates:
                            if len(all_candidates) >= max_candidates:
                                break
                            self._memory.record_candidate_task(candidate)
                            all_candidates.append(candidate)
                    except Exception as exc:
                        logger.warning(
                            "Operator %s raised on seed %s: %s",
                            operator.name,
                            seed.trace_id,
                            exc,
                        )

                if len(all_candidates) >= max_candidates:
                    break

            result.candidates_generated = len(all_candidates)
            self._log_step(
                cycle_id,
                "generate",
                "completed",
                candidates_generated=len(all_candidates),
            )
            result.steps_completed.append("generate")

            # ----------------------------------------------------------
            # Step 4: VALIDATE + PROMOTE
            # ----------------------------------------------------------
            self._log_step(cycle_id, "validate_promote", "started")
            promoted_count = 0

            # --- Critics: score failure traces before gate evaluation -----
            self._last_critic_latency_ms = self._run_critics(cycle_id, trace_rows)

            if self._promotion_pipeline is not None and all_candidates:
                gate_kwargs: dict[str, Any] = {}
                if eval_fn is not None:
                    gate_kwargs["eval_fn"] = eval_fn
                if model_ids is not None:
                    gate_kwargs["model_ids"] = model_ids
                gate_kwargs.update(self._config.get("gate_kwargs", {}))

                for candidate in all_candidates:
                    task_id = getattr(candidate, "task_id", "?")
                    try:
                        status, evidence = self._promotion_pipeline.run(candidate, **gate_kwargs)
                    except Exception as exc:
                        logger.error("PromotionPipeline.run() raised for task %s: %s", task_id, exc)
                        self._log_step(
                            cycle_id, "validate_promote", "error", task_id=task_id, error=str(exc)
                        )
                        continue
                    try:
                        self._memory.update_task_promotion_status(task_id, status, evidence)
                    except Exception as exc:
                        logger.warning("Could not persist promotion status for %s: %s", task_id, exc)
                    if status == "promoted":
                        promoted_count += 1

            result.candidates_promoted = promoted_count
            self._log_step(
                cycle_id,
                "validate_promote",
                "completed",
                candidates_promoted=promoted_count,
            )
            result.steps_completed.append("validate_promote")

            # ----------------------------------------------------------
            # Step 5: MONITOR (saturation check)
            # ----------------------------------------------------------
            self._log_step(cycle_id, "monitor", "started")
            saturated = self._check_saturation(threshold=saturation_threshold)
            result.tasks_saturated = saturated
            self._log_step(
                cycle_id,
                "monitor",
                "completed",
                tasks_saturated=saturated,
                saturation_threshold=saturation_threshold,
            )
            result.steps_completed.append("monitor")

            # ----------------------------------------------------------
            # Step 6: REPORT
            # ----------------------------------------------------------
            self._log_step(cycle_id, "report", "started")
            if self._report_generator is not None:
                try:
                    report_path = (
                        self._metrics_path.parent / f"report_{cycle_id}.md"
                    )
                    self._report_generator.generate(
                        output_path=str(report_path),
                        cycle_result=result,
                    )
                    logger.info("Report written to %s", report_path)
                except Exception as exc:
                    logger.warning("ReportGenerator.generate() raised: %s", exc)

            self._log_step(cycle_id, "report", "completed")
            result.steps_completed.append("report")

        except Exception as exc:
            result.error = str(exc)
            self._log_step(cycle_id, "cycle", "error", error=str(exc))
            logger.error("Evolution cycle %s failed: %s", cycle_id, exc)

        finally:
            result.finished_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)
            self._write_metrics(result)
            logger.info(
                "Evolution cycle %s finished — %d candidates, %d promoted, %d saturated",
                cycle_id,
                result.candidates_generated,
                result.candidates_promoted,
                result.tasks_saturated,
            )

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_step(self, cycle_id: str, step: str, status: str, **kwargs: Any) -> None:
        """Append one JSON line to evolution_loop_log.jsonl."""
        record = {
            "cycle_id": cycle_id,
            "step": step,
            "status": status,
            "timestamp": _now_iso(),
            **kwargs,
        }
        try:
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError as exc:
            logger.warning("Could not write evolution log: %s", exc)

    def _write_metrics(self, result: EvolutionCycleResult) -> None:
        """Write sepa_eval_metrics.json with system-level metrics."""
        # Compute aggregate trace count from DB.
        try:
            cur = self._memory._conn.execute("SELECT COUNT(*) FROM traces")
            traces_written = cur.fetchone()[0]
        except Exception:
            traces_written = 0

        # Compute aggregate candidate count.
        try:
            cur = self._memory._conn.execute("SELECT COUNT(*) FROM tasks")
            candidates_total = cur.fetchone()[0]
        except Exception:
            candidates_total = result.candidates_generated

        promotion_yield = (
            result.candidates_promoted / result.candidates_generated
            if result.candidates_generated > 0
            else 0.0
        )

        # Gate timeout count: count log lines with status=error for gate steps.
        gate_timeout_count = 0
        try:
            if self._log_path.exists():
                with self._log_path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            entry = json.loads(line)
                            if (
                                entry.get("step") == "validate_promote"
                                and entry.get("status") == "error"
                            ):
                                gate_timeout_count += 1
                        except json.JSONDecodeError:
                            pass
        except OSError:
            pass

        metrics = {
            "traces_written": traces_written,
            "candidates_generated": candidates_total,
            "promotion_yield": round(promotion_yield, 4),
            "critic_latency_ms": round(self._last_critic_latency_ms, 2),
            "gate_timeout_count": gate_timeout_count,
            "last_cycle_at": _now_iso(),
        }

        try:
            with self._metrics_path.open("w", encoding="utf-8") as fh:
                json.dump(metrics, fh, indent=2)
        except OSError as exc:
            logger.warning("Could not write metrics file: %s", exc)

    def _run_critics(self, cycle_id: str, trace_rows: list[dict]) -> float:
        """
        Run configured critics over failure trace rows and persist scores.

        Returns total wall-clock critic latency in milliseconds.  Any critic
        failure is logged and skipped; critics never abort the cycle.
        """
        if not self._critics or not trace_rows:
            return 0.0

        import time

        start = time.perf_counter()

        safety = self._critics.get("safety")
        if safety is not None:
            for row in trace_rows:
                trace_id = row.get("trace_id")
                try:
                    res = safety.evaluate(row)
                    self._memory.update_critic_score(
                        trace_id=trace_id,
                        critic_name="safety",
                        score=float(getattr(res, "score", 0.0)),
                        explanation=str(getattr(res, "explanation", "")),
                        confidence=float(getattr(res, "confidence", 1.0)),
                    )
                except Exception as exc:
                    logger.warning("SafetyCritic failed on trace %s: %s", trace_id, exc)

        robustness = self._critics.get("robustness")
        if robustness is not None:
            try:
                res = robustness.evaluate(trace_rows)
                self._log_step(
                    cycle_id,
                    "validate_promote",
                    "critic_robustness",
                    score=float(getattr(res, "score", 0.0)),
                    explanation=str(getattr(res, "explanation", ""))[:500],
                )
            except Exception as exc:
                logger.warning("RobustnessCritic failed: %s", exc)

        semantic = self._critics.get("semantic")
        if semantic is not None:
            for row in trace_rows:
                trace_id = row.get("trace_id")
                trace_path = row.get("trace_path")
                if not trace_path:
                    continue
                try:
                    trace = self._memory.load_trace_file(trace_path)
                    frames = _extract_frames(trace)
                    if not frames:
                        continue
                    res = semantic.judge(frames=frames, instruction=row.get("task_instruction", ""))
                    self._memory.update_critic_score(
                        trace_id=trace_id,
                        critic_name="semantic",
                        score=float(getattr(res, "score", 0.0)),
                        explanation=str(getattr(res, "explanation", "")),
                        confidence=float(getattr(res, "confidence", 1.0)),
                    )
                except Exception as exc:
                    logger.warning("SemanticCritic failed on trace %s: %s", trace_id, exc)

        return (time.perf_counter() - start) * 1000.0

    def _check_saturation(self, threshold: float = 0.95) -> int:
        """
        Query memory for promoted tasks where all models meet the SR threshold.
        Returns count of saturated tasks.
        """
        try:
            saturated_tasks = self._memory.get_saturated_tasks(threshold=threshold)
            return len(saturated_tasks)
        except Exception as exc:
            logger.warning("Saturation check failed: %s", exc)
            return 0
