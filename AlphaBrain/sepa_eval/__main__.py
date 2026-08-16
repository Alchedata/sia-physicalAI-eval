"""
SEPA-Eval CLI entry point.

Usage:
    python -m sepa_eval <command> [options]

Commands:
    run             Run the full 6-step evolution cycle.
    eval            Register a model and run evaluation.
    promote         Run the promotion pipeline only.
    report          Generate a Markdown eval report.
    export-hard-cases  Export failed episodes for continual learning.
    diff            Compare per-task SR between two model checkpoints.
    sync-models     Sync models.yaml into EvalMemory DB.
    prune           Delete trace files older than N days.
    review list     List candidates pending human review.
    review approve  Approve a candidate task by task_id.
    status          Show EvalMemory stats.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sepa_eval")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_memory_dir(args) -> str:
    """Return memory dir from --memory-dir arg or SEPA_MEMORY_DIR env var."""
    if hasattr(args, "memory_dir") and args.memory_dir:
        return args.memory_dir
    return os.environ.get("SEPA_MEMORY_DIR", "./eval_memory")


def _make_memory(memory_dir: str):
    """Instantiate EvalMemory from the given directory."""
    try:
        from sepa_eval.memory.eval_memory import EvalMemory
    except ImportError as exc:
        print(f"ERROR: Could not import EvalMemory: {exc}", file=sys.stderr)
        sys.exit(1)
    db_path = os.path.join(memory_dir, "eval.db")
    os.makedirs(memory_dir, exist_ok=True)
    return EvalMemory(db_path=db_path, memory_dir=os.path.join(memory_dir, "traces"))


def _load_config(config_path: str | None) -> dict:
    """Load YAML config if provided; return empty dict otherwise."""
    if not config_path:
        return {}
    try:
        import yaml  # type: ignore

        with open(config_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except ImportError:
        logger.warning("PyYAML not installed; ignoring --config %s", config_path)
        return {}
    except OSError as exc:
        logger.warning("Could not read config %s: %s", config_path, exc)
        return {}


def _build_real_eval(args) -> tuple:
    """
    Build (eval_fn, model_ids) from --real-eval / --policy / --models CLI options.

    Returns (None, None) when --real-eval is not set (default behaviour unchanged).
    """
    real_eval = getattr(args, "real_eval", None)
    if not real_eval:
        return None, None
    if real_eval != "libero":
        raise SystemExit(f"Unsupported --real-eval backend '{real_eval}' (only 'libero').")

    try:
        from sepa_eval.evalfn import make_libero_eval_fn, resolve_policy_fn
    except ImportError as exc:
        print(f"ERROR: sepa_eval.evalfn unavailable: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    policy_spec = getattr(args, "policy", None) or "random"
    try:
        policy_fn, default_model_id = resolve_policy_fn(policy_spec)
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: could not build policy '{policy_spec}': {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    eval_fn = make_libero_eval_fn(
        policy_fn,
        max_steps=int(getattr(args, "max_steps", None) or 60),
    )
    models_arg = getattr(args, "models", None)
    model_ids = [m.strip() for m in models_arg.split(",") if m.strip()] if models_arg else [default_model_id]
    logger.info("Real eval enabled: backend=libero policy=%s models=%s", policy_spec, model_ids)
    return eval_fn, model_ids


def _make_gates(memory_dir: str, gate_trials: int | None = None) -> list:
    """Instantiate the five promotion gates, optionally overriding eval-trial counts."""
    from sepa_eval.promotion.gates import (  # type: ignore
        DiscriminativePowerGate,
        HumanReviewGate,
        RedundancyGate,
        ReproducibilityGate,
        SolvabilityGate,
    )

    trial_kwargs = {"n_trials": gate_trials} if gate_trials else {}
    return [
        SolvabilityGate(**trial_kwargs),
        ReproducibilityGate(**trial_kwargs),
        RedundancyGate(),
        DiscriminativePowerGate(**trial_kwargs),
        HumanReviewGate(queue_path=os.path.join(memory_dir, "human_review_queue.jsonl")),
    ]


# ---------------------------------------------------------------------------
# Command: run
# ---------------------------------------------------------------------------


def cmd_run(args) -> int:
    memory_dir = _resolve_memory_dir(args)
    config = _load_config(getattr(args, "config", None))
    memory = _make_memory(memory_dir)

    try:
        from sepa_eval.mining.failure_cluster import FailureClusterer
        from sepa_eval.mining.seed_extractor import SeedExtractor
        from sepa_eval.orchestrator.evolution_loop import EvolutionLoopOrchestrator
        from sepa_eval.reporting.report_generator import ReportGenerator
    except ImportError as exc:
        print(f"ERROR: Missing dependency: {exc}", file=sys.stderr)
        memory.close()
        return 1

    # Optional mutation operators
    mutation_engine = []
    try:
        from sepa_eval.mutation.distractor_add import DistractorAdd
        from sepa_eval.mutation.instruction_paraphrase import InstructionParaphrase
        from sepa_eval.mutation.material_swap import MaterialSwap
        from sepa_eval.mutation.pose_perturbation import PosePerturbation

        mutation_engine = [
            PosePerturbation(),
            DistractorAdd(),
            InstructionParaphrase(),
            MaterialSwap(),
        ]
    except ImportError as exc:
        logger.warning("Some mutation operators could not be imported: %s", exc)

    # Optional promotion pipeline
    promotion_pipeline = None
    try:
        from sepa_eval.promotion.pipeline import PromotionPipeline  # type: ignore

        gates = _make_gates(memory_dir, gate_trials=getattr(args, "gate_trials", None))
        promotion_pipeline = PromotionPipeline(gates=gates)
    except ImportError:
        logger.warning("PromotionPipeline not available; promote step will be skipped.")

    eval_fn, model_ids = _build_real_eval(args)
    if eval_fn is not None:
        # RedundancyGate needs promoted_embeddings; supply an empty default so
        # the real-eval path can reach all gates (tasks table stores no embeddings).
        config.setdefault("gate_kwargs", {}).setdefault("promoted_embeddings", [])
        if promotion_pipeline is not None:
            # Main-thread gate execution: glfw offscreen rendering on macOS
            # crashes (SIGTRAP) when envs are created/stepped from worker threads.
            promotion_pipeline.run_inline = True

    report_generator = ReportGenerator(memory=memory)

    log_path = os.path.join(memory_dir, "evolution_loop_log.jsonl")
    metrics_path = os.path.join(memory_dir, "sepa_eval_metrics.json")

    # Optional critics (heuristic critics are always safe to enable; semantic
    # critic requires a reachable /judge endpoint and is enabled via config).
    critics = {}
    try:
        from sepa_eval.critics.robustness_critic import RobustnessCritic
        from sepa_eval.critics.safety_critic import SafetyCritic

        critics["safety"] = SafetyCritic()
        critics["robustness"] = RobustnessCritic()
    except ImportError as exc:
        logger.warning("Heuristic critics unavailable: %s", exc)
    if config.get("critics", {}).get("semantic", {}).get("enabled"):
        try:
            from sepa_eval.critics.semantic_critic import SemanticCritic

            critics["semantic"] = SemanticCritic(**config["critics"]["semantic"].get("kwargs", {}))
        except ImportError as exc:
            logger.warning("SemanticCritic unavailable: %s", exc)

    orchestrator = EvolutionLoopOrchestrator(
        memory=memory,
        clusterer=FailureClusterer(),
        seed_extractor=SeedExtractor(memory=memory),
        mutation_engine=mutation_engine,
        promotion_pipeline=promotion_pipeline,
        report_generator=report_generator,
        config=config,
        log_path=log_path,
        metrics_path=metrics_path,
        critics=critics,
    )

    print("Running SEPA-Eval evolution cycle...")
    result = orchestrator.run_cycle(eval_fn=eval_fn, model_ids=model_ids)

    print(f"\nCycle ID      : {result.cycle_id}")
    print(f"Steps         : {', '.join(result.steps_completed)}")
    print(f"Candidates    : {result.candidates_generated} generated, {result.candidates_promoted} promoted")
    print(f"Saturated tasks: {result.tasks_saturated}")
    if eval_fn is not None and hasattr(eval_fn, "close"):
        eval_fn.close()
    if result.error:
        print(f"ERROR         : {result.error}", file=sys.stderr)
        return 1

    memory.close()
    return 0


# ---------------------------------------------------------------------------
# Command: eval
# ---------------------------------------------------------------------------


def cmd_eval(args) -> int:
    memory_dir = _resolve_memory_dir(args)
    memory = _make_memory(memory_dir)

    model_id = args.model
    checkpoint = args.checkpoint
    benchmark = args.benchmark

    memory.register_model(
        model_id=model_id,
        framework="alphabrain",
        checkpoint=checkpoint,
        benchmarks=[benchmark],
    )

    print(f"Registered model '{model_id}' (checkpoint={checkpoint}, benchmark={benchmark}).")

    if getattr(args, "trace", False):
        print(
            "Note: --trace flag set. Attach your eval harness to write EpisodeTrace "
            "objects to memory using memory.record_trace()."
        )

    if getattr(args, "ci_mode", False):
        rc = _ci_check(memory, model_id=model_id)
        memory.close()
        return rc

    memory.close()
    return 0


# ---------------------------------------------------------------------------
# Command: promote
# ---------------------------------------------------------------------------


def cmd_promote(args) -> int:
    memory_dir = _resolve_memory_dir(args)
    memory = _make_memory(memory_dir)

    try:
        from sepa_eval.promotion.pipeline import PromotionPipeline  # type: ignore

        gates = _make_gates(memory_dir, gate_trials=getattr(args, "gate_trials", None))
    except ImportError as exc:
        print(
            f"ERROR: PromotionPipeline not available: {exc}\n" "Install optional promotion dependencies.",
            file=sys.stderr,
        )
        memory.close()
        return 1

    # Optional real-simulator eval_fn (--real-eval / --policy).
    eval_fn, model_ids = _build_real_eval(args)

    # run_inline: main-thread gate execution — glfw offscreen rendering on macOS
    # crashes (SIGTRAP) when the env is created/stepped from a worker thread.
    pipeline = PromotionPipeline(gates=gates, run_inline=eval_fn is not None)
    gate_kwargs: dict = {}
    if eval_fn is not None:
        gate_kwargs["eval_fn"] = eval_fn
        gate_kwargs["model_ids"] = model_ids
        # tasks table stores no embeddings; empty list makes RedundancyGate pass.
        gate_kwargs["promoted_embeddings"] = []

    statuses = [s.strip() for s in (getattr(args, "status", None) or "candidate").split(",") if s.strip()]

    # Fetch all candidate tasks (full rows so gates can inspect fields).
    try:
        from sepa_eval.memory.schema import CandidateTask

        placeholders = ",".join("?" for _ in statuses)
        cur = memory._conn.execute(
            f"""
            SELECT task_id, benchmark, instruction, scene_config,
                   mutation_lineage, promotion_status, created_at
            FROM tasks WHERE promotion_status IN ({placeholders})
            """,
            statuses,
        )
        candidates = []
        for row in cur.fetchall():
            # mutation_lineage column stores the mutation_type string
            candidates.append(
                CandidateTask(
                    task_id=row[0],
                    parent_task_id="",
                    benchmark=row[1] or "",
                    instruction=row[2] or "",
                    scene_config=json.loads(row[3]) if row[3] else {},
                    mutation_type=row[4] or "",
                    mutation_params={},
                    promotion_status=row[5] or "candidate",
                    created_at=row[6],
                )
            )
    except Exception as exc:
        print(f"ERROR querying candidates: {exc}", file=sys.stderr)
        memory.close()
        return 1

    if not candidates:
        print(f"No candidates pending promotion (status filter: {', '.join(statuses)}).")
        memory.close()
        return 0

    print(f"Running promotion pipeline on {len(candidates)} candidate(s)...")
    counts: dict[str, int] = {}
    for candidate in candidates:
        try:
            status, evidence = pipeline.run(candidate, **gate_kwargs)
        except Exception as exc:
            print(f"ERROR promoting {candidate.task_id}: {exc}", file=sys.stderr)
            counts["error"] = counts.get("error", 0) + 1
            continue
        try:
            memory.update_task_promotion_status(candidate.task_id, status, evidence)
        except Exception as exc:
            print(f"WARNING: could not persist status for {candidate.task_id}: {exc}", file=sys.stderr)
        counts[status] = counts.get(status, 0) + 1

    summary = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "nothing to do"
    print(f"Promotion results — {summary}")

    if eval_fn is not None and hasattr(eval_fn, "close"):
        eval_fn.close()
    memory.close()
    return 0


# ---------------------------------------------------------------------------
# Command: report
# ---------------------------------------------------------------------------


def cmd_report(args) -> int:
    memory_dir = _resolve_memory_dir(args)
    memory = _make_memory(memory_dir)

    try:
        from sepa_eval.reporting.report_generator import ReportGenerator
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        memory.close()
        return 1

    output_path = args.output
    rg = ReportGenerator(memory=memory)
    content = rg.generate(output_path=output_path)

    print(f"Report written to: {output_path}")
    print(f"Length: {len(content)} characters")
    memory.close()
    return 0


# ---------------------------------------------------------------------------
# Command: export-hard-cases
# ---------------------------------------------------------------------------


def cmd_export_hard_cases(args) -> int:
    memory_dir = _resolve_memory_dir(args)
    memory = _make_memory(memory_dir)

    try:
        from sepa_eval.exporter.hard_case_exporter import HardCaseExporter
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        memory.close()
        return 1

    fmt = getattr(args, "format", "lerobot") or "lerobot"
    exporter = HardCaseExporter(memory=memory, output_format=fmt)

    result = exporter.export(
        output_dir=args.output,
        model_id=getattr(args, "model", None),
    )

    print(f"Exported {result['episodes_exported']} episode(s) to {result['output_dir']} (format={result['format']})")
    memory.close()
    return 0


# ---------------------------------------------------------------------------
# Command: review list
# ---------------------------------------------------------------------------


def cmd_review_list(args) -> int:
    memory_dir = _resolve_memory_dir(args)
    memory = _make_memory(memory_dir)

    try:
        cur = memory._conn.execute(
            """
            SELECT task_id, benchmark, instruction, promotion_status, created_at
            FROM tasks
            WHERE promotion_status = 'candidate'
            ORDER BY created_at DESC
            LIMIT 50
            """
        )
        rows = [dict(r) for r in cur.fetchall()]
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        memory.close()
        return 1

    if not rows:
        print("No candidates pending human review.")
    else:
        print(f"{'task_id':<38} {'benchmark':<20} {'status':<12} instruction")
        print("-" * 100)
        for r in rows:
            instr = (r.get("instruction") or "")[:60]
            print(
                f"{r.get('task_id', ''):<38} "
                f"{r.get('benchmark', ''):<20} "
                f"{r.get('promotion_status', ''):<12} "
                f"{instr}"
            )

    memory.close()
    return 0


# ---------------------------------------------------------------------------
# Command: review approve
# ---------------------------------------------------------------------------


def cmd_review_approve(args) -> int:
    memory_dir = _resolve_memory_dir(args)
    memory = _make_memory(memory_dir)

    task_id = args.task_id
    try:
        memory.update_task_promotion_status(
            task_id=task_id,
            status="promoted",
            evidence={"approved_by": "human", "approved_via": "cli"},
        )
        print(f"Task '{task_id}' approved and promoted.")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        memory.close()
        return 1

    memory.close()
    return 0


# ---------------------------------------------------------------------------
# Command: status
# ---------------------------------------------------------------------------


def cmd_status(args) -> int:
    memory_dir = _resolve_memory_dir(args)
    memory = _make_memory(memory_dir)

    stats: dict = {}
    try:
        for table, label in [
            ("traces", "total_traces"),
            ("tasks", "total_tasks"),
            ("models", "registered_models"),
            ("critic_scores", "critic_score_entries"),
            ("failure_clusters", "failure_clusters"),
        ]:
            cur = memory._conn.execute(f"SELECT COUNT(*) FROM {table}")
            stats[label] = cur.fetchone()[0]

        for status in ("seed", "candidate", "promoted", "rejected", "archived"):
            cur = memory._conn.execute("SELECT COUNT(*) FROM tasks WHERE promotion_status = ?", (status,))
            stats[f"tasks_{status}"] = cur.fetchone()[0]

        cur = memory._conn.execute("SELECT COUNT(*) FROM traces WHERE success = 0")
        stats["failed_traces"] = cur.fetchone()[0]

        cur = memory._conn.execute("SELECT COUNT(*) FROM traces WHERE success = 1")
        stats["successful_traces"] = cur.fetchone()[0]

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        memory.close()
        return 1

    # Check sepa_eval_metrics.json
    metrics_path = os.path.join(memory_dir, "sepa_eval_metrics.json")
    metrics = {}
    if os.path.isfile(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as fh:
                metrics = json.load(fh)
        except Exception:
            pass

    print(f"\nSEPA-Eval Memory Status  ({memory_dir})")
    print("=" * 50)
    for k, v in stats.items():
        print(f"  {k:<30}: {v}")

    if metrics:
        print("\nLast Cycle Metrics:")
        for k, v in metrics.items():
            print(f"  {k:<30}: {v}")

    memory.close()
    return 0


# ---------------------------------------------------------------------------
# CI-check helper (used by cmd_eval --ci-mode and cmd_ci_check)
# ---------------------------------------------------------------------------


def _ci_check(memory, model_id: str | None = None) -> int:
    """
    Return 1 (failure) if:
      - A new failure cluster exists (failure_clusters table non-empty), or
      - Any task shows SR regression vs. the previous eval for the same model.
    Return 0 on success.
    """
    issues: list[str] = []

    # Check for failure clusters
    try:
        cur = memory._conn.execute("SELECT COUNT(*) FROM failure_clusters")
        n_clusters = cur.fetchone()[0]
        if n_clusters > 0:
            issues.append(f"{n_clusters} failure cluster(s) detected in the DB")
    except Exception as exc:
        logger.warning("CI check: could not query failure_clusters: %s", exc)

    # Check for SR regressions (requires model_task_results with ≥2 entries)
    if model_id:
        try:
            cur = memory._conn.execute(
                """
                SELECT task_id, success_rate, last_eval_at
                FROM model_task_results
                WHERE model_id = ?
                ORDER BY last_eval_at DESC
                """,
                (model_id,),
            )
            rows = [dict(r) for r in cur.fetchall()]

            # Group by task_id, compare most recent to previous
            by_task: dict[str, list[dict]] = {}
            for r in rows:
                by_task.setdefault(r["task_id"], []).append(r)

            for task_id, entries in by_task.items():
                if len(entries) >= 2:
                    latest_sr = float(entries[0]["success_rate"] or 0)
                    prev_sr = float(entries[1]["success_rate"] or 0)
                    if latest_sr < prev_sr - 0.05:
                        issues.append(f"Regression on task '{task_id}': " f"SR {prev_sr:.2f} → {latest_sr:.2f}")
        except Exception as exc:
            logger.warning("CI check: could not query model_task_results: %s", exc)

    if issues:
        print("CI FAILURE — the following issues were detected:", file=sys.stderr)
        for issue in issues:
            print(f"  ✗ {issue}", file=sys.stderr)
        return 1

    print("CI PASS — no new failure clusters or regressions detected.")
    return 0


# ---------------------------------------------------------------------------
# Command: diff
# ---------------------------------------------------------------------------


def cmd_diff(args) -> int:
    """Compare per-task success rate between two model checkpoints."""
    memory_dir = _resolve_memory_dir(args)
    memory = _make_memory(memory_dir)

    model_a = args.model_a
    model_b = args.model_b

    try:
        cur = memory._conn.execute(
            """
            SELECT task_id, benchmark, success_rate
            FROM model_task_results
            WHERE model_id = ?
            ORDER BY task_id
            """,
            (model_a,),
        )
        rows_a = {r["task_id"]: dict(r) for r in cur.fetchall()}

        cur = memory._conn.execute(
            """
            SELECT task_id, benchmark, success_rate
            FROM model_task_results
            WHERE model_id = ?
            ORDER BY task_id
            """,
            (model_b,),
        )
        rows_b = {r["task_id"]: dict(r) for r in cur.fetchall()}
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        memory.close()
        return 1

    all_tasks = sorted(set(rows_a) | set(rows_b))
    if not all_tasks:
        print(f"No model_task_results found for '{model_a}' or '{model_b}'.")
        memory.close()
        return 0

    print(f"\nSuccess Rate Comparison: {model_a}  vs  {model_b}")
    print(f"{'task_id':<38} {'benchmark':<20} {model_a:<12} {model_b:<12} {'delta':>8}")
    print("-" * 95)

    regressions = 0
    for task_id in all_tasks:
        ra = rows_a.get(task_id)
        rb = rows_b.get(task_id)
        sr_a = ra["success_rate"] if ra else None
        sr_b = rb["success_rate"] if rb else None
        benchmark = (ra or rb or {}).get("benchmark", "")

        if sr_a is not None and sr_b is not None:
            delta = sr_b - sr_a
            flag = "▼" if delta < -0.05 else ("▲" if delta > 0.05 else " ")
            if delta < -0.05:
                regressions += 1
            delta_str = f"{flag}{delta:+.2f}"
        else:
            delta_str = "N/A"

        print(
            f"{task_id:<38} "
            f"{benchmark:<20} "
            f"{(f'{sr_a:.2f}' if sr_a is not None else '—'):<12} "
            f"{(f'{sr_b:.2f}' if sr_b is not None else '—'):<12} "
            f"{delta_str:>8}"
        )

    print()
    if regressions:
        print(f"⚠  {regressions} task(s) regressed (SR drop > 0.05) in {model_b} vs {model_a}")
    else:
        print("✓  No significant regressions detected.")

    memory.close()
    return 1 if regressions and getattr(args, "fail_on_regression", False) else 0


# ---------------------------------------------------------------------------
# Command: sync-models
# ---------------------------------------------------------------------------


def cmd_sync_models(args) -> int:
    """Sync models.yaml into EvalMemory DB."""
    memory_dir = _resolve_memory_dir(args)
    memory = _make_memory(memory_dir)

    yaml_path = getattr(args, "yaml", None)

    try:
        from sepa_eval.registry.models_registry import ModelsRegistry
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        memory.close()
        return 1

    registry_kwargs: dict = {"memory": memory}
    if yaml_path:
        registry_kwargs["yaml_path"] = yaml_path

    registry = ModelsRegistry(**registry_kwargs)
    try:
        count = registry.sync_to_db()
        print(f"Synced {count} model(s) from YAML to DB.")
    except Exception as exc:
        print(f"ERROR syncing models: {exc}", file=sys.stderr)
        memory.close()
        return 1

    memory.close()
    return 0


# ---------------------------------------------------------------------------
# Command: prune
# ---------------------------------------------------------------------------


def cmd_prune(args) -> int:
    """Delete trace files older than retention_days. DB rows are kept."""
    memory_dir = _resolve_memory_dir(args)
    memory = _make_memory(memory_dir)

    retention_days = getattr(args, "retention_days", 90)
    deleted = memory.prune(retention_days=retention_days)
    print(f"Pruned {deleted} trace file(s) older than {retention_days} days.")

    memory.close()
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _add_real_eval_args(p: argparse.ArgumentParser) -> None:
    """Options shared by `run` and `promote` for real-simulator gate evaluation."""
    p.add_argument(
        "--real-eval",
        dest="real_eval",
        choices=["libero"],
        default=None,
        help="Evaluate promotion gates in a real simulator (currently: libero).",
    )
    p.add_argument(
        "--policy",
        default="random",
        metavar="SPEC",
        help="Policy for real eval: 'random' (default), 'model[:base_vlm_path]' or 'ws:<uri>'.",
    )
    p.add_argument(
        "--models",
        default=None,
        metavar="IDS",
        help="Comma-separated model ids to report gate SR under (default: policy-derived id).",
    )
    p.add_argument(
        "--gate-trials",
        dest="gate_trials",
        type=int,
        default=None,
        metavar="N",
        help="Override n_trials for Solvability/Reproducibility/DiscriminativePower gates.",
    )
    p.add_argument(
        "--max-steps",
        dest="max_steps",
        type=int,
        default=60,
        metavar="N",
        help="Max env steps per real-eval episode (default: 60).",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sepa_eval",
        description="SEPA-Eval: Self-Evolving Physical AI Evaluation System",
    )
    parser.add_argument(
        "--memory-dir",
        metavar="PATH",
        help="EvalMemory directory (default: SEPA_MEMORY_DIR env var or ./eval_memory)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # -- run ------------------------------------------------------------------
    p_run = sub.add_parser("run", help="Run the full 6-step evolution cycle.")
    p_run.add_argument("--config", metavar="PATH", help="Path to orchestrator.yaml")
    _add_real_eval_args(p_run)
    p_run.set_defaults(func=cmd_run)

    # -- eval -----------------------------------------------------------------
    p_eval = sub.add_parser("eval", help="Register a model and run evaluation.")
    p_eval.add_argument("--model", required=True, metavar="MODEL_ID")
    p_eval.add_argument("--checkpoint", required=True, metavar="PATH")
    p_eval.add_argument("--benchmark", required=True, metavar="NAME")
    p_eval.add_argument("--trace", action="store_true", help="Enable trace capture.")
    p_eval.add_argument(
        "--ci-mode",
        dest="ci_mode",
        action="store_true",
        help="Exit 1 if new failure clusters or SR regressions are detected.",
    )
    p_eval.add_argument("--config", metavar="PATH")
    p_eval.set_defaults(func=cmd_eval)

    # -- promote --------------------------------------------------------------
    p_promote = sub.add_parser("promote", help="Run promotion pipeline only.")
    p_promote.add_argument("--config", metavar="PATH")
    p_promote.add_argument(
        "--status",
        default="candidate",
        metavar="LIST",
        help="Comma-separated promotion_status values to re-run (default: candidate). "
        "Use 'candidate,deferred' to retry deferred tasks.",
    )
    _add_real_eval_args(p_promote)
    p_promote.set_defaults(func=cmd_promote)

    # -- report ---------------------------------------------------------------
    p_report = sub.add_parser("report", help="Generate a Markdown eval report.")
    p_report.add_argument("--output", required=True, metavar="PATH")
    p_report.add_argument("--config", metavar="PATH")
    p_report.set_defaults(func=cmd_report)

    # -- export-hard-cases ----------------------------------------------------
    p_export = sub.add_parser(
        "export-hard-cases",
        help="Export failed episodes for continual learning.",
    )
    p_export.add_argument("--model", metavar="MODEL_ID", default=None)
    p_export.add_argument("--output", required=True, metavar="PATH")
    p_export.add_argument(
        "--format",
        choices=["lerobot", "jsonl"],
        default="lerobot",
        metavar="FORMAT",
        help="Output format: lerobot (default) or jsonl.",
    )
    p_export.set_defaults(func=cmd_export_hard_cases)

    # -- review ---------------------------------------------------------------
    p_review = sub.add_parser("review", help="Human review commands.")
    review_sub = p_review.add_subparsers(dest="review_command", required=True)

    p_review_list = review_sub.add_parser("list", help="List pending human reviews.")
    p_review_list.set_defaults(func=cmd_review_list)

    p_review_approve = review_sub.add_parser("approve", help="Approve a candidate.")
    p_review_approve.add_argument("task_id", metavar="TASK_ID")
    p_review_approve.set_defaults(func=cmd_review_approve)

    # -- diff -----------------------------------------------------------------
    p_diff = sub.add_parser(
        "diff",
        help="Compare per-task success rate between two model checkpoints.",
    )
    p_diff.add_argument("model_a", metavar="MODEL_A")
    p_diff.add_argument("model_b", metavar="MODEL_B")
    p_diff.add_argument(
        "--fail-on-regression",
        dest="fail_on_regression",
        action="store_true",
        help="Exit 1 if any task shows SR regression > 0.05.",
    )
    p_diff.set_defaults(func=cmd_diff)

    # -- sync-models ----------------------------------------------------------
    p_sync = sub.add_parser(
        "sync-models",
        help="Sync models.yaml into EvalMemory DB.",
    )
    p_sync.add_argument("--yaml", metavar="PATH", help="Path to models.yaml")
    p_sync.set_defaults(func=cmd_sync_models)

    # -- prune ----------------------------------------------------------------
    p_prune = sub.add_parser(
        "prune",
        help="Delete trace files older than retention_days. DB rows are kept.",
    )
    p_prune.add_argument(
        "--retention-days",
        dest="retention_days",
        type=int,
        default=90,
        metavar="N",
        help="Delete trace files older than N days (default: 90).",
    )
    p_prune.set_defaults(func=cmd_prune)

    # -- status ---------------------------------------------------------------
    p_status = sub.add_parser("status", help="Show EvalMemory stats.")
    p_status.set_defaults(func=cmd_status)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
