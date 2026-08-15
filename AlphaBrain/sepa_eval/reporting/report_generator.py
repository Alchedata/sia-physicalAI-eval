"""
ReportGenerator: produce a Markdown eval report from EvalMemory data.

Uses Jinja2 when available; falls back to simple string formatting.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from sepa_eval.reporting.heatmap import build_task_model_heatmap, render_heatmap_markdown

logger = logging.getLogger(__name__)

_DEFAULT_TEMPLATE_DIR = Path(__file__).parent / "templates"
_TEMPLATE_NAME = "eval_report.md.jinja"


class ReportGenerator:
    """
    Generate a Markdown SEPA-Eval report.

    Sections:
      1. Capability Frontier — per-benchmark SR table per model
      2. Saturation Map — tasks by saturation level
      3. Failure Taxonomy — failure_type distribution per model
      4. Evolved Task Summary — seed → candidate → promoted counts
      5. Cross-Model Failure Heatmap
    """

    def __init__(self, memory, template_dir: str | None = None) -> None:
        self._memory = memory
        self._template_dir = Path(template_dir) if template_dir else _DEFAULT_TEMPLATE_DIR

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, output_path: str, cycle_result=None) -> str:
        """
        Generate a Markdown report, write it to output_path, and return the
        content string.
        """
        # Gather data
        capability_frontier = self._get_capability_frontier()
        failure_taxonomy = self._get_failure_taxonomy()
        saturation_map = self._get_saturation_map()

        # Task summary (seeds / candidates / promoted)
        seeds_analyzed, candidates_generated, tasks_promoted = self._get_task_summary()
        if cycle_result is not None:
            if hasattr(cycle_result, "candidates_generated"):
                candidates_generated = cycle_result.candidates_generated
            if hasattr(cycle_result, "candidates_promoted"):
                tasks_promoted = cycle_result.candidates_promoted

        # Heatmap
        heatmap_data = self._get_heatmap_data()
        heatmap = build_task_model_heatmap(heatmap_data)
        heatmap_table = render_heatmap_markdown(heatmap)

        # Render tables
        capability_table = _render_capability_table(capability_frontier)
        saturation_table = _render_saturation_table(saturation_map)
        failure_taxonomy_text = _render_failure_taxonomy(failure_taxonomy)

        cycle_date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        generated_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        context = {
            "cycle_date": cycle_date,
            "generated_at": generated_at,
            "capability_table": capability_table,
            "saturation_table": saturation_table,
            "failure_taxonomy": failure_taxonomy_text,
            "seeds_analyzed": seeds_analyzed,
            "candidates_generated": candidates_generated,
            "tasks_promoted": tasks_promoted,
            "heatmap_table": heatmap_table,
        }

        content = self._render(context)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(content, encoding="utf-8")
        logger.info("Report written to %s", out)
        return content

    # ------------------------------------------------------------------
    # Data gathering helpers
    # ------------------------------------------------------------------

    def _query(self, sql: str, params: tuple = ()) -> list[dict]:
        """Run a read-only query through the memory's public query API.

        Prefers ``EvalMemory.query()``; falls back to a raw connection only
        for legacy duck-typed memory objects that predate the read-only API.
        """
        query = getattr(self._memory, "query", None)
        if callable(query):
            return query(sql, params)
        cur = self._memory._conn.execute(sql, params)  # legacy fallback
        return [dict(row) for row in cur.fetchall()]

    def _get_capability_frontier(self) -> list[dict]:
        """Return rows from model_task_results for the capability frontier table."""
        try:
            return self._query(
                """
                SELECT model_id, benchmark, task_id,
                       success_rate, clean_success_rate, n_trials, last_eval_at
                FROM model_task_results
                ORDER BY benchmark, model_id, task_id
                """
            )
        except Exception as exc:
            logger.warning("Could not query model_task_results: %s", exc)
            return []

    def _get_failure_taxonomy(self) -> dict:
        """Return failure_type distribution per model: {model_id: {failure_type: count}}."""
        try:
            rows = self._query(
                """
                SELECT model_id, failure_type, COUNT(*) as cnt
                FROM traces
                WHERE success = 0 AND failure_type IS NOT NULL
                GROUP BY model_id, failure_type
                ORDER BY model_id, cnt DESC
                """
            )
            taxonomy: dict[str, dict[str, int]] = {}
            for row in rows:
                model = row["model_id"]
                ft = row["failure_type"]
                taxonomy.setdefault(model, {})[ft] = row["cnt"]
            return taxonomy
        except Exception as exc:
            logger.warning("Could not query failure taxonomy: %s", exc)
            return {}

    def _get_saturation_map(self) -> list[dict]:
        """Return tasks with their saturation status."""
        try:
            return self._query(
                """
                SELECT task_id, benchmark, promotion_status,
                       discriminative_power, saturation_flag
                FROM tasks
                ORDER BY saturation_flag DESC, discriminative_power ASC
                """
            )
        except Exception as exc:
            logger.warning("Could not query tasks table: %s", exc)
            return []

    def _get_task_summary(self) -> tuple[int, int, int]:
        """Return (seeds_analyzed, candidates_generated, tasks_promoted)."""
        try:
            rows = self._query(
                """
                SELECT promotion_status, COUNT(*) as cnt
                FROM tasks
                GROUP BY promotion_status
                """
            )
            counts: dict[str, int] = {}
            for row in rows:
                counts[row["promotion_status"]] = row["cnt"]

            seeds = counts.get("seed", 0)
            candidates = sum(counts.values())
            promoted = counts.get("promoted", 0)
            return seeds, candidates, promoted
        except Exception as exc:
            logger.warning("Could not compute task summary: %s", exc)
            return 0, 0, 0

    def _get_heatmap_data(self) -> list[dict]:
        """Return [{model_id, task_id, success_rate}] from model_task_results."""
        try:
            return self._query(
                """
                SELECT model_id, task_id, success_rate
                FROM model_task_results
                WHERE success_rate IS NOT NULL
                """
            )
        except Exception as exc:
            logger.warning("Could not query heatmap data: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self, context: dict) -> str:
        """Render using Jinja2 if available, else fall back to string formatting."""
        try:
            from jinja2 import Environment, FileSystemLoader  # type: ignore

            env = Environment(
                loader=FileSystemLoader(str(self._template_dir)),
                autoescape=False,
            )
            template = env.get_template(_TEMPLATE_NAME)
            return template.render(**context)
        except ImportError:
            logger.debug("Jinja2 not available; using built-in string formatting.")
        except Exception as exc:
            logger.warning("Jinja2 rendering failed (%s); falling back.", exc)

        return _fallback_render(context)


# ---------------------------------------------------------------------------
# Table renderers
# ---------------------------------------------------------------------------


def _render_capability_table(rows: list[dict]) -> str:
    if not rows:
        return "_No capability data available._"

    header = "| Model | Benchmark | Task | SR | Clean SR | Trials |"
    sep = "|-------|-----------|------|----|----------|--------|"
    lines = [header, sep]
    for r in rows:
        sr = f"{r.get('success_rate', 0.0):.3f}" if r.get("success_rate") is not None else "—"
        csr = f"{r.get('clean_success_rate', 0.0):.3f}" if r.get("clean_success_rate") is not None else "—"
        lines.append(
            f"| {r.get('model_id', '')} | {r.get('benchmark', '')} "
            f"| {r.get('task_id', '')} | {sr} | {csr} | {r.get('n_trials', '')} |"
        )
    return "\n".join(lines)


def _render_saturation_table(rows: list[dict]) -> str:
    if not rows:
        return "_No task data available._"

    header = "| Task | Benchmark | Status | Disc. Power | Saturated |"
    sep = "|------|-----------|--------|-------------|-----------|"
    lines = [header, sep]
    for r in rows:
        dp = f"{r.get('discriminative_power', 0.0):.3f}" if r.get("discriminative_power") is not None else "—"
        sat = "Yes" if r.get("saturation_flag") else "No"
        lines.append(
            f"| {r.get('task_id', '')} | {r.get('benchmark', '')} " f"| {r.get('promotion_status', '')} | {dp} | {sat} |"
        )
    return "\n".join(lines)


def _render_failure_taxonomy(taxonomy: dict) -> str:
    if not taxonomy:
        return "_No failure data available._"

    lines = []
    for model_id, ftypes in sorted(taxonomy.items()):
        lines.append(f"**{model_id}**")
        for ft, cnt in sorted(ftypes.items(), key=lambda x: -x[1]):
            lines.append(f"  - {ft}: {cnt}")
    return "\n".join(lines)


def _fallback_render(context: dict) -> str:
    """Pure-Python fallback report when Jinja2 is not installed."""
    return (
        f"# SEPA-Eval Report — {context['cycle_date']}\n\n"
        f"## 1. Capability Frontier\n{context['capability_table']}\n\n"
        f"## 2. Saturation Map\n{context['saturation_table']}\n\n"
        f"## 3. Failure Taxonomy\n{context['failure_taxonomy']}\n\n"
        f"## 4. Evolved Task Summary\n"
        f"- Seeds analyzed: {context['seeds_analyzed']}\n"
        f"- Candidates generated: {context['candidates_generated']}\n"
        f"- Tasks promoted: {context['tasks_promoted']}\n\n"
        f"## 5. Cross-Model Failure Heatmap\n{context['heatmap_table']}\n\n"
        f"---\n*Generated by SEPA-Eval on {context['generated_at']}*\n"
    )
