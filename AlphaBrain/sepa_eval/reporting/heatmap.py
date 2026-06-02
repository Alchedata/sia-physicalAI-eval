"""
Task x model success-rate heatmap builder and Markdown renderer.
"""
from __future__ import annotations


def build_task_model_heatmap(model_task_results: list[dict]) -> dict:
    """
    Build a task x model SR matrix.

    Parameters
    ----------
    model_task_results:
        List of dicts, each with keys: model_id, task_id, success_rate.

    Returns
    -------
    dict with keys:
        models            list[str]         — sorted unique model IDs
        tasks             list[str]         — sorted unique task IDs
        matrix            list[list[float]] - tasks x models; None where no data
        universal_failures list[str]        — task_ids where ALL models SR < 0.4
        model_specific_weaknesses dict[str, list[str]]
                                            — model_id → task_ids where only that
                                              model fails (SR < 0.4) and others do
                                              not (SR >= 0.4)
    """
    if not model_task_results:
        return {
            "models": [],
            "tasks": [],
            "matrix": [],
            "universal_failures": [],
            "model_specific_weaknesses": {},
        }

    models: list[str] = sorted({r["model_id"] for r in model_task_results})
    tasks: list[str] = sorted({r["task_id"] for r in model_task_results})

    # Build lookup: (task_id, model_id) → success_rate
    lookup: dict[tuple[str, str], float] = {}
    for r in model_task_results:
        lookup[(r["task_id"], r["model_id"])] = float(r["success_rate"])

    # Build matrix (tasks x models); use None for missing pairs.
    matrix: list[list[float | None]] = []
    for task_id in tasks:
        row: list[float | None] = []
        for model_id in models:
            row.append(lookup.get((task_id, model_id)))
        matrix.append(row)

    # Universal failures: all models SR < 0.4 (only tasks with at least one SR value)
    universal_failures: list[str] = []
    for task_id, row in zip(tasks, matrix, strict=False):
        non_none = [v for v in row if v is not None]
        if non_none and all(v < 0.4 for v in non_none):
            universal_failures.append(task_id)

    # Model-specific weaknesses: only *this* model fails (SR < 0.4) while all
    # others have SR >= 0.4.
    model_specific_weaknesses: dict[str, list[str]] = {m: [] for m in models}
    for task_id, row in zip(tasks, matrix, strict=False):
        for mi, model_id in enumerate(models):
            this_sr = row[mi]
            if this_sr is None or this_sr >= 0.4:
                continue
            # Check that all other models have SR >= 0.4 (or no data)
            others_ok = all(
                row[mj] is None or row[mj] >= 0.4
                for mj in range(len(models))
                if mj != mi
            )
            if others_ok:
                model_specific_weaknesses[model_id].append(task_id)

    return {
        "models": models,
        "tasks": tasks,
        "matrix": matrix,
        "universal_failures": universal_failures,
        "model_specific_weaknesses": model_specific_weaknesses,
    }


def render_heatmap_markdown(heatmap: dict) -> str:
    """
    Render a heatmap dict as a Markdown table.

    Color codes:
        SR > 0.8  → ✅
        SR 0.4-0.8 -> ⚠️
        SR < 0.4  → ❌
        No data    → —
    """
    models: list[str] = heatmap.get("models", [])
    tasks: list[str] = heatmap.get("tasks", [])
    matrix: list[list] = heatmap.get("matrix", [])

    if not models or not tasks:
        return "_No data available._"

    def _cell(sr) -> str:
        if sr is None:
            return "—"
        if sr > 0.8:
            return f"✅ {sr:.2f}"
        if sr >= 0.4:
            return f"⚠️ {sr:.2f}"
        return f"❌ {sr:.2f}"

    # Header row
    header = "| Task | " + " | ".join(models) + " |"
    separator = "|------|" + "------|" * len(models)

    rows = [header, separator]
    for task_id, row in zip(tasks, matrix, strict=False):
        cells = " | ".join(_cell(sr) for sr in row)
        rows.append(f"| `{task_id}` | {cells} |")

    # Append summary sections
    rows.append("")
    universal = heatmap.get("universal_failures", [])
    if universal:
        rows.append(
            "**Universal failures** (all models SR < 0.4): "
            + ", ".join(f"`{t}`" for t in universal)
        )

    weaknesses: dict[str, list[str]] = heatmap.get("model_specific_weaknesses", {})
    for model_id, task_ids in weaknesses.items():
        if task_ids:
            rows.append(
                f"**{model_id} only** (model-specific weakness): "
                + ", ".join(f"`{t}`" for t in task_ids)
            )

    return "\n".join(rows)
