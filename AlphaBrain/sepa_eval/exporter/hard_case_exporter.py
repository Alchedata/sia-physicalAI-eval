"""
HardCaseExporter: export failed rollout (obs, action) pairs for continual learning.

Supports two output formats:
  - "lerobot": parquet row groups, one per episode; meta/info.json with dataset metadata.
  - "jsonl":   one JSON line per episode.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class HardCaseExporter:
    """
    Export failed episodes from EvalMemory as a dataset consumable by
    AlphaBrain's continual learning pipeline.
    """

    def __init__(self, memory, output_format: str = "lerobot") -> None:
        if output_format not in ("lerobot", "jsonl"):
            raise ValueError(
                f"Unsupported output_format {output_format!r}. "
                "Choose 'lerobot' or 'jsonl'."
            )
        self._memory = memory
        self._output_format = output_format

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export(
        self,
        output_dir: str,
        model_id: str | None = None,
        benchmark: str | None = None,
        max_episodes: int = 1000,
    ) -> dict:
        """
        Query EvalMemory for failed traces and write them to output_dir.

        Parameters
        ----------
        output_dir:
            Destination directory.  Created if it does not exist.
        model_id:
            Optional filter: only export episodes from this model.
        benchmark:
            Optional filter: only export episodes from this benchmark.
        max_episodes:
            Maximum number of episodes to export.

        Returns
        -------
        dict with keys: episodes_exported, output_dir, format.
        Never raises on 0 episodes — returns episodes_exported=0.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        failed_rows = self._memory.get_failures(
            model_id=model_id,
            benchmark=benchmark,
            limit=max_episodes,
        )

        if not failed_rows:
            logger.info("HardCaseExporter: no failed traces found for export.")
            return {
                "episodes_exported": 0,
                "output_dir": str(out),
                "format": self._output_format,
            }

        # Load full trace files and build episode dicts.
        episodes: list[dict] = []
        for row in failed_rows:
            trace_path = row.get("trace_path")
            if not trace_path:
                logger.warning(
                    "trace_path missing for trace_id=%s; skipping.", row.get("trace_id")
                )
                continue
            try:
                trace = self._memory.load_trace_file(trace_path)
                episode = {
                    "episode_id": trace.identity.trace_id,
                    "instruction": trace.identity.task_instruction,
                    "model_id": trace.identity.model_id,
                    "benchmark": trace.identity.benchmark,
                    "task_id": trace.identity.task_id,
                    "steps": [
                        {
                            "obs": _coerce_json(obs),
                            "action": _coerce_json(act),
                        }
                        for obs, act in zip(
                            trace.rollout.observations, trace.rollout.actions
                        )
                    ],
                    "failure_type": trace.labels.failure_type,
                    "failure_step": trace.rollout.failure_step,
                    "episode_length": trace.rollout.episode_length,
                }
                episodes.append(episode)
            except Exception as exc:
                logger.warning(
                    "Could not load trace %s: %s", trace_path, exc
                )

        if not episodes:
            return {
                "episodes_exported": 0,
                "output_dir": str(out),
                "format": self._output_format,
            }

        if self._output_format == "lerobot":
            n_written = self._write_lerobot_format(episodes, out)
        else:
            n_written = self._write_jsonl_format(episodes, out)

        logger.info(
            "HardCaseExporter: exported %d episodes to %s (format=%s)",
            n_written,
            out,
            self._output_format,
        )

        return {
            "episodes_exported": n_written,
            "output_dir": str(out),
            "format": self._output_format,
        }

    # ------------------------------------------------------------------
    # Format writers
    # ------------------------------------------------------------------

    def _write_lerobot_format(self, episodes: list[dict], output_dir: Path) -> int:
        """
        Write episodes as parquet files in LeRobot dataset format.

        Layout:
            {output_dir}/data/episode_{i:06d}.parquet  — one file per episode
            {output_dir}/meta/info.json                — dataset metadata

        Each parquet file has columns: episode_id, step, obs_json, action_json.
        Falls back to JSONL if pyarrow/pandas is not installed.
        """
        data_dir = output_dir / "data"
        meta_dir = output_dir / "meta"
        data_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)

        pa = None
        pq = None
        _have_arrow = False
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore
            _have_arrow = True
        except ImportError:
            logger.warning(
                "pyarrow not installed; falling back to JSONL inside the lerobot "
                "layout (data/*.jsonl)."
            )

        n_written = 0
        for i, episode in enumerate(episodes):
            rows = []
            for step_idx, step in enumerate(episode.get("steps", [])):
                rows.append(
                    {
                        "episode_id": episode["episode_id"],
                        "step": step_idx,
                        "obs_json": json.dumps(step["obs"]),
                        "action_json": json.dumps(step["action"]),
                    }
                )

            if not rows:
                continue

            out_file = data_dir / f"episode_{i:06d}.parquet"
            if _have_arrow and pa is not None and pq is not None:
                table = pa.Table.from_pylist(rows)
                pq.write_table(table, str(out_file))
            else:
                # Graceful fallback: write JSONL with .parquet extension so
                # callers can still detect the file without crashing.
                jsonl_file = data_dir / f"episode_{i:06d}.parquet"
                with jsonl_file.open("w", encoding="utf-8") as fh:
                    for row in rows:
                        fh.write(json.dumps(row) + "\n")

            n_written += 1

        # Write metadata
        info = {
            "format": "lerobot",
            "n_episodes": n_written,
            "episodes": n_written,   # alias used by downstream consumers
            "total_steps": sum(
                len(ep.get("steps", [])) for ep in episodes[:n_written]
            ),
            "columns": ["episode_id", "step", "obs_json", "action_json"],
            "models": list({ep["model_id"] for ep in episodes}),
            "benchmarks": list({ep["benchmark"] for ep in episodes}),
        }
        with (meta_dir / "info.json").open("w", encoding="utf-8") as fh:
            json.dump(info, fh, indent=2)

        return n_written

    def _write_jsonl_format(self, episodes: list[dict], output_dir: Path) -> int:
        """
        Write each episode as one JSON line to {output_dir}/hard_cases.jsonl.
        Line format: {episode_id, instruction, steps: [{obs, action}]}.
        """
        out_file = output_dir / "hard_cases.jsonl"
        n_written = 0
        with out_file.open("w", encoding="utf-8") as fh:
            for episode in episodes:
                record = {
                    "episode_id": episode["episode_id"],
                    "instruction": episode["instruction"],
                    "model_id": episode["model_id"],
                    "benchmark": episode["benchmark"],
                    "task_id": episode["task_id"],
                    "failure_type": episode.get("failure_type"),
                    "failure_step": episode.get("failure_step"),
                    "steps": episode.get("steps", []),
                }
                fh.write(json.dumps(record) + "\n")
                n_written += 1
        return n_written


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce_json(obj: Any) -> Any:
    """Recursively convert numpy arrays / bytes to JSON-serialisable Python types."""
    type_name = type(obj).__name__
    if type_name == "ndarray":
        return obj.tolist()
    if isinstance(obj, bytes):
        return obj.hex()
    if isinstance(obj, dict):
        return {k: _coerce_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_coerce_json(v) for v in obj]
    return obj
