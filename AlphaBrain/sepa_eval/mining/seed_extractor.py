"""
Extracts a minimal-reproduction FailureSeed from a FailureCluster.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sepa_eval.mining.failure_cluster import FailureCluster


@dataclass
class FailureSeed:
    """A minimal reproduction extracted from a failure cluster."""

    trace_id: str
    benchmark: str
    task_id: str
    task_instruction: str
    failure_type: str | None
    failure_step: int | None
    scene_config: dict
    causal_factors: dict = field(default_factory=dict)
    """Causal factor weights, e.g. {"grasp": 0.7, "pose": 0.3}."""


class SeedExtractor:
    """
    Extracts a FailureSeed from the representative trace of a FailureCluster.

    The causal_factors dict is populated uniformly over the cluster's dominant
    failure_type (weight = 1.0).  If failure_type is None, causal_factors is
    left empty.

    If *memory* (an EvalMemory) is provided, the seed's scene_config is loaded
    from the trace's msgpack file when the DB row does not carry one — the
    ``traces`` table has no scene_config column, so without memory the
    scene_config would always be empty.
    """

    def __init__(self, memory=None) -> None:
        self._memory = memory

    def extract(
        self,
        cluster: FailureCluster,
        trace_rows: list[dict],
    ) -> FailureSeed:
        """
        Extract a FailureSeed from the representative trace of *cluster*.

        Parameters
        ----------
        cluster:
            A FailureCluster returned by FailureClusterer.cluster().
        trace_rows:
            The same list of trace row dicts that was used to build the cluster.
            Each row is expected to have the keys produced by
            EvalMemory.get_failures() (trace_id, benchmark, task_id,
            task_instruction, failure_type, failure_step, scene_config, …).
        """
        # Locate the representative row.
        rep_row = _find_row(trace_rows, cluster.representative_trace_id)

        # Causal factors: uniform weight 1.0 on the cluster's failure type.
        causal_factors: dict[str, float] = {}
        if cluster.failure_type is not None:
            causal_factors[cluster.failure_type] = 1.0

        scene_config = rep_row.get("scene_config") or {}
        if not scene_config and self._memory is not None:
            scene_config = self._load_scene_config(rep_row)

        return FailureSeed(
            trace_id=cluster.representative_trace_id,
            benchmark=rep_row.get("benchmark", ""),
            task_id=rep_row.get("task_id", ""),
            task_instruction=rep_row.get("task_instruction", ""),
            failure_type=cluster.failure_type,
            failure_step=rep_row.get("failure_step"),
            scene_config=scene_config,
            causal_factors=causal_factors,
        )

    def _load_scene_config(self, rep_row: dict) -> dict:
        """Best-effort scene_config recovery from the trace msgpack file."""
        trace_path = rep_row.get("trace_path")
        if not trace_path:
            return {}
        try:
            trace = self._memory.load_trace_file(trace_path)
            scene = getattr(trace, "scene", None)
            cfg = getattr(scene, "scene_config", None)
            return dict(cfg) if cfg else {}
        except Exception:
            return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_row(trace_rows: list[dict], trace_id: str) -> dict:
    """Return the first row whose trace_id matches; fall back to an empty dict."""
    for row in trace_rows:
        if row.get("trace_id") == trace_id:
            return row
    return {}
