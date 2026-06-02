"""
DBSCAN-based clustering of failure traces.

sklearn is imported lazily so the module can be imported without it installed
(e.g. in environments where only unit-testing utilities are available).
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

from sepa_eval.memory.schema import FAILURE_TYPES

# Canonical ordered list for one-hot encoding (deterministic order).
_FAILURE_TYPE_LIST: list[str] = sorted(FAILURE_TYPES)


@dataclass
class FailureCluster:
    """A single cluster of failure traces."""

    cluster_id: str
    failure_type: str | None
    member_trace_ids: list[str]
    representative_trace_id: str  # trace with median failure_step
    llm_summary: str | None = None


class FailureClusterer:
    """
    Clusters VLA failure traces using DBSCAN over a compact embedding vector.

    The embedding is:
        [failure_step / episode_length,   # 1 dim  (0.0 - 1.0)
         <failure_type one-hot>]           # 8 dims (one per canonical type)

    Total embedding size: 9 dimensions.
    """

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        last_n_runs: int = 5,
    ) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.last_n_runs = last_n_runs

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed_trace(self, trace_row: dict) -> list[float]:
        """
        Build a 9-dimensional embedding from a trace row dict.

        Expected keys (all optional — missing values are replaced with 0):
            - ``failure_step``    (int)
            - ``episode_length``  (int)
            - ``failure_type``    (str | None)
        """
        ep_len = trace_row.get("episode_length") or 0
        failure_step = trace_row.get("failure_step") or 0
        norm_step = (failure_step / ep_len) if ep_len > 0 else 0.0

        failure_type = trace_row.get("failure_type")
        one_hot = [
            1.0 if ft == failure_type else 0.0
            for ft in _FAILURE_TYPE_LIST
        ]

        return [norm_step, *one_hot]

    # ------------------------------------------------------------------
    # Window filtering
    # ------------------------------------------------------------------

    def filter_by_window(
        self,
        trace_rows: list[dict],
        last_n_runs: int | None = None,
    ) -> list[dict]:
        """
        Keep only traces from the most recent *last_n_runs* distinct
        eval_run_ids.  Recency is determined by the order in which
        eval_run_ids first appear when iterating trace_rows in reverse
        (most-recently-appended first).
        """
        n = last_n_runs if last_n_runs is not None else self.last_n_runs

        # Collect distinct run IDs in reverse insertion order.
        seen: list[str] = []
        for row in reversed(trace_rows):
            run_id = row.get("eval_run_id", "")
            if run_id not in seen:
                seen.append(run_id)
            if len(seen) >= n:
                break

        allowed = set(seen)
        return [r for r in trace_rows if r.get("eval_run_id", "") in allowed]

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def cluster(self, trace_rows: list[dict]) -> list[FailureCluster]:
        """
        Run DBSCAN and return a list of FailureCluster objects.

        - Returns [] if len(trace_rows) < min_samples (guard).
        - Noise points (DBSCAN label == -1) are excluded from clusters.
        - Cluster representative = trace with the median failure_step.
        """
        if len(trace_rows) < self.min_samples:
            return []

        # Lazy sklearn import.
        try:
            from sklearn.cluster import DBSCAN  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "scikit-learn is required for FailureClusterer.cluster(). "
                "Install it with: pip install scikit-learn"
            ) from exc

        embeddings = [self.embed_trace(row) for row in trace_rows]

        db = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="euclidean")
        labels: list[int] = db.fit_predict(embeddings).tolist()

        # Group trace rows by cluster label (skip noise = -1).
        clusters_map: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            if label == -1:
                continue
            clusters_map.setdefault(label, []).append(idx)

        result: list[FailureCluster] = []
        for cluster_label, member_indices in clusters_map.items():
            member_rows = [trace_rows[i] for i in member_indices]
            member_trace_ids = [r.get("trace_id", str(i)) for r, i in zip(member_rows, member_indices, strict=False)]

            # Majority-vote failure_type within the cluster.
            type_counts: dict[str | None, int] = {}
            for row in member_rows:
                ft = row.get("failure_type")
                type_counts[ft] = type_counts.get(ft, 0) + 1
            dominant_type: str | None = max(type_counts, key=lambda k: type_counts[k])

            # Representative = trace whose failure_step is closest to the median.
            failure_steps = [
                (row.get("failure_step") or 0, idx)
                for idx, row in zip(member_indices, member_rows, strict=False)
            ]
            steps_only = [fs for fs, _ in failure_steps]
            median_step = statistics.median(steps_only)
            rep_idx = min(
                failure_steps,
                key=lambda t: abs(t[0] - median_step),
            )[1]
            rep_trace_id = trace_rows[rep_idx].get("trace_id", str(rep_idx))

            result.append(
                FailureCluster(
                    cluster_id=f"cluster_{cluster_label}",
                    failure_type=dominant_type,
                    member_trace_ids=member_trace_ids,
                    representative_trace_id=rep_trace_id,
                    llm_summary=None,
                )
            )

        return result
