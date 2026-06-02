"""
SEPA-Eval failure mining subsystem.

Exports
-------
FailureClassifier   - heuristic per-type classifier
FailureClusterer    - DBSCAN clusterer over failure embeddings
FailureCluster      - dataclass: one cluster of failure traces
SeedExtractor       - extracts a minimal-reproduction FailureSeed
FailureSeed         - dataclass: minimal reproduction of a failure cluster
"""

from sepa_eval.mining.failure_classifier import FailureClassifier
from sepa_eval.mining.failure_cluster import FailureCluster, FailureClusterer
from sepa_eval.mining.seed_extractor import FailureSeed, SeedExtractor

__all__ = [
    "FailureClassifier",
    "FailureCluster",
    "FailureClusterer",
    "FailureSeed",
    "SeedExtractor",
]
