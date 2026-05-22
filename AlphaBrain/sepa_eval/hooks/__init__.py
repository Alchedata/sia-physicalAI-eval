"""
sepa_eval.hooks — BenchmarkAdapter protocol, TraceHook, and benchmark-specific
trace hook implementations.

Public API
----------
BenchmarkAdapter
    Runtime-checkable protocol that any benchmark env wrapper must satisfy.
TraceHook
    Base context-manager class that accumulates per-step data and flushes to
    EvalMemory on on_episode_end().
LiberoHook
    BenchmarkAdapter wrapping a LIBERO eval environment.
RobocasaHook
    BenchmarkAdapter wrapping a RoboCasa simulation environment.
run_libero_episode_with_trace
    Convenience function: run one LIBERO episode end-to-end with trace
    collection.
"""

try:
    from sepa_eval.hooks.base import BenchmarkAdapter, TraceHook
except ImportError:
    pass

try:
    from sepa_eval.hooks.libero_trace_hook import LiberoHook, run_libero_episode_with_trace
except ImportError:
    pass

try:
    from sepa_eval.hooks.robocasa_trace_hook import RobocasaHook
except ImportError:
    pass

__all__ = [
    "BenchmarkAdapter",
    "TraceHook",
    "LiberoHook",
    "RobocasaHook",
    "run_libero_episode_with_trace",
]
