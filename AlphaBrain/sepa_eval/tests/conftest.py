import pytest

from sepa_eval.memory.schema import (
    EpisodeTrace,
    TraceIdentity,
    SceneConfig,
    RolloutData,
    TraceLabels,
    TaskProvenance,
)


@pytest.fixture
def minimal_trace() -> EpisodeTrace:
    """Minimal EpisodeTrace for testing."""
    return EpisodeTrace(
        identity=TraceIdentity(
            trace_id="trace-001",
            eval_run_id="run-001",
            benchmark="libero_spatial",
            task_id="task-pick-cube",
            task_instruction="Pick up the cube",
            model_id="QwenOFT-test",
            model_version="v1.0",
        ),
        scene=SceneConfig(
            scene_config={"objects": [{"name": "cube", "target_pos": [0.1, 0.2, 0.3]}]},
            init_state=b"",
        ),
        rollout=RolloutData(
            observations=[{"gripper_state": [0.0], "qpos": [0.0] * 7}] * 10,
            actions=[[0.0] * 7] * 10,
            episode_length=10,
            success=False,
            failure_step=8,
        ),
        labels=TraceLabels(failure_type=None),
        provenance=TaskProvenance(),
    )


@pytest.fixture
def failed_trace(minimal_trace: EpisodeTrace) -> EpisodeTrace:
    """A failed trace with timeout-like length."""
    minimal_trace.rollout.episode_length = 500
    minimal_trace.rollout.success = False
    minimal_trace.rollout.failure_step = 500
    return minimal_trace
