import pytest


@pytest.mark.integration
def test_trace_pipeline_libero_mini(tmp_path):
    """
    Integration test: instantiate EvalMemory, write 3 synthetic traces,
    verify trace files exist and DB rows are correct.
    Does NOT require actual LIBERO simulation.
    """
    from sepa_eval.memory.eval_memory import EvalMemory
    from sepa_eval.memory.schema import (
        EpisodeTrace,
        TraceIdentity,
        SceneConfig,
        RolloutData,
        TraceLabels,
        TaskProvenance,
    )
    import uuid

    db_path = str(tmp_path / "test_eval.db")
    memory = EvalMemory(db_path=db_path, memory_dir=str(tmp_path / "traces"))

    for i in range(3):
        trace = EpisodeTrace(
            identity=TraceIdentity(
                trace_id=str(uuid.uuid4()),
                eval_run_id="run-001",
                benchmark="libero_spatial",
                task_id=f"task-{i}",
                task_instruction=f"Pick up object {i}",
                model_id="QwenOFT-test",
                model_version="v1.0",
            ),
            scene=SceneConfig(scene_config={}, init_state=b""),
            rollout=RolloutData(
                observations=[],
                actions=[],
                episode_length=20,
                success=(i % 2 == 0),
                failure_step=None if i % 2 == 0 else 15,
            ),
        )
        memory.record_trace(trace)

    failures = memory.get_failures(benchmark="libero_spatial")
    assert len(failures) >= 1  # at least one failed trace

    # Verify files
    trace_files = list((tmp_path / "traces").rglob("*.msgpack"))
    assert len(trace_files) == 3

    memory.close()


@pytest.mark.integration
def test_evolution_loop_e2e_libero_mini(tmp_path):
    """
    Stub for full evolution loop e2e test.
    Marked skip until Phase 5 is complete.
    """
    pytest.skip("Phase 5 not yet implemented — placeholder for e2e evolution loop test")
