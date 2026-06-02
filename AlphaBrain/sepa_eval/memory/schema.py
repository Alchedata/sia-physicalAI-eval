"""
Data schemas for SEPA-Eval. All persistent structures live here.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# ---------------------------------------------------------------------------
# EpisodeTrace sub-structs
# ---------------------------------------------------------------------------

@dataclass
class TraceIdentity:
    trace_id: str
    eval_run_id: str
    benchmark: str          # "libero_spatial", "robocasa_tabletop", etc.
    task_id: str
    task_instruction: str
    model_id: str           # framework name + checkpoint path hash
    model_version: str


@dataclass
class SceneConfig:
    scene_config: dict
    init_state: bytes       # simulator state snapshot (for replay)
    replay_mode: str = "exact"  # "exact" | "reseed"


@dataclass
class RolloutData:
    observations: list[dict]        # per-step: {images: {cam: np.uint8}, proprioception, timestamp}
    actions: list[Any]              # List[np.ndarray]
    episode_length: int
    success: bool
    failure_step: int | None = None  # first step where failure is detectable


@dataclass
class TraceLabels:
    critic_scores: dict = field(default_factory=dict)       # keyed by critic name
    failure_type: str | None = None                         # filled by failure mining
    failure_attribution: dict = field(default_factory=dict) # {"grasp": 0.7, ...}


@dataclass
class TaskProvenance:
    parent_task_id: str | None = None  # source seed if this is a mutation
    mutation_type: str | None = None   # "pose_perturb", "material_swap", etc.
    promotion_status: str = "seed"     # "seed"|"candidate"|"promoted"|"rejected"|"archived"


@dataclass
class EpisodeTrace:
    identity: TraceIdentity
    scene: SceneConfig
    rollout: RolloutData
    labels: TraceLabels = field(default_factory=TraceLabels)
    provenance: TaskProvenance = field(default_factory=TaskProvenance)


# ---------------------------------------------------------------------------
# CandidateTask (produced by mutation engine)
# ---------------------------------------------------------------------------

@dataclass
class CandidateTask:
    task_id: str
    parent_task_id: str           # source seed trace_id
    benchmark: str                # "libero_spatial", etc.
    instruction: str
    scene_config: dict            # full simulator init config
    mutation_type: str            # "PosePerturbation", "DistractorAdd", etc.
    mutation_params: dict         # operator-specific parameters
    promotion_status: str         # "candidate"|"promoted"|"rejected"|"archived"|"deferred"
    created_at: datetime
    embedding: bytes | None = None  # BGE-M3 encoded instruction
    promotion_evidence: dict = field(default_factory=dict)  # gate-by-gate evidence

    @classmethod
    def new(
        cls,
        parent_task_id: str,
        benchmark: str,
        instruction: str,
        scene_config: dict,
        mutation_type: str,
        mutation_params: dict,
    ) -> "CandidateTask":
        return cls(
            task_id=str(uuid.uuid4()),
            parent_task_id=parent_task_id,
            benchmark=benchmark,
            instruction=instruction,
            scene_config=scene_config,
            mutation_type=mutation_type,
            mutation_params=mutation_params,
            promotion_status="candidate",
            created_at=datetime.now(tz=__import__("datetime").timezone.utc).replace(tzinfo=None),
        )


# ---------------------------------------------------------------------------
# Failure types (canonical names used by FailureClassifier)
# ---------------------------------------------------------------------------

FAILURE_TYPES = frozenset({
    "grasp",
    "pose_estimation",
    "language_grounding",
    "contact_dynamics",
    "recovery",
    "out_of_reach",
    "distractor_confusion",
    "timeout",
})
