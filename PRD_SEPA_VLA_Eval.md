# PRD: Self-Evolving VLA Evaluation System (SEPA-Eval)
## Built on AlphaBrain

**Version**: 0.2  
**Date**: 2026-05-21  
**Scope**: VLA model evaluation only (world-model evaluation explicitly out of scope)

---

## 1. Problem Statement

AlphaBrain already provides a solid foundation for evaluating VLA policies across LIBERO, LIBERO-plus, Robocasa-Tabletop, and Robocasa365. Each benchmark evaluates success rate on a fixed task distribution using the client-server WebSocket architecture.

This static evaluation paradigm has three structural weaknesses:

1. **Saturation**: Once all frontier models exceed 95%+ SR on LIBERO suites, the benchmark stops discriminating.
2. **Misalignment**: Binary success rate misses fragile success, unsafe contact, brittle grasp recovery, and language-edge-case failures.
3. **Silent drift**: Scores continue to look meaningful even when they no longer predict real deployment behavior.

The paper *Self-Evolving Agents for Physical AI Evaluation* (SEPA-Eval) proposes an agentic evaluation infrastructure that observes failures, generates new tests, validates them, promotes them into the benchmark distribution, and tracks when existing evals become obsolete.

This PRD defines the requirements to implement SEPA-Eval's VLA evaluation components on top of AlphaBrain.

---

## 2. Goals

| # | Goal |
|---|------|
| G1 | Extend AlphaBrain's eval loop to emit **structured rollout traces** (observation, action, contact event, success label, failure moment) stored to an evaluation memory database. |
| G2 | Implement a **failure mining** module that clusters traces by failure type (perception, grasp, physics, language, recovery) and extracts reproducible failure seeds. |
| G3 | Implement a **scenario mutation engine** that generates task variants from failure seeds: pose perturbation, material change, distractor addition, instruction paraphrase, extended horizon. |
| G4 | Implement a **critic ensemble** for VLA rollout scoring: semantic critic (VLM-judge), safety critic (force/collision heuristics), robustness critic (success across perturbations). |
| G5 | Implement a **test promotion pipeline** that validates generated scenarios and promotes them into the active benchmark distribution based on reproducibility, non-redundancy, and discriminative power. |
| G6 | Implement a **self-evolution loop** orchestrating the full Evaluate → Diagnose → Generate → Validate → Promote → Monitor → Report cycle. |
| G7 | Produce a **reporting dashboard** that surfaces capability frontiers, recurring failure patterns, saturation signals, and model comparison across evolved benchmark slices. |
| G8 | Implement a **hard-case exporter** (`sepa_eval export-hard-cases`) that exports failed rollout (obs, action) pairs from EvalMemory as a dataset consumable by AlphaBrain's continual learning pipeline, closing the eval → training feedback loop. |

---

## 3. Non-Goals

- World model evaluation of any kind (video generation quality, physics consistency of generated video, downstream world-model utility)
- Sim-to-real calibration against physical robots (deferred to a later phase)
- New robot embodiment support beyond what AlphaBrain already supports
- Training/fine-tuning VLA models (this is purely an evaluation system)
- New simulator integration beyond LIBERO, Robocasa-Tabletop, and Robocasa365

---

## 4. Background: AlphaBrain Current State

### 4.1 Evaluation Infrastructure

AlphaBrain already provides:

| Asset | Location | Notes |
|-------|----------|-------|
| LIBERO eval loop | `benchmarks/LIBERO/eval/eval_libero.py` | Full episode runner, SR tracking, JSON logs |
| Robocasa eval loop | `benchmarks/Robocasa_tabletop/eval/simulation_env.py` | Gym-wrapped, video recording |
| LIBERO-plus eval | `benchmarks/LIBERO-plus/eval/` | Extended LIBERO distribution |
| Robocasa365 eval | `benchmarks/Robocasa365/eval/` | Large-scale household tasks |
| Benchmark adapters | `model2libero_interface.py`, `model2robocasa_interface.py` | Bridge env quirks to generic server |
| Policy server | WebSocket server in `deployment/model_server/` | Model-agnostic inference endpoint |
| Frameworks | `AlphaBrain/model/framework/` | QwenOFT, QwenPI, PaliGemmaOFT, NeuroVLA, ACT, CosmosPolicy, etc. |

### 4.2 Key Gap

Current eval produces only: **per-episode success boolean + aggregate SR**. There is no trace storage, no failure classification, no scenario generation, and no evolution loop.

---

## 5. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SEPA-Eval VLA System                            │
│                                                                     │
│  ┌──────────────┐    ┌───────────────────────┐    ┌─────────────┐  │
│  │  Enhanced    │───▶│   Evaluation Memory   │◀───│  Reporting  │  │
│  │  Eval Loops  │    │  (trace store + DB)   │    │  Dashboard  │  │
│  │  (existing   │    └──────────┬────────────┘    └─────────────┘  │
│  │   + hooks)   │               │                                   │
│  └──────────────┘    ┌──────────▼────────────┐                     │
│                      │   Failure Mining &    │                     │
│                      │   Attribution Module  │                     │
│                      └──────────┬────────────┘                     │
│                                 │                                   │
│                      ┌──────────▼────────────┐                     │
│                      │  Scenario Mutation    │                     │
│                      │  Engine               │                     │
│                      └──────────┬────────────┘                     │
│                                 │                                   │
│  ┌──────────────┐    ┌──────────▼────────────┐                     │
│  │  Critic      │───▶│  Validation &         │                     │
│  │  Ensemble    │    │  Promotion Pipeline   │                     │
│  └──────────────┘    └──────────┬────────────┘                     │
│                                 │                                   │
│                      ┌──────────▼────────────┐                     │
│                      │  Self-Evolution Loop  │                     │
│                      │  Orchestrator         │                     │
│                      └───────────────────────┘                     │
└─────────────────────────────────────────────────────────────────────┘
```

The system is built as a layer **on top of** existing eval infrastructure. Existing `eval_libero.py` and `simulation_env.py` entry points are extended with trace emission hooks; no existing logic is broken.

**Benchmark adapter protocol** (`sepa_eval/hooks/base.py`): All hooks implement this 4-method protocol so the orchestrator talks only to the protocol, not to each eval loop directly:

```python
class BenchmarkAdapter(Protocol):
    def reset(self) -> dict: ...          # returns initial obs
    def step(self, action) -> tuple: ... # returns (obs, done, info)
    def get_task_id(self) -> str: ...
    def get_scene_config(self) -> dict: ...
```

Each benchmark hook (`LiberoHook`, `RobocasaHook`, etc.) implements this protocol. Adding a new benchmark = adding one file that implements these 4 methods.

---

## 6. Component Specifications

### 6.1 Enhanced Eval Loops with Trace Hooks

**Location**: `sepa_eval/hooks/`

**What changes**: The existing eval loops (`eval_libero.py`, `simulation_env.py`) gain optional trace emission that is off by default and enabled via a `--trace` flag or config option.

**Trace schema per episode** (msgpack, stored in eval memory). Uses nested sub-dataclasses to prevent the flat-struct growth problem:

```python
@dataclass
class TraceIdentity:
    trace_id:         str          # UUID
    eval_run_id:      str          # groups traces from one eval run
    benchmark:        str          # "libero_spatial", "robocasa_tabletop", etc.
    task_id:          str          # task name or hash
    task_instruction: str
    model_id:         str          # framework name + checkpoint path hash
    model_version:    str

@dataclass
class SceneConfig:
    scene_config:     dict         # object list, poses, materials
    init_state:       bytes        # simulator state snapshot (for replay)
    replay_mode:      str = "exact" # "exact" | "reseed"

@dataclass
class RolloutData:
    observations:     List[dict]   # per-step: {images: {cam: np.uint8}, proprioception, timestamp}
    actions:          List[np.ndarray]
    episode_length:   int
    success:          bool
    failure_step:     int | None   # first step where failure is detectable

@dataclass
class TraceLabels:
    critic_scores:       dict = field(default_factory=dict)  # keyed by critic name
    failure_type:        str | None = None                   # filled by failure mining
    failure_attribution: dict = field(default_factory=dict)  # {"grasp": 0.7, ...}

@dataclass
class TaskProvenance:
    parent_task_id:   str | None = None   # source seed if this is a mutation
    mutation_type:    str | None = None   # "pose_perturb", "material_swap", etc.
    promotion_status: str = "seed"        # "seed"|"candidate"|"promoted"|"rejected"|"archived"

@dataclass
class EpisodeTrace:
    identity:    TraceIdentity
    scene:       SceneConfig
    rollout:     RolloutData
    labels:      TraceLabels      = field(default_factory=TraceLabels)
    provenance:  TaskProvenance   = field(default_factory=TaskProvenance)
```

**Implementation notes**:
- Default (`--store-obs=False`): store only failure-window frames (last 10 steps before failure). Observations stored inline after `on_episode_end()` — no blocking I/O during the episode.
- Full obs (`--store-obs=True`): accumulate raw numpy arrays in memory during the episode; compress all at once in `on_episode_end()` (batch compression avoids per-step blocking). Images: PNG. Proprioception: float16.
- The trace hook is a context manager wrapping the episode loop; does not alter control flow.
- All trace writes use `.tmp` path first; rename to final path only on successful `on_episode_end()`. Orphaned `.tmp` files cleaned by `EvalMemory.fsck()`.

---

### 6.2 Evaluation Memory

**Location**: `sepa_eval/memory/`

**Backend**: SQLite (local, single-machine) with a JSON/Parquet trace file store alongside it. Designed to later upgrade to a remote store without changing the query interface.

**Tables**:

```sql
-- Core trace index (lightweight; full trace lives in files)
CREATE TABLE traces (
    trace_id          TEXT PRIMARY KEY,
    eval_run_id       TEXT,
    benchmark         TEXT,
    task_id           TEXT,
    task_instruction  TEXT,
    model_id          TEXT,
    model_version     TEXT,
    success           INTEGER,
    failure_step      INTEGER,
    episode_length    INTEGER,
    failure_type      TEXT,
    promotion_status  TEXT,
    parent_task_id    TEXT,
    mutation_type     TEXT,
    created_at        TIMESTAMP,
    trace_path        TEXT           -- path to full trace file
);

-- Task registry (seed tasks + evolved tasks)
CREATE TABLE tasks (
    task_id           TEXT PRIMARY KEY,
    benchmark         TEXT,
    instruction       TEXT,
    scene_config      TEXT,          -- JSON
    mutation_lineage  TEXT,          -- JSON list of ancestor task IDs
    promotion_status  TEXT,          -- "seed", "candidate", "promoted", "retired"
    discriminative_power REAL,       -- fraction of runs where models disagree on outcome
    saturation_flag   INTEGER,       -- 1 if all frontier models solve this
    created_at        TIMESTAMP
);

-- Critic judgments (separate table for retroactive scoring)
CREATE TABLE critic_scores (
    trace_id          TEXT,
    critic_name       TEXT,
    score             REAL,
    explanation       TEXT,
    confidence        REAL,
    scored_at         TIMESTAMP
);

-- Model performance summary (per task, per model)
CREATE TABLE model_task_results (
    model_id          TEXT,
    task_id           TEXT,
    benchmark         TEXT,
    n_trials          INTEGER,
    success_rate      REAL,
    clean_success_rate REAL,         -- success without unsafe contact flag
    avg_episode_length REAL,
    last_eval_at      TIMESTAMP,
    PRIMARY KEY (model_id, task_id)
);
```

**Key query capabilities** (exposed via `EvalMemory` Python API):
- `memory.get_failures(benchmark=..., model_id=..., failure_type=...)` → `List[EpisodeTrace]`
- `memory.get_saturated_tasks(threshold=0.95)` → tasks where all models ≥ threshold SR
- `memory.get_discriminative_tasks(min_spread=0.2)` → tasks still separating models
- `memory.get_failure_clusters()` → cluster labels + representative traces
- `memory.record_trace(trace)` → write
- `memory.update_critic_score(trace_id, critic_name, score)` → retroactive labeling

---

### 6.3 Failure Mining & Attribution Module

**Location**: `sepa_eval/mining/`

**Inputs**: Failed episode traces from evaluation memory.

**Outputs**: For each failure, an `FailureRecord` with:
- `failure_type`: one of `{grasp, pose_estimation, language_grounding, contact_dynamics, recovery, out_of_reach, distractor_confusion, timeout}`
- `failure_step`: step index where failure is first detectable
- `causal_factors`: dict of factor → attribution weight (summing to 1.0)
- `minimal_reproduction`: dict describing the smallest scene context that reproduces the failure

**Attribution heuristics** (rule-based, no training required in Phase 1):

| Failure type | Detection heuristic |
|---|---|
| `grasp` | Episode ends in grasping subtask; gripper state oscillates without stable closure |
| `pose_estimation` | Action target diverges from object position; object visible but not approached |
| `language_grounding` | Robot acts on wrong object class; detected by semantic critic |
| `contact_dynamics` | Object slips post-grasp; success at placement fails due to contact |
| `recovery` | Failed retry of same action ≥ 2 times without replanning |
| `out_of_reach` | Robot workspace limit reached; end-effector at joint limits |
| `distractor_confusion` | Multiple similar objects present; wrong one selected |
| `timeout` | Episode length hit without progress |

**`failure_step` detection**: Each failure type has a `FailureStepDetector` with its own detection heuristic (e.g., for `grasp`: scan action history for gripper oscillation without stable closure). Fallback: `failure_step = episode_length` if no heuristic fires.

**Clustering**: DBSCAN over a failure embedding (concatenate: task embedding, failure_step/total_steps, causal_factor vector). Cluster representatives are selected as the trace with median failure_step in each cluster.

**Windowed clustering**: By default, cluster only traces from the most recent `cluster_window.last_n_runs` eval runs (default: 5). This bounds DBSCAN input size regardless of total history. Configure in `sepa_eval/configs/mutation.yaml`:

```yaml
failure_mining:
  cluster_window:
    last_n_runs: 5    # only cluster traces from the 5 most recent eval runs
  min_cluster_size: 5 # skip clustering if fewer than 5 failures
```

**Cluster LLM summarizer**: After clustering, each cluster's representative trace (final N frames + critic scores + causal factor vector) is passed to the VLM server with a summarization prompt. The model produces a one-paragraph natural-language explanation: *what tends to go wrong, at which stage, and why.* The summary is stored in the `failure_clusters` table and included in the evaluation report. Reuses the existing policy server infrastructure — same endpoint as the semantic critic, different prompt.

```sql
-- Added to schema.py
CREATE TABLE failure_clusters (
    cluster_id        TEXT PRIMARY KEY,
    eval_run_id       TEXT,
    failure_type      TEXT,
    centroid          BLOB,           -- serialized embedding
    representative_trace_id TEXT,
    member_count      INTEGER,
    llm_summary       TEXT,           -- auto-generated human-readable explanation
    summarized_at     TIMESTAMP
);
```

---

### 6.4 Scenario Mutation Engine

**Location**: `sepa_eval/mutation/`

**Purpose**: Given a failure seed, produce N candidate task variants that isolate the causal factor.

**Mutation operators**:

| Operator | Description | Applicable simulators |
|---|---|---|
| `PosePerturbation` | Randomize target object pose within ±δ of seed pose | LIBERO, Robocasa |
| `MaterialSwap` | Change object material (color, texture, friction coefficient) | LIBERO, Robocasa |
| `DistractorAdd` | Insert N additional same-category objects into the scene | LIBERO, Robocasa |
| `InstructionParaphrase` | Rephrase instruction using LLM while preserving intent | All |
| `HorizonExtension` | Chain the seed task as a subtask inside a longer sequence | LIBERO Long |
| `LightingChange` | Modify ambient light parameters (brightness, direction) | Robocasa |
| `OcclusionAdd` | Place a partially occluding object between camera and target | LIBERO, Robocasa |
| `GripperSwap` | Change robot embodiment (where simulator supports it) | Robocasa |

**Configuration**:
```yaml
# sepa_eval/configs/mutation.yaml
mutation:
  operators:
    - name: PosePerturbation
      delta_pos: [0.02, 0.05, 0.10]    # meters; generates 3 variants
      delta_rot_deg: [5, 15, 30]
    - name: DistractorAdd
      counts: [1, 2, 3]
    - name: InstructionParaphrase
      n_variants: 3
      llm_model: "gpt-4o-mini"         # or local model
  max_variants_per_seed: 10
  causally_isolate: true               # change one factor at a time
```

**Output**: A `CandidateTask` containing the simulator init config, instruction, and provenance metadata. Candidate tasks are written to the `tasks` table with `promotion_status = "candidate"`.

---

### 6.5 Critic Ensemble

**Location**: `sepa_eval/critics/`

Three critics are required in Phase 1; Phase 2 adds more.

#### 6.5.1 Semantic Critic (VLM-Judge)

**Purpose**: Given a rollout video or final-frame image + instruction, judge whether the task was semantically completed.

**Implementation**:
- Input: final N frames + instruction text
- Model: GPT-4o Vision or Qwen2.5-VL (configurable; uses the same model server AlphaBrain already runs)
- Prompt: structured rubric asking (1) did the correct object move? (2) is the final state consistent with the instruction? (3) was there unnecessary contact with other objects?
- Output: `{completion: float, object_correct: bool, collateral_damage: bool, explanation: str}`

**Calibration**: Compare against ground-truth `success` labels from the simulator. Flag systematic disagreement cases for human review.

#### 6.5.2 Safety Critic

**Purpose**: Flag unsafe contact, excessive force, dropped objects, and fragile-success patterns.

**Implementation** (heuristic, no model required):
- Reads per-step joint torque / force data from simulator if available
- Flags: torque spike > threshold, object dropped mid-episode (success=True but object displaced), repeated grasp retries
- Output: `{unsafe_contact: bool, force_spike: bool, fragile_success: bool, safety_score: float}`

#### 6.5.3 Robustness Critic

**Purpose**: Score a policy's success rate across a family of variants of the same task.

**Implementation**:
- Input: `List[EpisodeTrace]` for all variants of a task family
- Computes: SR baseline, SR under each mutation type, Δ SR per perturbation class
- Output: `{robustness_score: float, sensitive_to: List[str], variant_breakdown: dict}`

#### 6.5.4 (Phase 2) Physics Critic

Rules-based physics plausibility check (gravity, support, containment) using simulator state queries. Not required for Phase 1.

---

### 6.6 Validation & Promotion Pipeline

**Location**: `sepa_eval/promotion/`

A generated `CandidateTask` is promoted to `"promoted"` status only when it passes ALL promotion gates:

| Gate | Check | Pass threshold |
|---|---|---|
| **Solvability** | At least one AlphaBrain model achieves SR ≥ 0.5 on N=10 trials | SR ≥ 0.5 |
| **Reproducibility** | The target failure recurs in ≥ 60% of trials for at least one failing model | ≥ 0.6 failure rate |
| **Non-redundancy** | Cosine similarity between this task embedding and any promoted task < threshold | sim < 0.85 |
| **Discriminative power** | SR spread between best and worst model ≥ 0.2 on N=20 trials each | Δ SR ≥ 0.2 |
| **Deployment relevance** | Human reviewer labels the scenario as realistic (async review queue, not blocking) | Approved |

Candidates that fail solvability are **discarded**. Candidates that fail non-redundancy are **archived**. Candidates that fail discriminative power are **deferred** (re-checked after the next model release).

**Promotion orchestration**:
```
CandidateTask
  → SolvabilityCheck (runs N=10 trials via AlphaBrain eval loop)
  → ReproducibilityCheck (runs N=20 trials for failing models)
  → RedundancyCheck (embedding similarity against task DB)
  → DiscriminativePowerCheck (computes SR spread across registered models)
  → HumanReviewQueue (async; does not block automated pipeline)
  → PromotionDecision → write to tasks table
```

**Parallelization**: Gate calls are dispatched through a `ThreadPoolExecutor` (max `max_parallel_eval_workers` workers from orchestrator config). Each gate calls `batch_eval(candidate, n=K)` which submits to the pool and returns a `Future`. Pipeline awaits all futures for a gate before advancing. Gate timeout: `gate_timeout_minutes: 120` — on timeout, candidate is marked `deferred`, not `failed`.

---

### 6.7 Self-Evolution Loop Orchestrator

**Location**: `sepa_eval/orchestrator/`

**CandidateTask schema** (defined in `sepa_eval/memory/schema.py`):

```python
@dataclass
class CandidateTask:
    task_id:          str           # UUID
    parent_task_id:   str           # source seed trace_id
    benchmark:        str           # "libero_spatial", etc.
    instruction:      str
    scene_config:     dict          # full simulator init config
    mutation_type:    str           # "PosePerturbation", "DistractorAdd", etc.
    mutation_params:  dict          # operator-specific parameters
    promotion_status: str           # "candidate"|"promoted"|"rejected"|"archived"|"deferred"
    created_at:       datetime
    embedding:        bytes | None  # BGE-M3 encoded instruction (set after creation)
    promotion_evidence: dict        # gate-by-gate pass/fail evidence for audit trail
```

The orchestrator runs the seven-step SEPA-Eval cycle as a configurable pipeline:

```
Step 1: EVALUATE
  Run all registered models on the current task distribution (promoted tasks).
  Emit traces to evaluation memory.

Step 2: DIAGNOSE
  Run failure mining on new traces.
  Identify clusters, extract seeds for each unresolved failure cluster.

Step 3: GENERATE
  For each failure seed, run the mutation engine.
  Write CandidateTasks to the task table.

Step 4: VALIDATE + PROMOTE
  Run the promotion pipeline on all candidates.
  Promoted tasks enter the active benchmark distribution.

Step 5: MONITOR OBSOLESCENCE
  For every promoted task, check:
    - Is SR > 0.95 for all registered models? → mark saturated.
    - Has discriminative power dropped below 0.1? → mark stale.
    - Emit saturation warnings in the report.

Step 6: REPORT
  Generate the evaluation report (see §6.8).
```

**Cadence config**:
```yaml
# sepa_eval/configs/orchestrator.yaml
orchestrator:
  full_cycle_schedule: "weekly"          # cron or manual trigger
  quick_eval_on_new_model: true          # runs Step 1 only on new model registration
  promotion_pipeline_schedule: "daily"   # runs Steps 4-5 independently
  max_candidates_per_cycle: 50
  max_parallel_eval_workers: 4
```

**Entry point**:
```bash
python -m sepa_eval run --config sepa_eval/configs/orchestrator.yaml
python -m sepa_eval promote          # run promotion only
python -m sepa_eval report           # generate report only
python -m sepa_eval eval --model QwenOFT --checkpoint path/to/ckpt  # register + quick eval
python -m sepa_eval export-hard-cases --model QwenOFT --output data/hard_cases/  # export failed rollouts
```

---

### 6.8 Reporting Dashboard

**Location**: `sepa_eval/reporting/`

**Format**: Markdown + JSON export (consumable by the existing docs/mkdocs setup). Optional: Streamlit app for interactive exploration.

**Report sections**:

1. **Capability Frontier**: Per-benchmark SR for all registered models on the current promoted task distribution. Highlights which tasks discriminate frontier models.

2. **Saturation Map**: Tasks grouped by saturation level. Color codes: green (still discriminating), yellow (approaching saturation), red (saturated).

3. **Failure Taxonomy**: Distribution of failure types across last N eval runs per model. Shows whether new model versions shift failure distributions.

4. **Evolved Task Summary**: Count of seeds → candidates → promoted tasks in the last cycle. Provenance table for each promoted task.

5. **Model Regression Check**: For each model version, which newly promoted tasks expose regressions versus the previous version?

6. **Critic Agreement Matrix**: Correlation between semantic critic, safety critic, and simulator success label. Disagreement cells are flagged for investigation.

7. **Cross-Model Failure Heatmap**: A task × model matrix where each cell shows the success rate for that (task, model) combination. Color-coded: green (SR ≥ 0.8), yellow (0.4–0.8), red (< 0.4). Cells where **all** models fail highlight universal benchmark gaps; cells where only one model fails highlight model-specific weaknesses. Replaces the per-model per-benchmark tables with a single comparative view and is the primary output of each evolution cycle.

---

## 7. New Directory Structure

```
AlphaBrain/
├── sepa_eval/                          ← NEW: all SEPA-Eval components
│   ├── __init__.py
│   ├── __main__.py                     ← CLI entry point
│   ├── configs/
│   │   ├── mutation.yaml
│   │   ├── orchestrator.yaml
│   │   └── critics.yaml
│   ├── hooks/
│   │   ├── __init__.py
│   │   ├── base.py                     ← BenchmarkAdapter protocol (4-method interface)
│   │   ├── libero_trace_hook.py        ← wraps eval_libero.py episode loop
│   │   └── robocasa_trace_hook.py      ← wraps simulation_env.py episode loop
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── eval_memory.py              ← EvalMemory class (SQLite + file store, WAL mode)
│   │   └── schema.py                   ← EpisodeTrace (nested), CandidateTask, TaskRecord
│   ├── mining/
│   │   ├── __init__.py
│   │   ├── failure_classifier.py       ← heuristic failure type detector
│   │   ├── failure_cluster.py          ← DBSCAN clustering
│   │   └── seed_extractor.py           ← minimal reproduction extractor
│   ├── mutation/
│   │   ├── __init__.py
│   │   ├── base_operator.py
│   │   ├── pose_perturbation.py
│   │   ├── material_swap.py
│   │   ├── distractor_add.py
│   │   ├── instruction_paraphrase.py
│   │   └── horizon_extension.py
│   ├── critics/
│   │   ├── __init__.py
│   │   ├── semantic_critic.py          ← VLM-judge via API or local model
│   │   ├── safety_critic.py            ← heuristic force/contact checks
│   │   └── robustness_critic.py        ← variant SR aggregator
│   ├── promotion/
│   │   ├── __init__.py
│   │   ├── gates.py                    ← solvability, reproducibility, etc.
│   │   ├── pipeline.py                 ← ordered gate runner
│   │   └── human_review_queue.py       ← async review queue
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   └── evolution_loop.py
│   ├── exporter/
│   │   ├── __init__.py
│   │   └── hard_case_exporter.py    ← export failed rollouts as training data
│   ├── tests/                          ← pytest tests alongside each module
│   │   ├── test_eval_memory.py
│   │   ├── test_failure_classifier.py
│   │   ├── test_failure_cluster.py
│   │   ├── test_mutation.py
│   │   ├── test_critics.py
│   │   ├── test_promotion.py
│   │   ├── test_exporter.py
│   │   └── test_integration.py
│   └── reporting/
│       ├── __init__.py
│       ├── report_generator.py
│       ├── heatmap.py               ← task × model SR matrix generator
│       └── templates/
│           └── eval_report.md.jinja
└── benchmarks/
    ├── LIBERO/eval/
    │   └── eval_libero.py              ← MODIFIED: accepts --trace flag
    └── Robocasa_tabletop/eval/
        └── simulation_env.py           ← MODIFIED: accepts trace_hook kwarg
```

---

## 8. Integration with Existing AlphaBrain Infrastructure

### 8.1 Minimal changes to existing eval code

The trace hook is a drop-in context manager. Existing eval scripts are modified only to:
1. Accept an optional `--trace` / `--trace-dir` CLI argument
2. Call `trace_hook.on_step(obs, action)` and `trace_hook.on_episode_end(success)` inside the episode loop

No changes to the WebSocket policy server, framework modules, or training code.

### 8.2 Model registration

SEPA-Eval maintains a registry of models to include in comparative evaluation. Registration via:

```python
from sepa_eval.memory import EvalMemory
memory = EvalMemory("./eval_memory.db")
memory.register_model(
    model_id="QwenOFT-v1.2",
    framework="QwenOFT",
    checkpoint="path/to/checkpoint",
    benchmarks=["libero_spatial", "libero_long", "robocasa_tabletop"]
)
```

### 8.3 Semantic critic: `/judge` endpoint

The existing WebSocket policy server (`deployment/model_server/`) needs a new HTTP endpoint for the semantic critic and cluster summarizer. This is a Phase 2 scope addition to the model server.

**New endpoint**: `POST /judge`

```json
// Request
{
  "images": ["<base64 PNG>", ...],   // up to 10 frames
  "instruction": "Pick up the mug",
  "prompt": "Did the robot complete the task described in the instruction? ..."
}

// Response
{
  "text": "<model's judgment text>",
  "completion_score": 0.85,
  "object_correct": true,
  "collateral_damage": false
}
```

This endpoint calls the already-loaded VLM backbone in generation mode (not action prediction mode). No additional GPU memory needed.

**Circular bias routing**: When `model_id` of the model under evaluation matches the VLM family of the critic (e.g., evaluating `QwenOFT` with a Qwen2.5-VL critic), the semantic critic automatically routes to GPT-4o-mini Vision instead. This is configured in `critics.yaml`:

```yaml
critics:
  semantic:
    default_model: "local"   # uses /judge endpoint
    fallback_model: "gpt-4o-mini"
    circular_bias_patterns:  # when evaluated model matches any of these, use fallback
      - "Qwen*"
      - "qwen*"
```

---

## 9. Implementation Phases

### Phase 1: Trace Collection + Failure Mining (Weeks 1–4)

**Goal**: Get structured failure data into a queryable store.

| Task | Owner area | Deliverable |
|---|---|---|
| Implement `EpisodeTrace` dataclass and `EvalMemory` (SQLite + file store) | Memory | `sepa_eval/memory/` |
| Add trace hooks to `eval_libero.py` | Hooks | `eval_libero.py --trace` works end-to-end |
| Add trace hooks to `simulation_env.py` (Robocasa) | Hooks | Same for Robocasa |
| Implement `FailureClassifier` with 8 heuristic failure types | Mining | Classifies ≥ 80% of failures into known types |
| Implement `FailureCluster` using DBSCAN | Mining | Clusters failures; human-readable cluster names |
| CLI: `python -m sepa_eval eval` runs existing model + collects traces | CLI | End-to-end trace pipeline |
| Basic report: failure distribution per model per benchmark | Reporting | Markdown report |

**Success criteria**:
- Running `eval_libero.py --trace` on any registered model produces a populated `EvalMemory` database
- Failure types are assigned for ≥ 80% of failed episodes
- Clustering produces ≤ 20 clusters for a 500-episode eval run on LIBERO

---

### Phase 2: Semantic + Safety Critics (Weeks 5–7)

**Goal**: Score rollouts beyond binary success.

| Task | Deliverable |
|---|---|
| Implement `SemanticCritic` using GPT-4o-mini Vision API | VLM judge scoring final-frame |
| Calibrate semantic critic against simulator ground truth (should agree ≥ 90%) | Calibration report |
| Implement `SafetyCritic` using joint torque + object-displacement heuristics | Per-episode safety flag |
| Add `clean_success_rate` computation (success AND no safety flags) | New metric in report |
| Update `EvalMemory` with critic score tables | DB schema update |
| Implement `HardCaseExporter` — export failed (obs, action) pairs as LeRobot or HuggingFace dataset format for continual learning | `sepa_eval export-hard-cases --model X --output path/` |
| Add cluster LLM summarizer to `FailureCluster` | Natural-language failure description per cluster |
| Implement cross-model failure heatmap in reporting | task × model SR matrix in Markdown + JSON export |

**Success criteria**:
- Semantic critic agrees with simulator success label ≥ 90% on LIBERO test set
- `clean_success_rate` is lower than raw SR for at least one model (demonstrating fragile success detection)
- LLM cluster summaries are generated for all clusters from a LIBERO eval run; human spot-check rates ≥ 80% as accurate
- Cross-model heatmap is included in the auto-generated report; universal failure tasks are visually distinct
- `sepa_eval export-hard-cases --model X` produces a valid LeRobot-format dataset of ≥ 10 failed episodes

---

### Phase 3: Mutation Engine + Candidate Generation (Weeks 8–11)

**Goal**: Automatically generate new task variants from failure seeds.

| Task | Deliverable |
|---|---|
| Implement `PosePerturbation` operator for LIBERO | Generates ≥ 3 variants per seed |
| Implement `DistractorAdd` operator for LIBERO | Inserts same-category distractors |
| Implement `InstructionParaphrase` operator (LLM-based) | 3 paraphrases per instruction |
| Implement `MaterialSwap` for Robocasa (texture/friction change) | For Robocasa seeds |
| Integrate `SeedExtractor` → `MutationEngine` pipeline | CLI: `sepa_eval generate --seed trace_id` |
| Write CandidateTasks to task table | DB write confirmed |

**Success criteria**:
- From a LIBERO failure seed, the engine generates ≥ 5 syntactically valid candidate tasks
- At least one generated candidate produces a different SR profile than the original task (validated by running 10-trial eval)

---

### Phase 4: Validation & Promotion (Weeks 12–14)

**Goal**: Gatekeep candidates; build an evolving task distribution.

| Task | Deliverable |
|---|---|
| Implement `SolvabilityGate` | Runs N=10 trials via AlphaBrain eval loop |
| Implement `ReproducibilityGate` | Checks failure recurrence rate |
| Implement `RedundancyGate` | Task embedding similarity check |
| Implement `DiscriminativePowerGate` | SR spread across models |
| Implement `HumanReviewQueue` | Simple JSON queue + CLI review tool |
| Implement full `PromotionPipeline` | Sequenced gate runner with status logging |
| Update reporting to include evolved task summary | Report section 4 |

**Success criteria**:
- Promotion pipeline processes 50 candidates and promotes ≥ 3 to the benchmark distribution
- All promoted tasks are confirmed solvable and reproducible by re-running eval

---

### Phase 5: Self-Evolution Loop + Full Orchestration (Weeks 15–18)

**Goal**: Automate the full Evaluate → Diagnose → Generate → Promote → Monitor → Report cycle.

| Task | Deliverable |
|---|---|
| Implement `EvolutionLoopOrchestrator` | Runs all 6 steps sequentially |
| Implement saturation detection | Flags tasks where all models ≥ 95% SR |
| Implement discriminative power decay monitoring | Alerts when power drops below 0.1 |
| Add `RobustnessCritic` (variant family aggregator) | Robustness scores per model |
| Streamlit reporting dashboard | Interactive model comparison view |
| Weekly schedule trigger (cron or GitHub Actions) | Automated weekly cycle |
| End-to-end integration test with LIBERO + Robocasa | CI test |

**Success criteria**:
- One complete evolution cycle runs end-to-end without manual intervention
- At least 5 tasks are promoted from the first full cycle
- Saturation detection correctly identifies LIBERO Spatial (currently 97%+ for all models) as saturated
- Report is generated automatically after each cycle

---

## 10. Metrics for the Evaluation System Itself

Following the paper's "evaluator metrics" category:

| Metric | Definition | Target |
|---|---|---|
| **Failure discovery rate** | Novel unique failure clusters found per cycle / total failure clusters observed | ≥ 2 new clusters per cycle after first 3 cycles |
| **Promotion yield** | Candidates promoted / total candidates generated | ≥ 10% |
| **Solvability accuracy** | Fraction of promoted tasks that remain solvable after re-validation | ≥ 95% |
| **Discriminative power retention** | Fraction of promoted tasks still discriminating models after 3 eval cycles | ≥ 70% |
| **Semantic critic calibration** | Agreement with simulator ground truth | ≥ 90% |
| **Saturation detection recall** | Fraction of actually saturated tasks correctly flagged | ≥ 90% |

---

## 11. Technical Constraints & Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Memory backend | SQLite + local file store | Zero-dependency, single-machine; upgrade path to PostgreSQL later |
| Trace compression | PNG (images) + float16 (actions) | 10× smaller than raw float32; lossless for images |
| Semantic critic model | GPT-4o-mini Vision or local Qwen2.5-VL | Reuses existing model server; avoids extra infra |
| Mutation operators | Simulator-level (not generative) | Reproducible; no world model dependency |
| Promotion automation | Fully automated except human relevance review | Human review is async and non-blocking |
| Task embedding | Instruction sentence embedding (BGE-M3) | Small, fast, no GPU needed; swappable |
| Clustering algorithm | DBSCAN | No preset cluster count; handles noise |
| CLI framework | `tyro` (already used in AlphaBrain eval) | Consistent with existing codebase conventions |

---

## 12. Open Questions

1. **Simulator state serialization**: Does LIBERO support deterministic init-state replay from a saved state snapshot? (Needed for `init_state` field in trace schema.) → Verify with LIBERO env API. Fallback: store full scene XML + object pose list; add `replay_mode: exact|reseed` config.

2. **Robocasa texture/friction API**: Does the Robocasa-Tabletop gym wrapper expose material parameter overrides at init time? → Check `robosuite` XML injection API. Fallback: skip `MaterialSwap` for Robocasa in Phase 3.

3. **Instruction paraphrase quality**: How do we verify that a paraphrase preserves the intended task rather than changing it? → Add a back-translation check and semantic similarity gate (cosine sim ≥ 0.9). Also: sanitize instruction text before LLM call (prompt injection guard).

4. **Multi-model comparative eval budget**: Running all registered models on all promoted tasks for discriminative power checks could be expensive. → Add a budget cap (max N model×task combinations per cycle) and prioritize newly promoted tasks.

5. **Human review queue**: Who reviews? What is the SLA? → For initial deployment, the research team reviews async; 3-business-day SLA. Promotion is not blocked by pending reviews.

6. **SemanticCritic endpoint**: The existing policy server exposes `predict_action`; a VQA/judge endpoint does not yet exist. → Add `/judge` endpoint to the model server in Phase 2. Alternatively, invoke the underlying model directly in a separate process.

7. **Circular evaluation bias**: When evaluating QwenOFT, the semantic critic should not use the same Qwen2.5-VL backbone. → Default critic: GPT-4o-mini. If evaluating a Qwen model, GPT-4o-mini is required. Document in `critics.yaml`.

8. **`CandidateTask` schema**: The mutation engine outputs `CandidateTask` objects but the dataclass is not yet defined. → Define in `sepa_eval/memory/schema.py` before Phase 3 begins (minimum: `task_id, parent_task_id, benchmark, instruction, scene_config: dict, mutation_type, mutation_params: dict, promotion_status`).

---

## 12b. Issues Requiring PRD Updates (From Architecture Review)

The following issues were identified during CEO review and must be addressed before implementation:

| ID | Issue | Priority | Resolution |
|----|-------|----------|------------|
| A1 | No `BenchmarkAdapter` protocol — hooks will diverge | High | Add `base.py` adapter protocol to `sepa_eval/hooks/` |
| A2 | `CandidateTask` unspecified | **Blocking Phase 3** | Define in `schema.py` |
| A3 | Promotion gates are blocking inline calls | High | Gates must dispatch to async eval worker pool |
| A4 | Dual consistency (DB + file store) crash safety | High | Write file first, then DB; add `fsck()` |
| A5 | `SemanticCritic` needs `/judge` endpoint | **Blocking Phase 2** | Add to policy server scope |
| Q3 | Zero tests specified for `sepa_eval/` | **Blocking Phase 1 ship** | Add test plan (see §9b) |
| O1 | No system-level metrics for the eval loop | Medium | Add `sepa_eval_metrics.json` output |
| O2 | No structured logging for orchestrator steps | Medium | Add `evolution_loop_log.jsonl` |
| L2 | No promotion evidence audit trail | Medium | Add `promotion_evidence` JSON column to tasks table |

## 9b. Test Plan (Added Post-Review, Expanded v2)

All tests in `sepa_eval/tests/`. Run: `pytest sepa_eval/tests/ -v`. Target: ~24 tests covering happy paths, crash paths, and API failure paths.

### Unit Tests

**EvalMemory (`test_eval_memory.py`)**
| Test | Description | Phase |
|------|-------------|-------|
| `test_record_trace_happy_path` | write .tmp → rename → DB row committed | Phase 1 |
| `test_record_trace_file_write_fails` | no DB row committed, .tmp cleaned up | Phase 1 |
| `test_record_trace_db_commit_fails` | file deleted (compensating write), no orphan | Phase 1 |
| `test_get_failures_empty_db` | returns `[]` not exception | Phase 1 |
| `test_update_critic_score_unknown_id` | raises `ValueError` | Phase 1 |
| `test_concurrent_writes_wal_mode` | 4 concurrent writes, all committed, no data loss | Phase 1 |
| `test_eval_memory_fsck` | orphan `.tmp` file detected and removed | Phase 1 |

**Failure Classifier (`test_failure_classifier.py`)**
| Test | Description | Phase |
|------|-------------|-------|
| `test_classify_8types` | returns one of 8 known failure types for each fixture trace | Phase 1 |
| `test_failure_step_per_type` | `failure_step` computed for each type (not None for classified) | Phase 1 |
| `test_fallback_unclassifiable` | returns `failure_type=None`, `failure_step=episode_length` | Phase 1 |

**Clustering (`test_failure_cluster.py`)**
| Test | Description | Phase |
|------|-------------|-------|
| `test_dbscan_empty_guard` | 0–4 failures → `[]`, no crash | Phase 1 |
| `test_cluster_window` | only last N runs' traces included | Phase 1 |
| `test_llm_summarizer_fails` | `llm_summary=null`, cluster still returned | Phase 2 |

**Mutation Operators (`test_mutation.py`)**
| Test | Description | Phase |
|------|-------------|-------|
| `test_pose_perturbation_valid_config` | output pose within ±δ of seed | Phase 3 |
| `test_instruction_paraphrase_mock_llm` | output passes cosine similarity gate ≥ 0.9 | Phase 3 |
| `test_instruction_paraphrase_llm_fails` | raises `MutationError`, does not crash | Phase 3 |
| `test_instruction_paraphrase_injection_guard` | instruction wrapped in `[START]...[END]` delimiters | Phase 3 |

**Critics (`test_critics.py`)**
| Test | Description | Phase |
|------|-------------|-------|
| `test_semantic_critic_judge_200` | `/judge` 200 → `CriticResult` with scores | Phase 2 |
| `test_semantic_critic_judge_500` | `/judge` 500 → raises `CriticError` | Phase 2 |
| `test_circular_bias_routing` | evaluating `QwenOFT` model → routes to GPT-4o-mini | Phase 2 |

**Promotion Gates (`test_promotion.py`)**
| Test | Description | Phase |
|------|-------------|-------|
| `test_promotion_gates_mock` | all 4 gates pass on fixture candidate | Phase 4 |
| `test_gate_timeout_deferred` | gate timeout → status=`deferred`, not `failed` | Phase 4 |

**Hard-Case Exporter (`test_exporter.py`)**
| Test | Description | Phase |
|------|-------------|-------|
| `test_hard_case_exporter_lerobot_format` | export() produces valid LeRobot-format dataset | Phase 2 |
| `test_exporter_zero_failures` | 0 failed episodes → empty dataset, no crash | Phase 2 |

### Integration Tests

| Test | Description | Phase |
|------|-------------|-------|
| `test_trace_pipeline_libero_mini` | eval_libero.py --trace (10 mini episodes) → trace files + DB rows | Phase 1 |
| `test_evolution_loop_e2e_libero_mini` | 10-episode eval → failure cluster → candidate → promoted | Phase 5 |

### Eval Tests (run on CI schedule, not on every PR)
| Test | Description | Trigger |
|------|-------------|---------|
| `test_semantic_critic_calibration` | ≥90% agreement with simulator ground truth on LIBERO-Spatial 100-ep set | prompt template changes, model upgrade |

Full test plan artifact: `~/.gstack/projects/Alchedata-sia-physicalAI-eval/fei-main-eng-review-test-plan-*.md`

---

## 13. Out of Scope (Explicitly Deferred)

- **Real-robot calibration**: Comparing simulation outcomes against physical robot execution (SEPA-Eval Phase 4)
- **World model as proxy evaluator**: Using video generation models to rank VLA policies without simulation
- **Physics critic**: Rule-based physics plausibility checker on simulator state (Phase 2 stretch goal)
- **New simulator support**: Adding Isaac Lab, GENESIS, or other simulators
- **Cross-embodiment testing**: Evaluating the same task on different robot morphologies
- **Automatic benchmark retirement**: Formal process for removing promoted tasks that have become stale (defer to after first full cycle)
- **FAISS ANN index**: Windowed DBSCAN sufficient for Phase 1-4; defer to Phase 5+ (TODOS.md ENG-P2)
- **SQLite → DuckDB/Parquet migration**: Design migration path before Phase 4; no action Phase 1-3 (TODOS.md ENG-P1)

---

## GSTACK REVIEW REPORT

Generated by `/plan-eng-review` — 2026-05-21

### CEO Review Summary

| Section | Verdict | Key Decisions |
|---------|---------|---------------|
| Scope challenge | Accepted as-is (complexity noted) | VLA-only, 5 phases |
| Cherry-picks | 3/4 accepted (CP3, CP4, CP5) | Hard-case exporter, cross-model heatmap, LLM cluster summarizer |
| Architecture concerns | 5 flagged, 5 resolved | See CEO plan |
| Open questions expanded | 5 → 8 | OQ6-OQ8 added |
| CEO Review Score | 5.9/10 → 7.8/10 post-decisions | 3 blocking items resolved |

### Eng Review Summary

| Section | Issues Found | Issues Resolved | Decision Quality |
|---------|-------------|-----------------|-----------------|
| Scope challenge | Accepted | — | — |
| Architecture | 5 | 5 | All resolved via AskUserQuestion |
| Code Quality | 4 | 4 | All resolved (3 obvious fixes + 1 security fix) |
| Test Coverage | 1 | 1 | 9 tests → 24 tests (crash paths added) |
| Performance | 2 | 2 | DBSCAN windowing + batch compression |
| Outside Voice | — | — | PASS WITH CONCERNS (GPT-4.1) |

### Issues Log

| ID | Category | Description | Decision |
|----|----------|-------------|----------|
| Arch 1.1 | Architecture | No BenchmarkAdapter protocol | Add 4-method protocol to hooks/base.py |
| Arch 1.2 | Architecture | CandidateTask undefined | Defined in schema.py (10 fields + evidence dict) |
| Arch 1.3 | Architecture | Promotion gates block 29h | ThreadPoolExecutor; gate_timeout_minutes=120 → deferred |
| Arch 1.4 | Architecture | Dual-write crash safety | .tmp→rename→DB; compensating write on DB fail; fsck() |
| Arch 1.5 | Architecture | No text endpoint for semantic critic | Add /judge endpoint to policy server (Phase 2) |
| CQ 2.1 | Code Quality | EpisodeTrace flat 25+ field struct | Nested sub-dataclasses: 5 groups |
| CQ 2.2 | Code Quality | failure_step undefined algorithm | FailureStepDetector per type; fallback=episode_length |
| CQ 2.3 | Code Quality | Prompt injection in InstructionParaphrase | [START]...[END] delimited template |
| CQ 2.4 | Code Quality | No SQLite WAL mode | PRAGMA journal_mode=WAL; busy_timeout=5000 at init |
| Test 3.1 | Tests | 16% path coverage (9/~38 tests) | Expand to ~24 tests; crash paths + API failures added |
| Perf 4.1 | Performance | DBSCAN O(n²) unbounded | Windowed: last_n_runs=5 config; FAISS deferred to Phase 5+ |
| Perf 4.2 | Performance | PNG compression blocking episode loop | Batch compress at on_episode_end() for --store-obs=True |

### Review Readiness Dashboard

| Dimension | Before Eng Review | After Eng Review |
|-----------|------------------|-----------------|
| Architecture completeness | 6/10 | 9/10 |
| Code quality spec | 5/10 | 8/10 |
| Test coverage spec | 2/10 | 7/10 |
| Performance readiness | 5/10 | 8/10 |
| Security | 7/10 | 9/10 (injection guard added) |
| **Overall** | **5.0/10** | **8.2/10** |

### Blocking Items (all resolved)

| Item | Status |
|------|--------|
| A2: CandidateTask schema | ✅ Defined in PRD §6.7 |
| A5: /judge endpoint | ✅ Full spec in PRD §8.3 |
| Q3: Test plan | ✅ 24 tests in PRD §9b |

### Remaining High-Priority Before Phase 1

- T1: Lock in trace hook call signature (context manager, on_episode_end flush)
- T2: Lock in file store layout (eval_memory/{run_id}/{trace_id}.msgpack)
- T3: Add SEPA_MEMORY_DIR env var
- OQ1: Test LIBERO init-state replay determinism

### Next Step Recommendation

Start Phase 1 implementation with two parallel lanes:
- **Lane A**: `sepa_eval/memory/` — EvalMemory class, schema.py, WAL mode, fsck()
- **Lane B** (after schema stable): `sepa_eval/hooks/` — BenchmarkAdapter protocol, LiberoHook

First PR: `sepa_eval/memory/` with 7 unit tests passing.

