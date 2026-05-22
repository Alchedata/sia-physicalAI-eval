# SEPA-Eval: Self-evolving Evaluation System for Physical AI 

SEPA-Eval is a self-evolving evaluation system for vision-language-action (VLA) models built on top of AlphaBrain. Instead of treating a benchmark as fixed, SEPA-Eval records failures, clusters them, generates harder task variants, validates those candidates through promotion gates, and feeds successful tasks back into the benchmark distribution.

This repository contains:

- `AlphaBrain/`: the upstream training, deployment, and benchmark framework
- `AlphaBrain/sepa_eval/`: the SEPA-Eval package that is now implemented in this repo
- `PRD_SEPA_VLA_Eval.md`: the product and system spec
- `TODOS.md`: the status tracker for implemented and pending work
- `paper/`: the white paper source

## Current Status

The repository is no longer just a plan. The core SEPA-Eval package exists under `AlphaBrain/sepa_eval` and includes working implementations for:

- trace hooks and benchmark adapters
- persistent memory with SQLite WAL mode plus msgpack trace files
- failure classification and clustering
- seed extraction and multiple mutation operators
- promotion gates and asynchronous promotion pipeline execution
- orchestrator logging and metrics output
- reporting, hard-case export, model registry sync, and review CLI commands
- package-level tests under `AlphaBrain/sepa_eval/tests`

What is still intentionally incomplete or pending external verification:

- long-term storage migration planning beyond SQLite
- ANN indexing for larger trace volumes
- simulator-specific validation such as LIBERO state replay and RoboCasa material override behavior
- final integration details around external judge and critic backends

For the most up-to-date task checklist, see [TODOS.md](TODOS.md).

## Architecture

SEPA-Eval extends AlphaBrain's existing evaluation flow with a closed loop:

```text
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

1. Evaluate a model on a benchmark and record traces.
2. Diagnose failures using heuristic classifiers and clustering.
3. Extract representative seeds from failure clusters.
4. Generate mutated candidate tasks.
5. Validate candidates with promotion gates.
6. Promote accepted tasks and report the new benchmark frontier.

At the code level, the package is organized as:

```text
AlphaBrain/sepa_eval/
  hooks/         BenchmarkAdapter protocol + trace hooks
  memory/        EvalMemory, schema, trace persistence
  mining/        Failure classifier, clustering, seed extraction
  mutation/      Scenario mutation operators
  critics/       Semantic, safety, and robustness critics
  promotion/     Promotion gates, pipeline, human review queue
  orchestrator/  Evolution loop
  exporter/      Hard-case dataset export
  registry/      models.yaml-backed model registry
  reporting/     Markdown report generation and heatmaps
  configs/       Orchestrator, mutation, critics, and model config
  tests/         Unit and integration tests
```

## Implemented Features

### Memory and Trace Storage

- `EvalMemory` stores metadata in SQLite and rollout traces as msgpack files.
- SQLite is opened with `PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000`.
- Trace writes use a crash-safe `.tmp` to atomic rename flow.
- `EvalMemory.fsck()` removes orphan temp files.
- `EvalMemory.prune()` deletes old trace files while preserving database rows.

### Hooks and Adapters

- `BenchmarkAdapter` is defined as a protocol with version pinning via `PROTOCOL_VERSION`.
- `TraceHook` is a context manager that accumulates per-step data and flushes on `on_episode_end()`.
- Benchmark-specific hooks exist for LIBERO and RoboCasa.

### Failure Mining and Mutation

- Heuristic failure detectors exist for timeout, out-of-reach, grasp, contact dynamics, recovery, pose estimation, distractor confusion, and language grounding.
- Failure clustering is implemented with DBSCAN over compact trace embeddings.
- Seed extraction is implemented from representative cluster traces.
- Mutation operators currently include pose perturbation, distractor injection, instruction paraphrase, material swap, and horizon extension.

### Promotion and Orchestration

- Promotion gates run under `ThreadPoolExecutor` timeouts and defer instead of hard-failing on timeout.
- Promotion evidence is recorded per task.
- The evolution loop writes `evolution_loop_log.jsonl` and `sepa_eval_metrics.json`.
- Review workflows are exposed through CLI commands.

### Reporting and Export

- Markdown report generation is implemented.
- Task-model heatmap generation is implemented.
- Failed episode export supports `jsonl` and LeRobot-style output layouts.
- A YAML-backed model registry is implemented through `configs/models.yaml` and DB sync.

## Installation

The Python package lives under `AlphaBrain/`, so install from there or point pip at that subdirectory.

### Base install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ./AlphaBrain
```

### Dev install

```bash
pip install -e "./AlphaBrain[dev]"
```

### Common optional dependencies

SEPA-Eval uses several optional packages depending on which features you run:

```bash
pip install msgpack scikit-learn jinja2 pyyaml openai pyarrow pytest
```

Notes:

- `msgpack` is required for trace persistence.
- `scikit-learn` is required for failure clustering.
- `jinja2` is used for templated report rendering.
- `pyyaml` is used for config and model registry loading.
- `openai` is used by the instruction paraphrase operator.
- `pyarrow` enables parquet output for LeRobot-style export.

If you want to run AlphaBrain benchmark evaluation end to end, also follow the environment setup in `AlphaBrain/.env.example` and the benchmark-specific Python environment requirements described in [CLAUDE.md](CLAUDE.md).

## Quick Start

### 1. Inspect memory status

```bash
cd AlphaBrain
python -m sepa_eval --memory-dir ../eval_memory status
```

### 2. Register or sync models

```bash
cd AlphaBrain
python -m sepa_eval --memory-dir ../eval_memory sync-models
```

### 3. Run the evolution loop

```bash
cd AlphaBrain
python -m sepa_eval --memory-dir ../eval_memory run
```

### 4. Generate a report

```bash
cd AlphaBrain
python -m sepa_eval --memory-dir ../eval_memory report --output ../eval_memory/report.md
```

### 5. Review pending tasks

```bash
cd AlphaBrain
python -m sepa_eval --memory-dir ../eval_memory review list
python -m sepa_eval --memory-dir ../eval_memory review approve <task_id>
```

## AlphaBrain Harness Integration

SEPA-Eval now exposes the main hook surface needed to attach trace collection to a real AlphaBrain benchmark run, but this should be treated as a wiring interface rather than a fully simulator-verified path.

What is implemented today:

- `LiberoHook` and `run_libero_episode_with_trace()` in `AlphaBrain/sepa_eval/hooks/libero_trace_hook.py`
- `RobocasaHook` and `run_robocasa_episode_with_trace()` in `AlphaBrain/sepa_eval/hooks/robocasa_trace_hook.py`
- `TraceHook` in `AlphaBrain/sepa_eval/hooks/base.py` for per-step accumulation and final trace flush
- `EvalMemory.record_trace()` in `AlphaBrain/sepa_eval/memory/eval_memory.py` for persistent storage

Current verification level:

- the hook and memory layers are implemented and test-covered
- the current integration test verifies trace persistence with synthetic traces
- the full LIBERO evolution-loop e2e test is still marked as a placeholder, so real-simulator end-to-end verification is not complete yet

If you want to wire SEPA-Eval into a real AlphaBrain eval harness today, the minimal pattern is:

```python
from sepa_eval.hooks.libero_trace_hook import run_libero_episode_with_trace
from sepa_eval.memory.eval_memory import EvalMemory
from sepa_eval.memory.schema import TraceIdentity

memory = EvalMemory(db_path="../eval_memory/eval.db", memory_dir="../eval_memory/traces")

trace = run_libero_episode_with_trace(
  env=libero_env,
  policy_fn=policy_fn,
  memory=memory,
  identity=TraceIdentity(
    trace_id="trace-001",
    eval_run_id="run-001",
    benchmark="libero_spatial",
    task_id="some_task",
    task_instruction="pick up the red cup",
    model_id="model_x",
    model_version="v1",
  ),
)
```

In practice, the missing work is not the trace API itself; it is validating the exact simulator-specific state capture, replay fidelity, and benchmark-runner insertion points inside AlphaBrain's real evaluation scripts.

## CLI Overview

The current CLI entry point is `python -m sepa_eval` and supports:

- `run`: run the closed-loop evolution cycle
- `eval`: register a model and run CI-style checks
- `promote`: run the promotion pipeline on candidate tasks
- `report`: generate a Markdown report
- `export-hard-cases`: export failed episodes
- `diff`: compare task-level success rates between two models
- `sync-models`: sync `configs/models.yaml` into the database
- `prune`: remove old trace files
- `review list`: list candidate tasks awaiting review
- `review approve`: mark a candidate as promoted
- `status`: show memory and metrics status

By default, SEPA-Eval uses `SEPA_MEMORY_DIR` or `./eval_memory/` as its storage root.

## Development and Testing

The current SEPA-Eval tests live in `AlphaBrain/sepa_eval/tests` and cover:

- memory and trace persistence
- critics
- mutation operators
- failure classification and clustering
- promotion pipeline behavior
- exporter and registry logic
- integration-level flows

Run the package tests with:

```bash
cd AlphaBrain
pytest sepa_eval/tests
```

Repository-wide linting for the AlphaBrain package is:

```bash
cd AlphaBrain
ruff check .
black --check .
```

## Repository Layout

```text
.
├── AlphaBrain/              AlphaBrain framework and the sepa_eval package
├── paper/                   White paper source
├── PRD_SEPA_VLA_Eval.md     Product and system specification
├── TODOS.md                 Implementation status tracker
├── CLAUDE.md                Repo-specific engineering notes
└── README.md                This file
```

## Known Gaps

The current package is substantial, but this repo should still be treated as an active implementation rather than a finished product release. In particular:

- some external integrations are specified more completely than they are validated in this repo
- AlphaBrain benchmark execution still depends on simulator-specific environment setup
- long-term scale work is intentionally deferred until trace volume justifies it

## Related Documents

- [PRD_SEPA_VLA_Eval.md](PRD_SEPA_VLA_Eval.md)
- [TODOS.md](TODOS.md)
- [CLAUDE.md](CLAUDE.md)
- [AlphaBrain/README.md](AlphaBrain/README.md)

## License

The repository includes AlphaBrain under the MIT license. See `AlphaBrain/LICENSE` for the packaged framework license.
