<div align="center">

<img src="assets/alchedata_logo.jpeg" alt="Alchedata" width="220"/>

# SEPA-Eval

### Self-Evolving Physical AI Evaluation

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](AlphaBrain/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Built on AlphaBrain](https://img.shields.io/badge/Built%20on-AlphaBrain-teal.svg)](AlphaBrain/README.md)
[![Paper](https://img.shields.io/badge/Paper-White%20Paper-orange.svg)](paper/main.pdf)

<p align="center">
  <img src="assets/data_flywheel.png" width="72%" alt="SEPA-Eval Closed-Loop Data Flywheel"/>
</p>

**SEPA-Eval** turns a static VLA benchmark into a living one. It records rollout failures, clusters them, mutates the worst-case scenarios into harder variants, validates them through a critic ensemble, and automatically promotes them into the active benchmark distribution — closing the loop between evaluation and continual learning.

[Architecture](#architecture) · [Installation](#installation) · [Quick Start](#quick-start) · [CLI Reference](#cli-reference) · [Testing](#testing-the-package) · [Demo](#customer-demo-workflow) · [Paper](#citation)

</div>

---

## The Problem with Static Benchmarks

| Problem | What Happens |
|---|---|
| **Saturation** | Frontier models exceed 95%+ SR on LIBERO; the benchmark stops discriminating |
| **Misalignment** | Binary success rate misses fragile success, unsafe contact, and brittle grasp recovery |
| **Silent drift** | Scores look meaningful long after they stop predicting real deployment behavior |

SEPA-Eval solves all three by making the benchmark itself a first-class output of the evaluation process.

---

## Key Capabilities

- **Structured trace recording** — every episode emits observations, actions, contact events, success labels, and detected failure moments to a crash-safe SQLite + msgpack store
- **Failure mining** — heuristic detectors (timeout, out-of-reach, grasp, contact dynamics, recovery, pose, distractor, language) feed into DBSCAN clustering over compact trace embeddings
- **Scenario mutation** — pose perturbation, material swap, distractor injection, instruction paraphrase, and horizon extension operators generate harder task candidates from failure seeds
- **Critic ensemble** — semantic critic (VLM judge), safety critic (force/collision heuristics), and robustness critic score each candidate independently
- **Promotion pipeline** — async gate execution with configurable timeouts; evidence recorded per task; human review queue for borderline decisions
- **Hard-case export** — failed episodes exported as `jsonl` or LeRobot-style parquet datasets, ready for AlphaBrain's continual learning pipeline
- **Self-evolution loop** — a single orchestrator command runs the full Evaluate → Diagnose → Generate → Validate → Promote → Report cycle
- **`/judge` VLM endpoint** — the policy server exposes an opt-in HTTP judge endpoint (`--judge_port` / `SEPA_JUDGE_PORT`) so the semantic critic can reuse the loaded VLA/VLM; graceful GPT-4o-mini fallback
- **Mutation replay** — pose-mutated scene configs are re-applied to LIBERO via MuJoCo init-state injection; distractor mutations via BDDL generation with explicit degradation reporting

---

## Architecture

<p align="center">
  <img src="assets/sepa_eval_architecture.png" width="55%" alt="SEPA-Eval System Architecture"/>
</p>

The closed-loop pipeline runs six stages on every evolution cycle:

1. **Evaluate** — run VLA policies on the current benchmark distribution (LIBERO / RoboCasa) via AlphaBrain's WebSocket eval harness; record structured rollout traces
2. **Diagnose** — classify failures by type and cluster traces with DBSCAN to extract reproducible failure seeds
3. **Generate** — apply mutation operators to failure seeds to produce harder task candidates
4. **Validate** — score candidates through the critic ensemble; defer on timeout rather than hard-fail
5. **Promote** — admit candidates that clear all gates into the live benchmark distribution; log evidence per task
6. **Report** — surface capability frontiers, saturation signals, and model comparison across evolved benchmark slices

### Package Structure

```
AlphaBrain/sepa_eval/
  hooks/          BenchmarkAdapter protocol + LiberoHook, RobocasaHook
  memory/         EvalMemory (SQLite WAL + msgpack trace store), schema
  mining/         FailureStepDetector, DBSCAN clustering, seed extraction, LLM summarizer
  mutation/       PosePerturbation, MaterialSwap, DistractorAdd, InstructionParaphrase, HorizonExtension
  critics/        Semantic critic (VLM), safety critic, robustness critic
  promotion/      Validation pipeline, promotion gates, human review queue
  replay/         LIBERO init-state replay for mutated scene configs
  orchestrator/   Self-evolution loop
  exporter/       Hard-case dataset export (jsonl + LeRobot parquet)
  registry/       models.yaml-backed model registry + DB sync
  reporting/      Markdown report generation, task-model heatmaps
  configs/        mutation.yaml, orchestrator defaults, model configs
  tests/          Unit and integration tests
```

---

## Installation

### SEPA-Eval package only

The `sepa_eval` package lives inside `AlphaBrain/`. Install from that subdirectory.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ./AlphaBrain
pip install -e "./AlphaBrain[dev]"   # adds black, ruff, pre-commit
```

All runtime dependencies (`msgpack`, `scikit-learn`, `jinja2`, `pyyaml`, `openai`, `requests`, `pyarrow`, `Pillow`, `numpy`) are declared in `pyproject.toml` and installed automatically. After installation, the `sepa-eval` CLI is available directly on your PATH.

| Optional extra | Required for |
|---|---|
| `openai` | Instruction paraphrase mutation + GPT-4o-mini fallback critic |
| `pyarrow` | LeRobot-style parquet export (graceful JSONL fallback if absent) |
| `pytest` | Running the test suite |

### Full benchmark setup (LIBERO + RoboCasa)

Running SEPA-Eval against real simulators requires three separate Python environments and several dataset downloads. See **[SETUP.md](SETUP.md)** for the complete step-by-step guide covering:

- AlphaBrain conda environment
- LIBERO simulator installation and data preparation
- RoboCasa Tabletop environment (robosuite 1.5.1 + tabletop fork + assets)
- RoboCasa 365 environment
- `.env` variable reference with correct path conventions
- Verification commands and common pitfalls

---

## Quick Start

All commands run from the `AlphaBrain/` directory.

All commands use the `sepa-eval` CLI installed with the package, or equivalently `python -m sepa_eval`.

### 1. Check memory status

```bash
sepa-eval --memory-dir ./eval_memory status
```

### 2. Register / sync models

```bash
sepa-eval --memory-dir ./eval_memory sync-models
```

### 3. Run the evolution loop

```bash
sepa-eval --memory-dir ./eval_memory run
```

### 4. Generate a Markdown report

```bash
sepa-eval --memory-dir ./eval_memory report --output ./eval_memory/report.md
```

### 5. Review pending promotion candidates

```bash
sepa-eval --memory-dir ./eval_memory review list
sepa-eval --memory-dir ./eval_memory review approve <task_id>
```

---

## CLI Reference

| Command | Description |
|---|---|
| `run` | Run the full closed-loop evolution cycle |
| `eval` | Register a model and run CI-style evaluation checks |
| `promote` | Run the promotion pipeline on pending candidates |
| `report` | Generate a Markdown report with heatmaps |
| `export-hard-cases` | Export failed episodes as a training dataset |
| `diff` | Compare task-level success rates between two models |
| `sync-models` | Sync `configs/models.yaml` into the database |
| `prune` | Remove old trace files while preserving DB rows |
| `review list` | List candidates awaiting human review |
| `review approve` | Promote a candidate manually |
| `status` | Show memory statistics and evolution metrics |

Default storage root: `SEPA_MEMORY_DIR` env var, or `./eval_memory/`.

---

## Wiring into an AlphaBrain Eval Run

```python
from sepa_eval.hooks.libero_trace_hook import run_libero_episode_with_trace
from sepa_eval.memory.eval_memory import EvalMemory
from sepa_eval.memory.schema import TraceIdentity

memory = EvalMemory(
    db_path="../eval_memory/eval.db",
    memory_dir="../eval_memory/traces",
)

trace = run_libero_episode_with_trace(
    env=libero_env,
    policy_fn=policy_fn,
    memory=memory,
    identity=TraceIdentity(
        trace_id="trace-001",
        eval_run_id="run-001",
        benchmark="libero_spatial",
        task_id="pick_red_cup",
        task_instruction="pick up the red cup",
        model_id="alphabrain-v2",
        model_version="v2.1",
    ),
)
```

RoboCasa: replace `libero_trace_hook` with `robocasa_trace_hook` and `run_robocasa_episode_with_trace`.

---

## Customer Demo Workflow

A deterministic, simulator-free walkthrough using synthetic traces.

### 1. Generate synthetic traces

```bash
cd AlphaBrain
python ../demo/generate_libero_traces.py --output-dir ../demo/demo_eval_memory
```

### 2. Run the guided demo

```bash
cd demo
./run_customer_demo.sh               # interactive, pauses between stages
./run_customer_demo.sh --no-pause    # fully automatic
./run_customer_demo.sh --regenerate  # rebuild traces first
```

### 3. Open the HTML dashboard

```bash
python demo/render_report_html.py \
  --input  demo/output/report.md \
  --output demo/output/report.html
open demo/output/report.html
```

Artifacts produced: `demo/output/report.md`, `demo/output/report.html`, `demo/demo_eval_memory/` (SQLite + msgpack traces).

---

## Testing the Package

Everything below runs **without any simulator, GPU, or model checkpoint** — it is the fastest way to verify the package works on your machine.

### 1. Install and run the test suite

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e "./AlphaBrain[dev]"
pip install pytest

pytest AlphaBrain/sepa_eval/tests -q
# expected: 140 passed, 1 skipped (the skip is the live-simulator e2e)
```

> **Known environment pitfall:** on some Anaconda installs an outdated `threadpoolctl`
> makes scikit-learn's DBSCAN crash with `AttributeError: 'NoneType' object has no
> attribute 'split'`. Fix with `pip install -U threadpoolctl`.

The suite covers **141 tests** across all modules, including a *contract test layer*
(`test_contract_real_components.py`) that assembles the real orchestrator, memory,
mining, mutation, critics, and promotion pipeline together — only simulator rollouts
and LLM calls are stubbed.

### 2. Smoke-test the CLI on an empty memory

```bash
cd AlphaBrain
export SEPA_MEMORY_DIR=$(mktemp -d)/mem
sepa-eval run        # full 6-step cycle on empty memory — should finish cleanly
sepa-eval status     # prints metrics incl. critic_latency_ms
```

### 3. End-to-end walkthrough with synthetic data

Use the [Customer Demo Workflow](#customer-demo-workflow) above: it generates 200
synthetic LIBERO traces and drives the full mine → mutate → validate → promote →
report pipeline, producing a browsable HTML dashboard.

### 4. Test the `/judge` endpoint (optional, no model needed)

```bash
pytest AlphaBrain/sepa_eval/tests/test_judge_endpoint.py -v
```

This spins up the real HTTP judge server with a fake framework and exercises the
`SemanticCritic` client against it, including the 501 → GPT-4o-mini fallback path.

### Lint

```bash
ruff check AlphaBrain/sepa_eval/
black --check AlphaBrain/
```

Production code passes ruff with zero violations.

---

## Current Status

**Implemented and test-covered (141 tests, ruff clean):**
- Trace hooks and benchmark adapters (LIBERO, RoboCasa)
- Persistent memory (SQLite WAL + msgpack) with crash-safe writes, portable
  relative trace paths, `user_version` schema migrations, true critic-score
  upserts, secondary indexes, thread-safe shared connection, and `fsck` orphan /
  broken-link reporting
- Failure classification, clustering, and seed extraction (scene_config recovered
  from msgpack traces)
- Full mutation operator suite (5 operators)
- Critic ensemble wired into the evolution loop (safety + robustness always on;
  semantic critic opt-in via config)
- Promotion gates + async pipeline with per-candidate status/evidence persistence
- `/judge` HTTP endpoint in the policy server (opt-in via `--judge_port` /
  `SEPA_JUDGE_PORT`), matching the `SemanticCritic` client contract
- LIBERO init-state replay for pose mutations + BDDL distractor injection
  (`sepa_eval/replay/`), with explicit degradation reporting
- WebSocket policy client with per-request timeout and bounded reconnect/retry
- Orchestrator loop, real critic-latency metrics, and review CLI
- Reporting with task×model heatmaps, hard-case export, and model registry
- `sepa-eval` CLI entry point; all runtime deps declared in `pyproject.toml`

**Pending / in progress:**
- Full end-to-end evolution loop verified against a live simulator (all code
  paths are in place; needs a LIBERO environment + model checkpoint)
- RoboCasa material override fidelity (robosuite texture injection)
- Long-term storage migration beyond SQLite (DuckDB/Parquet, ANN indexing for
  large trace volumes)

See [TODOS.md](TODOS.md) for the detailed implementation checklist.

---

## Citation

If you use SEPA-Eval in your research, please cite:

```bibtex
@article{alchedata2026sepa,
  title   = {Self-Evolving Agents for Physical AI Evaluation},
  author  = {Alchedata},
  year    = {2026},
  note    = {White paper. \url{https://github.com/alchedata/sepa-eval}}
}
```

---

## Repository Layout

```
.
├── AlphaBrain/              AlphaBrain VLA framework + sepa_eval package
│   └── sepa_eval/           SEPA-Eval implementation
├── demo/                    Synthetic trace generator and HTML dashboard renderer
├── paper/                   ACM acmart white paper source
├── docs_analysis/           Architecture evaluation reports + improvement plan
├── assets/                  Diagrams and logos
├── PRD_SEPA_VLA_Eval.md     Product and system specification
└── TODOS.md                 Implementation status tracker
```

---

## License

SEPA-Eval is released under the [MIT License](AlphaBrain/LICENSE).

---

<div align="center">

<img src="assets/alchedata_logo.jpeg" alt="Alchedata" width="160"/>

**[Alchedata](https://alchedata.com)** — DATA INFRA 2.0 for Physical AI

*Closed-loop eval → data → better checkpoints, automatically.*

</div>
