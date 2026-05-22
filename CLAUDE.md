# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repository Is

This repo implements **SEPA-Eval** — a self-evolving VLA evaluation system built on top of **AlphaBrain**. The system extends AlphaBrain's static benchmark evaluation with a closed-loop pipeline that observes failures, generates harder test variants, validates them, and promotes them into the live benchmark distribution.

The full specification lives in [PRD_SEPA_VLA_Eval.md](PRD_SEPA_VLA_Eval.md). Open tasks are tracked in [TODOS.md](TODOS.md).

## Repository Layout

| Path | Contents |
|---|---|
| `AlphaBrain/` | VLA framework (training, evaluation, deployment, benchmarks) |
| `paper/` | ACM `acmart` LaTeX source for the SEPA-Eval white paper |
| `PRD_SEPA_VLA_Eval.md` | Full product requirements doc (v0.2) |
| `TODOS.md` | Prioritized implementation checklist generated from CEO + Eng review |
| `AlphaBrain/sepa_eval/` | Implemented SEPA-Eval Python package |

## SEPA-Eval Package Structure

The `sepa_eval/` package now lives under `AlphaBrain/sepa_eval/`. Its current layout is:

```
AlphaBrain/sepa_eval/
  hooks/          # BenchmarkAdapter protocol + LiberoHook, RobocasaHook
  memory/         # EvalMemory (SQLite + msgpack file store), schema.py
  mining/         # failure clustering, FailureStepDetector, LLM summarizer
  mutation/       # PosePerturbation, MaterialSwap, DistractorAdd, etc.
  critics/        # semantic critic (VLM), safety critic, robustness critic
  promotion/      # validation pipeline + promotion gates
  orchestrator/   # self-evolution loop
  exporter/       # hard-case export for continual learning
  registry/       # models.yaml-backed model registry
  reporting/      # report generation + heatmaps
  configs/        # mutation.yaml, eval defaults
  tests/          # unit and integration tests
```

Current implementation status:
- Core memory, hooks, mining, mutation, promotion, reporting, exporter, and registry modules are implemented.
- The CLI entry point lives in `AlphaBrain/sepa_eval/__main__.py`.
- End-to-end real-simulator benchmark wiring is partially surfaced through hook helpers, but full simulator-backed evolution-loop verification is still pending.

Key invariants:
- All benchmarks implement `BenchmarkAdapter` (4 methods: `reset`, `step`, `get_task_id`, `get_scene_config`). New benchmarks = one new file implementing this protocol.
- Trace writes use `.tmp`→rename pattern; `EvalMemory.fsck()` cleans orphans.
- SQLite must be opened with `PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000`.
- Promotion gates run async via `ThreadPoolExecutor`; timeouts defer (don't fail) the candidate.
- The policy server's `/judge` endpoint is reused for semantic critic and LLM cluster summarization.

## AlphaBrain Architecture

AlphaBrain uses a **client-server WebSocket architecture** separating the simulation environment from model inference:

- **Policy server** (`AlphaBrain/deployment/model_server/server_policy.py`): loads a checkpoint, serves predictions over WebSocket
- **Benchmark clients** (`AlphaBrain/benchmarks/*/eval/`): run the simulation, call the server for actions
- **Framework adapters** (`AlphaBrain/AlphaBrain/model/framework/`): model-specific inference logic (QwenOFT, QwenPI, PaliGemmaOFT, NeuroVLA, ACT, CosmosPolicy, etc.)
- **`*2model_interface.py` files**: per-benchmark wrappers handling action space conversion and simulator quirks

Training entry point: `AlphaBrain/AlphaBrain/training/train_alphabrain.py` (launched via accelerate).

## Common Commands

All commands run from `AlphaBrain/` unless noted.

**Setup:**
```bash
cp .env.example .env   # fill in PRETRAINED_MODELS_DIR, LIBERO_DATA_ROOT, LIBERO_HOME, etc.
pip install -e .
pip install -e ".[dev]"  # adds black, ruff, pre-commit
```

**Fine-tune a VLA model:**
```bash
bash scripts/run_finetune.sh <mode>
# e.g.: bash scripts/run_finetune.sh qwen_oft
# Config in configs/finetune_config.yaml; modes override it at the bottom of that file
```

**Evaluate a checkpoint:**
```bash
bash scripts/run_eval.sh <mode> [config_file]
# e.g.: bash scripts/run_eval.sh libero_eval
# Automatically: starts policy server → runs benchmark client → shuts down server
# Results land in results/evaluation/<benchmark>/<checkpoint_slug>/
```

**Evaluate all LIBERO suites:**
```bash
# Set TASK_SUITE=libero_all in config or mode; run_eval.sh runs all 4 suites sequentially
bash scripts/run_eval.sh libero_eval
```

**Lint:**
```bash
ruff check .
black --check .
```

**Paper (from `paper/`):**
```bash
make          # pdflatex + bibtex (3-pass)
make clean    # remove aux files only
make realclean  # also removes PDF and .bbl
```

## Configuration System

`AlphaBrain/configs/finetune_config.yaml` is the single config entry point. It has a `modes:` block where each mode name maps to overrides for `framework`, `datasets`, `training`, and `eval`. The `scripts/parse_config.py` script resolves the selected mode and exports shell variables consumed by the training/eval scripts.

Priority (low → high): model defaults → dataset defaults → trainer defaults → mode overrides.

Environment variables in `.env` are referenced in YAML via `${oc.env:VAR}` and are sourced automatically by the shell scripts.

## Multi-Python Environment Note

Different benchmarks require different conda environments:
- `LIBERO_PYTHON`: Python with LIBERO simulation deps
- `ROBOCASA_TABLETOP_PYTHON`: Python with RoboCasa deps
- `SERVER_PYTHON` (default): the AlphaBrain Python with model deps

These are set in `.env` and automatically used by `run_eval.sh`.

## Linting Configuration

- Line length: 121 characters (black + ruff)
- Target: Python 3.10+
- Ruff rules: A, B, E, F, I, RUF, W (F722 ignored; `__init__.py` E402/F401 ignored)
