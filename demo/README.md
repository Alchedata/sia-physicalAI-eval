# SEPA-Eval Demo

This directory contains everything needed to run an offline demo of the
SEPA-Eval pipeline without a live GPU or simulator.

## Quick Start

```bash
# 1. Install the package (once)
cd AlphaBrain
pip install -e . --ignore-requires-python

# 2. Generate 100 synthetic LIBERO episode traces
python ../demo/generate_synthetic_traces.py

# All downstream commands use the generated data in demo/demo_eval_memory/
MEM=../demo/demo_eval_memory

# 3. Show the capability frontier + heatmap report
python -m sepa_eval --memory-dir $MEM report

# 4. Compare the two models side-by-side
python -m sepa_eval --memory-dir $MEM diff QwenOFT-v2.1 NeuroVLA-v1.2

# 5. Show tasks awaiting human review (mutation candidates from the evolution loop)
python -m sepa_eval --memory-dir $MEM review list

# 6. Show memory stats (traces, clusters, promoted tasks)
python -m sepa_eval --memory-dir $MEM status
```

## What the Data Contains

| Item | Count |
|---|---|
| Episode traces (msgpack) | 100 |
| Benchmarks | `libero_spatial` (60 eps) · `libero_goal` (40 eps) |
| Models | `QwenOFT-v2.1` · `NeuroVLA-v1.2` |
| Tasks | 20 seed tasks + 3 evolution mutations |
| Failed episodes | 25 (8 failure types) |
| Failure clusters | 9 |
| Saturated tasks | 8 (all models ≥ 90 % SR) |

## Demo Narrative

| Slide | What to show | CLI command |
|---|---|---|
| **Problem** | libero_spatial is nearly saturated — both models exceed 90% SR | `report` → Saturation Map section |
| **Frontier** | libero_goal reveals the gap: QwenOFT 74% vs NeuroVLA 44% | `diff QwenOFT-v2.1 NeuroVLA-v1.2` |
| **Diagnosis** | 5 distinct failure clusters on the goal suite | `report` → Failure Taxonomy |
| **Heatmap** | Per-task weakness of each model | `report` → Cross-Model Heatmap |
| **Evolution** | SEPA-Eval promoted 2 harder variants and queued 1 for review | `review list` |
| **Loop closes** | Export failures as continual-learning dataset | `export-hard-cases` |

## Re-generating

```bash
# Fresh run (deletes existing demo_eval_memory/)
rm -rf demo/demo_eval_memory
python demo/generate_synthetic_traces.py

# Different seed
python demo/generate_synthetic_traces.py --seed 123

# Custom output directory
python demo/generate_synthetic_traces.py --output-dir /tmp/sepa_demo
```

## Notes

- All traces are fully valid `EpisodeTrace` msgpack files readable by
  `EvalMemory.load_trace_file()`.
- Observations contain `gripper_state`, `qpos`, and tiny camera byte buffers so
  the `FailureStepDetector` classifiers (`sepa_eval mine`) produce correct labels.
- The script is deterministic: `--seed 42` always produces identical output.
