#!/bin/bash
# =============================================================================
# LIBERO Evaluation Script for NeuroVLA
#
# Loads a NeuroVLA checkpoint and evaluates on one or all LIBERO task suites.
# Optionally enables online R-STDP test-time adaptation on the SNN head.
#
# Usage:
#   bash run_eval_libero.sh --pretrained <ckpt>                         # default: libero_goal, 10 trials
#   bash run_eval_libero.sh --pretrained <ckpt> --suite libero_spatial
#   bash run_eval_libero.sh --pretrained <ckpt> --suite all             # run all 4 suites
#   bash run_eval_libero.sh --pretrained <ckpt> --trials 50 --gpu 1
#   bash run_eval_libero.sh --pretrained <ckpt> --online-stdp            # enable online STDP
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

[ -f .env ] && { set -a; source .env; set +a; }

# ---------- defaults ----------
PRETRAINED=""
SUITE="libero_goal"
TRIALS="${NUM_TRIALS:-10}"
SEED="${SEED:-7}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
VIDEO_OUT="${VIDEO_OUT:-results/evaluation/brain_inspired_eval_$(date +%Y%m%d_%H%M%S)}"
ONLINE_STDP=false
CONDA_ENV="${CONDA_ENV:-alphabrain}"

# Online STDP defaults (only used if --online-stdp)
STDP_LR="${STDP_LR:-5e-5}"
STDP_WARMUP="${STDP_WARMUP:-0}"
STDP_MAX_DEV="${STDP_MAX_DEV:-0.2}"
STDP_ROLLBACK="${STDP_ROLLBACK:-0.5}"
STDP_RESET_PER_TASK=true

# ---------- parse CLI ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pretrained|--ckpt) PRETRAINED="$2"; shift 2 ;;
        --suite)             SUITE="$2"; shift 2 ;;
        --trials)            TRIALS="$2"; shift 2 ;;
        --seed)              SEED="$2"; shift 2 ;;
        --gpu)               GPU="$2"; shift 2 ;;
        --video-out)         VIDEO_OUT="$2"; shift 2 ;;
        --online-stdp)       ONLINE_STDP=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$PRETRAINED" ]; then
    echo "ERROR: --pretrained <checkpoint_path> is required"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="${PROJECT_ROOT}:${LIBERO_HOME:-/path/to/LIBERO}${PYTHONPATH:+:$PYTHONPATH}"

# ---------- expand suite list ----------
if [ "$SUITE" = "all" ]; then
    SUITES=(libero_goal libero_spatial libero_object libero_10)
else
    SUITES=("$SUITE")
fi

# ---------- per-suite eval loop ----------
for S in "${SUITES[@]}"; do
    OUT_DIR="${VIDEO_OUT}/${S}"
    mkdir -p "$OUT_DIR"

    echo "=============================================="
    echo "  LIBERO Eval — NeuroVLA"
    echo "  Suite:        $S"
    echo "  Checkpoint:   $PRETRAINED"
    echo "  Trials/task:  $TRIALS"
    echo "  GPU:          $GPU"
    echo "  Online STDP:  $ONLINE_STDP"
    echo "  Output:       $OUT_DIR"
    echo "=============================================="

    CMD=(
        benchmarks/LIBERO/eval/eval_libero_online_stdp.py
        --pretrained-path "$PRETRAINED"
        --task-suite-name "$S"
        --num-trials-per-task "$TRIALS"
        --video-out-path "$OUT_DIR"
        --seed "$SEED"
    )

    if [ "$ONLINE_STDP" = true ]; then
        CMD+=(
            --stdp-lr "$STDP_LR"
            --stdp-warmup-episodes "$STDP_WARMUP"
            --stdp-max-deviation "$STDP_MAX_DEV"
            --stdp-rollback-shrink "$STDP_ROLLBACK"
        )
        [ "$STDP_RESET_PER_TASK" = true ] && CMD+=(--stdp-reset-per-task)
    else
        CMD+=(--no-stdp)
    fi

    # Robust python invocation: prefer direct conda env python, fall back to `conda run`
    ENV_PY=""
    for __p in /root/miniconda3/envs/$CONDA_ENV/bin/python /opt/conda/envs/$CONDA_ENV/bin/python; do
        if [ -x "$__p" ]; then ENV_PY="$__p"; break; fi
    done
    if [ -n "$ENV_PY" ] && [ -x "$ENV_PY" ]; then
        "$ENV_PY" "${CMD[@]}"
    elif command -v conda >/dev/null 2>&1; then
        conda run -n "$CONDA_ENV" python "${CMD[@]}"
    else
        echo "ERROR: neither conda env python nor 'conda' command found for env=$CONDA_ENV" >&2
        exit 1
    fi

    # Print summary
    if [ -f "$OUT_DIR/eval_results.json" ]; then
        SR=$("$ENV_PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print(f'{d[\"success_rate\"]*100:.1f}%')" "$OUT_DIR/eval_results.json" 2>/dev/null || echo ERR)
        echo ">>> $S success rate: $SR"
    fi
done

echo ""
echo "=============================================="
echo "  All evaluations done"
echo "  Output root: $VIDEO_OUT"
echo "=============================================="
