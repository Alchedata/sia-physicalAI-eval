#!/usr/bin/env bash
# =============================================================================
# Cosmos Policy Evaluation (Server-Client, LIBERO)
#
# Starts deployment/model_server/server_policy_cosmos.py then runs
# benchmarks/LIBERO/eval/eval_libero_cosmos.py against it.
#
# Usage:
#   # Quick smoke test (1 trial per task = 10 episodes)
#   bash scripts/run_world_model/eval/eval_cosmos_policy.sh
#
#   # Full eval (50 trials per task = 500 episodes)
#   NUM_TRIALS=50 bash scripts/run_world_model/eval/eval_cosmos_policy.sh
#
#   # Use a custom checkpoint
#   CKPT_DIR=results/training/<run>/checkpoints/steps_40000 #     bash scripts/run_world_model/eval/eval_cosmos_policy.sh
#
# Environment variables:
#   CKPT_DIR       policy checkpoint (default: data/pretrained_models/Cosmos-Policy-LIBERO-Predict2-2B)
#   PRETRAINED_DIR base Cosmos model for VAE (default: data/pretrained_models/Cosmos-Predict2-2B-Video2World)
#   TASK_SUITE     libero_goal | libero_spatial | libero_object | libero_10
#   NUM_TRIALS     trials per task (default: 1)
#   GPU_ID         GPU index (default: 0)
#   PORT           websocket port (default: 10093)
#   HOST           host (default: 127.0.0.1)
#   PYTHON         python interpreter (default: python)
#   LIBERO_HOME    LIBERO repo path (required)
# =============================================================================
set -euo pipefail

CKPT_DIR="${CKPT_DIR:-data/pretrained_models/Cosmos-Policy-LIBERO-Predict2-2B}"
PRETRAINED_DIR="${PRETRAINED_DIR:-data/pretrained_models/Cosmos-Predict2-2B-Video2World}"
TASK_SUITE="${TASK_SUITE:-libero_goal}"
NUM_TRIALS="${NUM_TRIALS:-1}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-10093}"
HOST="${HOST:-127.0.0.1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

LIBERO_HOME="${LIBERO_HOME:?set LIBERO_HOME to your LIBERO repo path}"
PYTHON="${PYTHON:-python}"

export PYTHONPATH="${PROJECT_ROOT}:${LIBERO_HOME}:${PYTHONPATH:-}"
export MUJOCO_GL=egl

TIMESTAMP=$(date +%m%d_%H%M%S)
TRIAL_TAG=$( [ "${NUM_TRIALS}" -gt 1 ] && echo "${NUM_TRIALS}trials" || echo "quick" )
EVAL_OUT_DIR="results/evaluation/cosmos/${TIMESTAMP}-SC-${TASK_SUITE}-${TRIAL_TAG}"
mkdir -p "${EVAL_OUT_DIR}/videos"

SERVER_LOG="${EVAL_OUT_DIR}/server.log"
EVAL_LOG="${EVAL_OUT_DIR}/eval.log"

echo "============================================================"
echo "  Cosmos Policy Eval (server-client)"
echo "  Checkpoint : ${CKPT_DIR}"
echo "  Task Suite : ${TASK_SUITE}"
echo "  Num Trials : ${NUM_TRIALS} per task"
echo "  Server     : GPU ${GPU_ID}, ${HOST}:${PORT}"
echo "  Output     : ${EVAL_OUT_DIR}"
echo "============================================================"

echo "[1/2] Starting Cosmos Policy server on GPU ${GPU_ID}, port ${PORT} ..."
CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} deployment/model_server/server_policy_cosmos.py     --ckpt_dir "${CKPT_DIR}"     --pretrained_dir "${PRETRAINED_DIR}"     --port "${PORT}"     > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

cleanup() {
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[info] Shutting down server ..."
        kill "${SERVER_PID}" 2>/dev/null
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "[info] Waiting for server on port ${PORT} ..."
MAX_WAIT=300
WAITED=0
while true; do
    ${PYTHON} -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('${HOST}', ${PORT})); s.close()" 2>/dev/null && break
    kill -0 "${SERVER_PID}" 2>/dev/null || { echo "[error] Server died. Check: ${SERVER_LOG}"; tail -20 "${SERVER_LOG}"; exit 1; }
    [ ${WAITED} -ge ${MAX_WAIT} ] && { echo "[error] Timeout after ${MAX_WAIT}s"; exit 1; }
    sleep 3
    WAITED=$((WAITED + 3))
done
echo "[ok] Server ready (${WAITED}s)"

echo ""
echo "[2/2] Running evaluation on ${TASK_SUITE} ..."
${PYTHON} benchmarks/LIBERO/eval/eval_libero_cosmos.py     --ckpt_dir "${CKPT_DIR}"     --host "${HOST}"     --port "${PORT}"     --task_suite_name "${TASK_SUITE}"     --num_trials_per_task "${NUM_TRIALS}"     --video_out_path "${EVAL_OUT_DIR}/videos"     --job_name "cosmos_${TASK_SUITE}"     2>&1 | tee "${EVAL_LOG}"

echo ""
echo "[ok] Evaluation complete!"
echo "  Results : ${EVAL_OUT_DIR}"
echo "  Eval log: ${EVAL_LOG}"
echo "  Server  : ${SERVER_LOG}"
