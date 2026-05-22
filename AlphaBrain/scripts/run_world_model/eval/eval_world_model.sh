#!/usr/bin/env bash
# =============================================================================
# World Model + GR00T Evaluation (Server-Client, LIBERO)
#
# Spins up deployment/model_server/server_policy.py with the given checkpoint
# and runs benchmarks/LIBERO/eval/eval_libero.py against it.
#
# Usage:
#   CKPT=results/training/<run>/checkpoints/steps_30000 #     bash scripts/run_world_model/eval/eval_world_model.sh
#
#   # Different task suite and 50 trials per task
#   CKPT=... TASK_SUITE=libero_spatial NUM_TRIALS=50 #     bash scripts/run_world_model/eval/eval_world_model.sh
#
#   # Enable side-by-side predicted-frame visualization
#   CKPT=... PREDICT_VIDEO=true #     bash scripts/run_world_model/eval/eval_world_model.sh
#
# Environment variables:
#   CKPT           checkpoint directory (required)
#   TASK_SUITE     libero_goal | libero_spatial | libero_object | libero_10
#   NUM_TRIALS     trials per task (default: 3)
#   GPU_ID         GPU index for the server (default: 0)
#   PORT           websocket server port (default: 5694)
#   HOST           server host (default: 127.0.0.1)
#   PREDICT_VIDEO  true/false — render side-by-side rollout+prediction mp4 (default: false)
#   PYTHON         python interpreter to use (default: python)
#   LIBERO_HOME    path to the LIBERO repo checkout (required)
# =============================================================================
set -euo pipefail

CKPT="${CKPT:?Error: CKPT must be set to a checkpoint directory}"
TASK_SUITE="${TASK_SUITE:-libero_goal}"
NUM_TRIALS="${NUM_TRIALS:-3}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-5694}"
HOST="${HOST:-127.0.0.1}"
PREDICT_VIDEO="${PREDICT_VIDEO:-false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PYTHON:-python}"
LIBERO_HOME="${LIBERO_HOME:?set LIBERO_HOME to your LIBERO repo path}"

export PYTHONPATH="${PROJECT_ROOT}:${LIBERO_HOME}:${PYTHONPATH:-}"
export MUJOCO_GL=egl

CKPT_NAME=$(basename $(dirname $(dirname "${CKPT}")))_$(basename "${CKPT}")
OUT_DIR="results/evaluation/${TASK_SUITE}/${CKPT_NAME}-$(date +%m%d_%H%M%S)"
mkdir -p "${OUT_DIR}/videos"

echo "============================================================"
echo "  World Model Eval (server-client)"
echo "  Checkpoint : ${CKPT}"
echo "  Task Suite : ${TASK_SUITE}"
echo "  Num Trials : ${NUM_TRIALS}"
echo "  GPU        : ${GPU_ID}, Port: ${PORT}"
echo "  Output     : ${OUT_DIR}"
echo "============================================================"

echo "[1/2] Starting server ..."
CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} deployment/model_server/server_policy.py     --ckpt_path "${CKPT}" --port ${PORT} --use_bf16     > "${OUT_DIR}/server.log" 2>&1 &
SERVER_PID=$!

cleanup() {
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[info] Shutting down server ..."
        kill "${SERVER_PID}" 2>/dev/null
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "Waiting for server ..."
WAITED=0
while true; do
    ${PYTHON} -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('${HOST}', ${PORT})); s.close()" 2>/dev/null && break
    kill -0 "${SERVER_PID}" 2>/dev/null || { echo "[error] Server died!"; tail -20 "${OUT_DIR}/server.log"; exit 1; }
    [ ${WAITED} -ge 300 ] && { echo "[error] Timeout after 300s"; exit 1; }
    sleep 3; WAITED=$((WAITED+3))
done
echo "Server ready (${WAITED}s)"

echo "[2/2] Running eval ..."
PREDICT_VIDEO_FLAG=""
if [ "${PREDICT_VIDEO}" = "true" ]; then
    PREDICT_VIDEO_FLAG="--args.predict-video"
    echo "[info] predict-video enabled — will render side-by-side mp4 visualization"
fi

${PYTHON} benchmarks/LIBERO/eval/eval_libero.py     --args.pretrained-path "${CKPT}"     --args.host "${HOST}"     --args.port ${PORT}     --args.task-suite-name "${TASK_SUITE}"     --args.num-trials-per-task "${NUM_TRIALS}"     --args.video-out-path "${OUT_DIR}/videos"     ${PREDICT_VIDEO_FLAG}     2>&1 | tee "${OUT_DIR}/eval.log"

echo ""
echo "[ok] Eval complete: ${OUT_DIR}"
