#!/usr/bin/env bash
# =============================================================================
# World Model + QwenGR00T Training (consolidated)
#
# Trains a world-model visual backbone (Cosmos 2.0 / Cosmos 2.5 / V-JEPA 2 /
# WAN 2.2) joint with a GR00T FlowMatching DiT action decoder on LIBERO.
#
# Usage:
#   MODEL=cos2        bash scripts/run_world_model/train/run_world_model.sh
#   MODEL=cos25_4gpu  bash scripts/run_world_model/train/run_world_model.sh
#   MODEL=vjepa       bash scripts/run_world_model/train/run_world_model.sh
#   MODEL=wan22       RESUME=true bash scripts/run_world_model/train/run_world_model.sh
#
#   # 1-GPU smoke test (override batch/steps/ds config)
#   CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 PER_DEVICE_BATCH=2 GRAD_ACCUM=1 \
#     MAX_STEPS=3 SAVE_INTERVAL=1000 \
#     DEEPSPEED_CONFIG=configs/deepspeed/accelerate_zero2_1gpu.yaml \
#     MODEL=cos2 bash scripts/run_world_model/train/run_world_model.sh
#
# Environment variables (override defaults):
#   MODEL            cos2 | cos25_4gpu | vjepa | wan22  (default: cos2)
#   NUM_GPUS         number of GPUs                     (default: 4)
#   MASTER_PORT      distributed training port          (defaults per model)
#   RESUME           "true" to resume from latest       (default: false)
#   CONFIG_YAML      override the recipe file           (default: configs/models/config_${MODEL}.yaml)
#   DEEPSPEED_CONFIG accelerate launch config file      (default: configs/deepspeed/accelerate_zero2.yaml)
#   PER_DEVICE_BATCH per-GPU batch size override        (default: from recipe)
#   GRAD_ACCUM       gradient accumulation steps        (default: from recipe)
#   MAX_STEPS        total training steps               (default: from recipe)
#   SAVE_INTERVAL    ckpt save every N steps            (default: from recipe)
#   LOG_FREQ         console/wandb log every N steps    (default: from recipe)
# =============================================================================
set -euo pipefail

# -- Resolve project root -----------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

# -- Verify editable install points here --------------------------------------
PY_PATH="$(python -c 'import AlphaBrain; print(AlphaBrain.__file__)' 2>/dev/null || true)"
if [[ "${PY_PATH}" != "${PROJECT_ROOT}"* ]]; then
    echo "[warn] AlphaBrain not loaded from ${PROJECT_ROOT} (current: ${PY_PATH})"
    echo "       Run: cd ${PROJECT_ROOT} && pip install -e . --no-deps"
fi

# -- Clean stale __pycache__ (avoids torch.compile / import cache surprises) --
find . -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

# -- Model selection ----------------------------------------------------------
MODEL="${MODEL:-cos2}"
case "${MODEL}" in
    cos2)        DEFAULT_PORT=29500 ;;
    cos25_4gpu)  DEFAULT_PORT=29501 ;;
    vjepa)       DEFAULT_PORT=29502 ;;
    wan22)       DEFAULT_PORT=29503 ;;
    *)
        echo "[error] Unknown MODEL='${MODEL}'. Valid: cos2 | cos25_4gpu | vjepa | wan22"
        exit 1
        ;;
esac

NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-${DEFAULT_PORT}}"
CONFIG_YAML="${CONFIG_YAML:-configs/models/config_${MODEL}.yaml}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed/accelerate_zero2.yaml}"
RESUME="${RESUME:-false}"

if [ ! -f "${CONFIG_YAML}" ]; then
    echo "[error] Config not found: ${CONFIG_YAML}"
    exit 1
fi
if [ ! -f "${DEEPSPEED_CONFIG}" ]; then
    echo "[error] Deepspeed config not found: ${DEEPSPEED_CONFIG}"
    exit 1
fi

# Cos 2.5 is sensitive to batch size; warn if user tries to scale up naively.
if [ "${MODEL}" = "cos25_4gpu" ] && [ "${NUM_GPUS}" != "4" ]; then
    echo "[warn] MODEL=cos25_4gpu is tuned for 4 GPUs; scaling NUM_GPUS may degrade results."
    sleep 3
fi

# -- Runtime env (tune for your cluster; overridable) -------------------------
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_2,mlx5_3}"
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-10000}"
export NCCL_SOCKET_TIMEOUT_MS="${NCCL_SOCKET_TIMEOUT_MS:-360000}"

# -- Build CLI overrides (only when env var is set) ---------------------------
OVERRIDE_ARGS=()
[ "${RESUME}" = "true" ]                && OVERRIDE_ARGS+=("trainer.is_resume=true")
[ -n "${PER_DEVICE_BATCH:-}" ]          && OVERRIDE_ARGS+=("datasets.vla_data.per_device_batch_size=${PER_DEVICE_BATCH}")
[ -n "${GRAD_ACCUM:-}" ]                && OVERRIDE_ARGS+=("trainer.gradient_accumulation_steps=${GRAD_ACCUM}")
[ -n "${MAX_STEPS:-}" ]                 && OVERRIDE_ARGS+=("trainer.max_train_steps=${MAX_STEPS}")
[ -n "${SAVE_INTERVAL:-}" ]             && OVERRIDE_ARGS+=("trainer.save_interval=${SAVE_INTERVAL}")
[ -n "${LOG_FREQ:-}" ]                  && OVERRIDE_ARGS+=("trainer.logging_frequency=${LOG_FREQ}")

echo "============================================================"
echo "  World Model Training"
echo "  MODEL         : ${MODEL}"
echo "  Project       : ${PROJECT_ROOT}"
echo "  Recipe        : ${CONFIG_YAML}"
echo "  Deepspeed cfg : ${DEEPSPEED_CONFIG}"
echo "  NUM_GPUS      : ${NUM_GPUS}"
echo "  MASTER_PORT   : ${MASTER_PORT}"
echo "  Resume        : ${RESUME}"
echo "  Overrides     : ${OVERRIDE_ARGS[*]:-<none>}"
echo "============================================================"

python -m accelerate.commands.launch \
    --config_file "${DEEPSPEED_CONFIG}" \
    --num_processes "${NUM_GPUS}" \
    --main_process_port "${MASTER_PORT}" \
    AlphaBrain/training/train_alphabrain.py \
    --config_yaml "${CONFIG_YAML}" \
    "${OVERRIDE_ARGS[@]}"
