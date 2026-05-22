#!/bin/bash
set -euo pipefail
cd "${ALPHABRAIN_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"  # e.g. /path/to/AlphaBrain

[ -f .env ] && { set -a; source .env; set +a; }

# AlphaBrain isn't pip-installed; prepend repo root so `python <path>/script.py`
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

export LIBERO_PYTHON="${LIBERO_PYTHON:-/path/to/envs/libero/bin/python}"
export LIBERO_HOME="${LIBERO_HOME:-/path/to/LIBERO}"
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"

GPU_IDS=${1:-"0,2,3,4,5,1"}
PRETRAIN_GPU="${GPU_IDS%%,*}"   # first GPU in the list

CKPT_PATH="results/training/QwenOFT-5traj-libero_goal/final_model"
PRETRAIN_DIR="results/rlt_training_TD3/5traj_alltasks_pretrain/pretrain"
ENCODER_PATH="${PRETRAIN_DIR}/checkpoints/pretrain_best/encoder.pt"

RUN_NAME="rlt_5traj_alltasks_release"
TIMESTAMP=$(date +%m%d_%H%M)
OUTPUT_DIR="results/action_token_training_TD3/${RUN_NAME}_${TIMESTAMP}/rl_offpolicy"

if [ ! -d "${CKPT_PATH}" ]; then
    echo "ERROR: 5-traj VLA ckpt not found: ${CKPT_PATH}"
    exit 1
fi

# ── Phase 1: pretrain encoder (only if missing) ────────────────
if [ ! -f "${ENCODER_PATH}" ]; then
    echo "============================================================"
    echo " Phase 1: encoder not found — pretraining (GPU ${PRETRAIN_GPU}, ~5 min)"
    echo "   ckpt: ${CKPT_PATH}"
    echo "   out:  ${PRETRAIN_DIR}"
    echo "============================================================"
    CUDA_VISIBLE_DEVICES=${PRETRAIN_GPU} python AlphaBrain/training/reinforcement_learning/trainers/train.py \
        --phase pretrain \
        --ckpt_path ${CKPT_PATH} \
        --output_dir ${PRETRAIN_DIR} \
        --suite libero_goal \
        --all_tasks \
        --bottleneck_dim 256 \
        --encoder_layers 2 \
        --encoder_heads 4 \
        --pretrain_n_obs 3000 \
        --pretrain_steps_per_reset 20 \
        --pretrain_epochs 500 \
        --pretrain_lr 1e-4 \
        --pretrain_batch_size 32 \
        --vla_extract_batch_size 16 \
        --num_envs_per_task 8 \
        --seed 42 \
        --use_wandb \
        --wandb_project AlphaBrain_RLT \
        --run_name action_token_5traj_alltasks_pretrain
    if [ ! -f "${ENCODER_PATH}" ]; then
        echo "ERROR: pretrain finished but encoder missing at ${ENCODER_PATH}"
        exit 1
    fi
    echo "[Phase 1] encoder saved -> ${ENCODER_PATH}"
fi

# ── Phase 2: off-policy TD3 RL ─────────────────────────────────
export CUDA_VISIBLE_DEVICES=${GPU_IDS}

python AlphaBrain/training/reinforcement_learning/trainers/train.py \
    --phase rl_offpolicy \
    --ckpt_path ${CKPT_PATH} \
    --encoder_path ${ENCODER_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --suite libero_goal \
    --all_tasks \
    --use_steplock \
    --rollout_gpus 0,1,2,3,4 \
    --train_gpu 5 \
    --bottleneck_dim 256 \
    --encoder_layers 2 \
    --encoder_heads 4 \
    --actor_hidden_dim 512 \
    --critic_hidden_dim 512 \
    --ref_dropout 0.5 \
    --fixed_std 0.1 \
    --G_per_task 30 \
    --group_size 1 \
    --num_envs_per_task 10 \
    --reward_coef 5.0 \
    --lr_actor 3e-4 \
    --lr_critic 3e-4 \
    --gamma 0.99 \
    --max_grad_norm 1.0 \
    --buffer_capacity 1000000 \
    --buffer_warmup 1024 \
    --warmup_iters 5 \
    --td_updates_per_iter 10000 \
    --utd_ratio 10.0 \
    --td_batch_size 1024 \
    --tau 0.005 \
    --beta 1.0 \
    --actor_update_freq 2 \
    --target_noise_std 0.2 \
    --target_noise_clip 0.5 \
    --max_iter 400 \
    --eval_interval 20 \
    --eval_n_episodes 20 \
    --save_interval 50 \
    --save_video_interval 100 \
    --seed 42 \
    --use_wandb \
    --wandb_project AlphaBrain_RLT \
    --run_name "${RUN_NAME}" \
    --log_interval 1
