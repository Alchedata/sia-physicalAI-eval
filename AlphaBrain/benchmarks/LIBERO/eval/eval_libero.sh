#!/bin/bash
# =========================================================================================
# LIBERO 评估 — 统一入口的便捷 wrapper
# 等价于: bash scripts/run_eval.sh libero_eval
#
# 也可直接运行此脚本进行独立评估（不依赖 finetune_config.yaml）
# =========================================================================================

# 获取项目根目录（相对于此脚本的位置）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

# 优先使用统一入口
if [ -f scripts/run_eval.sh ]; then
    exec bash scripts/run_eval.sh libero_eval "$@"
fi

# === 回退: 独立模式（当统一入口不可用时） ===

# 加载环境变量
if [ -f .env ]; then
    set -a; source .env; set +a
fi

export LIBERO_HOME="${LIBERO_HOME:-../LIBERO}"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
LIBERO_PYTHON="${LIBERO_PYTHON:-python}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${LIBERO_HOME}"
export PYTHONPATH="$(pwd):${PYTHONPATH}"

host="127.0.0.1"
base_port=5694
your_ckpt="results/training/0309_libero4in1_qwen3oft_sf_cc/checkpoints/steps_100"

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')

LOG_DIR="logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p ${LOG_DIR}

task_suite_name=libero_goal
num_trials_per_task=50
video_out_path="results/${task_suite_name}/${folder_name}"

${LIBERO_PYTHON} ./benchmarks/LIBERO/eval/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    --args.host "$host" \
    --args.port $base_port \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path"
