#!/usr/bin/env bash
# =========================================================================================
# LIBERO QwenOFT 训练 — 统一入口的便捷 wrapper
# 等价于: bash scripts/run_finetune.sh qwen_oft
#
# 也可直接运行此脚本进行独立训练（不依赖 finetune_config.yaml）
# =========================================================================================

# 获取项目根目录（相对于此脚本的位置）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

# 优先使用统一入口
if [ -f scripts/run_finetune.sh ]; then
    exec bash scripts/run_finetune.sh qwen_oft "$@"
fi

# === 回退: 独立模式（当统一入口不可用时） ===

# 加载环境变量
if [ -f .env ]; then
    set -a; source .env; set +a
fi

export NCCL_IB_HCA=mlx5_2,mlx5_3
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000
export WANDB_MODE=disabled

Framework_name=QwenOFT
freeze_module_list=''
base_vlm="${PRETRAINED_MODELS_DIR:-data/pretrained_models}/Qwen2.5-VL-3B-Instruct"
config_yaml=./configs/train_recipes/QwenGR00T_LIBERO.yaml
libero_data_root="${LIBERO_DATA_ROOT}"
dataset_mix=libero_goal
output_root_dir=./results/training
run_id=0309_libero4in1_qwen3oft_sf_cc

output_dir=${output_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/

accelerate launch \
  --config_file configs/deepspeed/accelerate_zero2.yaml \
  --num_processes 2 \
  AlphaBrain/training/train_alphabrain.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${libero_data_root} \
  --datasets.vla_data.dataset_mix ${dataset_mix} \
  --datasets.vla_data.per_device_batch_size 2 \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 200 \
  --trainer.save_interval 100 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 201 \
  --output_root_dir ${output_root_dir} \
  --run_id ${run_id}
