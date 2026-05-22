#!/bin/bash
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo
# # origin0309: 旧的单文件checkpoint路径（需要base_vlm，冗余权重读取）
# your_ckpt=results/training/0309_libero4in1_qwen3oft_safetentor/checkpoints/steps_200_model.safetensors
# lpt0309: 新的自包含目录checkpoint路径（单一路径推断，无需base_vlm）
# your_ckpt=results/training/0309_libero4in1_qwen3oft_sf_cc/checkpoints/steps_100
your_ckpt=results/training/Qwen2.5-VL-GR00T-LIBERO-4in1/checkpoints/steps_30000_pytorch_model.pt
gpu_id=0
port=5694
################# star Policy Server ######################

# export DEBUG=true
CUDA_VISIBLE_DEVICES=$gpu_id python deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16

# #################################