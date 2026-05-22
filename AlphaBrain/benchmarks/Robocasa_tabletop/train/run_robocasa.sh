export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  # timeout set to 1 hour (unit: seconds)

Framework_name=QwenPI_v2
base_vlm=./data/pretrained_models/Qwen3-VL-4B-Instruct
freeze_module_list='' # just for fast debug, sota is under fully FT, i.g., freeze_module_list=""
DIT_TYPE="DiT-B"
data_root_dir=./data/datasets/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim
dataset_mix=fourier_gr1_unified_1000


output_root_dir=./data/checkpoints
run_id=debug_alphabrain_qwen3fast_fourier_gr1_unified_1000

export WANDB_MODE=disabled

output_dir=${output_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/

accelerate launch \
  --config_file configs/deepspeed/accelerate_zero2.yaml \
  --num_processes 8 \
  AlphaBrain/training/train_alphabrain.py \
  --config_yaml ./configs/train_recipes/QwenGR00T_Robocasa.yaml \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.action_model.action_model_type ${DIT_TYPE} \
  --datasets.vla_data.data_root_dir ${data_root_dir} \
  --datasets.vla_data.dataset_mix ${dataset_mix} \
  --datasets.vla_data.per_device_batch_size 8 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 100 \
  --trainer.learning_rate.base 3e-5 \
  --output_root_dir ${output_root_dir} \
  --run_id ${run_id} \
  --wandb_project AlphaBrain_robocasa \
  --wandb_entity jinhuiye \
  # --is_debug True


