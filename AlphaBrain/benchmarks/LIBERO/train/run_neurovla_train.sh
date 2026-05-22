

# 加载环境变量
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# export NCCL_SOCKET_IFNAME=bond0 # TODO 这里会socket报错
export NCCL_IB_HCA=mlx5_2,mlx5_3

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000  # timeout set to 1 hour (unit: seconds)
export NCCL_SOCKET_TIMEOUT_MS=360000
###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=NeuroVLA
freeze_module_list=''
base_vlm=data/pretrained_models/Qwen2.5-VL-3B-Instruct
config_yaml=./configs/train_recipes/NeuroVLA_LIBERO.yaml
libero_data_root="${LIBERO_DATA_ROOT}"
dataset_mix=libero_goal
output_root_dir=./results/training
run_id=neurovla_libero_goal
# === End of environment variable configuration ===
###########################################################################################


export WANDB_MODE=disabled

output_dir=${output_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/


accelerate launch \
  --config_file configs/deepspeed/accelerate_zero2.yaml \
  --num_processes 2 \
  AlphaBrain/training/train_alphabrain.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${libero_data_root}\
  --datasets.vla_data.dataset_mix ${dataset_mix} \
  --datasets.vla_data.per_device_batch_size 2 \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 200 \
  --trainer.save_interval 100 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 201 \
  --output_root_dir ${output_root_dir} \
  --run_id ${run_id} \
  # --wandb_project AlphaBrain_Libero \
  # --is_debug True



##### Multi-Server Multi-GPU training script #####
  # accelerate launch \
  #   --config_file configs/deepspeed/accelerate_zero2.yaml \
  #   --main_process_ip $MASTER_ADDR \
  #   --main_process_port $MASTER_PORT \
  #   --machine_rank $SLURM_PROCID \
  #   --num_machines $SLURM_NNODES \
  #   --num_processes=${TOTAL_GPUS} \
  #   AlphaBrain/training/train_alphabrain.py \
  #   --config_yaml ${config_yaml} \
  #   --framework.name ${Framework_name} \
  #   --framework.qwenvl.base_vlm ${base_vlm} \
  #   --output_root_dir ${output_root_dir} \
  #   --run_id ${run_id} \
  #   --wandb_project your_project \
  #   --wandb_entity your_name
##### Multi-Server Multi-GPU training script #####
