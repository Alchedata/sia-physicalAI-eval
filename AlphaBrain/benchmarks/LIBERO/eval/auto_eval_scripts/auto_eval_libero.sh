#!/bin/bash

cd "${ALPHABRAIN_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"  # e.g. /path/to/AlphaBrain
SCRIPT_PATH="./benchmarks/LIBERO/eval/auto_eval_scripts/eval_libero_parall.sh"
your_ckpt="${YOUR_CKPT:-/path/to/your/checkpoint.pt}"  # override via env
run_index_base=346

#####################################################
task_suite_name=libero_10 # align with your model
run_index=$((run_index_base + 0))
bash $SCRIPT_PATH $your_ckpt $task_suite_name $run_index &
#####################################################

sleep 15
#####################################################
task_suite_name=libero_goal # align with your model
run_index=$((run_index_base + 1))
bash $SCRIPT_PATH $your_ckpt $task_suite_name $run_index &
#####################################################
sleep 15
#####################################################
task_suite_name=libero_object # align with your model
run_index=$((run_index_base + 2))
bash $SCRIPT_PATH $your_ckpt $task_suite_name $run_index &
#####################################################
sleep 15
####################################################
task_suite_name=libero_spatial # align with your model
run_index=$((run_index_base + 3))
bash $SCRIPT_PATH $your_ckpt $task_suite_name $run_index &
#####################################################

