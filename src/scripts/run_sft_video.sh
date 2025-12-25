#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Source the central configuration
source "${REPO_ROOT}/config/run_config.sh"

# Set compiler variables to use GCC instead of the default Intel compiler
export CC=gcc
export CXX=g++

# Set PYTHONPATH to include the project's src directory for module discovery
# This allows python to find the 'configs' module from the project root
export PYTHONPATH="${PYTHONPATH}:${REPO_ROOT}/src/r1-v"

# Set other environment variables
export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export WANDB_MODE="offline"

# Variables are now loaded from run_config.sh
MODEL_PATH="${SFT_MODEL_PATH}"
EXP_NAME="${SFT_EXP_NAME}"
OUT_DIR="${SFT_OUT_DIR}"

# Ensure DATA_ROOT is available to Python
export DATA_ROOT

# Create output directory if it doesn't exist
mkdir -p "$OUT_DIR"

# Run SFT from the project root
cd "$REPO_ROOT"

# Run SFT with torchrun
# Configured for 4 GPUs (0,1,2,3)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/r1-v/src/open_r1/sft_multi_task.py \
    --output_dir "$OUT_DIR" \
    --model_name_or_path "$MODEL_PATH" \
    --dataset_name "${DATA_ROOT}/json_data/STGR-SFT.json" \
    --deepspeed "src/r1-v/local_scripts/zero2.json" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name "$EXP_NAME" \
    --save_steps 500 \
    --max_grad_norm 5 \
    --save_only_model true
