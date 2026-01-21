#!/usr/bin/env bash

# Central run config: change paths/model IDs here.
# Shell scripts source this; Python eval scripts read MODEL_PATH/LLM_PATH env vars.

# --- Data Configuration ---
# TODO: Ensure dataset is downloaded to this location
export DATA_ROOT="/hkfs/work/workspace/scratch/uzivy-open-o3-data/sft_rl"

# --- Model Configuration ---
SFT_MODEL_PATH="/home/hk-project-p0024638/uzivy/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203"
#SFT_MODEL_PATH="/home/hk-project-p0024638/uzivy/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
GRPO_MODEL_PATH="${SFT_MODEL_PATH}"
EVAL_MODEL_PATH="${GRPO_MODEL_PATH}"
LLM_PATH="Qwen/Qwen2.5-7B-Instruct" # Text-only LLM for V-STAR grading

# --- Experiment & Output Configuration ---
CHECKPOINT_DIR="/hkfs/work/workspace/scratch/uzivy-checkpoints/"
SFT_EXP_NAME="sft"
SFT_OUT_DIR="${CHECKPOINT_DIR}${SFT_EXP_NAME}"

GRPO_EXP_NAME="rl"
GRPO_OUT_DIR="${CHECKPOINT_DIR}${GRPO_EXP_NAME}"

EVAL_EXP_NAME="${GRPO_EXP_NAME}"
