#!/usr/bin/env bash

# Central run config: change paths/model IDs here.
# Shell scripts source this; Python eval scripts read MODEL_PATH/LLM_PATH env vars.

# --- Data Configuration ---
# TODO: Ensure dataset is downloaded to this location
export DATA_ROOT="/hkfs/home/project/hk-project-p0024638/uzivy/datasets"

# --- Model Configuration ---
SFT_MODEL_PATH="/hkfs/home/project/hk-project-p0024638/uzivy/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203"
GRPO_MODEL_PATH="${SFT_MODEL_PATH}"
EVAL_MODEL_PATH="${GRPO_MODEL_PATH}"
LLM_PATH="Qwen/Qwen3-8B" # Text-only LLM for V-STAR grading

# --- Experiment & Output Configuration ---
SFT_EXP_NAME="sft"
SFT_OUT_DIR="/hkfs/home/project/hk-project-p0024638/uzivy/checkpoints/${SFT_EXP_NAME}"

GRPO_EXP_NAME="rl"
GRPO_OUT_DIR="/hkfs/home/project/hk-project-p0024638/uzivy/checkpoints/${GRPO_EXP_NAME}"

EVAL_EXP_NAME="${GRPO_EXP_NAME}"
