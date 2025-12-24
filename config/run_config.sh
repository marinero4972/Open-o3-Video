#!/usr/bin/env bash

# Central run config: change paths/model IDs here.
# Shell scripts source this; Python eval scripts read MODEL_PATH/LLM_PATH env vars.
SFT_MODEL_PATH="Qwen/Qwen3-VL-2B-Instruct"
GRPO_MODEL_PATH="/path/to/ckpts/sft/"
EVAL_MODEL_PATH="${GRPO_MODEL_PATH}"
LLM_PATH="Qwen/Qwen3-8B" # Text-only LLM for V-STAR grading

SFT_EXP_NAME="sft"
SFT_OUT_DIR="/path/to/ckpts/${SFT_EXP_NAME}"
GRPO_EXP_NAME="rl"
GRPO_OUT_DIR="/path/to/ckpts/${GRPO_EXP_NAME}"
EVAL_EXP_NAME="${GRPO_EXP_NAME}"
