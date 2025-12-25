#!/bin/bash
#SBATCH --job-name=grade_vstar
#SBATCH --output=logs/slurm_grading_%j.log
#SBATCH --error=logs/slurm_grading_%j.err
#SBATCH --account=hk-project-p0024638
#SBATCH --partition=accelerated
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=76
#SBATCH --time=00:30:00
#SBATCH --mem=240G

# --- Setup Environment ---
set -e  # Exit on error

# Set explicit paths to avoid Slurm spool directory issues
REPO_ROOT="/hkfs/home/project/hk-project-p0024638/uzivy/Bachelor-thesis"
SCRIPT_DIR="${REPO_ROOT}/eval/scripts"
EVAL_ROOT="${REPO_ROOT}/eval"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Repo Root: $REPO_ROOT"
echo "Eval Root: $EVAL_ROOT"

# Activate Virtual Environment
if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
    echo "Activated virtual environment at ${REPO_ROOT}/.venv"
else
    echo "Error: Virtual environment not found at ${REPO_ROOT}/.venv"
    exit 1
fi

# Source global config to get EXP_NAME and LLM_PATH
source "${REPO_ROOT}/config/run_config.sh"

# --- Fix Permission/Cache Issues ---
export TORCHINDUCTOR_CACHE_DIR="${HOME}/.cache/torch_inductor_cache"
export TRITON_CACHE_DIR="${HOME}/.cache/triton_cache"
export VLLM_CACHE_DIR="${HOME}/.cache/vllm"
export VLLM_NO_USAGE_STATS=1

# Clean up cache to prevent stale lock issues
rm -rf "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_DIR"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_DIR"

# Change directory to Bachelor-thesis/eval so that "./test/..." paths work
cd "$EVAL_ROOT"
mkdir -p logs/vstar_logs

# Create symlink to latest log
ln -sf slurm_grading_${SLURM_JOB_ID}.log logs/latest_grading.log

echo "Starting V-STAR Grading (Optimization: vLLM Batched)..."
echo "Using Result File: ./logs/vstar_logs/${EXP_NAME}_vstar.json"
echo "Using Judge Model: $LLM_PATH"

# Run ONLY the grading script
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./test/eval_vstar.py \
    --result_file  "./logs/vstar_logs/${EXP_NAME}_vstar.json" \
    --model_path "$LLM_PATH" \
    > "./logs/vstar_logs/eval_${EXP_NAME}_vstar.log" 2>&1

echo "Grading finished. Check results in ./logs/vstar_logs/eval_${EXP_NAME}_vstar.log"
