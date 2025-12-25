#!/bin/bash
#SBATCH --job-name=grade_dev
#SBATCH --output=/hkfs/home/project/hk-project-p0024638/uzivy/Bachelor-thesis/eval/logs/latest.log
#SBATCH --error=/hkfs/home/project/hk-project-p0024638/uzivy/Bachelor-thesis/eval/logs/latest.err
#SBATCH --account=hk-project-p0024638
#SBATCH --partition=dev_accelerated
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=76
#SBATCH --time=00:30:00
#SBATCH --mem=240G

# --- Setup Environment ---
set -e

# Read Tag from Argument (default to "eval" if not provided)
TAG=${1:-"eval"}

REPO_ROOT="/hkfs/home/project/hk-project-p0024638/uzivy/Bachelor-thesis"
EVAL_ROOT="${REPO_ROOT}/eval"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Experiment Tag: $TAG"

# Activate Virtual Environment
if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
else
    echo "Error: Virtual environment not found"
    exit 1
fi

# Source global config
source "${REPO_ROOT}/config/run_config.sh"
EXP_NAME="${EVAL_EXP_NAME}"

# --- Fix Permission/Cache Issues ---
export TORCHINDUCTOR_CACHE_DIR="${HOME}/.cache/torch_inductor_cache"
export TRITON_CACHE_DIR="${HOME}/.cache/triton_cache"
export VLLM_CACHE_DIR="${HOME}/.cache/vllm"
export VLLM_NO_USAGE_STATS=1

# Clean up cache
rm -rf "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_DIR"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_DIR"

# Change directory to eval root
cd "$EVAL_ROOT"

# Run grading
echo "Starting V-STAR Grading on DEV partition..."
echo "Result file: ./logs/vstar_logs/${EXP_NAME}_vstar.json"

CUDA_VISIBLE_DEVICES=0,1,2,3 python ./test/eval_vstar.py \
    --result_file  "./logs/vstar_logs/${EXP_NAME}_vstar.json" \
    --model_path "$LLM_PATH" \
    --tag "$TAG" \
    

echo "Grading finished."
