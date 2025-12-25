#!/bin/bash
#SBATCH --job-name=sft_video
#SBATCH --output=/hkfs/home/project/hk-project-p0024638/uzivy/Bachelor-thesis/src/scripts/logs/latest.log
#SBATCH --error=/hkfs/home/project/hk-project-p0024638/uzivy/Bachelor-thesis/src/scripts/logs/latest.err
#SBATCH --account=hk-project-p0024638
#SBATCH --partition=accelerated
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=76
#SBATCH --time=04:00:00
#SBATCH --mem=240G

# --- Setup Environment ---
set -e  # Exit on error

# Set explicit paths
REPO_ROOT="/hkfs/home/project/hk-project-p0024638/uzivy/Bachelor-thesis"
SCRIPT_DIR="${REPO_ROOT}/src/scripts"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Repo Root: $REPO_ROOT"

# Activate Virtual Environment
if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
    echo "Activated virtual environment at ${REPO_ROOT}/.venv"
else
    echo "Error: Virtual environment not found at ${REPO_ROOT}/.venv"
    exit 1
fi

# --- Fix Permission/Cache Issues ---
export TORCHINDUCTOR_CACHE_DIR="${HOME}/.cache/torch_inductor_cache"
export TRITON_CACHE_DIR="${HOME}/.cache/triton_cache"
export VLLM_CACHE_DIR="${HOME}/.cache/vllm"
export VLLM_NO_USAGE_STATS=1

# Clean up cache
rm -rf "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_DIR"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_DIR"

# Change directory to REPO_ROOT for running the script
cd "$REPO_ROOT"

# Run the SFT script
echo "Starting SFT training..."
bash "${SCRIPT_DIR}/run_sft_video.sh"

echo "SFT training finished."