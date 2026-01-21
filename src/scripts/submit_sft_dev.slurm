#!/bin/bash
#SBATCH --job-name=sft_video_dev
#SBATCH --output=/hkfs/home/project/hk-project-p0024638/uzivy/Bachelor-thesis/src/scripts/logs/latest_dev.log
#SBATCH --error=/hkfs/home/project/hk-project-p0024638/uzivy/Bachelor-thesis/src/scripts/logs/latest_dev.err
#SBATCH --account=hk-project-p0024638
#SBATCH --partition=dev_accelerated
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:15:00
#SBATCH --mem=480G

# --- Setup Environment ---
set -e  # Exit on error

# Load CUDA 12.9 for PyTorch 2.9 compatibility
module load devel/cuda/12.9

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
export VIDEO_CACHE_DIR="/hkfs/work/workspace/scratch/uzivy-open-o3-data/video_cache_dev_nocache/${SLURM_JOB_ID}"
export PYTHONUNBUFFERED=1
export VIDEO_FPS="${VIDEO_FPS:-2.0}"
export VIDEO_MIN_FRAMES="${VIDEO_MIN_FRAMES:-4}"
export VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES:-8}"
export VIDEO_MIN_PIXELS="${VIDEO_MIN_PIXELS:-$((64*28*28))}"
export VIDEO_MAX_PIXELS_PER_FRAME="${VIDEO_MAX_PIXELS_PER_FRAME:-$((64*28*28))}"

# Clean up cache
rm -rf "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_DIR"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_DIR"
rm -rf "$VIDEO_CACHE_DIR"
mkdir -p "$VIDEO_CACHE_DIR"

# Change directory to REPO_ROOT for running the script
cd "$REPO_ROOT"

# Run the SFT script
echo "Starting SFT training on DEV..."
bash "${SCRIPT_DIR}/run_sft_video.sh"

echo "SFT training finished."
