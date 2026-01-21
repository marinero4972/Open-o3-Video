#!/bin/bash
#SBATCH --job-name=sft_video
#SBATCH --output=/hkfs/home/project/hk-project-p0024638/uzivy/Bachelor-thesis/src/scripts/logs/latest.log
#SBATCH --error=/hkfs/home/project/hk-project-p0024638/uzivy/Bachelor-thesis/src/scripts/logs/latest.err
#SBATCH --account=hk-project-p0024638
#SBATCH --partition=accelerated
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:10:00
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
export VIDEO_CACHE_DIR="/hkfs/work/workspace/scratch/uzivy-open-o3-data/video_cache_accel_nocache/${SLURM_JOB_ID}"
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

# Source config to get SFT_OUT_DIR
source "${REPO_ROOT}/config/run_config.sh"

# --- Checkpoint Setup ---
export SFT_OUT_DIR="${CHECKPOINT_DIR}${SFT_EXP_NAME}_accel_${SLURM_JOB_ID}"
mkdir -p "$SFT_OUT_DIR"

# Find latest checkpoint
LATEST_CHECKPOINT=$(ls -td "${SFT_OUT_DIR}"/checkpoint-* 2>/dev/null | head -n 1)

SFT_ARGS=""
if [ -n "$LATEST_CHECKPOINT" ]; then
    echo "Resuming from checkpoint: $LATEST_CHECKPOINT"
    SFT_ARGS="--resume_from_checkpoint $LATEST_CHECKPOINT"
else
    echo "No checkpoint found, starting from scratch."
fi

# Run the SFT script
echo "Starting SFT training..."
bash "${SCRIPT_DIR}/run_sft_video.sh" $SFT_ARGS

echo "SFT training finished."
