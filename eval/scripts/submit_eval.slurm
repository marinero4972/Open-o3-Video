#!/bin/bash
#SBATCH --job-name=eval_vstar
#SBATCH --output=logs/slurm_eval_%j.log
#SBATCH --error=logs/slurm_eval_%j.err
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

# Set explicit paths to avoid Slurm spool directory issues
REPO_ROOT="/hkfs/home/project/hk-project-p0024638/uzivy/Bachelor-thesis"
SCRIPT_DIR="${REPO_ROOT}/eval/scripts"
EVAL_ROOT="${REPO_ROOT}/eval"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Repo Root: $REPO_ROOT"
echo "Eval Root: $EVAL_ROOT"

# Activate Virtual Environment
# Assuming .venv is in the REPO_ROOT
if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
    echo "Activated virtual environment at ${REPO_ROOT}/.venv"
else
    echo "Error: Virtual environment not found at ${REPO_ROOT}/.venv"
    exit 1
fi

# --- Fix Permission/Cache Issues ---
# Force Torch/Triton to use a persistent user directory to avoid Slurm spool/scratch issues
export TORCHINDUCTOR_CACHE_DIR="${HOME}/.cache/torch_inductor_cache"
export TRITON_CACHE_DIR="${HOME}/.cache/triton_cache"
export VLLM_CACHE_DIR="${HOME}/.cache/vllm"
export VLLM_NO_USAGE_STATS=1

# Nuke ALL possible cache locations (including defaults) to remove stale scratch paths
rm -rf "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_DIR"
rm -rf "${HOME}/.cache/torch" "${HOME}/.triton"

mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_DIR"

# Change directory to Bachelor-thesis/eval so that "./test/..." paths in eval_all.sh work
cd "$EVAL_ROOT"
mkdir -p logs/vstar_logs

# Create symlink to latest log
ln -sf slurm_eval_${SLURM_JOB_ID}.log logs/latest.log

# Run the evaluation script
# We call it via bash explicitly
echo "Starting evaluation..."
bash "${SCRIPT_DIR}/eval_all.sh"

echo "Evaluation finished."
