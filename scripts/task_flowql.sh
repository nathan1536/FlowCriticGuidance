#!/bin/bash
#SBATCH --job-name=flowql
#SBATCH --output=logs/flowql_%j.out
#SBATCH --error=logs/flowql_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-1-employee

# =============================================================================
# FlowQL Training - SLURM Job Script
# =============================================================================
#
# Usage:
#   sbatch task_flowql.sh [alg_name] [task_name] [addition_info] [seed]
#
# Example:
#   sbatch task_flowql.sh flowql_adroit_door adroit_door_image flowql_v1 0
#
# Default values if not provided:
#   alg_name: flowql_adroit_door
#   task_name: adroit_door_image
#   addition_info: flowql_cluster
#   seed: 0
#
# Prerequisites:
#   1. Build the container: sudo apptainer build maniflow.sif Apptainer.def
#   2. Upload maniflow.sif to the cluster home directory
#   3. Upload ManiFlow_Policy code to the cluster
#   4. Upload replay buffer zarr to the correct path
#   5. Create logs directory: mkdir -p logs
# =============================================================================

# Print job information
echo "=============================================="
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Running on host: $(hostname)"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "=============================================="

# =============================================================================
# Configuration - Modify these paths for your cluster
# =============================================================================

# Path to the Apptainer/Singularity image
CONTAINER_IMAGE="${HOME}/maniflow.sif"

# Project directory (where ManiFlow_Policy is located on the cluster)
PROJECT_DIR="${HOME}/ManiFlow_Policy"

# WandB cache directory
WANDB_DIR="${HOME}/.wandb"

# =============================================================================
# Parse Arguments
# =============================================================================

ALG_NAME=${1:-"flowql_adroit_door"}
TASK_NAME=${2:-"adroit_door_image"}
ADDITION_INFO=${3:-"flowql_cluster"}
SEED=${4:-0}

echo "Configuration:"
echo "  Algorithm: ${ALG_NAME}"
echo "  Task: ${TASK_NAME}"
echo "  Additional info: ${ADDITION_INFO}"
echo "  Seed: ${SEED}"
echo "  GPUs: 1"

# =============================================================================
# Environment Setup
# =============================================================================

# Create output directories if they don't exist
mkdir -p "${PROJECT_DIR}/ManiFlow/data/outputs"
mkdir -p "${WANDB_DIR}"
mkdir -p logs

# GPU
GPU_IDS="0"

# Check if container exists
if [ ! -f "${CONTAINER_IMAGE}" ]; then
    echo "ERROR: Container image not found at ${CONTAINER_IMAGE}"
    echo "Please upload the container first:"
    echo "  scp maniflow.sif username@cluster:~/"
    exit 1
fi

# Check if project directory exists
if [ ! -d "${PROJECT_DIR}" ]; then
    echo "ERROR: Project directory not found at ${PROJECT_DIR}"
    echo "Please upload your code first:"
    echo "  rsync -avz ManiFlow_Policy/ username@cluster:~/ManiFlow_Policy/"
    exit 1
fi

export WANDB_API_KEY=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
export WANDB_PROJECT="FlowQL"

# Check for WandB API key
if [ -z "${WANDB_API_KEY}" ]; then
    echo "WARNING: WANDB_API_KEY not set. Using offline mode."
    WANDB_MODE="offline"
else
    WANDB_MODE="online"
fi

# =============================================================================
# Install Third-Party Packages (first run only)
# =============================================================================

echo ""
echo "Setting up third-party packages..."
echo "=============================================="

apptainer exec --nv \
    --writable-tmpfs \
    --bind "${PROJECT_DIR}:/workspace" \
    --env PYTHONPATH="/workspace/ManiFlow:/workspace/third_party/gym-0.21.0:/workspace/third_party/Metaworld:/workspace/third_party/r3m" \
    --pwd /workspace \
    "${CONTAINER_IMAGE}" \
    bash -c "
        # Install gym 0.21.0
        cd /workspace/third_party/gym-0.21.0 && pip install --user -e . -q
        # Install Metaworld
        cd /workspace/third_party/Metaworld && pip install --user -e . -q
        # Install mjrl (Adroit dependency)
        cd /workspace/third_party/rrl-dependencies/mjrl && pip install --user -e . -q
        # Install mj_envs (Adroit env registration)
        cd /workspace/third_party/rrl-dependencies/mj_envs && pip install --user -e . -q
        # Install r3m
        cd /workspace/third_party/r3m && pip install --user -e . -q
        # Install ManiFlow
        cd /workspace/ManiFlow && pip install --user -e . -q
        pip install --user numpy==1.26.4
        echo 'Third-party packages installed successfully!'
    "

# =============================================================================
# Run Training
# =============================================================================

echo ""
echo "Starting FlowQL training at $(date)"
echo "=============================================="

echo "Cleaning Windows line endings from shell scripts..."
sed -i 's/\r$//' "${PROJECT_DIR}/scripts/train_flowql.sh"

# Run the training script inside the container
apptainer exec --nv \
    --writable-tmpfs \
    --bind "${PROJECT_DIR}:/workspace" \
    --bind "${WANDB_DIR}:/root/.wandb" \
    --env WANDB_API_KEY="${WANDB_API_KEY}" \
    --env WANDB_DIR="/root/.wandb" \
    --env MUJOCO_GL="egl" \
    --env PYTHONPATH="/workspace/ManiFlow:/workspace/third_party/gym-0.21.0:/workspace/third_party/Metaworld:/workspace/third_party/r3m" \
    --pwd /workspace \
    "${CONTAINER_IMAGE}" \
    /bin/bash -lc /workspace/scripts/train_flowql.sh \
        "${ALG_NAME}" \
        "${TASK_NAME}" \
        "${ADDITION_INFO}" \
        "${SEED}" \
        "${GPU_IDS}"

# Capture exit status
EXIT_STATUS=$?

# =============================================================================
# Job Completion
# =============================================================================

echo ""
echo "=============================================="
echo "Job finished at $(date)"
echo "Exit status: ${EXIT_STATUS}"
echo "=============================================="

if [ ${EXIT_STATUS} -eq 0 ]; then
    echo "FlowQL training completed successfully!"
else
    echo "FlowQL training failed with exit code ${EXIT_STATUS}"
fi

exit ${EXIT_STATUS}
