#!/bin/bash
#SBATCH --job-name=maniflow-2d
#SBATCH --output=logs/maniflow_%j.out
#SBATCH --error=logs/maniflow_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-1-employee

# =============================================================================
# ManiFlow 2D MetaWorld Training - SLURM Job Script
# =============================================================================
#
# Usage:
#   sbatch task.sh [alg_name] [task_name] [addition_info] [seed]
#
# Example:
#   sbatch task.sh maniflow_image_timm_policy_metaworld_2d metaworld_multitask_mp_2d exp1 42
#
# Default values if not provided:
#   alg_name: maniflow_image_timm_policy_metaworld_2d
#   task_name: metaworld_multitask_mp_2d
#   addition_info: cluster
#   seed: 0
#
# Prerequisites:
#   1. Build the container: sudo apptainer build maniflow.sif Apptainer.def
#   2. Upload maniflow.sif to the cluster home directory
#   3. Upload ManiFlow_Policy code to the cluster
#   4. Set WANDB_API_KEY environment variable or run `wandb login`
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

ALG_NAME=${1:-"maniflow_image_timm_policy_metaworld_2d"}
TASK_NAME=${2:-"metaworld_multitask_mp_2d"}
ADDITION_INFO=${3:-"cluster"}
SEED=${4:-0}

echo "Configuration:"
echo "  Algorithm: ${ALG_NAME}"
echo "  Task: ${TASK_NAME}"
echo "  Additional info: ${ADDITION_INFO}"
echo "  Seed: ${SEED}"
echo "  GPUs: 4"

# =============================================================================
# Environment Setup
# =============================================================================

# Create output directories if they don't exist
mkdir -p "${PROJECT_DIR}/ManiFlow/data/outputs"
mkdir -p "${WANDB_DIR}"
mkdir -p logs

# GPU device string for multi-GPU training
GPU_IDS="0_1_2_3"

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

# Install packages inside the container (writes to user site-packages)
# Only needed: gym-0.21.0, Metaworld, r3m, ManiFlow
apptainer exec --nv \
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
echo "Starting training at $(date)"
echo "=============================================="

# Run the training script inside the container
apptainer exec --nv \
    --bind "${PROJECT_DIR}:/workspace" \
    --bind "${WANDB_DIR}:/root/.wandb" \
    --env WANDB_API_KEY="${WANDB_API_KEY}" \
    --env WANDB_DIR="/root/.wandb" \
    --env MUJOCO_GL="egl" \
    --env CUDA_VISIBLE_DEVICES="0,1,2,3" \
    --env PYTHONPATH="/workspace/ManiFlow:/workspace/third_party/gym-0.21.0:/workspace/third_party/Metaworld:/workspace/third_party/r3m" \
    --pwd /workspace \
    "${CONTAINER_IMAGE}" \
    bash scripts/train_eval_metaworld_2d.sh \
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
    echo "Training completed successfully!"
else
    echo "Training failed with exit code ${EXIT_STATUS}"
fi

exit ${EXIT_STATUS}
