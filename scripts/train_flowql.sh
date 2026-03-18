
# Usage:
# bash scripts/train_flowql.sh flowql_adroit_door adroit_door_image flowql_test 0 0

DEBUG=False
save_ckpt=True
train=True
eval=False

alg_name=${1-flowql_adroit_door}
task_name=${2-adroit_door_image}
addition_info=${3-flowql_test}
seed=${4-0}
gpu_id=${5-0}

# Process task name (remove _image or _pointcloud suffix)
processed_task_name=${task_name}
if [[ $task_name == *"_image"* ]]; then
    processed_task_name=${task_name//_image/}
elif [[ $task_name == *"_pointcloud"* ]]; then
    processed_task_name=${task_name//_pointcloud/}
fi

# Setup paths and configuration
base_path="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="${base_path}/ManiFlow/data/outputs/${exp_name}_seed${seed}"
config_name=${alg_name}

# Environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VK_ICD_FILENAMES="${SCRIPT_DIR}/nvidia_icd.json"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

# Set wandb mode based on debug flag
if [ $DEBUG = True ]; then
    wandb_mode=offline
    echo -e "\033[33m=== DEBUG MODE ===\033[0m"
else
    wandb_mode=online
    echo -e "\033[33m=== TRAINING MODE ===\033[0m"
fi

# Print configuration
echo -e "\033[33mAlgorithm: ${alg_name}\033[0m"
echo -e "\033[33mTask: ${task_name}\033[0m"
echo -e "\033[33mGPU ID: ${gpu_id}\033[0m"
echo -e "\033[33mTrain: ${train}, Eval: ${eval}\033[0m"

# Change to workspace directory
cd ManiFlow/maniflow/workspace

# Training phase
if [ $train = True ]; then
    echo -e "\033[32m=== Starting FlowQL Training ===\033[0m"
    python train_flowql_adroit_workspace.py \
        --config-name=${config_name}.yaml \
        task=${task_name} \
        hydra.run.dir=${run_dir} \
        training.debug=$DEBUG \
        training.seed=${seed} \
        training.device="cuda:0" \
        exp_name=${exp_name} \
        logging.mode=${wandb_mode} \
        checkpoint.save_ckpt=${save_ckpt}

    if [ $? -eq 0 ]; then
        echo -e "\033[32m=== Training completed successfully ===\033[0m"
    else
        echo -e "\033[31m=== Training failed ===\033[0m"
        exit 1
    fi
else
    echo -e "\033[33m=== Skipping Training ===\033[0m"
fi

# Evaluation phase
if [ $eval = True ]; then
    echo -e "\033[32m=== Starting Evaluation ===\033[0m"
    python train_flowql_adroit_workspace.py \
        --config-name=${config_name}.yaml \
        task=${task_name} \
        hydra.run.dir=${run_dir} \
        training.debug=$DEBUG \
        training.seed=${seed} \
        training.device="cuda:0" \
        exp_name=${exp_name} \
        logging.mode=${wandb_mode} \
        checkpoint.save_ckpt=${save_ckpt}

    if [ $? -eq 0 ]; then
        echo -e "\033[32m=== Evaluation completed successfully ===\033[0m"
    else
        echo -e "\033[31m=== Evaluation failed ===\033[0m"
        exit 1
    fi
else
    echo -e "\033[33m=== Skipping Evaluation ===\033[0m"
fi

echo -e "\033[32m=== Script completed ===\033[0m"
