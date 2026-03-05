
# 2D MetaWorld train+eval launcher (RGB image + agent_pos).
#
# Usage examples:
# bash scripts/train_eval_metaworld_2d.sh maniflow_image_timm_policy_metaworld_2d metaworld_multitask_mp_2d debug 0 1_2_3
# bash scripts/train_eval_metaworld_2d.sh maniflow_image_timm_policy_metaworld_2d metaworld_multitask_2d debug 0 1
#
# Notes:
# - alg_name should match a config under ManiFlow/maniflow/config/, e.g. `maniflow_image_timm_policy_metaworld_2d`
# - task_name should match a task config under ManiFlow/maniflow/config/task/, e.g. `metaworld_multitask_mp_2d`

DEBUG=False
save_ckpt=True
train=True
eval=True # only set eval=True when save_ckpt=True
eval_mode=best

alg_name=${1}
task_name=${2}
addition_info=${3}
seed=${4}
gpu_id=${5}
eval_env_processes=8  # Number of parallel processes for evaluation in total, adjust based on your GPU memory


# Setup paths and configuration
base_path="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dataset_path=${base_path}/ManiFlow/data
exp_name=${task_name}-${alg_name}-acgd_0.3_normtask-${addition_info}
run_dir="${base_path}/ManiFlow/data/outputs/${exp_name}_seed${seed}"
config_name=${alg_name}


# Environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VK_ICD_FILENAMES="${SCRIPT_DIR}/nvidia_icd.json"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export PYTHONPATH="${base_path}/ManiFlow:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=$(echo $gpu_id | tr '_' ',')

# env_device setup for multi-gpu eval
IFS='_' read -ra gpu_id_arr <<< "$gpu_id"
num_gpus=${#gpu_id_arr[@]}
env_device=""
for i in $(echo $gpu_id | tr "_" "\n")
do
    env_device+="\"cuda:${i}\","
done
env_device=[${env_device%,}]  # Remove trailing comma
echo -e "\033[33mEnvironment devices: ${env_device}\033[0m"


# 27 simple tasks + 11 medium tasks + 10 hard tasks --> 48 tasks in total
task_list=(
    "bin-picking"
    "coffee-pull"
    "door-lock"
    "peg-insert-side"
    "pick-place"
)
task_list_str=$(IFS=,; echo "${task_list[*]}")


# Set wandb mode based on debug flag
if [ $DEBUG = True ]; then
    wandb_mode=offline
    echo -e "\033[33m=== DEBUG MODE ===\033[0m"
else
    wandb_mode=online
    echo -e "\033[33m=== TRAINING MODE ===\033[0m"
fi


# Print configuration
echo -e "\033[33mTask config: ${task_name}\033[0m"
echo -e "\033[33mAlg config: ${alg_name}\033[0m"
echo -e "\033[33mGPU ID: ${gpu_id}\033[0m"
echo -e "\033[33mTrain: ${train}, Eval: ${eval}\033[0m"


# Change to workspace directory
cd ManiFlow/maniflow/workspace


# Training phase
if [ $train = True ]; then
    echo -e "\033[32m=== Starting Training (2D) ===\033[0m"
    python train_maniflow_metaworld_multitask_ddp_workspace_2d.py \
        --config-name=${config_name}.yaml \
        training.num_gpus=${num_gpus} \
        training.distributed=True \
        task=${task_name} \
        task.dataset.data_path=${dataset_path} \
        task.task_names=[${task_list_str}] \
        hydra.run.dir=${run_dir} \
        training.debug=$DEBUG \
        training.seed=${seed} \
        training.device="cuda:0" \
        training.env_device=${env_device} \
        exp_name=${exp_name} \
        logging.mode=${wandb_mode} \
        checkpoint.save_ckpt=${save_ckpt} \
        +eval_mode=${eval_mode}

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
    echo -e "\033[32m=== Starting Evaluation (2D) ===\033[0m"
    python eval_maniflow_metaworld_workspace_2d.py \
        --config-name=${config_name}.yaml \
        task=${task_name} \
        task.dataset.data_path=${dataset_path} \
        task.task_names=[${task_list_str}] \
        +task.env_runner.max_processes=${eval_env_processes} \
        hydra.run.dir=${run_dir} \
        training.debug=$DEBUG \
        training.seed=${seed} \
        training.device="cuda:0" \
        training.env_device=${env_device} \
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


