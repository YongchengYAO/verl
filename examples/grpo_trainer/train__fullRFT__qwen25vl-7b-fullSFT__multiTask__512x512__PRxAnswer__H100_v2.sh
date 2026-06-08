#! /bin/bash

set -x
ENGINE=${1:-vllm}
# If DRY_RUN=1, script will skip heavy setup and training steps and only print actions.
DRY_RUN=${DRY_RUN:-0}
if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN=1: skipping heavy setup and training. Use DRY_RUN=0 to run for real."
fi
# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export VLLM_USE_FLASHINFER_SAMPLER=0

# HuggingFace login: export token so all Ray/vLLM subprocesses inherit it
HF_TOKEN_FILE="$HOME/.cache/huggingface/token"
if [ -f "$HF_TOKEN_FILE" ]; then
    export HF_TOKEN=$(cat "$HF_TOKEN_FILE")
    echo "HF_TOKEN loaded from $HF_TOKEN_FILE"
else
    echo "WARNING: HF token file not found at $HF_TOKEN_FILE. Private model downloads may fail."
fi


# NOTE: conda env creation/installation and CUDA library-path setup happen after the
# experiment config below (see "Set up env") so the env exists before we activate it.


# Define experiment name
exp_name="medvision__fullRFT__qwen25vl-7b-fullSFT__multiTask__512x512__PRxAnswer__normL2-mean"

# Define directories (derived from this script's location: examples/grpo_trainer/ -> repo root is two levels up)
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
verl_dir="$(cd "$script_dir/../.." && pwd)"
workspace_root="$script_dir"
log_root=$workspace_root/log
rollout_data_dir=$log_root/medvision_multi_tasks/$exp_name/rollout_data
validation_data_dir=$log_root/medvision_multi_tasks/$exp_name/validation_data
default_local_dir=$workspace_root/checkpoints/medvision_multi_tasks/$exp_name

# Data
# Check MedVision on how to prepare the verl datasets: https://github.com/YongchengYAO/MedVision
# This script expects dataset variant: ds__AD0_D1000000_TL0_all1000000__resized-hw-512x512
dataset_root="${DATASET_ROOT:?Set DATASET_ROOT to your prepared verl dataset directory (see https://github.com/YongchengYAO/MedVision)}"
if ls "$dataset_root/shards/"train_shard_*.parquet 1>/dev/null 2>&1; then
    dataset_train="$dataset_root/shards/train_shard_*.parquet"
else
    dataset_train="$dataset_root/train_verl.parquet"
fi
dataset_val=$dataset_root/validation_verl.parquet

# Model: HF model id or local checkpoint path (this stage continues from the AD-TL RFT checkpoint)
base_model_hf="${BASE_MODEL:?Set BASE_MODEL to a HF model id or local checkpoint path}"

# Training
epoch=10

# (Optional) Custom reward function
# process reward: mean normalized L2 distance for AD/TL localization steps; no process reward for detection
reward_function_path=$verl_dir/verl/utils/reward_score/medvision_rewards/medvision_general.py
reward_function_name=compute_score_exp_decay_PRxAnswer_v2

# (Optional) Custom dataset class
custom_cls_path=$verl_dir/verl/utils/dataset/medvision_dataset.py
custom_cls_name=MedVisionDataset

# Wandb
wandb_project=MedVision_fullRFT_GRPO_verl


# Set up env: create + install the conda env (idempotent), then activate it in THIS shell
# (setup_conda_verl.sh runs in a subshell, so its activation does not propagate here).
if [ "$DRY_RUN" != "1" ]; then
    bash $verl_dir/setup_conda_verl.sh
else
    echo "(DRY_RUN) would run: bash $verl_dir/setup_conda_verl.sh"
fi
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME:-verl}"


# ------
# Set environment variables for CUDA and library paths (after the env is installed)
# ------
# Fix for missing libcudart.so when using vLLM/FlashInfer with conda/pip installed CUDA
PYTHON_SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
# Add nvidia package paths to LD_LIBRARY_PATH and LIBRARY_PATH
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$PYTHON_SITE_PACKAGES/nvidia/cuda_runtime/lib:$PYTHON_SITE_PACKAGES/nvidia/cublas/lib:$PYTHON_SITE_PACKAGES/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$CONDA_PREFIX/lib:$PYTHON_SITE_PACKAGES/nvidia/cuda_runtime/lib:$PYTHON_SITE_PACKAGES/nvidia/cublas/lib:$PYTHON_SITE_PACKAGES/nvidia/cudnn/lib:$LIBRARY_PATH"

# Find libcudart.so.12 and create symlink
LIBCUDART_PATH=$(find "$CONDA_PREFIX" -name "libcudart.so.12" 2>/dev/null | head -n 1)
if [ -z "$LIBCUDART_PATH" ]; then
    echo "WARNING: libcudart.so.12 not found in $CONDA_PREFIX"
else
    echo "Found libcudart.so.12 at $LIBCUDART_PATH"
    if [ ! -f "$CONDA_PREFIX/lib/libcudart.so" ]; then
        ln -s "$LIBCUDART_PATH" "$CONDA_PREFIX/lib/libcudart.so"
        echo "Created symlink $CONDA_PREFIX/lib/libcudart.so -> $LIBCUDART_PATH"
    fi
    # Also try to link in torch/lib if it exists, as that is in the linker path
    TORCH_LIB="$PYTHON_SITE_PACKAGES/torch/lib"
    if [ -d "$TORCH_LIB" ] && [ ! -f "$TORCH_LIB/libcudart.so" ]; then
        ln -s "$LIBCUDART_PATH" "$TORCH_LIB/libcudart.so"
        echo "Created symlink $TORCH_LIB/libcudart.so -> $LIBCUDART_PATH"
    fi
fi

# Force linker to look in CONDA_PREFIX/lib
export LDFLAGS="-L$CONDA_PREFIX/lib $LDFLAGS"

export CUDA_HOME="$CONDA_PREFIX"
# ------


# Add optional arguments for logging rollout and validation data
# trainer.validation_data_dir=$validation_data_dir \
# trainer.rollout_data_dir=$rollout_data_dir \

# # Add optional arguments for custom datasets
# data.custom_cls.path=$custom_cls_path \
# data.custom_cls.name=$custom_cls_name \

# Other configs:
# actor_rollout_ref.model.use_shm=True \
# trainer.max_actor_ckpt_to_keep=5 \


# 4x H100 (80 GB) full finetuning settings
# Differences from the H200 variant:
#   ppo_micro_batch_size_per_gpu:        4 -> 2  (less activation memory per GPU)
#   rollout.log_prob_micro_batch_size_per_gpu: 4 -> 2
#   ref.log_prob_micro_batch_size_per_gpu:     4 -> 2
#   gpu_memory_utilization:              0.55 -> 0.50  (less absolute VRAM headroom)
if [ "$DRY_RUN" != "1" ]; then
    export RAY_memory_usage_threshold=0.98
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files=$dataset_train \
        data.val_files=$dataset_val \
        data.train_batch_size=256 \
        data.max_prompt_length=4096 \
        data.max_response_length=4096 \
        data.filter_overlong_prompts=False \
        data.truncation='error' \
        data.image_key=images \
        actor_rollout_ref.model.path=$base_model_hf \
        actor_rollout_ref.actor.optim.lr=3e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=128 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.01 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra'] \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.name=$ENGINE \
        +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.50 \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.rollout.enforce_eager=False \
        actor_rollout_ref.rollout.free_cache_engine=False \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.rollout.load_format="safetensors" \
        actor_rollout_ref.rollout.layered_summon=True \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.ref.fsdp_config.param_offload=False \
        algorithm.use_kl_in_reward=False \
        trainer.critic_warmup=0 \
        trainer.logger='["console","wandb"]' \
        trainer.project_name=$wandb_project \
        trainer.experiment_name=$exp_name \
        trainer.n_gpus_per_node=4 \
        trainer.nnodes=1 \
        trainer.save_freq=10 \
        trainer.test_freq=10 \
        trainer.total_epochs=$epoch \
        trainer.default_local_dir=$default_local_dir \
        trainer.validation_data_dir=$validation_data_dir \
        custom_reward_function.path=$reward_function_path \
        custom_reward_function.name=$reward_function_name \
        +reward_model.use_reward_loop=True \
        data.custom_cls.path=$custom_cls_path \
        data.custom_cls.name=$custom_cls_name \
        $@
else
    echo "(DRY_RUN) would run training: python3 -m verl.trainer.main_ppo with dataset_train=$dataset_train dataset_val=$dataset_val"
fi


# ------
# Merge latest checkpoint to HF model
# ------
# Find the latest checkpoint (largest global step number)
latest_ckpt=$(ls -d "$default_local_dir"/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -n 1)

if [ -z "$latest_ckpt" ]; then
    echo "WARNING: No checkpoints found in $default_local_dir. Skipping model merge."
else
    echo "Latest checkpoint: $latest_ckpt"
    actor_dir="$latest_ckpt/actor"
    merged_dir="$actor_dir/merged_hf_model"
    if [ "$DRY_RUN" != "1" ]; then
        python -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "$actor_dir" \
            --target_dir "$merged_dir"
        echo "Merged HF model saved to: $merged_dir"
    else
        echo "(DRY_RUN) would run: python -m verl.model_merger merge --backend fsdp --local_dir $actor_dir --target_dir $merged_dir"
    fi
fi
