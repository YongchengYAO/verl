#! /bin/bash
# ================================================================================================
# MedVision RFT (GRPO) -- multi-task RFT
#
# Settings:
#   experiment ....... multi-task RFT ablation: one stage over the 121K mixture, T=8 task mixing, epoch-level curriculum (../../CURRICULUM_FILTERING.md)
#   base model ....... full-SFT CoT checkpoint (BASE_MODEL_HF default: YongchengYAO/MedVision-V0-7B-dev0630, a private repo)
#   dataset variant .. ds__AD5500_D110000_TL5500_all121000__resized-hw-512x512
#   reward ........... format_reward=soft, composition=multiplicative
#
# Usage:
#   DATASET_ROOT=<data_dir>/verl_datasets/qwen25vl/ds__AD5500_D110000_TL5500_all121000__resized-hw-512x512 \
#   BASE_MODEL_PATH=/path/to/full-SFT-checkpoint \
#   bash examples/grpo_trainer/train__rft-multitask.sh
#   # from a Hub model instead (e.g. the public MedVision-V0):
#   DATASET_ROOT=... BASE_MODEL_HF=YongchengYAO/MedVision-V0-7B bash examples/grpo_trainer/train__rft-multitask.sh
#   # without the curriculum (task mixing only):
#   DATASET_ROOT=... BASE_MODEL_PATH=... bash examples/grpo_trainer/train__rft-multitask.sh +data.curriculum.enable=False
#
# Environment variables:
#   DATASET_ROOT     prepared verl dataset directory (required; see https://github.com/YongchengYAO/MedVision#-training-rft)
#   BASE_MODEL_PATH  local checkpoint directory                                  -- exactly one of
#   BASE_MODEL_HF    Hugging Face repo id, downloaded into models/<name> first   -- the two
#   EXP_NAME         experiment name (default: rft-multitask); checkpoints and logs land under
#                    checkpoints|log/<EXP_NAME>; reuse a name to resume that run
#   ENGINE           rollout engine (default: vllm)
#   ENV_NAME         conda env name (default: verl)
#   DRY_RUN=1        print the actions (env setup, model download, training command, merge) without running them
# Trailing arguments are Hydra overrides for verl.trainer.main_ppo, e.g. trainer.total_epochs=5.
# Reward options: ../../REWARDS.md; recipe table and paper mapping: ../../README.md.
# ================================================================================================

set -x
ENGINE=${ENGINE:-vllm}
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
# Pod-injected HF_TOKEN may carry a trailing newline, which is illegal in HTTP auth headers (breaks vLLM/HF Hub requests)
if [ -n "${HF_TOKEN:-}" ]; then
    export HF_TOKEN="$(printf '%s' "$HF_TOKEN" | tr -d '[:space:]')"
fi


# NOTE: conda env creation/installation and CUDA library-path setup happen after the
# experiment config below (see "Set up env") so the env exists before we activate it.


# Experiment name (see the header): checkpoints|log/$exp_name; reuse a name to resume.
exp_name="${EXP_NAME:-rft-multitask}"

# Define directories (derived from this script's location: examples/grpo_trainer/ -> repo root is two levels up)
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
verl_dir="$(cd "$script_dir/../.." && pwd)"
workspace_root="$script_dir"
log_root=$workspace_root/log
rollout_data_dir=$log_root/$exp_name/rollout_data
validation_data_dir=$log_root/$exp_name/validation_data
default_local_dir=$workspace_root/checkpoints/$exp_name

# Data
# Check MedVision on how to prepare the verl datasets: https://github.com/YongchengYAO/MedVision
# This script expects dataset variant: ds__AD5500_D110000_TL5500_all121000__resized-hw-512x512
dataset_root="${DATASET_ROOT:?Set DATASET_ROOT to your prepared verl dataset directory (see https://github.com/YongchengYAO/MedVision)}"
if ls "$dataset_root/shards/"train_shard_*.parquet 1>/dev/null 2>&1; then
    dataset_train="$dataset_root/shards/train_shard_*.parquet"
else
    dataset_train="$dataset_root/train_verl.parquet"
fi
dataset_val=$dataset_root/validation_verl.parquet

# Base model -- set exactly one of:
#   BASE_MODEL_PATH  local checkpoint directory (a full-SFT CoT checkpoint)
#   BASE_MODEL_HF    Hugging Face repo id, downloaded once into models/<name> and trained from that local copy.
#                    Never hand a Hub id to verl directly: ~15 Ray/vLLM workers fetch it concurrently, and one
#                    transient hub failure leaves a vLLM server without a processor -> CUDA device-side assert.
if [ -n "${BASE_MODEL_PATH:-}" ] && [ -n "${BASE_MODEL_HF:-}" ]; then
    echo "ERROR: set only one of BASE_MODEL_PATH and BASE_MODEL_HF"; exit 1
fi
if [ -n "${BASE_MODEL_PATH:-}" ]; then
    base_model_dir="$BASE_MODEL_PATH"
    base_model_download_hf=""
else
    # Default = the paper's base: the final full-SFT CoT checkpoint (PRIVATE repo). Use your own SFT
    # checkpoint, or the public YongchengYAO/MedVision-V0-7B, if you cannot access it.
    base_model_download_hf="${BASE_MODEL_HF:-YongchengYAO/MedVision-V0-7B-dev0630}"
    base_model_dir="$script_dir/models/$(basename "$base_model_download_hf")"
fi

# Training
epoch=10

# Temperature-based multitask sampler (rebalances task sampling like SFT's TemperatureSamplerSFTTrainer)
# T=1: proportional to task counts (no rebalancing); larger T flattens task probabilities; SFT used T=5
# T=8 on the 110K/5.5K/5.5K mix -> Detection 42.1%, AD 29.0%, TL 29.0% per epoch
# Tasks are grouped via task_group_map: angle + distance merged into one AD task (matches SFT's AD/Detection/TL)
temperature_sampler_enable=True
temperature_sampler_T=8

# Curriculum sample filtering (epoch-level hard-example mining; see verl/utils/dataset/curriculum.py)
# At each epoch end, the top 20% of the current training set by EMA reward (gated on
# EMA answer-error < gate: MRE < 0.1 for A/D + T/L; overlap error (1-CIoU)/2 < 0.25 for
# detection) moves to a per-task easy pool and is dropped from training.
# A retention mix-in of most-recently-promoted easy samples ramps up with the solved
# fraction (full 30% easy / 70% hard once 50% of the task is solved), and each task keeps
# at least 10% of its samples active (anti-extinction floor), so the training set shrinks
# with the hard pool without any task vanishing from training.
# Requires the reward loop (+reward_model.use_reward_loop=True, set below) for per-sample
# answer_error. NOTE: the LR schedule horizon is computed from the first epoch's length,
# so with shrinking epochs a decaying schedule will not fully decay (constant LR is fine).
curriculum_enable=True
curriculum_easy_top_frac=0.20
curriculum_mre_gate=0.10
curriculum_threshold_frac=0.50
curriculum_mixin_easy_frac=0.30
curriculum_demote_easy=True  # re-demote mixed-in easy samples whose MRE regresses (False: easy is sticky)
curriculum_ema_alpha=0.4  # EMA weight for per-sample score/error evidence (1.0 = single-epoch evidence)
curriculum_promote_patience=1  # consecutive observed passing epochs to promote (2 = stricter, but coverage-limited on big tasks)
curriculum_demote_patience=1  # consecutive failing audits required to demote
curriculum_demote_margin=1.5  # demote only past margin*mre_gate (hysteresis vs promotion gate)
curriculum_audit_frac=0.05  # rotating easy-pool audit slots per epoch, as a fraction of the hard pool
curriculum_mixin_ramp=True  # ramp the easy mix-in with the solved fraction (False: binary phase at threshold_frac)
curriculum_task_floor_frac=0.10  # minimum active fraction of each task's original size (anti task-extinction)
curriculum_detection_gate=0.25  # detection promotion gate on overlap error (1-CIoU)/2; 0.25 <=> EMA CIoU>0.5 (~IoU>0.5)

# Reward (see REWARDS.md): one entry point configured through reward_kwargs (Hydra overrides below).
#   format_reward ... soft (default: 0.8 * reasoning-structure score + 0.2 * answer-format check) | binary
#   composition ..... multiplicative (default: r = format + process * answer) | additive (r = format + process + answer)
#   process reward: worst-point localization error of the CoT steps (A/D, T/L); answer reward: relative error (A/D, T/L)
#   or CIoU overlap error (detection, which has no process reward: r = format + answer)
reward_function_path=$verl_dir/verl/utils/reward_score/medvision_rewards/medvision_general.py
reward_function_name=compute_score
reward_format=soft
reward_composition=multiplicative

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

# Fetch the Hub base model into its local directory (idempotent; see BASE_MODEL_HF above).
if [ -n "$base_model_download_hf" ]; then
    if [ "$DRY_RUN" != "1" ]; then
        python "$script_dir/download_hf_model.py" --repo_id "$base_model_download_hf" --local_dir "$base_model_dir" \
            || { echo "ERROR: model download failed"; exit 1; }
    else
        echo "(DRY_RUN) would download $base_model_download_hf to $base_model_dir"
    fi
fi


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
    if [ "$DRY_RUN" != "1" ] && [ ! -f "$CONDA_PREFIX/lib/libcudart.so" ]; then
        ln -s "$LIBCUDART_PATH" "$CONDA_PREFIX/lib/libcudart.so"
        echo "Created symlink $CONDA_PREFIX/lib/libcudart.so -> $LIBCUDART_PATH"
    fi
    # Also try to link in torch/lib if it exists, as that is in the linker path
    TORCH_LIB="$PYTHON_SITE_PACKAGES/torch/lib"
    if [ "$DRY_RUN" != "1" ] && [ -d "$TORCH_LIB" ] && [ ! -f "$TORCH_LIB/libcudart.so" ]; then
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


# 4x H200 (140 GB) full finetuning settings
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
        actor_rollout_ref.model.path=$base_model_dir \
        actor_rollout_ref.actor.optim.lr=3e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=128 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.01 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra'] \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.name=$ENGINE \
        +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.rollout.enforce_eager=False \
        actor_rollout_ref.rollout.free_cache_engine=False \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.rollout.load_format="safetensors" \
        actor_rollout_ref.rollout.layered_summon=True \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
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
        +custom_reward_function.reward_kwargs.format_reward=$reward_format \
        +custom_reward_function.reward_kwargs.composition=$reward_composition \
        +reward_model.use_reward_loop=True \
        data.custom_cls.path=$custom_cls_path \
        data.custom_cls.name=$custom_cls_name \
        data.seed=1024 \
        +data.temperature_sampler.enable=$temperature_sampler_enable \
        +data.temperature_sampler.T=$temperature_sampler_T \
        +data.temperature_sampler.task_key=ability \
        "+data.temperature_sampler.task_group_map='medvision-angle:AD,medvision-distance:AD'" \
        +data.curriculum.enable=$curriculum_enable \
        +data.curriculum.easy_top_frac=$curriculum_easy_top_frac \
        +data.curriculum.mre_gate=$curriculum_mre_gate \
        +data.curriculum.threshold_frac=$curriculum_threshold_frac \
        +data.curriculum.mixin_easy_frac=$curriculum_mixin_easy_frac \
        +data.curriculum.demote_easy=$curriculum_demote_easy \
        +data.curriculum.ema_alpha=$curriculum_ema_alpha \
        +data.curriculum.promote_patience=$curriculum_promote_patience \
        +data.curriculum.demote_patience=$curriculum_demote_patience \
        +data.curriculum.demote_margin=$curriculum_demote_margin \
        +data.curriculum.audit_frac=$curriculum_audit_frac \
        +data.curriculum.mixin_ramp=$curriculum_mixin_ramp \
        +data.curriculum.task_floor_frac=$curriculum_task_floor_frac \
        +data.curriculum.detection_gate=$curriculum_detection_gate \
        +data.curriculum.task_key=ability \
        "+data.curriculum.task_group_map='medvision-angle:AD,medvision-distance:AD'" \
        "$@"
    train_status=$?
else
    echo "(DRY_RUN) would run training: python3 -m verl.trainer.main_ppo with dataset_train=$dataset_train dataset_val=$dataset_val"
    train_status=0
fi

# Skip the checkpoint merge if training did not finish cleanly, so we never merge a
# stale/older checkpoint after a crash (this script has no `set -e`, so without this
# guard it would fall through to the merge step regardless of the training exit code).
if [ "$train_status" -ne 0 ]; then
    echo "ERROR: training exited with status $train_status; skipping checkpoint merge."
    exit "$train_status"
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
