#! /bin/bash

# Run RFT (GRPO) from the public HF model YongchengYAO/MedVision-V0-7B
# on the 121K multi-task dataset (AD 5.5K + Detection 110K + TL 5.5K, 512x512),
# with curriculum sample filtering enabled (see CURRICULUM_FILTERING.md).
#
# Identical to run__...__PRxAnswer__H200_v3.sh except it launches the curriculum
# variant of the train script (temperature sampler + epoch-level easy-sample
# filtering, ramped retention mix-in, per-task floor, easy-pool audits).
#
# Coverage note for this dataset mix: the temperature sampler (T=8) gives
# Detection ~42% of the 121K draws/epoch spread over 110K samples, so only
# ~37% of Detection samples are observed per epoch (with replacement). The
# default curriculum_promote_patience=1 lets each task filter as soon as a
# sample passes (anti-luck handled by the EMA gate + top-fraction cap);
# raising it to 2 would coverage-limit Detection promotion (~5 epochs just to
# observe a sample twice) while AD/TL (~6 draws per sample per epoch) would
# be unaffected.
#
# ----------------------------------------------------------------------------
# Hyperparameters (defined in the train script; listed here for reference)
#
# Training (GRPO full finetuning, 1 node x 4 H200, vLLM rollout):
#   model ......................... YongchengYAO/MedVision-V0-7B (pre-downloaded locally)
#   dataset ....................... 121K multi-task (AD 5.5K / Detection 110K / TL 5.5K), 512x512
#   epochs ........................ 10 (epoch length shrinks as the curriculum filters)
#   train_batch_size .............. 256 prompts/step (= generation batch; drop_last)
#   ppo_mini_batch_size ........... 128
#   ppo_micro_batch_size_per_gpu .. 4 (actor; log-prob/ref micro = 4)
#   rollout.n ..................... 8 rollouts per prompt (GRPO group size)
#   max_prompt / max_response ..... 4096 / 4096 tokens (truncation='error')
#   lr ............................ 3e-6, constant (no decay -> immune to the
#                                   shrinking-epoch LR-horizon caveat)
#   KL ............................ kl_loss_coef=0.01 (low_var_kl); use_kl_in_reward=False
#   entropy_coeff ................. 0
#   rollout engine ................ vLLM, TP=2, gpu_memory_utilization=0.55,
#                                   enforce_eager=False, free_cache_engine=False
#   reward ........................ compute_score_exp_decay_PRxAnswer_v3
#                                   (r = format + process*answer; max-normL2 process
#                                   reward; answer reward = MRE for A/D+T/L, CIoU overlap
#                                   for detection), async reward loop enabled (required by
#                                   the curriculum for per-sample answer_error)
#   seed .......................... 1024
#   save_freq / test_freq ......... every 10 steps (step-based: relatively denser
#                                   per epoch as epochs shrink)
#
# Temperature multitask sampler (composes with the curriculum):
#   T=8 over `ability`, angle+distance merged into AD via task_group_map
#   -> Detection 42.1% / AD 29.0% / TL 29.0% of draws per epoch; weights are
#   recomputed on the curriculum's active subset each epoch, reseeded seed+epoch.
#
# Curriculum sample filtering (all defaults; see CURRICULUM_FILTERING.md):
#   enable ............ True
#   easy_top_frac ..... 0.20   top fraction of the current set eligible for promotion/epoch
#   mre_gate .......... 0.10   EMA answer-MRE below this = "easy" for A/D+T/L (matches benchmark MRE<0.1)
#   detection_gate .... 0.25   EMA overlap error (1-CIoU)/2 below this = "easy" for detection (~IoU>0.5)
#   threshold_frac .... 0.50   solved fraction at which the mix-in reaches full strength
#   mixin_easy_frac ... 0.30   retention mix-in ceiling (30% easy / 70% hard)
#   mixin_ramp ........ True   ramp the mix-in with the solved fraction (no phase cliff)
#   demote_easy ....... True   re-demote regressed mixed-in/audited easy samples
#   ema_alpha ......... 0.4    evidence EMA weight (~2.5-epoch memory)
#   promote_patience .. 1      observed passing epochs required to promote
#   demote_patience ... 1      failing audits required to demote
#   demote_margin ..... 1.5    hysteresis: demote only at EMA MRE >= 0.15
#   audit_frac ........ 0.05   rotating easy-pool audit slots (fraction of the hard pool)
#   task_floor_frac ... 0.10   per-task minimum active fraction (anti-extinction)
#   task_key .......... ability (group map: medvision-angle/-distance -> AD)
# ----------------------------------------------------------------------------

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Download the model to a local dir first (idempotent), then train from the local path.
# Loading from the hub id at runtime is fragile: ~15 Ray processes fetch the repo
# concurrently, and a single transient hub failure leaves a vLLM server without a
# processor, which silently skips image-token dedup and crashes rollout with a
# CUDA device-side assert (masked_scatter: totalElements <= srcSize).
model_dir="$script_dir/models/MedVision-V0-7B"
if [ -n "${HF_TOKEN:-}" ]; then
    export HF_TOKEN="$(printf '%s' "$HF_TOKEN" | tr -d '[:space:]')"
fi
eval "$(conda shell.bash hook)"
conda activate verl
python "$script_dir/dev-medvision/download_hf_model.py" \
    --repo_id YongchengYAO/MedVision-V0-7B \
    --local_dir "$model_dir" || { echo "ERROR: model download failed"; exit 1; }

export BASE_MODEL="$model_dir"
export DATASET_ROOT="/mnt/vincent-pvc-rwm/Github/MedVision/Data/verl_datasets/qwen25vl/ds__AD5500_D110000_TL5500_all121000__resized-hw-512x512"

# NOTE: the train script runs `set -x`, so the tee'd log can contain secrets
# (e.g. HF_TOKEN). Logs are gitignored (*.log) — keep it that way.
bash "$script_dir/train__fullRFT__qwen25vl-7b-fullSFT__multiTask__512x512__PRxAnswer__curriculum__H200.sh" "$@" 2>&1 \
    | tee "$script_dir/run__fullRFT__MedVision-V0-7B__multiTask__512x512__PRxAnswer__curriculum__H200.log"
