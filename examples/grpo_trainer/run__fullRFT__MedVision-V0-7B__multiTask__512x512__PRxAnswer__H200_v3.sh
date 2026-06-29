#! /bin/bash

# Run RFT (GRPO) from the public HF model YongchengYAO/MedVision-V0-7B
# on the 121K multi-task dataset (AD 5.5K + Detection 110K + TL 5.5K, 512x512).

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

bash "$script_dir/train__fullRFT__qwen25vl-7b-fullSFT__multiTask__512x512__PRxAnswer__H200_v3.sh" "$@" 2>&1 \
    | tee "$script_dir/run__fullRFT__MedVision-V0-7B__multiTask__512x512__PRxAnswer__H200_v3.log"
