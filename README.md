# verl (MedVision RFT fork)

A fork of [volcengine/verl](https://github.com/volcengine/verl) extended for **reinforcement
fine-tuning (RFT) of MedVision models** — quantitative medical image analysis with Vision
Language Models. See the companion project: [MedVision](https://github.com/YongchengYAO/MedVision).

> **Looking for the original verl documentation?** The upstream README (installation,
> architecture, general usage, model/algorithm support) is preserved verbatim at
> **[`VERL_README.md`](./VERL_README.md)**.

> All MedVision-specific changes live on the **`medvision-rl`** branch (this branch), rebased on
> top of upstream verl. The `main` branch tracks upstream unchanged.

---

## What this fork adds

GRPO full-RFT training on the three MedVision quantitative tasks:

- **A/D** — angle / distance biometric measurement from landmarks
- **T/L** — tumor / lesion size (major & minor axis lengths)
- **Detection** — bounding-box localization

Custom reward functions score both the final `<answer>` value and the intermediate landmark
estimates:

```
r = r_format + r_process * r_answer        # process/answer rewards = exp(-error)
```

`r_process` rewards accurate intermediate landmark localization (A/D and T/L only; detection has
no process reward).

**Curriculum sample filtering** (optional): epoch-level hard-example mining that drops reliably
solved samples from training (per-task easy/hard pools, EMA + patience promotion evidence,
a retention mix-in that ramps to 30/70 with the solved fraction, a per-task anti-extinction
floor, and rotating easy-pool audits with hysteresis-guarded demotion) so
rollout compute concentrates on samples that still carry GRPO gradient. See
**[`CURRICULUM_FILTERING.md`](./CURRICULUM_FILTERING.md)** for the algorithm, configuration,
default-threshold rationale, checkpoint/resume behavior, and limitations.

---

## New components

### Reward functions — `verl/utils/reward_score/medvision_rewards/`
| File | Purpose |
|------|---------|
| `medvision_ad.py` | A/D reward: parses `<answer>`, scores landmark localization + final value |
| `medvision_tl.py` | T/L reward: two-axis (major / minor) measurement scoring |
| `medvision_general.py` | Shared entry points (`compute_score_exp_decay*`); dispatches by `ability` |
| `reward_fn.py` | Reward-shaping helpers (`scaled_sigmoid`, `gaussian_proxy`, `exp_decay`) |
| `registry.py` | Maps task name → reward function |

### Dataset — `verl/utils/dataset/medvision_dataset.py`
`MedVisionDataset` (subclass of `RLHFDataset`). MedVision parquet stores each prompt as
structured `list[dict]` content with the image in a separate `images` column; the
`_build_messages` override normalizes this into the `<image>`-placeholder form the upstream async
AgentLoop expects, so images are injected unchanged from the parquet. Images are **pre-sized
during data prep** (the prompt embeds pixel-size info), so no extra resize is introduced. Also
supports optional `max_samples` subsampling, and hosts the curriculum-filtering hooks
(`init_curriculum` / `on_batch_end` / `advance_curriculum`, see
[`CURRICULUM_FILTERING.md`](./CURRICULUM_FILTERING.md)).

### Training recipes — `examples/grpo_trainer/`
Qwen2.5-VL-7B full-RFT GRPO scripts. Paths are **not hardcoded** — `verl_dir`/`workspace_root`
derive from the script location, and dataset/model come from the `DATASET_ROOT` / `BASE_MODEL`
environment variables (scripts fail fast with a helpful message if unset).

- **Sequential single-task** (4×H200), run A/D → T/L → detection:
  `…__AD__…__H200.sh`, `…__AD-TL__…__H200.sh`, `…__AD-TL-D__…__H200.sh`
- **Multi-task** (single mixed run): `…__multiTask__…__H200_v2.sh` / `_v3.sh`. **v2 = mean**,
  **v3 = max** normalized-L2 process reward.
- **Multi-task + curriculum filtering**: `…__multiTask__…__curriculum__H200.sh` — v3 plus
  epoch-level easy-sample filtering (see [`CURRICULUM_FILTERING.md`](./CURRICULUM_FILTERING.md)).

### Environment — repo root
- `setup_conda_verl.sh` — creates the conda `verl` env (Python 3.12) and installs the full pinned
  stack (torch / vLLM / SGLang / flash-attn + CUDA libs as pip wheels). No system CUDA/cuDNN or
  root required.
- `requirements_medvision.txt` — frozen `pip freeze` (257 pinned packages) of a known-good
  environment, kept for reproducibility.

### Modified upstream files (minimal, behavior-additive)
| File | Change |
|------|--------|
| `verl/workers/reward_manager/naive.py` | Inject top-level `ability` into `extra_info` so reward fns can route by task (copy, no mutation) |
| `verl/experimental/reward_loop/reward_manager/naive.py` | Same `ability` injection on the async reward-loop path |
| `verl/trainer/ppo/ray_trainer.py` | Log extra/auxiliary reward components; curriculum epoch-boundary hooks (pool advance, dataloader rebuild, `curriculum.json` save/resume) |
| `verl/utils/dataset/curriculum.py` | **New** — curriculum pools + active-subset sampler ([docs](./CURRICULUM_FILTERING.md)) |
| `verl/model_merger/base_model_merger.py` | Patch merged `config.json` to `dtype=bfloat16` |
| `verl/utils/fsdp_utils.py` | LoRA: `get_peft_model_state_dict(..., save_embedding_layers=False)` |
| `verl/protocol.py` | Make `ray` import optional (lightweight debug environments) |
| `verl/utils/dataset/rl_dataset.py` | Import/formatting only |

---

## Quick start

```bash
# 1. Set up the conda env (installs the full pinned stack via uv, pip fallback; then verl).
#    No system CUDA/cuDNN or root needed. Optional: ENV_NAME=<name>, RUN_SYSTEM_FIXES=1 (GLIBC fix).
bash setup_conda_verl.sh

# 2. Prepare the verl-format MedVision dataset (see the MedVision repo for data prep):
#    https://github.com/YongchengYAO/MedVision#-training-rft

# 3. Run a GRPO RFT stage. DATASET_ROOT and BASE_MODEL are required.
#    Use DRY_RUN=1 to preview the commands without launching training.
DATASET_ROOT=</path/to/verl_datasets/qwen25vl/ds__AD5500_D0_TL0_all5500__resized-hw-512x512> \
BASE_MODEL=</path/to/full-SFT-checkpoint-or-HF-id> \
bash examples/grpo_trainer/train__fullRFT__qwen25vl-7b-fullSFT__AD__512x512__PRxAnswer__H200.sh
```

Each script prints the dataset variant it expects (a comment near `dataset_root`), supports a
`shards/` → `train_verl.parquet` layout fallback, and merges the latest checkpoint to an HF model
at the end of the run.

---

## Environment setup details (`setup_conda_verl.sh`)

The setup installs a single, fully pinned, known-good environment from `requirements_medvision.txt`,
which captures the **entire stack as pip wheels** — `torch`, `vllm`, `sglang`, `flash-attn` (exact
wheel + sha256), `flashinfer`, and all CUDA libraries (`nvidia-*-cu12`).

1. **One-shot, reproducible install.** Runs `uv pip install -r requirements_medvision.txt`
   (falling back to plain `pip`), then `pip install --no-deps -e .` for verl itself.
2. **No system CUDA/cuDNN, no root.** CUDA runtime/libraries and flash-attn come from pip wheels.
   The only host prerequisite is an **NVIDIA driver new enough for CUDA 12.8**.
3. **Optional host GLIBC fix.** If flash-attn reports `GLIBC_2.32' not found`, re-run with
   `RUN_SYSTEM_FIXES=1` (requires `sudo`) to `apt install libc6 zlib1g-dev libtinfo-dev`.
4. **Single source for env creation.** The training scripts call `setup_conda_verl.sh`, then
   activate the env and set CUDA library paths before launching — no duplicated bootstrap.

Use `ENV_NAME=<name> bash setup_conda_verl.sh` to install into a differently-named env.
