# verl (MedVision RFT fork)

A fork of [volcengine/verl](https://github.com/volcengine/verl) extended for **reinforcement
fine-tuning (RFT) of MedVision models** — quantitative medical image analysis with Vision
Language Models. It implements the RFT stage of the paper
[MedVision: Benchmarking Quantitative Medical Image Analysis](https://arxiv.org/abs/2511.18676)
(EMNLP 2026): the task-specific rewards, the multi-task training mixture, and the epoch-level
curriculum. Companion project: [MedVision](https://github.com/YongchengYAO/MedVision) (dataset,
benchmark, SFT, and the parquet builders that produce the RFT data consumed here).

> **Looking for the original verl documentation?** The upstream README (installation,
> architecture, general usage, model/algorithm support) is preserved verbatim at
> **[`VERL_README.md`](./VERL_README.md)**.

> All MedVision-specific changes live on the **`medvision-rl`** branch (this branch), rebased on
> top of upstream verl. The `main` branch tracks upstream unchanged.

---

## News

- **[Sep 1, 2026]** Refactor codebase (aligned with our [EMNLP26 paper](https://arxiv.org/abs/2511.18676)) 
  - sequential and multi-task RFT scripts
  - `REWARDS.md`: process reward, multiplicative reward composition
  - `CURRICULUM_FILTERING.md`: curriculum learning for multi-task RFT

## What this fork adds

GRPO full-parameter RFT on the three MedVision quantitative tasks:

- **A/D** — angle / distance biometric measurement from landmarks
- **T/L** — tumor / lesion size (major & minor axis lengths)
- **Detection** — bounding-box localization

Three components of the paper are implemented on top of upstream verl:

| Paper | Implementation |
|---|---|
| Appendix *Details of RFT via GRPO* — reward components and Algorithm 2 (reward computation for one rollout) | [`verl/utils/reward_score/medvision_rewards/`](./verl/utils/reward_score/medvision_rewards/); every variant, how to select it, and each experiment's true configuration in [`REWARDS.md`](./REWARDS.md) |
| Appendix *Temperature-Scaled Task Mixing* | [`verl/utils/dataset/temperature_sampler.py`](./verl/utils/dataset/temperature_sampler.py) |
| Appendix *Epoch-Level Curriculum Learning for Multi-Task RFT* — Algorithm 3 | [`verl/utils/dataset/curriculum.py`](./verl/utils/dataset/curriculum.py) + [`CURRICULUM_FILTERING.md`](./CURRICULUM_FILTERING.md) |
| Training configuration (GRPO, 8 rollouts, batch 256 / mini-batch 128, lr 3e-6, KL 0.01, 4096/4096 tokens) and the two ablations (multi-task RFT; additive vs. multiplicative reward) | [`examples/grpo_trainer/`](./examples/grpo_trainer/) recipes (table below) |

### Rewards

Each rollout is scored by up to three components in [0, 1]. The two accuracy components map an
error `e` to a reward with `ρ(e) = exp(−e)`; an unparseable value earns 0. Predicted coordinates
are relative positions in [0, 1] and measurement errors are relative errors, so all errors are
dimensionless and comparable across tasks.

| Component | A/D and T/L | Detection |
|---|---|---|
| `r_format` | **soft** (default): 0.8 · reasoning-structure score of the `<think>` / `<step-k-*>` blocks + 0.2 · binary answer-format check; **binary**: the answer check alone | binary answer-format check (no CoT steps) |
| `r_process` | mean over the CoT steps (A/D: 3, T/L: 4) of `ρ(step error)`. Localization steps (landmarks / axis endpoints in `<step-k-answer>`): displacement of the worst-localized point from its ground truth, divided by √2. Measurement steps: relative error | — (the answer is a single localization step) |
| `r_answer` | `ρ(mean relative error)` of the values in `<answer>` | `ρ((1 − CIoU) / 2)`: the overlap error of the predicted box (CIoU = IoU − center-distance penalty − aspect-ratio penalty, a graded signal even for non-overlapping boxes) |

```
r = r_format + r_process * r_answer     # composition=multiplicative (default)
r = r_format + r_process + r_answer     # composition=additive (reward-design ablation)
r = r_format + r_answer                 # detection (no process reward)
```

One entry point, `medvision_general.compute_score`, dispatches on the sample's `ability`
(`medvision-tl`, `medvision-angle`, `medvision-distance`, `medvision-detection`) and is configured
from the recipe through `+custom_reward_function.reward_kwargs.*`: `format_reward` (`soft` |
`binary`), `composition` (`multiplicative` | `additive`) and `reward_mapping_func`. Every call also emits the raw per-sample errors (`answer_error`,
`localization_error`, `measurement_error`; NaN where not applicable) — the curriculum consumes
`answer_error`, and all of them are logged as `critic/rewards/*`. Options, presets and the
configuration of the paper's runs: [`REWARDS.md`](./REWARDS.md).

### Temperature-scaled task mixing

The 121K multi-task mixture is 110K detection / 5.5K T/L / 5.5K A/D. Task `t` with `N_t` samples
is drawn with probability `P(t) ∝ N_t^(1/T)` (every sample of the task gets weight `P(t)/N_t`);
one epoch draws `len(dataset)` samples **with replacement**, seeded by `data.seed`. `T=1`
reproduces the natural proportions (90.9 / 4.5 / 4.5 %), larger `T` flattens toward uniform;
`T=8` (used in the multi-task RFT runs) gives Detection 42.1 % / A/D 29.0 % / T/L 29.0 % of draws.
Angle and distance are merged into one A/D task via `task_group_map`. Wired into
`create_rl_sampler` (`verl/trainer/main_ppo.py`); with fewer than two distinct tasks it falls
back to the stock sampler.

```bash
+data.temperature_sampler.enable=True +data.temperature_sampler.T=8 \
+data.temperature_sampler.task_key=ability \
"+data.temperature_sampler.task_group_map='medvision-angle:AD,medvision-distance:AD'"
```

### Curriculum sample filtering

Optional epoch-level hard-example mining for multi-task RFT: samples the policy reliably solves
are moved out of training so rollout compute concentrates on samples that still carry GRPO
gradient. Per task, samples live in a hard pool (under training) or an easy pool (solved); at
every epoch end the drawn hard samples are ranked by EMA reward and the top fraction `f` whose
EMA answer error clears the task's gate (MRE < 0.1 for A/D and T/L; overlap error < 0.25, i.e.
IoU > 0.5, for detection) is promoted, a retention mix-in of recently promoted samples ramps up
to 30 % easy / 70 % hard with the solved fraction, a rotating audit slice re-validates the easy
pool with hysteresis-guarded demotion, and a per-task floor keeps every task represented. The
temperature weights are recomputed over the active set every epoch, so epochs shorten as the
pools shrink. The paper's symbols (`f, m, p*, λ, a, φ, α, ν±, g_t`) map one-to-one onto the
`+data.curriculum.*` options — see the correspondence table and the implementation notes in
**[`CURRICULUM_FILTERING.md`](./CURRICULUM_FILTERING.md)** (algorithm, configuration,
default-threshold rationale, checkpoint/resume behavior, monitoring, limitations).

Full pool membership is dumped per epoch to `{default_local_dir}/curriculum_pools/epoch_NNNN.json`
(sample `idx` = post-filter dataset row position — unique and resume-stable within a run, but
positional, not a MedVision case ID) and can be plotted with
[`examples/grpo_trainer/curriculum-learning/plot_curriculum_pools.py`](./examples/grpo_trainer/curriculum-learning/plot_curriculum_pools.py).

---

## New components

### Reward functions — `verl/utils/reward_score/medvision_rewards/`
| File | Purpose |
|------|---------|
| `medvision_general.py` | `compute_score` — the configurable multi-task entry point (format variant, composition); format / answer / process reward dispatch by `ability` |
| `medvision_tl.py`, `medvision_ad.py` | Per-task CoT step layout and ground-truth mapping, soft reasoning-structure score, process reward |
| `process_reward.py` | Generic worst-point process reward over the CoT steps |
| `parsing.py` | Tags, number / coordinate patterns, `<answer>` and `<step-k-answer>` parsing, the soft-structure rubric |
| `reward_fn.py` | Error metrics (relative error, worst-point normalized-L2, `cal_ciou` / `cal_ciou_reward_error`) and the ρ maps (`exp_decay`, `scaled_sigmoid`, `gaussian_proxy`) |
| `registry.py` | Plots the available error-to-reward maps |

### Dataset and sampling — `verl/utils/dataset/`
| File | Purpose |
|------|---------|
| `medvision_dataset.py` | `MedVisionDataset` (subclass of `RLHFDataset`): normalizes the parquet's structured `list[dict]` prompt + separate `images` column into the `<image>`-placeholder form the upstream async AgentLoop expects (images are pre-sized during data prep; no extra resize); optional `max_samples`; hosts the curriculum hooks `init_curriculum` / `on_batch_end` / `advance_curriculum` |
| `temperature_sampler.py` | **New** — temperature-scaled task mixing (`+data.temperature_sampler.*`) |
| `curriculum.py` | **New** — `CurriculumManager`: per-task pools, evidence record, active-set sampler ([docs](./CURRICULUM_FILTERING.md)) |

### Training recipes — `examples/grpo_trainer/`
Five scripts, one per experiment of the paper — full-parameter GRPO of Qwen2.5-VL-7B on 4×H200
with vLLM rollouts. Paths derive from the script location; the dataset and the base model come from
environment variables (a script fails fast with a message if they are missing), and any trailing
arguments are passed to `verl.trainer.main_ppo` as Hydra overrides.

| Script | Experiment | Base model | Dataset variant (`DATASET_ROOT`) | Reward (`compute_score`) |
|---|---|---|---|---|
| `train__rft-sequential__1-AD.sh` | MedVision-V0 stage 1: A/D RFT | full-SFT CoT checkpoint | `ds__AD5500_D0_TL0_all5500__resized-hw-512x512` | `format_reward=soft`, `composition=multiplicative` |
| `train__rft-sequential__2-TL.sh` | stage 2: T/L RFT | stage 1's `global_step_N/actor/merged_hf_model` | `ds__AD0_D0_TL5500_all5500__resized-hw-512x512` | `format_reward=soft`, `composition=multiplicative` |
| `train__rft-sequential__3-detection.sh` | stage 3: detection RFT → **MedVision-V0** | stage 2's `global_step_N/actor/merged_hf_model` | `ds__AD0_D1000000_TL0_all1000000__resized-hw-512x512` (1M detection set) | `format_reward=soft`, `composition=multiplicative` |
| `train__rft-multitask.sh` | **multi-task RFT** ablation (Table *RFT data-mixture ablation*): one stage over the 121K mixture, T=8 task mixing, epoch-level curriculum | full-SFT CoT checkpoint: `BASE_MODEL_PATH=<dir>` or `BASE_MODEL_HF=<repo>` (default `YongchengYAO/MedVision-V0-7B-dev0630`, private) | `ds__AD5500_D110000_TL5500_all121000__resized-hw-512x512` | `format_reward=soft`, `composition=multiplicative` |
| `train__rft-multitask__additive-reward.sh` | **additive-reward** ablation (Table *RFT reward-design ablation*): identical to the row above except the combination rule | full-SFT CoT checkpoint: `BASE_MODEL_PATH=<dir>` or `BASE_MODEL_HF=<repo>` (default `YongchengYAO/MedVision-V0-7B-dev0630`, private) | `ds__AD5500_D110000_TL5500_all121000__resized-hw-512x512` | `format_reward=soft`, `composition=additive` |

Plain multi-task RFT without the curriculum is `train__rft-multitask.sh +data.curriculum.enable=False`
(the temperature sampler stays on). Alongside the scripts: `download_hf_model.py` (fetches a Hub
base model into `models/<name>`), `curriculum-learning/plot_curriculum_pools.py` (renders a run's
`curriculum_pools/` snapshots); the upstream verl GRPO examples live in `unused/`.

Each script takes the base model as **either** a local checkpoint (`BASE_MODEL_PATH`) **or** a Hub
repo id (`BASE_MODEL_HF`), which it downloads once into `models/<name>` before training — a Hub id
must never reach verl directly, because ~15 Ray/vLLM workers would fetch it concurrently and one
transient failure leaves a vLLM server without a processor (CUDA device-side assert).

| Variable | Used by | Meaning |
|---|---|---|
| `DATASET_ROOT` | all (required) | Prepared verl dataset directory, e.g. `…/verl_datasets/qwen25vl/ds__AD5500_D110000_TL5500_all121000__resized-hw-512x512` ([data prep](https://github.com/YongchengYAO/MedVision#-training-rft)); a `shards/train_shard_*.parquet` layout is also accepted |
| `BASE_MODEL_PATH` | all | Local checkpoint directory (stage 1: your full-SFT checkpoint; stages 2–3: the previous stage's `global_step_N/actor/merged_hf_model`). Mutually exclusive with `BASE_MODEL_HF` |
| `BASE_MODEL_HF` | all | Hugging Face repo id, downloaded into `models/<name>` and trained from there. Sequential stages require one of the two variables; the multi-task recipes default to the paper's base `YongchengYAO/MedVision-V0-7B-dev0630` — the final full-SFT CoT checkpoint (private) — so set it to the public `YongchengYAO/MedVision-V0-7B` or your own SFT checkpoint if you cannot access it |
| `EXP_NAME` | all | Experiment name (checkpoints and logs under `checkpoints|log/$EXP_NAME`); defaults to the script's name, e.g. `rft-multitask`. Reuse a name to resume that run |
| `ENGINE` | all | Rollout engine (default `vllm`) |
| `ENV_NAME` | all | conda env name (default `verl`) |
| `DRY_RUN=1` | all | Print the actions (env setup, training command, merge) without running them |
| *trailing arguments* | all | Hydra overrides appended to the training command, e.g. `+data.curriculum.enable=False`, `+custom_reward_function.reward_kwargs.format_reward=binary`, `trainer.total_epochs=5` |

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
| `verl/trainer/main_ppo.py` | `create_rl_sampler`: temperature multitask sampler branch (`+data.temperature_sampler.*`) |
| `verl/trainer/ppo/ray_trainer.py` | Log extra/auxiliary reward components (NaN-aware); curriculum epoch-boundary hooks (pool advance, dataloader rebuild, `curriculum.json` save/resume, per-epoch pool snapshots) |
| `verl/trainer/ppo/metric_utils.py` | NaN-aware validation-metric aggregation (per-task error keys are NaN where not applicable) |
| `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | Fail loudly on a multimodal request when the HF processor failed to load (otherwise image-token dedup is silently skipped and rollout crashes with a CUDA device-side assert) |
| `verl/model_merger/base_model_merger.py` | Patch merged `config.json` to `dtype=bfloat16` |
| `verl/utils/fsdp_utils.py` | LoRA: `get_peft_model_state_dict(..., save_embedding_layers=False)` |
| `verl/protocol.py` | Make `ray` import optional (lightweight debug environments) |
| `verl/utils/dataset/rl_dataset.py` | Import/formatting only |
| `requirements.txt`, `setup.py` | Pin `transformers<5` (5.x writes nested Qwen2.5-VL configs that the pinned MedVision eval stack cannot read) |

---

## Quick start

```bash
# 1. Set up the conda env (installs the full pinned stack via uv, pip fallback; then verl).
#    No system CUDA/cuDNN or root needed. Optional: ENV_NAME=<name>, RUN_SYSTEM_FIXES=1 (GLIBC fix).
bash setup_conda_verl.sh

# 2. Prepare the verl-format MedVision dataset (see the MedVision repo for data prep):
#    https://github.com/YongchengYAO/MedVision#-training-rft

# 3a. Sequential RFT (MedVision-V0): run the three stages in order, each from the previous
#     stage's merged checkpoint. DATASET_ROOT plus BASE_MODEL_PATH (local checkpoint) or
#     BASE_MODEL_HF (Hub id, downloaded first) are required. DRY_RUN=1 previews the commands.
DATASET_ROOT=</path/to/verl_datasets/qwen25vl/ds__AD5500_D0_TL0_all5500__resized-hw-512x512> \
BASE_MODEL_PATH=</path/to/full-SFT-checkpoint> \
bash examples/grpo_trainer/train__rft-sequential__1-AD.sh   # then __2-TL.sh, then __3-detection.sh

# 3b. Multi-task RFT with temperature-scaled task mixing + curriculum (paper ablation).
#     Base model = a full-SFT CoT checkpoint, given as BASE_MODEL_PATH (local directory) or
#     BASE_MODEL_HF (Hub repo id). Set one explicitly.
DATASET_ROOT=</path/to/verl_datasets/qwen25vl/ds__AD5500_D110000_TL5500_all121000__resized-hw-512x512> \
BASE_MODEL_PATH=</path/to/full-SFT-checkpoint> \
bash examples/grpo_trainer/train__rft-multitask.sh 2>&1 | tee multitask.log

# ... or from a Hub model, e.g. the public MedVision-V0:
DATASET_ROOT=</path/to/verl_datasets/qwen25vl/ds__AD5500_D110000_TL5500_all121000__resized-hw-512x512> \
BASE_MODEL_HF=YongchengYAO/MedVision-V0-7B \
bash examples/grpo_trainer/train__rft-multitask.sh 2>&1 | tee multitask.log
# additive-reward ablation: same variables, train__rft-multitask__additive-reward.sh
```

Each script prints the dataset variant it expects (a comment near `dataset_root`), supports a
`shards/` → `train_verl.parquet` layout fallback, and merges the latest checkpoint to an HF model
at the end of the run.

---

## Tests

The MedVision additions are covered by CPU-only unit tests (no GPU; they load the pure-Python
modules by file when the full verl/torch stack is absent):

```bash
python -m pytest tests/utils/dataset/test_curriculum_on_cpu.py \
                 tests/utils/reward_score/test_medvision_ciou_on_cpu.py \
                 tests/utils/reward_score/test_medvision_reward_on_cpu.py
```

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

---

## Citation

```
@misc{yao2026medvisionbenchmarkingquantitativemedical,
      title={MedVision: Benchmarking Quantitative Medical Image Analysis},
      author={Yongcheng Yao and Yongshuo Zong and Raman Dutt and Yongxin Yang and Sotirios A Tsaftaris and Timothy Hospedales},
      year={2026},
      eprint={2511.18676},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.18676},
}
```
