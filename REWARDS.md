# MedVision RFT rewards

The reward used for GRPO fine-tuning on the three MedVision tasks (A/D measurement, T/L size,
detection): one configurable entry point, its options, and how a recipe selects them.

## 1. The reward

```
r = r_format + r_process · r_answer     composition = multiplicative   (default)
r = r_format + r_process + r_answer     composition = additive
```

Every component lies in [0, 1]; the two accuracy components map an error `e` to a reward with
`ρ(e) = exp(−e)` (an exact prediction earns 1, an unparseable value 0). Predicted coordinates are
relative image positions in [0, 1] and measurement errors are relative errors, so all errors are
dimensionless and comparable across tasks.

| Component | A/D and T/L | Detection |
|---|---|---|
| `r_format` | **soft** (default): `0.8 · reasoning-structure score + 0.2 · binary answer check`; **binary**: the answer check alone (§3) | binary answer check (no CoT steps) |
| `r_process` | mean over the CoT steps of `ρ(step error)`: worst-point displacement / √2 for landmark / endpoint steps, relative error for measurement steps (§4) | — (the answer is a single localization step); `r = r_format + r_answer` |
| `r_answer` | `ρ(mean relative error)` of the `<answer>` values | `ρ((1 − CIoU) / 2)` of the predicted box (§5) |

The defaults (soft format, multiplicative) give `r = r_format + r_process · r_answer` for A/D and
T/L and `r = r_format + r_answer` for detection — the reward of the paper's Algorithm 2.

## 2. Selecting and configuring the reward (CLI)

The entry point is `compute_score` in `verl/utils/reward_score/medvision_rewards/medvision_general.py`;
verl loads it from the file and forwards `reward_kwargs` to it:

```bash
custom_reward_function.path=$verl_dir/verl/utils/reward_score/medvision_rewards/medvision_general.py \
custom_reward_function.name=compute_score \
+custom_reward_function.reward_kwargs.format_reward=soft \
+custom_reward_function.reward_kwargs.composition=multiplicative \
+reward_model.use_reward_loop=True     # async reward loop: the only path that propagates the per-sample
                                       # answer_error the curriculum needs (set by the multi-task recipes)
```

The recipes expose the first two options as `reward_format` / `reward_composition` shell variables.

| `reward_kwargs` option | Values | Default | Effect |
|---|---|---|---|
| `format_reward` | `soft`, `binary` | `soft` | format-reward variant (§3) |
| `composition` | `multiplicative`, `additive` | `multiplicative` | how process and answer rewards combine |
| `reward_mapping_func` | `exp_decay`, `scaled_sigmoid`, `gaussian_proxy` | `exp_decay` | `ρ` (§6) |

Named settings expressed through these options:

| Setting | Options |
|---|---|
| paper default (MedVision-V0 stages, multi-task RFT) | defaults |
| additive-reward ablation | `composition=additive` |
| binary format check only | `format_reward=binary` |

Dispatch inside `compute_score` uses fields the reward manager injects into `extra_info`:
`ability` (`medvision-tl`, `medvision-angle`, `medvision-distance`, `medvision-detection`),
`metric_type` (`angle` / `distance`), and the ground-truth landmarks written by the MedVision
parquet builders (`landmark_P{1..4}_wh` for T/L; `landmark_{1,2}_wh` for distance;
`line_{1,2}_point_{1,2}_wh` for angle). Every call returns the same keys — `score`,
`format_reward`, `process_reward`, `answer_reward`, `answer_error`, `localization_error`,
`measurement_error` (NaN where not applicable) — as the reward loop requires.

## 3. Format reward

**Binary answer check** (`parsing.match_answer`): 1 if the `<answer>` block holds exactly the
task's number of non-negative decimals separated by commas (2 for T/L, 1 for A/D, 4 for detection),
optionally wrapped in `( )` or `[ ]` — so both the SFT answer template `<answer> (a, b) </answer>`
and the benchmark instruction's `a, b` pass; anything else (units, text, wrong count) scores 0.

**Soft reasoning-structure score** (`parsing.soft_format_score`, task step layout from §4): partial
credit for the `<think>` block (required — 0 without it), each step's `<step-k-reasoning>` tag, each
step's well-formed `<step-k-answer>` tag, the numbers found for each step (proportional to the
expected count: 2 per coordinate, 1 per scalar), and the step order (full credit when every step
is present in order, half when all but one are), normalized to [0, 1]. The soft format reward is
`0.8 · structure + 0.2 · answer check`.

## 4. Process reward

The process reward supervises the `<step-k-answer>` blocks of the CoT (`process_reward.py`):
each step's value is parsed, compared with its ground truth, mapped through ρ (unparseable step →
0), and the process reward is the **mean over the steps**.

| Task (module) | Steps (`<step-k-answer>`) |
|---|---|
| T/L (`medvision_tl`) | 1 major-axis endpoints `(x1, y1), (x2, y2)`; 2 minor-axis endpoints; 3 major length; 4 minor length |
| A/D distance (`medvision_ad`, `metric_type=distance`) | 1 landmark 1 `(x, y)`; 2 landmark 2; 3 distance |
| A/D angle (`metric_type=angle`) | 1 line-1 endpoints; 2 line-2 endpoints; 3 angle |
| Detection | — |

- Localization steps: Euclidean displacement of the **worst-localized point** from its ground truth
  in relative coordinates, divided by √2 (the unit-square diagonal) so the error lies in [0, 1];
  two-endpoint steps take the better of the two endpoint orderings.
- Measurement steps: relative error.
- `localization_error` / `measurement_error` in the output are the mean raw errors over the
  parsed localization / measurement steps (NaN if none parsed).

## 5. Answer reward

| Task | Parsed from `<answer>` | Error |
|---|---|---|
| A/D | 1 number | relative error `\|ŷ − y\| / y` |
| T/L | 2 numbers (major, minor) | mean relative error over the two values |
| Detection | 4 numbers `x1, y1, x2, y2` (lower-left, upper-right; relative) | overlap error `(1 − CIoU) / 2`, `CIoU = IoU − d²/c² − α·v` (`reward_fn.cal_ciou`) — a graded signal even for non-overlapping boxes |

`reward = ρ(error)`; an unparseable answer gives `reward = 0`, `error = NaN`. Value parsing is
tolerant: the strict pattern first, then the last N numbers inside `<answer> … </answer>`.

## 6. Error-to-reward maps (`reward_fn.py`)

| `reward_mapping_func` | Formula | Parameters |
|---|---|---|
| `exp_decay` | `exp(−k·e)` | k = 1 (the paper's ρ) |
| `scaled_sigmoid` | `sigmoid(−k·(e − ½))` | k = 10 |
| `gaussian_proxy` | `exp(−e² / (2·var))` | var = 0.5 |

## Code layout

| File | Contents |
|---|---|
| `medvision_general.py` | `compute_score` (entry point: format variant, composition), `cal_format_reward`, `cal_answer_reward_error`, `cal_process_reward_error` (dispatch by `ability`), `build_error_info` |
| `medvision_tl.py`, `medvision_ad.py` | per-task CoT step layout and ground-truth mapping, `match_reasoning` (soft score), `cal_process_reward_error` |
| `process_reward.py` | generic worst-point process reward over a list of step targets |
| `parsing.py` | tags and number / coordinate patterns, `<answer>` parsing and check, `<step-k-answer>` parsing, soft-structure rubric |
| `reward_fn.py` | error metrics (relative error, worst-point normalized-L2, CIoU) and the ρ maps |
| `registry.py` | plots the ρ maps |

Tests: `tests/utils/reward_score/test_medvision_reward_on_cpu.py`, `test_medvision_ciou_on_cpu.py`
(CPU only; load the modules by file when the full verl stack is absent).
