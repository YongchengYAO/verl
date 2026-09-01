# Copyright 2025 Individual Contributor: Yongcheng Yao
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multi-task reward for MedVision RFT (GRPO): one configurable entry point, ``compute_score``.

    r = r_format + r_process * r_answer     composition="multiplicative" (default)
    r = r_format + r_process + r_answer     composition="additive"

Components, each in [0, 1]:

* ``r_format`` -- ``"soft"`` (default): 0.8 * reasoning-structure score + 0.2 * binary ``<answer>``
  format check; ``"binary"``: the ``<answer>`` check alone. Detection has no CoT steps and always
  uses the binary check.
* ``r_process`` -- mean over the CoT steps of ``rho(step error)``: worst-point localization error
  for landmark / endpoint steps, relative error for measurement steps (A/D and T/L only; detection
  has no process reward, so its score is ``r_format + r_answer``).
* ``r_answer`` -- ``rho(mean relative error)`` of the ``<answer>`` values (A/D, T/L) or
  ``rho((1 - CIoU) / 2)`` of the predicted box (detection). ``rho(e) = exp(-e)`` by default.

Selected from a recipe through verl's custom reward function config::

    custom_reward_function.path=.../medvision_general.py custom_reward_function.name=compute_score
    +custom_reward_function.reward_kwargs.format_reward=soft          # soft | binary
    +custom_reward_function.reward_kwargs.composition=multiplicative  # multiplicative | additive

Every call returns the same keys (``score``, the three components, and the raw errors
``answer_error`` / ``localization_error`` / ``measurement_error``, NaN where not applicable), as
required by the reward loop and consumed by the curriculum (``answer_error``).
"""

from verl.utils.reward_score.medvision_rewards import medvision_ad, medvision_tl
from verl.utils.reward_score.medvision_rewards.parsing import match_answer, parse_answer, parse_ground_truth
from verl.utils.reward_score.medvision_rewards.reward_fn import (
    cal_ciou_reward_error,
    cal_MRE_error,
    cal_reward_from_error_or_zero,
)

ABILITIES = ["medvision-tl", "medvision-angle", "medvision-distance", "medvision-detection"]
NUM_ANSWER_VALUES = {"medvision-tl": 2, "medvision-angle": 1, "medvision-distance": 1, "medvision-detection": 4}
# Tasks with CoT steps (process reward + soft format score) and the module implementing them.
PROCESS_MODULES = {"medvision-tl": medvision_tl, "medvision-angle": medvision_ad, "medvision-distance": medvision_ad}
FORMAT_REWARD_VARIANTS = ("soft", "binary")
COMPOSITIONS = ("multiplicative", "additive")
SOFT_FORMAT_ALPHA = 0.8  # weight of the reasoning-structure score in the soft format reward


def _ability(kwargs):
    ability = kwargs.get("ability")
    assert ability in ABILITIES, f"[Error] ability should be one of {ABILITIES}, but got {ability!r}."
    return ability


def cal_format_reward(solution, format_reward="soft", **kwargs):
    """
    Format reward in [0, 1].

    Args:
        solution: model response (text).
        format_reward: "soft" (0.8 * reasoning-structure score + 0.2 * binary answer check) or
            "binary" (answer check only). Detection always uses the binary check.
        kwargs: ``extra_info`` of the sample (``ability``, ``metric_type``, ...).
    """
    ability = _ability(kwargs)
    assert format_reward in FORMAT_REWARD_VARIANTS, (
        f"[Error] format_reward should be one of {FORMAT_REWARD_VARIANTS}, but got {format_reward!r}."
    )
    answer_ok = match_answer(solution, NUM_ANSWER_VALUES[ability])
    if format_reward == "binary" or ability not in PROCESS_MODULES:
        return answer_ok
    structure = PROCESS_MODULES[ability].match_reasoning(solution, **kwargs)
    return SOFT_FORMAT_ALPHA * structure + (1 - SOFT_FORMAT_ALPHA) * answer_ok


def cal_answer_reward_error(solution, ground_truth, reward_mapping_func="exp_decay", **kwargs):
    """
    Answer reward and raw answer error of the final ``<answer>``: mean relative error for A/D and
    T/L, CIoU overlap error ``(1 - CIoU) / 2`` for detection.

    Returns:
        (reward, error); ``(0.0, NaN)`` when the answer cannot be parsed.
    """
    ability = _ability(kwargs)
    num_values = NUM_ANSWER_VALUES[ability]
    gt = parse_ground_truth(ground_truth)
    assert len(gt) == num_values, (
        f"[Error] ground truth {ground_truth!r} has {len(gt)} values; {ability} expects {num_values}."
    )
    pred = parse_answer(solution, num_values)
    if pred is None:
        return 0.0, float("nan")
    if ability == "medvision-detection":
        return cal_ciou_reward_error(pred, gt, reward_mapping_func)
    error = cal_MRE_error(pred, gt)
    return cal_reward_from_error_or_zero(error, reward_mapping_func), error


def cal_process_reward_error(solution, ground_truth, reward_mapping_func="exp_decay", **kwargs):
    """(reward, localization_error, measurement_error) of the CoT steps (A/D and T/L only)."""
    ability = _ability(kwargs)
    assert ability in PROCESS_MODULES, f"[Error] {ability} has no process reward."
    return PROCESS_MODULES[ability].cal_process_reward_error(solution, ground_truth, reward_mapping_func, **kwargs)


def build_error_info(answer_error, localization_error=None, measurement_error=None):
    """
    Error-logging dict with a FIXED key set regardless of ability: the reward loop takes the key
    list from the first sample of a batch and indexes every sample with it
    (verl/experimental/reward_loop/reward_loop.py), so non-applicable values are NaN, never
    omitted. The trainer aggregates them with NaN-aware means.
    """
    nan = float("nan")
    info = {"answer_error": answer_error}
    if localization_error is not None or measurement_error is not None:
        info["localization_error"] = nan if localization_error is None else localization_error
        info["measurement_error"] = nan if measurement_error is None else measurement_error
    return info


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info,
    format_reward="soft",
    composition="multiplicative",
    reward_mapping_func="exp_decay",
):
    """
    Reward of one rollout (see the module docstring for the formula).

    Args:
        data_source: unused (verl custom-reward signature).
        solution_str: model response (text).
        ground_truth: ground-truth answer string.
        extra_info: per-sample metadata; must contain ``ability`` (injected by the reward manager),
            plus ``metric_type`` and the ground-truth landmarks for A/D and T/L.
        format_reward: "soft" (default) or "binary".
        composition: "multiplicative" (default) or "additive" combination of process and answer rewards.
        reward_mapping_func: error-to-reward map ("exp_decay" default, "scaled_sigmoid", "gaussian_proxy").

    Returns:
        dict with ``score`` (training reward), ``format_reward``, ``process_reward``,
        ``answer_reward``, ``answer_error``, ``localization_error``, ``measurement_error``.
    """
    assert extra_info is not None, (
        "[Error] extra_info cannot be None: the reward manager injects 'ability' into extra_info "
        "(see workers/reward_manager/naive.py and experimental/reward_loop/reward_manager/naive.py)."
    )
    assert composition in COMPOSITIONS, f"[Error] composition should be one of {COMPOSITIONS}, but got {composition!r}."
    ability = _ability(extra_info)

    f = cal_format_reward(solution_str, format_reward=format_reward, **extra_info)
    a, answer_error = cal_answer_reward_error(solution_str, ground_truth, reward_mapping_func, **extra_info)

    if ability in PROCESS_MODULES:
        p, localization_error, measurement_error = cal_process_reward_error(
            solution_str, ground_truth, reward_mapping_func, **extra_info
        )
        task_term = p * a if composition == "multiplicative" else p + a
    else:
        p, localization_error, measurement_error = 0.0, float("nan"), float("nan")
        task_term = a

    return {
        "score": f + task_term,
        "format_reward": f,
        "process_reward": p,
        "answer_reward": a,
        **build_error_info(answer_error, localization_error, measurement_error),
    }
