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

import re

from verl.utils.reward_score.medvision_rewards.medvision_ad import (
    cal_process_reward_error_v2 as cal_ad_process_reward_error_v2,
)
from verl.utils.reward_score.medvision_rewards.medvision_ad import (
    cal_process_reward_error_v3 as cal_ad_process_reward_error_v3,
)
from verl.utils.reward_score.medvision_rewards.medvision_tl import (
    cal_process_reward_error_v2 as cal_tl_process_reward_error_v2,
)
from verl.utils.reward_score.medvision_rewards.medvision_tl import (
    cal_process_reward_error_v3 as cal_tl_process_reward_error_v3,
)
from verl.utils.reward_score.medvision_rewards.reward_fn import (
    cal_ciou_reward_error,
    cal_MRE_error,
    cal_reward_from_error_or_zero,
    extract_last_k_nums,
)

# All supported abilities; all except detection have process (CoT step) rewards
ABILITIES = ["medvision-tl", "medvision-angle", "medvision-distance", "medvision-detection"]


def build_error_info(answer_error, localization_error=None, measurement_error=None):
    """
    Builds the error-logging dict with a FIXED key set regardless of ability.

    A fixed key set is required because the reward loop takes the key list from the
    first sample of a batch and indexes every sample with it
    (see verl/experimental/reward_loop/reward_loop.py).

    Keys:
      - answer_error: MRE of the final answer, shared across all tasks
      - if process errors are given: localization_error / measurement_error,
        shared across AD and TL (normalized L2 and MRE definitions are consistent
        across these tasks); NaN for detection, whose localization is already
        captured by its answer error

    NaN values are excluded by the np.nanmean aggregation during multi-task training.

    Args:
        answer_error: MRE of the final answer (NaN if unparseable).
        localization_error: Localization error of the CoT steps (None to omit process keys).
        measurement_error: Measurement error of the CoT steps (None to omit process keys).

    Returns:
        A dict of error metrics.
    """
    nan = float("nan")
    info = {"answer_error": answer_error}

    if localization_error is not None or measurement_error is not None:
        info["localization_error"] = nan if localization_error is None else localization_error
        info["measurement_error"] = nan if measurement_error is None else measurement_error

    return info


# Tag helpers: strict tags (no spaces inside <>), flexible spaces between structures
def tag(name):
    return rf"<{name}>"


def end(name):
    return rf"</{name}>"


def get_answer_pattern_k_values(num_target_values):
    """
    Returns a regex pattern for answer with num_target_values values.
    """
    # Patterns for answer extraction and format checking
    PATTERN_NON_NEG_REAL = r"\d+(?:\.\d+)?"
    PATTERN_NON_NEG_REAL_GROUP = rf"({PATTERN_NON_NEG_REAL})"

    values = ",".join([f"\s*{PATTERN_NON_NEG_REAL}\s*" for _ in range(num_target_values)])
    pattern = rf"\s*{tag('answer')}\s*{values}\s*{end('answer')}\s*"

    values_group = ",".join([f"\s*{PATTERN_NON_NEG_REAL_GROUP}\s*" for _ in range(num_target_values)])
    pattern_group = rf"\s*{tag('answer')}\s*{values_group}\s*{end('answer')}\s*"
    return pattern, pattern_group


def match_answer(content, num_target_values):
    """
    Return 1 if <answer> block matches required format (case-sensitive, flexible spacing), else 0.

    Examples of valid formats (case-sensitive) for tasks with 2 target values:
        "<answer> (3, 5) </answer>"
        "<answer>(10,20)</answer>"
        "<answer> ( 0 , 0.5 ) </answer>"

    Args:
        content: The content string to evaluate.

    Returns:
        1 if the format matches, 0 otherwise.
    """
    # Remove leading/trailing spaces and brackets from content
    content_clean = content.strip()
    content_clean = re.sub(r"^[\[\{\(\s]+", "", content_clean)
    content_clean = re.sub(r"[\]\}\)\s]+$", "", content_clean)
    pattern = get_answer_pattern_k_values(num_target_values)[0]
    return 1 if re.search(pattern, content_clean, re.VERBOSE) else 0


def _to_float(*gs):
    return tuple(float(x) for x in gs)


def cal_format_reward(solution, **kwargs):
    """
    Reward function that checks if the model response has a specific format.

    Args:
        solution: model responses (text)

    Returns:
        a scalar reward
    """
    # Validate ability and determine number of target values based on ability
    ability = kwargs.get("ability")
    assert ability in ABILITIES, f"[Error] ability should be one of {ABILITIES}, but got {ability}."
    if ability in ["medvision-tl"]:
        num_target_values = 2
    elif ability in ["medvision-angle", "medvision-distance"]:
        num_target_values = 1
    elif ability in ["medvision-detection"]:
        num_target_values = 4

    # Check format using regex pattern matching
    answer_format_reward = match_answer(solution, num_target_values)

    return answer_format_reward


def cal_answer_reward_error(solution, ground_truth, reward_mapping_func="exp_decay", **kwargs):
    """
    Calculates the answer reward and raw error from the extracted final answer of the
    model response (solution). The error metric depends on the task: A/D and T/L use MRE
    (Mean Relative Error) over the scalar/axis-length answer, while detection uses the
    CIoU overlap error (1 - CIoU) / 2 over the 4 box coordinates.

    Args:
        solution: model responses (text)
        ground_truth: ground truth strings
        reward_mapping_func: Reward mapping function name.

    Returns:
        A tuple (reward, answer_error): for A/D and T/L answer_error is the MRE; for
        detection it is the CIoU overlap error (NaN if the answer is unparseable).
    """
    # Validate ability and determine number of target values based on ability
    ability = kwargs.get("ability")
    assert ability in ABILITIES, f"[Error] ability should be one of {ABILITIES}, but got {ability}."
    if ability in ["medvision-tl"]:
        num_target_values = 2
    elif ability in ["medvision-angle", "medvision-distance"]:
        num_target_values = 1
    elif ability in ["medvision-detection"]:
        num_target_values = 4
    pattern_group = get_answer_pattern_k_values(num_target_values)[1]

    # Extract ground truth coordinates
    gt_string = ground_truth.strip()
    gt_parts = [
        part.strip()
        for part in gt_string.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(",")
    ]
    gt_float = [float(part) for part in gt_parts if part]
    num_gt = len(gt_float)

    # Sanity check to ensure the number of extracted ground truth values matches expected number based on ability
    assert num_gt == num_target_values, (
        f"[Error] The number of target values extracted from ground truth ({num_gt}) does not match "
        f"expected number based on ability ({num_target_values}). Please check the format of the "
        "ground truth and ensure it contains the correct number of values."
    )

    # Extract predicted values
    try:
        # Directly extract specified number of values using GROUP pattern
        pattern_answer = re.compile(pattern_group, re.DOTALL)
        ma = pattern_answer.search(solution)
        pred_float = list(_to_float(*ma.groups()))
    except Exception:
        try:
            # Extract content within <answer>...</answer> and parse last num_gt numbers
            pattern_answer = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
            ma = pattern_answer.search(solution)
            answer_content = ma.group(1).strip()
            # Parse prediction string to extract num_gt numbers
            pred_string_parsed = extract_last_k_nums(
                answer_content, num_gt
            )  # returns a string with num_gt numbers seperated by comma or empty string
            pred_parts = [
                part.strip()
                for part in pred_string_parsed.replace("(", "")
                .replace(")", "")
                .replace("[", "")
                .replace("]", "")
                .split(",")
            ]
            pred_float = [float(part) for part in pred_parts if part]
            if len(pred_float) != num_gt:
                return 0.0, float("nan")
        except Exception:
            return 0.0, float("nan")

    # Compute the answer reward/error. Detection scores the 4 box coordinates with CIoU
    # (overlap-aware and position/scale-fair, unlike coordinate MRE which is biased toward
    # the origin); the returned answer_error is the overlap error (1 - CIoU) / 2. A/D and
    # T/L keep MRE over their scalar/axis-length answers.
    if ability == "medvision-detection":
        return cal_ciou_reward_error(pred_float, gt_float, reward_mapping_func)

    error = cal_MRE_error(pred_float, gt_float)
    reward = cal_reward_from_error_or_zero(error, reward_mapping_func)

    return reward, error


def cal_answer_reward(solution, ground_truth, reward_mapping_func="exp_decay", **kwargs):
    """
    Calculates the answer reward (see cal_answer_reward_error).

    Returns:
        a scalar reward
    """
    return cal_answer_reward_error(solution, ground_truth, reward_mapping_func, **kwargs)[0]


def compute_score_exp_decay(
    data_source,
    solution_str,
    ground_truth,
    extra_info,
):
    """
    Computes the scores for the given solution against the ground truth.

    Args:
        data_source: The source of the data.
        solution_str: The solution (completions).
        ground_truth: The ground truth.
        extra_info: Extra information for reward calculation.

    Returns:
        A dictionary containing the calculated rewards.
    """
    assert extra_info is not None, (
        "[Error] extra_info cannot be None since we have injected the filed 'ability' into "
        "extra_info in workers/reward_manager/naive.py. Please check the code there for details."
    )

    format_reward = cal_format_reward(solution_str, **extra_info)
    answer_reward, answer_error = cal_answer_reward_error(
        solution_str, ground_truth, reward_mapping_func="exp_decay", **extra_info
    )
    reward = format_reward + answer_reward

    return {
        "score": reward,
        "format_reward": format_reward,
        "answer_reward": answer_reward,
        **build_error_info(answer_error),
    }


def compute_score_scaled_sigmoid(
    data_source,
    solution_str,
    ground_truth,
    extra_info,
):
    """
    Computes the scores for the given solution against the ground truth.

    Args:
        data_source: The source of the data.
        solution_str: The solution (completions).
        ground_truth: The ground truth.
        extra_info: Extra information for reward calculation.

    Returns:
        A dictionary containing the calculated rewards.
    """
    assert extra_info is not None, (
        "[Error] extra_info cannot be None since we have injected the filed 'ability' into "
        "extra_info in workers/reward_manager/naive.py. Please check the code there for details."
    )

    format_reward = cal_format_reward(solution_str, **extra_info)
    answer_reward, answer_error = cal_answer_reward_error(
        solution_str, ground_truth, reward_mapping_func="scaled_sigmoid", **extra_info
    )
    reward = format_reward + answer_reward

    return {
        "score": reward,
        "format_reward": format_reward,
        "answer_reward": answer_reward,
        **build_error_info(answer_error),
    }


def compute_score_gaussian_proxy(
    data_source,
    solution_str,
    ground_truth,
    extra_info,
):
    """
    Computes the scores for the given solution against the ground truth.

    Args:
        data_source: The source of the data.
        solution_str: The solution (completions).
        ground_truth: The ground truth.
        extra_info: Extra information for reward calculation.

    Returns:
        A dictionary containing the calculated rewards.
    """
    assert extra_info is not None, (
        "[Error] extra_info cannot be None since we have injected the filed 'ability' into "
        "extra_info in workers/reward_manager/naive.py. Please check the code there for details."
    )

    format_reward = cal_format_reward(solution_str, **extra_info)
    answer_reward, answer_error = cal_answer_reward_error(
        solution_str, ground_truth, reward_mapping_func="gaussian_proxy", **extra_info
    )
    reward = format_reward + answer_reward

    return {
        "score": reward,
        "format_reward": format_reward,
        "answer_reward": answer_reward,
        **build_error_info(answer_error),
    }


def compute_score_exp_decay_PRxAnswer_v2(
    data_source,
    solution_str,
    ground_truth,
    extra_info,
):
    """
    Multi-task PRxAnswer reward using mean normalized L2 process reward for AD and TL.

    Routes process reward based on ability:
      - medvision-tl:              format + TL_process_v2 * answer
      - medvision-angle/distance:  format + AD_process_v2 * answer
      - medvision-detection:       format + answer  (no process reward)

    Args:
        data_source: The source of the data.
        solution_str: The solution (completions).
        ground_truth: The ground truth.
        extra_info: Extra information including the 'ability' field.

    Returns:
        A dictionary containing the calculated rewards.
    """
    assert extra_info is not None, (
        "[Error] extra_info cannot be None since we have injected the field 'ability' into "
        "extra_info in workers/reward_manager/naive.py. Please check the code there for details."
    )

    ability = extra_info.get("ability")
    format_reward = cal_format_reward(solution_str, **extra_info)
    answer_reward, answer_error = cal_answer_reward_error(
        solution_str, ground_truth, reward_mapping_func="exp_decay", **extra_info
    )

    if ability == "medvision-tl":
        process_reward, localization_error, measurement_error = cal_tl_process_reward_error_v2(
            solution_str, ground_truth, reward_mapping_func="exp_decay", **extra_info
        )
        score = format_reward + process_reward * answer_reward
    elif ability in ["medvision-angle", "medvision-distance"]:
        process_reward, localization_error, measurement_error = cal_ad_process_reward_error_v2(
            solution_str, ground_truth, reward_mapping_func="exp_decay", **extra_info
        )
        score = format_reward + process_reward * answer_reward
    else:
        # medvision-detection: no process reward
        process_reward = 0.0
        localization_error = measurement_error = float("nan")
        score = format_reward + answer_reward

    return {
        "score": score,
        "format_reward": format_reward,
        "process_reward": process_reward,
        "answer_reward": answer_reward,
        **build_error_info(answer_error, localization_error, measurement_error),
    }


def compute_score_exp_decay_PRxAnswer_v3(
    data_source,
    solution_str,
    ground_truth,
    extra_info,
):
    """
    Multi-task PRxAnswer reward using max normalized L2 process reward for AD and TL.

    Routes process reward based on ability:
      - medvision-tl:              format + TL_process_v3 * answer
      - medvision-angle/distance:  format + AD_process_v3 * answer
      - medvision-detection:       format + answer  (no process reward)

    Args:
        data_source: The source of the data.
        solution_str: The solution (completions).
        ground_truth: The ground truth.
        extra_info: Extra information including the 'ability' field.

    Returns:
        A dictionary containing the calculated rewards.
    """
    assert extra_info is not None, (
        "[Error] extra_info cannot be None since we have injected the field 'ability' into "
        "extra_info in workers/reward_manager/naive.py. Please check the code there for details."
    )

    ability = extra_info.get("ability")
    format_reward = cal_format_reward(solution_str, **extra_info)
    answer_reward, answer_error = cal_answer_reward_error(
        solution_str, ground_truth, reward_mapping_func="exp_decay", **extra_info
    )

    if ability == "medvision-tl":
        process_reward, localization_error, measurement_error = cal_tl_process_reward_error_v3(
            solution_str, ground_truth, reward_mapping_func="exp_decay", **extra_info
        )
        score = format_reward + process_reward * answer_reward
    elif ability in ["medvision-angle", "medvision-distance"]:
        process_reward, localization_error, measurement_error = cal_ad_process_reward_error_v3(
            solution_str, ground_truth, reward_mapping_func="exp_decay", **extra_info
        )
        score = format_reward + process_reward * answer_reward
    else:
        # medvision-detection: no process reward
        process_reward = 0.0
        localization_error = measurement_error = float("nan")
        score = format_reward + answer_reward

    return {
        "score": score,
        "format_reward": format_reward,
        "process_reward": process_reward,
        "answer_reward": answer_reward,
        **build_error_info(answer_error, localization_error, measurement_error),
    }
