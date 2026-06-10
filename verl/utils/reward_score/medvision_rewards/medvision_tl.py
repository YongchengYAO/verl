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

import numpy as np

from verl.utils.reward_score.medvision_rewards.reward_fn import (
    cal_MAE_error,
    cal_MRE_error,
    cal_norm_L2_error,
    cal_norm_L2_max_error,
    cal_reward_from_error_or_zero,
    extract_last_k_nums,
    safe_nanmean,
)


# Tag helpers: strict tags (no spaces inside <>), flexible spaces between structures
def tag(name):
    return rf"<{name}>"


def end(name):
    return rf"</{name}>"


# Step reasoning patterns
def step_reasoning(k):
    return rf"{tag(f'step-{k}-reasoning')}.*?{end(f'step-{k}-reasoning')}"


# =========================================================
# Define patterns for MedVision-TL (tumor/lesion size measurement) task
# =========================================================
# Paterns for answer extraction and format checking
PATTERN_NON_NEG_REAL = r"\d+(?:\.\d+)?"
PATTERN_NON_NEG_REAL_GROUP = rf"({PATTERN_NON_NEG_REAL})"
PATTERN_RELATIVE_POSITION = r"(?:0(?:\.\d+)?|1(?:\.0+)?)"  # [0, 1] inclusive
PATTERN_RELATIVE_POSITION_GROUP = rf"({PATTERN_RELATIVE_POSITION})"
PATTERN_COORD = rf"\(\s*{PATTERN_RELATIVE_POSITION}\s*,\s*{PATTERN_RELATIVE_POSITION}\s*\)"
PATTERN_COORD_GROUP = rf"\(\s*{PATTERN_RELATIVE_POSITION_GROUP}\s*,\s*{PATTERN_RELATIVE_POSITION_GROUP}\s*\)"
PATTERN_ANSWER_2_VALUES = (
    rf"\s*{tag('answer')}\s*\(\s*{PATTERN_NON_NEG_REAL}\s*,\s*{PATTERN_NON_NEG_REAL}\s*\)\s*{end('answer')}\s*"
)
PATTERN_ANSWER_2_VALUES_GROUP = rf"\s*{tag('answer')}\s*\(\s*{PATTERN_NON_NEG_REAL_GROUP}\s*,\s*{PATTERN_NON_NEG_REAL_GROUP}\s*\)\s*{end('answer')}\s*"  # noqa: E501

# Reasoning patterns
PATTERNS_STEP_REASONING = {}
for k in range(1, 5):
    PATTERNS_STEP_REASONING[k] = step_reasoning(k)

# Step answer patterns
# Steps 1 & 2: Expect two coordinates (x1, y1), (x2, y2)
# Steps 3 & 4: Expect a single scalar value
PATTERNS_STEP_ANSWER = {
    1: rf".*?{tag('step-1-answer')}.*?{PATTERN_COORD}\s*,\s*{PATTERN_COORD}.*?{end('step-1-answer')}.*?",
    2: rf".*?{tag('step-2-answer')}.*?{PATTERN_COORD}\s*,\s*{PATTERN_COORD}.*?{end('step-2-answer')}.*?",
    3: rf".*?{tag('step-3-answer')}.*?{PATTERN_NON_NEG_REAL}.*?{end('step-3-answer')}.*?",
    4: rf".*?{tag('step-4-answer')}.*?{PATTERN_NON_NEG_REAL}.*?{end('step-4-answer')}.*?",
}
PATTERNS_STEP_ANSWER_GROUP = {
    1: rf".*?{tag('step-1-answer')}.*?{PATTERN_COORD_GROUP}\s*,\s*{PATTERN_COORD_GROUP}.*?{end('step-1-answer')}.*?",
    2: rf".*?{tag('step-2-answer')}.*?{PATTERN_COORD_GROUP}\s*,\s*{PATTERN_COORD_GROUP}.*?{end('step-2-answer')}.*?",
    3: rf".*?{tag('step-3-answer')}.*?{PATTERN_NON_NEG_REAL_GROUP}.*?{end('step-3-answer')}.*?",
    4: rf".*?{tag('step-4-answer')}.*?{PATTERN_NON_NEG_REAL_GROUP}.*?{end('step-4-answer')}.*?",
}
# =========================================================


def match_reasoning(content):
    """
    Soft (non-binary) reward in [0, 1] for how well `content` matches the required <think> format.

    The scoring criteria are:
    1. Presence of <think>...</think> block.
    2. Presence of step-by-step reasoning and answer blocks (steps 1 to 4).
    3. Correct formatting of answers within steps (coordinates for steps 1-2, scalars for steps 3-4).
    4. Logical ordering of steps (Step 1 -> Step 2 -> Step 3 -> Step 4).

    Args:
        content: The content string to evaluate.

    Returns:
        A score between 0.0 and 1.0 indicating the quality of the reasoning format.
    """

    # re.DOTALL to allow multiline matches; do not use IGNORECASE so tags are case-sensitive
    flags = re.DOTALL

    # --- Scoring Weights (Total = 17) ---
    W_THINK = 1
    W_ORDER = 2
    W_STEP = {
        1: {"reason": 1, "answer_tag": 1, "format": 2},  # 2 coords
        2: {"reason": 1, "answer_tag": 1, "format": 2},  # 2 coords
        3: {"reason": 1, "answer_tag": 1, "format": 1},  # 1 float
        4: {"reason": 1, "answer_tag": 1, "format": 1},  # 1 float
    }
    max_score = W_THINK + W_ORDER + sum(d["reason"] + d["answer_tag"] + d["format"] for d in W_STEP.values())
    score = 0.0

    # --- Evaluation ---

    # 1) <think> block presence and extraction
    think_match = re.search(rf"{tag('think')}(.*?){end('think')}", content, flags)
    if not think_match:
        return 0.0
    score += W_THINK
    body = think_match.group(1)

    # 2) Step-by-step scoring
    step_spans = {}
    for k in (1, 2, 3, 4):
        # A. Reasoning block
        reasoning_pattern = PATTERNS_STEP_REASONING[k]
        reasoning_match = re.search(reasoning_pattern, body, flags)
        if reasoning_match:
            score += W_STEP[k]["reason"]
            step_spans[f"r{k}"] = reasoning_match.span()

        # B. Answer block (tag presence)
        answer_pattern = PATTERNS_STEP_ANSWER[k]
        answer_match = re.search(answer_pattern, body, flags)
        if answer_match:
            score += W_STEP[k]["answer_tag"]
            step_spans[f"a{k}"] = answer_match.span()

        # C. Format partial credit
        # If answer tag is missing but partial structure exists near reasoning block
        region = None
        if answer_match:
            region = answer_match.group(0)
        elif reasoning_match:
            # local window: from end of reasoning to some chars ahead
            start = reasoning_match.end()
            window_size = 2000 if k in (1, 2) else 1000
            region = body[start : start + window_size]

        if region:
            if k in (1, 2):
                # Step 1/2: count coords found
                found_coords = re.findall(PATTERN_NON_NEG_REAL, region, flags)
                # up to 2 coords expected
                score += min(len(found_coords), 2) * (W_STEP[k]["format"] / 2.0)
            elif k in (3, 4):
                # Step 3/4: numeric float presence
                if re.search(PATTERN_NON_NEG_REAL, region, flags):
                    score += W_STEP[k]["format"]

    # 3) Order check: step1 → step2 → step3 → step4
    # We determine the position of a step by the earliest occurrence of its reasoning or answer block.
    def get_step_positions(keys_list):
        positions = []
        for key_group in keys_list:
            step_indices = []
            for key in key_group:
                if key in step_spans:
                    step_indices.append(step_spans[key][0])
            positions.append(min(step_indices) if step_indices else None)
        return positions

    # Check order: s1 <= s2 <= s3 <= s4
    s1, s2, s3, s4 = get_step_positions([("r1", "a1"), ("r2", "a2"), ("r3", "a3"), ("r4", "a4")])

    if all(p is not None for p in (s1, s2, s3, s4)) and s1 <= s2 <= s3 <= s4:
        score += W_ORDER
    elif sum(p is not None for p in (s1, s2, s3, s4)) >= 3:
        # Partial order credit if most steps are present and sorted
        seq = [p for p in (s1, s2, s3, s4) if p is not None]
        if seq == sorted(seq):
            score += W_ORDER * 0.5

    return max(0.0, min(1.0, score / max_score))


def match_answer(content):
    """
    Return 1 if <answer> block matches required format (case-sensitive, flexible spacing), else 0.

    Examples of valid formats (case-sensitive):
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
    return 1 if re.search(PATTERN_ANSWER_2_VALUES, content_clean, re.VERBOSE) else 0


def _to_float(*gs):
    return tuple(float(x) for x in gs)


def cal_process_reward_error(solution, ground_truth, reward_mapping_func="exp_decay", **kwargs):
    """
    Calculates the process reward together with the raw process errors.

    The process reward evaluates the step-by-step reasoning and intermediate answers in the model's response (solution).
    The reward for an unparseable step is 0; its error is NaN (excluded from the error means).

    Args:
        solution: model responses (text)
        ground_truth: ground truth string.

    Returns:
        A tuple (reward, localization_error, measurement_error) where
        localization_error is the mean MAE over the landmark steps (1-2) and
        measurement_error is the mean MRE over the length steps (3-4).
    """

    # Step patterns: reasoning + answer
    patterns_step = {}
    for k in range(1, 5):
        # NOTE: Use GROUP patterns to extract numeric values
        patterns_step[k] = rf"{PATTERNS_STEP_REASONING[k]}\s*{PATTERNS_STEP_ANSWER_GROUP[k]}"
    pattern_step1 = re.compile(patterns_step[1], re.DOTALL)
    pattern_step2 = re.compile(patterns_step[2], re.DOTALL)
    pattern_step3 = re.compile(patterns_step[3], re.DOTALL)
    pattern_step4 = re.compile(patterns_step[4], re.DOTALL)

    # NOTE:
    # ------
    # The landmark coordinates are in (w, h) format, not the conventional (h, w) format for image space indexing.
    # Such conversion is achieved in the dataset building recipe from MedVision (https://github.com/YongchengYAO/MedVision)
    # ------
    # P1 and P2 are the major axis landmarks, P3 and P4 are the minor axis landmarks
    gt_p1_wh = np.array(kwargs.get("landmark_P1_wh"))
    gt_p2_wh = np.array(kwargs.get("landmark_P2_wh"))
    gt_p3_wh = np.array(kwargs.get("landmark_P3_wh"))
    gt_p4_wh = np.array(kwargs.get("landmark_P4_wh"))

    # Extract ground truth coordinates
    gt_string = ground_truth.strip()
    gt_parts = [
        part.strip()
        for part in gt_string.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(",")
    ]
    gt_float = [float(part) for part in gt_parts if part]

    pred_major_wh = None
    pred_minor_wh = None
    err_s1 = err_s2 = err_s3 = err_s4 = float("nan")

    try:
        # NOTE: MAE reward should be used for normalized coordinates (step 1 and step 2)
        # NOTE: MRE reward should be used in length estimation (step 3 and step 4)

        # --- parse step 1: Major Axis Endpoints
        m1 = pattern_step1.search(solution)
        if m1:
            x1_major, y1_major, x2_major, y2_major = _to_float(m1.group(1), m1.group(2), m1.group(3), m1.group(4))
            pred_major_wh = [
                x1_major,
                y1_major,
                x2_major,
                y2_major,
            ]
            # Calculate MAE for both orderings of points (P1, P2) vs (P2, P1)
            err_s1 = min(
                cal_MAE_error(
                    pred_major_wh,
                    [gt_p1_wh[0], gt_p1_wh[1], gt_p2_wh[0], gt_p2_wh[1]],
                ),
                cal_MAE_error(
                    pred_major_wh,
                    [gt_p2_wh[0], gt_p2_wh[1], gt_p1_wh[0], gt_p1_wh[1]],
                ),
            )
        reward_s1 = cal_reward_from_error_or_zero(err_s1, reward_mapping_func)

        # --- parse step 2: Minor Axis Endpoints
        m2 = pattern_step2.search(solution)
        if m2:
            x1_minor, y1_minor, x2_minor, y2_minor = _to_float(m2.group(1), m2.group(2), m2.group(3), m2.group(4))
            pred_minor_wh = [
                x1_minor,
                y1_minor,
                x2_minor,
                y2_minor,
            ]
            # Calculate MAE for both orderings of points (P3, P4) vs (P4, P3)
            err_s2 = min(
                cal_MAE_error(
                    pred_minor_wh,
                    [gt_p3_wh[0], gt_p3_wh[1], gt_p4_wh[0], gt_p4_wh[1]],
                ),
                cal_MAE_error(
                    pred_minor_wh,
                    [gt_p4_wh[0], gt_p4_wh[1], gt_p3_wh[0], gt_p3_wh[1]],
                ),
            )
        reward_s2 = cal_reward_from_error_or_zero(err_s2, reward_mapping_func)

        # --- parse step 3: Major Axis Length
        m3 = pattern_step3.search(solution)
        if m3:
            major_axis_length = _to_float(m3.group(1))
            err_s3 = cal_MRE_error(
                [major_axis_length],
                [gt_float[0]],
            )  # gt_float[0] is the GT major axis length
        reward_s3 = cal_reward_from_error_or_zero(err_s3, reward_mapping_func)

        # --- parse step 4: Minor Axis Length
        m4 = pattern_step4.search(solution)
        if m4:
            minor_axis_length = _to_float(m4.group(1))
            err_s4 = cal_MRE_error(
                [minor_axis_length],
                [gt_float[1]],
            )  # gt_float[1] is the GT minor axis length
        reward_s4 = cal_reward_from_error_or_zero(err_s4, reward_mapping_func)

        reward = np.mean([reward_s1, reward_s2, reward_s3, reward_s4])

    except Exception as e:
        print(f"Exception in cal_process_reward: {e}")
        reward = 0.0
        err_s1 = err_s2 = err_s3 = err_s4 = float("nan")

    return reward, safe_nanmean([err_s1, err_s2]), safe_nanmean([err_s3, err_s4])


def cal_process_reward(solution, ground_truth, reward_mapping_func="exp_decay", **kwargs):
    """
    Calculates the process reward (see cal_process_reward_error).

    Returns:
        a scalar reward
    """
    return cal_process_reward_error(solution, ground_truth, reward_mapping_func, **kwargs)[0]


def cal_process_reward_error_v2(solution, ground_truth, reward_mapping_func="exp_decay", **kwargs):
    """
    Variant of cal_process_reward_error using normalized L2 distance for localization steps.

    Steps 1 & 2 (landmark coordinate prediction) use cal_norm_L2_error instead of cal_MAE_error.
    Steps 3 & 4 (axis length estimation) are unchanged (cal_MRE_error).

    Args:
        solution: model responses (text)
        ground_truth: ground truth string.

    Returns:
        A tuple (reward, localization_error, measurement_error); see cal_process_reward_error.
    """

    # Step patterns: reasoning + answer
    patterns_step = {}
    for k in range(1, 5):
        # NOTE: Use GROUP patterns to extract numeric values
        patterns_step[k] = rf"{PATTERNS_STEP_REASONING[k]}\s*{PATTERNS_STEP_ANSWER_GROUP[k]}"
    pattern_step1 = re.compile(patterns_step[1], re.DOTALL)
    pattern_step2 = re.compile(patterns_step[2], re.DOTALL)
    pattern_step3 = re.compile(patterns_step[3], re.DOTALL)
    pattern_step4 = re.compile(patterns_step[4], re.DOTALL)

    # NOTE:
    # ------
    # The landmark coordinates are in (w, h) format, not the conventional (h, w) format for image space indexing.
    # Such conversion is achieved in the dataset building recipe from MedVision (https://github.com/YongchengYAO/MedVision)
    # ------
    # P1 and P2 are the major axis landmarks, P3 and P4 are the minor axis landmarks
    gt_p1_wh = np.array(kwargs.get("landmark_P1_wh"))
    gt_p2_wh = np.array(kwargs.get("landmark_P2_wh"))
    gt_p3_wh = np.array(kwargs.get("landmark_P3_wh"))
    gt_p4_wh = np.array(kwargs.get("landmark_P4_wh"))

    # Extract ground truth coordinates
    gt_string = ground_truth.strip()
    gt_parts = [
        part.strip()
        for part in gt_string.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(",")
    ]
    gt_float = [float(part) for part in gt_parts if part]

    pred_major_wh = None
    pred_minor_wh = None
    err_s1 = err_s2 = err_s3 = err_s4 = float("nan")

    try:
        # NOTE: norm_L2 reward should be used for normalized coordinates (step 1 and step 2)
        # NOTE: MRE reward should be used in length estimation (step 3 and step 4)

        # --- parse step 1: Major Axis Endpoints
        m1 = pattern_step1.search(solution)
        if m1:
            x1_major, y1_major, x2_major, y2_major = _to_float(m1.group(1), m1.group(2), m1.group(3), m1.group(4))
            pred_major_wh = [
                x1_major,
                y1_major,
                x2_major,
                y2_major,
            ]
            # Calculate norm_L2 for both orderings of points (P1, P2) vs (P2, P1)
            err_s1 = min(
                cal_norm_L2_error(
                    pred_major_wh,
                    [gt_p1_wh[0], gt_p1_wh[1], gt_p2_wh[0], gt_p2_wh[1]],
                ),
                cal_norm_L2_error(
                    pred_major_wh,
                    [gt_p2_wh[0], gt_p2_wh[1], gt_p1_wh[0], gt_p1_wh[1]],
                ),
            )
        reward_s1 = cal_reward_from_error_or_zero(err_s1, reward_mapping_func)

        # --- parse step 2: Minor Axis Endpoints
        m2 = pattern_step2.search(solution)
        if m2:
            x1_minor, y1_minor, x2_minor, y2_minor = _to_float(m2.group(1), m2.group(2), m2.group(3), m2.group(4))
            pred_minor_wh = [
                x1_minor,
                y1_minor,
                x2_minor,
                y2_minor,
            ]
            # Calculate norm_L2 for both orderings of points (P3, P4) vs (P4, P3)
            err_s2 = min(
                cal_norm_L2_error(
                    pred_minor_wh,
                    [gt_p3_wh[0], gt_p3_wh[1], gt_p4_wh[0], gt_p4_wh[1]],
                ),
                cal_norm_L2_error(
                    pred_minor_wh,
                    [gt_p4_wh[0], gt_p4_wh[1], gt_p3_wh[0], gt_p3_wh[1]],
                ),
            )
        reward_s2 = cal_reward_from_error_or_zero(err_s2, reward_mapping_func)

        # --- parse step 3: Major Axis Length
        m3 = pattern_step3.search(solution)
        if m3:
            major_axis_length = _to_float(m3.group(1))
            err_s3 = cal_MRE_error(
                [major_axis_length],
                [gt_float[0]],
            )  # gt_float[0] is the GT major axis length
        reward_s3 = cal_reward_from_error_or_zero(err_s3, reward_mapping_func)

        # --- parse step 4: Minor Axis Length
        m4 = pattern_step4.search(solution)
        if m4:
            minor_axis_length = _to_float(m4.group(1))
            err_s4 = cal_MRE_error(
                [minor_axis_length],
                [gt_float[1]],
            )  # gt_float[1] is the GT minor axis length
        reward_s4 = cal_reward_from_error_or_zero(err_s4, reward_mapping_func)

        reward = np.mean([reward_s1, reward_s2, reward_s3, reward_s4])

    except Exception as e:
        print(f"Exception in cal_process_reward_v2: {e}")
        reward = 0.0
        err_s1 = err_s2 = err_s3 = err_s4 = float("nan")

    return reward, safe_nanmean([err_s1, err_s2]), safe_nanmean([err_s3, err_s4])


def cal_process_reward_v2(solution, ground_truth, reward_mapping_func="exp_decay", **kwargs):
    """
    Calculates the v2 process reward (see cal_process_reward_error_v2).

    Returns:
        a scalar reward
    """
    return cal_process_reward_error_v2(solution, ground_truth, reward_mapping_func, **kwargs)[0]


def cal_format_reward(solution, alpha=0.8):
    """
    Reward function that checks if the model response has a specific format.

    Args:
        solution: model responses (text)
        alpha: Weight for reasoning format reward. Defaults to 0.8.

    Returns:
        a scalar reward
    """
    reason_format_reward = match_reasoning(solution)
    answer_format_reward = match_answer(solution)
    reward = alpha * reason_format_reward + (1 - alpha) * answer_format_reward

    return reward


def cal_answer_reward_error(solution, ground_truth, reward_mapping_func="exp_decay", **kwargs):
    """
    Calculates the MRE (Mean Relative Error) reward and raw MRE from the extracted final answer
    from the model response (solution).

    Args:
        solution: model responses (text)
        ground_truth: ground truth strings
        reward_mapping_func: Reward mapping function name.

    Returns:
        A tuple (reward, answer_error) where answer_error is the MRE
        (NaN if the answer is unparseable).
    """

    # Extract ground truth coordinates
    gt_string = ground_truth.strip()
    gt_parts = [
        part.strip()
        for part in gt_string.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(",")
    ]
    gt_float = [float(part) for part in gt_parts if part]
    num_gt = len(gt_float)

    # Extract predicted values
    try:
        # Directly extract specified number of values using GROUP pattern
        pattern_answer = re.compile(PATTERN_ANSWER_2_VALUES_GROUP, re.DOTALL)
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

    # Compute MRE reward
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
    extra_info=None,
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
    if extra_info is None:
        extra_info = {}

    format_reward = cal_format_reward(solution_str)
    process_reward, localization_error, measurement_error = cal_process_reward_error(
        solution_str, ground_truth, reward_mapping_func="exp_decay", **extra_info
    )
    answer_reward, answer_error = cal_answer_reward_error(
        solution_str, ground_truth, reward_mapping_func="exp_decay", **extra_info
    )
    reward = format_reward + process_reward + answer_reward

    return {
        "score": reward,
        "format_reward": format_reward,
        "process_reward": process_reward,
        "answer_reward": answer_reward,
        "localization_error": localization_error,
        "measurement_error": measurement_error,
        "answer_error": answer_error,
    }


def compute_score_exp_decay_PRxAnswer(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
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
    if extra_info is None:
        extra_info = {}

    format_reward = cal_format_reward(solution_str)
    process_reward, localization_error, measurement_error = cal_process_reward_error(
        solution_str, ground_truth, reward_mapping_func="exp_decay", **extra_info
    )
    answer_reward, answer_error = cal_answer_reward_error(
        solution_str, ground_truth, reward_mapping_func="exp_decay", **extra_info
    )
    reward = format_reward + process_reward * answer_reward

    return {
        "score": reward,
        "format_reward": format_reward,
        "process_reward": process_reward,
        "answer_reward": answer_reward,
        "localization_error": localization_error,
        "measurement_error": measurement_error,
        "answer_error": answer_error,
    }


def cal_process_reward_error_v3(solution, ground_truth, reward_mapping_func="exp_decay", **kwargs):
    """
    Variant of cal_process_reward_error using max normalized L2 distance for localization steps.

    Steps 1 & 2 (landmark coordinate prediction) use cal_norm_L2_max_error: the localization
    error per step is the worst-case (max) per-point normalized L2 distance instead of the mean.
    Steps 3 & 4 (axis length estimation) are unchanged (cal_MRE_error).

    Args:
        solution: model responses (text)
        ground_truth: ground truth string.

    Returns:
        A tuple (reward, localization_error, measurement_error); see cal_process_reward_error.
    """

    # Step patterns: reasoning + answer
    patterns_step = {}
    for k in range(1, 5):
        # NOTE: Use GROUP patterns to extract numeric values
        patterns_step[k] = rf"{PATTERNS_STEP_REASONING[k]}\s*{PATTERNS_STEP_ANSWER_GROUP[k]}"
    pattern_step1 = re.compile(patterns_step[1], re.DOTALL)
    pattern_step2 = re.compile(patterns_step[2], re.DOTALL)
    pattern_step3 = re.compile(patterns_step[3], re.DOTALL)
    pattern_step4 = re.compile(patterns_step[4], re.DOTALL)

    # NOTE:
    # ------
    # The landmark coordinates are in (w, h) format, not the conventional (h, w) format for image space indexing.
    # Such conversion is achieved in the dataset building recipe from MedVision (https://github.com/YongchengYAO/MedVision)
    # ------
    # P1 and P2 are the major axis landmarks, P3 and P4 are the minor axis landmarks
    gt_p1_wh = np.array(kwargs.get("landmark_P1_wh"))
    gt_p2_wh = np.array(kwargs.get("landmark_P2_wh"))
    gt_p3_wh = np.array(kwargs.get("landmark_P3_wh"))
    gt_p4_wh = np.array(kwargs.get("landmark_P4_wh"))

    # Extract ground truth coordinates
    gt_string = ground_truth.strip()
    gt_parts = [
        part.strip()
        for part in gt_string.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(",")
    ]
    gt_float = [float(part) for part in gt_parts if part]

    pred_major_wh = None
    pred_minor_wh = None
    err_s1 = err_s2 = err_s3 = err_s4 = float("nan")

    try:
        # NOTE: norm_L2_max reward should be used for normalized coordinates (step 1 and step 2)
        # NOTE: MRE reward should be used in length estimation (step 3 and step 4)

        # --- parse step 1: Major Axis Endpoints
        m1 = pattern_step1.search(solution)
        if m1:
            x1_major, y1_major, x2_major, y2_major = _to_float(m1.group(1), m1.group(2), m1.group(3), m1.group(4))
            pred_major_wh = [
                x1_major,
                y1_major,
                x2_major,
                y2_major,
            ]
            # Calculate norm_L2_max for both orderings of points (P1, P2) vs (P2, P1)
            err_s1 = min(
                cal_norm_L2_max_error(
                    pred_major_wh,
                    [gt_p1_wh[0], gt_p1_wh[1], gt_p2_wh[0], gt_p2_wh[1]],
                ),
                cal_norm_L2_max_error(
                    pred_major_wh,
                    [gt_p2_wh[0], gt_p2_wh[1], gt_p1_wh[0], gt_p1_wh[1]],
                ),
            )
        reward_s1 = cal_reward_from_error_or_zero(err_s1, reward_mapping_func)

        # --- parse step 2: Minor Axis Endpoints
        m2 = pattern_step2.search(solution)
        if m2:
            x1_minor, y1_minor, x2_minor, y2_minor = _to_float(m2.group(1), m2.group(2), m2.group(3), m2.group(4))
            pred_minor_wh = [
                x1_minor,
                y1_minor,
                x2_minor,
                y2_minor,
            ]
            # Calculate norm_L2_max for both orderings of points (P3, P4) vs (P4, P3)
            err_s2 = min(
                cal_norm_L2_max_error(
                    pred_minor_wh,
                    [gt_p3_wh[0], gt_p3_wh[1], gt_p4_wh[0], gt_p4_wh[1]],
                ),
                cal_norm_L2_max_error(
                    pred_minor_wh,
                    [gt_p4_wh[0], gt_p4_wh[1], gt_p3_wh[0], gt_p3_wh[1]],
                ),
            )
        reward_s2 = cal_reward_from_error_or_zero(err_s2, reward_mapping_func)

        # --- parse step 3: Major Axis Length
        m3 = pattern_step3.search(solution)
        if m3:
            major_axis_length = _to_float(m3.group(1))
            err_s3 = cal_MRE_error(
                [major_axis_length],
                [gt_float[0]],
            )  # gt_float[0] is the GT major axis length
        reward_s3 = cal_reward_from_error_or_zero(err_s3, reward_mapping_func)

        # --- parse step 4: Minor Axis Length
        m4 = pattern_step4.search(solution)
        if m4:
            minor_axis_length = _to_float(m4.group(1))
            err_s4 = cal_MRE_error(
                [minor_axis_length],
                [gt_float[1]],
            )  # gt_float[1] is the GT minor axis length
        reward_s4 = cal_reward_from_error_or_zero(err_s4, reward_mapping_func)

        reward = np.mean([reward_s1, reward_s2, reward_s3, reward_s4])

    except Exception as e:
        print(f"Exception in cal_process_reward_v3: {e}")
        reward = 0.0
        err_s1 = err_s2 = err_s3 = err_s4 = float("nan")

    return reward, safe_nanmean([err_s1, err_s2]), safe_nanmean([err_s3, err_s4])


def cal_process_reward_v3(solution, ground_truth, reward_mapping_func="exp_decay", **kwargs):
    """
    Calculates the v3 process reward (see cal_process_reward_error_v3).

    Returns:
        a scalar reward
    """
    return cal_process_reward_error_v3(solution, ground_truth, reward_mapping_func, **kwargs)[0]


def compute_score_exp_decay_PRxAnswer_v2(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
):
    """
    Variant of compute_score_exp_decay_PRxAnswer using normalized L2 process reward.

    Identical to compute_score_exp_decay_PRxAnswer except the process reward uses
    cal_process_reward_v2, which measures localization error via normalized L2 distance
    (sqrt(dx^2+dy^2)/sqrt(2)) instead of MAE.

    Args:
        data_source: The source of the data.
        solution_str: The solution (completions).
        ground_truth: The ground truth.
        extra_info: Extra information for reward calculation.

    Returns:
        A dictionary containing the calculated rewards.
    """
    if extra_info is None:
        extra_info = {}

    format_reward = cal_format_reward(solution_str)
    process_reward, localization_error, measurement_error = cal_process_reward_error_v2(
        solution_str, ground_truth, reward_mapping_func="exp_decay", **extra_info
    )
    answer_reward, answer_error = cal_answer_reward_error(
        solution_str, ground_truth, reward_mapping_func="exp_decay", **extra_info
    )
    reward = format_reward + process_reward * answer_reward

    return {
        "score": reward,
        "format_reward": format_reward,
        "process_reward": process_reward,
        "answer_reward": answer_reward,
        "localization_error": localization_error,
        "measurement_error": measurement_error,
        "answer_error": answer_error,
    }


def compute_score_exp_decay_PRxAnswer_v3(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
):
    """
    Variant of compute_score_exp_decay_PRxAnswer using max normalized L2 process reward.

    Identical to compute_score_exp_decay_PRxAnswer except the process reward uses
    cal_process_reward_v3, which measures localization error via the max (worst-case)
    per-point normalized L2 distance instead of the mean.

    Args:
        data_source: The source of the data.
        solution_str: The solution (completions).
        ground_truth: The ground truth.
        extra_info: Extra information for reward calculation.

    Returns:
        A dictionary containing the calculated rewards.
    """
    if extra_info is None:
        extra_info = {}

    format_reward = cal_format_reward(solution_str)
    process_reward, localization_error, measurement_error = cal_process_reward_error_v3(
        solution_str, ground_truth, reward_mapping_func="exp_decay", **extra_info
    )
    answer_reward, answer_error = cal_answer_reward_error(
        solution_str, ground_truth, reward_mapping_func="exp_decay", **extra_info
    )
    reward = format_reward + process_reward * answer_reward

    return {
        "score": reward,
        "format_reward": format_reward,
        "process_reward": process_reward,
        "answer_reward": answer_reward,
        "localization_error": localization_error,
        "measurement_error": measurement_error,
        "answer_error": answer_error,
    }


def compute_score_exp_decay_morePR(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
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
    if extra_info is None:
        extra_info = {}

    format_reward = cal_format_reward(solution_str)
    process_reward, localization_error, measurement_error = cal_process_reward_error(
        solution_str, ground_truth, reward_mapping_func="exp_decay", **extra_info
    )
    answer_reward, answer_error = cal_answer_reward_error(
        solution_str, ground_truth, reward_mapping_func="exp_decay", **extra_info
    )
    reward = 0.1 * format_reward + process_reward + 0.1 * answer_reward

    return {
        "score": reward,
        "format_reward": format_reward,
        "process_reward": process_reward,
        "answer_reward": answer_reward,
        "localization_error": localization_error,
        "measurement_error": measurement_error,
        "answer_error": answer_error,
    }


def compute_score_exp_decay_wo_proc(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
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
    if extra_info is None:
        extra_info = {}

    format_reward = cal_format_reward(solution_str)
    answer_reward, answer_error = cal_answer_reward_error(
        solution_str, ground_truth, reward_mapping_func="exp_decay", **extra_info
    )
    reward = format_reward + answer_reward

    return {
        "score": reward,
        "format_reward": format_reward,
        "answer_reward": answer_reward,
        "answer_error": answer_error,
    }


def compute_score_scaled_sigmoid(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
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
    if extra_info is None:
        extra_info = {}

    format_reward = cal_format_reward(solution_str)
    process_reward, localization_error, measurement_error = cal_process_reward_error(
        solution_str, ground_truth, reward_mapping_func="scaled_sigmoid", **extra_info
    )
    answer_reward, answer_error = cal_answer_reward_error(
        solution_str, ground_truth, reward_mapping_func="scaled_sigmoid", **extra_info
    )
    reward = format_reward + process_reward + answer_reward

    return {
        "score": reward,
        "format_reward": format_reward,
        "process_reward": process_reward,
        "answer_reward": answer_reward,
        "localization_error": localization_error,
        "measurement_error": measurement_error,
        "answer_error": answer_error,
    }


def compute_score_gaussian_proxy(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
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
    if extra_info is None:
        extra_info = {}

    format_reward = cal_format_reward(solution_str)
    process_reward, localization_error, measurement_error = cal_process_reward_error(
        solution_str, ground_truth, reward_mapping_func="gaussian_proxy", **extra_info
    )
    answer_reward, answer_error = cal_answer_reward_error(
        solution_str, ground_truth, reward_mapping_func="gaussian_proxy", **extra_info
    )
    reward = format_reward + process_reward + answer_reward

    return {
        "score": reward,
        "format_reward": format_reward,
        "process_reward": process_reward,
        "answer_reward": answer_reward,
        "localization_error": localization_error,
        "measurement_error": measurement_error,
        "answer_error": answer_error,
    }
