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
    cal_MAE_reward,
    cal_MRE_reward,
    cal_norm_L2_max_reward,
    cal_norm_L2_reward,
    extract_last_k_nums,
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
# Define patterns for MedVision-AD (angle/distance measurement) task
# =========================================================
# Paterns for answer extraction and format checking
PATTERN_NON_NEG_REAL = r"\d+(?:\.\d+)?"
PATTERN_NON_NEG_REAL_GROUP = rf"({PATTERN_NON_NEG_REAL})"
PATTERN_RELATIVE_POSITION = r"(?:0(?:\.\d+)?|1(?:\.0+)?)"  # [0, 1] inclusive
PATTERN_RELATIVE_POSITION_GROUP = rf"({PATTERN_RELATIVE_POSITION})"
PATTERN_COORD = rf"\(\s*{PATTERN_RELATIVE_POSITION}\s*,\s*{PATTERN_RELATIVE_POSITION}\s*\)"
PATTERN_COORD_GROUP = rf"\(\s*{PATTERN_RELATIVE_POSITION_GROUP}\s*,\s*{PATTERN_RELATIVE_POSITION_GROUP}\s*\)"
PATTERN_ANSWER_1_VALUE = rf"\s*{tag('answer')}\s*{PATTERN_NON_NEG_REAL}\s*{end('answer')}\s*"
PATTERN_ANSWER_1_VALUE_GROUP = rf"\s*{tag('answer')}\s*{PATTERN_NON_NEG_REAL_GROUP}\s*{end('answer')}\s*"

# ------
# NOTE:
# The number of reasoning steps below is defined in COT_INSTRUCT_ANGLE and COT_INSTRUCT_DISTANCE
# imported from medvision_bm.sft.sft_prompts
# ------
# Reasoning patterns
PATTERNS_STEP_REASONING_DISTANCE = {}
for k in range(1, 4):
    PATTERNS_STEP_REASONING_DISTANCE[k] = step_reasoning(k)
PATTERNS_STEP_REASONING_ANGLE = {}
for k in range(1, 4):
    PATTERNS_STEP_REASONING_ANGLE[k] = step_reasoning(k)

# Step answer patterns for distance estimation tasks
# Steps 1 & 2: Expect a coordinate (x1, y1)
# Step 3: Expect a single scalar value
PATTERNS_STEP_ANSWER_DISTANCE = {
    1: rf".*?{tag('step-1-answer')}.*?{PATTERN_COORD}.*?{end('step-1-answer')}.*?",
    2: rf".*?{tag('step-2-answer')}.*?{PATTERN_COORD}.*?{end('step-2-answer')}.*?",
    3: rf".*?{tag('step-3-answer')}.*?{PATTERN_NON_NEG_REAL}.*?{end('step-3-answer')}.*?",
}
PATTERNS_STEP_ANSWER_DISTANCE_GROUP = {
    1: rf".*?{tag('step-1-answer')}.*?{PATTERN_COORD_GROUP}.*?{end('step-1-answer')}.*?",
    2: rf".*?{tag('step-2-answer')}.*?{PATTERN_COORD_GROUP}.*?{end('step-2-answer')}.*?",
    3: rf".*?{tag('step-3-answer')}.*?{PATTERN_NON_NEG_REAL_GROUP}.*?{end('step-3-answer')}.*?",
}

# Step answer patterns for angle estimation tasks
# Steps 1 & 2: Expect 2 coordinates (x1, y1), (x2, y2)
# Step 3: Expect a single scalar value
PATTERNS_STEP_ANSWER_ANGLE = {
    1: rf".*?{tag('step-1-answer')}.*?{PATTERN_COORD}\s*,\s*{PATTERN_COORD}.*?{end('step-1-answer')}.*?",
    2: rf".*?{tag('step-2-answer')}.*?{PATTERN_COORD}\s*,\s*{PATTERN_COORD}.*?{end('step-2-answer')}.*?",
    3: rf".*?{tag('step-3-answer')}.*?{PATTERN_NON_NEG_REAL}.*?{end('step-3-answer')}.*?",
}
PATTERNS_STEP_ANSWER_ANGLE_GROUP = {
    1: rf".*?{tag('step-1-answer')}.*?{PATTERN_COORD_GROUP}\s*,\s*{PATTERN_COORD_GROUP}.*?{end('step-1-answer')}.*?",
    2: rf".*?{tag('step-2-answer')}.*?{PATTERN_COORD_GROUP}\s*,\s*{PATTERN_COORD_GROUP}.*?{end('step-2-answer')}.*?",
    3: rf".*?{tag('step-3-answer')}.*?{PATTERN_NON_NEG_REAL_GROUP}.*?{end('step-3-answer')}.*?",
}
# ------
# =========================================================


def match_reasoning_distance_task(content, **kwargs):
    """
    Soft (non-binary) reward in [0, 1] for how well `content` matches the required <think> format.

    The scoring criteria are:
    1. Presence of <think>...</think> block.
    2. Presence of step-by-step reasoning and answer blocks (steps 1 to 3).
    3. Correct formatting of answers within steps (coordinates for steps 1-2, scalars for step 3).
    4. Logical ordering of steps (Step 1 -> Step 2 -> Step 3).

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
    for k in (1, 2, 3):
        # A. Reasoning block
        reasoning_pattern = PATTERNS_STEP_REASONING_DISTANCE[k]
        reasoning_match = re.search(reasoning_pattern, body, flags)
        if reasoning_match:
            score += W_STEP[k]["reason"]
            step_spans[f"r{k}"] = reasoning_match.span()

        # B. Answer block (tag presence)§
        answer_pattern = PATTERNS_STEP_ANSWER_DISTANCE[k]
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
            elif k == 3:
                # Step 3: numeric float presence
                if re.search(PATTERN_NON_NEG_REAL, region, flags):
                    score += W_STEP[k]["format"]

    # 3) Order check: step1 → step2 → step3
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

    # Check order: s1 <= s2 <= s3
    s1, s2, s3 = get_step_positions([("r1", "a1"), ("r2", "a2"), ("r3", "a3")])

    if all(p is not None for p in (s1, s2, s3)) and s1 <= s2 <= s3:
        score += W_ORDER
    elif sum(p is not None for p in (s1, s2, s3)) >= 2:
        # Partial order credit if most steps are present and sorted
        seq = [p for p in (s1, s2, s3) if p is not None]
        if seq == sorted(seq):
            score += W_ORDER * 0.5

    return max(0.0, min(1.0, score / max_score))


def match_reasoning_angle_task(content, **kwargs):
    """
    Soft (non-binary) reward in [0, 1] for how well `content` matches the required <think> format.

    The scoring criteria are:
    1. Presence of <think>...</think> block.
    2. Presence of step-by-step reasoning and answer blocks (steps 1 to 3).
    3. Correct formatting of answers within steps (coordinates for steps 1-2, scalars for step 3).
    4. Logical ordering of steps (Step 1 -> Step 2 -> Step 3).

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
    for k in (1, 2, 3):
        # A. Reasoning block
        reasoning_pattern = PATTERNS_STEP_REASONING_ANGLE[k]
        reasoning_match = re.search(reasoning_pattern, body, flags)
        if reasoning_match:
            score += W_STEP[k]["reason"]
            step_spans[f"r{k}"] = reasoning_match.span()

        # B. Answer block (tag presence)§
        answer_pattern = PATTERNS_STEP_ANSWER_ANGLE[k]
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
                # up to 2 coords (4 numbers) expected
                score += min(len(found_coords), 4) * (W_STEP[k]["format"] / 4.0)
            elif k == 3:
                # Step 3: numeric float presence
                if re.search(PATTERN_NON_NEG_REAL, region, flags):
                    score += W_STEP[k]["format"]

    # 3) Order check: step1 → step2 → step3
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

    # Check order: s1 <= s2 <= s3
    s1, s2, s3 = get_step_positions([("r1", "a1"), ("r2", "a2"), ("r3", "a3")])

    if all(p is not None for p in (s1, s2, s3)) and s1 <= s2 <= s3:
        score += W_ORDER
    elif sum(p is not None for p in (s1, s2, s3)) >= 2:
        # Partial order credit if most steps are present and sorted
        seq = [p for p in (s1, s2, s3) if p is not None]
        if seq == sorted(seq):
            score += W_ORDER * 0.5

    return max(0.0, min(1.0, score / max_score))


def match_reasoning(content, **kwargs):
    # Task type
    metric_type = kwargs.get("metric_type", None)
    assert metric_type is not None, "metric_type not found in kwargs"

    if metric_type == "distance":
        return match_reasoning_distance_task(content, **kwargs)
    elif metric_type == "angle":
        return match_reasoning_angle_task(content, **kwargs)
    else:
        raise ValueError(f"Unsupported metric_type: {metric_type}")


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
    return 1 if re.search(PATTERN_ANSWER_1_VALUE, content_clean, re.VERBOSE) else 0


def _to_float(*gs):
    return tuple(float(x) for x in gs)


def cal_process_reward_distance_task(solution, ground_truth, reward_mapping_func="exp_decay", **kwargs):
    """
    Calculates the process reward for MedVision distance estimation tasks.

    The process reward evaluates the step-by-step reasoning and intermediate answers in the model's response (solution).

    Args:
        solution: model responses (text)
        ground_truth: ground truth string.

    Returns:
        a scalar reward

    NOTE:
        - The number of reasoning steps is hardcoded, see "pattern_step{1,2,3}"
        - The number of reasoning steps below is defined in COT_INSTRUCT_DISTANCE
          imported from medvision_bm.sft.sft_prompts
    """
    # Step patterns: reasoning + answer
    patterns_step = {}
    for k in range(1, 4):
        # NOTE: Use GROUP patterns to extract numeric values
        patterns_step[k] = rf"{PATTERNS_STEP_REASONING_DISTANCE[k]}\s*{PATTERNS_STEP_ANSWER_DISTANCE_GROUP[k]}"
    pattern_step1 = re.compile(patterns_step[1], re.DOTALL)
    pattern_step2 = re.compile(patterns_step[2], re.DOTALL)
    pattern_step3 = re.compile(patterns_step[3], re.DOTALL)

    # NOTE:
    # ------
    # The landmark coordinates are in (w, h) format, not the conventional (h, w) format for image space indexing.
    # Such conversion is achieved in the dataset building recipe from MedVision (https://github.com/YongchengYAO/MedVision)
    # ------
    gt_lm1_wh = np.array(kwargs.get("landmark_1_wh"))
    gt_lm2_wh = np.array(kwargs.get("landmark_2_wh"))

    # Extract ground truth coordinates
    gt_string = ground_truth.strip()
    gt_parts = [
        part.strip()
        for part in gt_string.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(",")
    ]
    gt_float = [float(part) for part in gt_parts if part]

    try:
        # NOTE:
        # To ensure the input to reward function are in [0,1]:
        #   - MAE reward should be used for normalized coordinates (step 1 and step 2)
        #   - MRE reward should be used in distance estimation (step 3)

        # --- parse step 1: landmark 1 coordinate
        m1 = pattern_step1.search(solution)
        if m1:
            pred_lm1_wh = list(_to_float(m1.group(1), m1.group(2)))
            # Calculate MAE
            reward_s1 = cal_MAE_reward(
                pred_lm1_wh,
                [gt_lm1_wh[0], gt_lm1_wh[1]],
                reward_mapping_func,
            )
        else:
            reward_s1 = 0

        # --- parse step 2: landmark 2 coordinate
        m2 = pattern_step2.search(solution)
        if m2:
            pred_lm2_wh = list(_to_float(m2.group(1), m2.group(2)))
            # Calculate MAE
            reward_s2 = cal_MAE_reward(
                pred_lm2_wh,
                [gt_lm2_wh[0], gt_lm2_wh[1]],
                reward_mapping_func,
            )
        else:
            reward_s2 = 0

        # --- parse step 3: the target distance
        m3 = pattern_step3.search(solution)
        if m3:
            pred_distance = list(_to_float(m3.group(1)))
            reward_s3 = cal_MRE_reward(
                pred_distance,
                [gt_float[0]],
                reward_mapping_func,
            )
        else:
            reward_s3 = 0

        reward = np.mean([reward_s1, reward_s2, reward_s3])

    except Exception as e:
        print(f"Exception in cal_process_reward: {e}")
        reward = 0.0

    return reward


def cal_process_reward_angle_task(solution, ground_truth, reward_mapping_func="exp_decay", **kwargs):
    """
    Calculates the process reward for MedVision angle estimation tasks.

    The process reward evaluates the step-by-step reasoning and intermediate answers in the model's response (solution).

    Args:
        solution: model responses (text)
        ground_truth: ground truth string.

    Returns:
        a scalar reward

    NOTE:
        - The number of reasoning steps is hardcoded, see "pattern_step{1,2,3}"
        - The number of reasoning steps below is defined in COT_INSTRUCT_ANGLE
          imported from medvision_bm.sft.sft_prompts
    """
    # Step patterns: reasoning + answer
    patterns_step = {}
    for k in range(1, 4):
        # NOTE: Use GROUP patterns to extract numeric values
        patterns_step[k] = rf"{PATTERNS_STEP_REASONING_ANGLE[k]}\s*{PATTERNS_STEP_ANSWER_ANGLE_GROUP[k]}"
    pattern_step1 = re.compile(patterns_step[1], re.DOTALL)
    pattern_step2 = re.compile(patterns_step[2], re.DOTALL)
    pattern_step3 = re.compile(patterns_step[3], re.DOTALL)

    # NOTE:
    # ------
    # The landmark coordinates are in (w, h) format, not the conventional (h, w) format for image space indexing.
    # Such conversion is achieved in the dataset building recipe from MedVision (https://github.com/YongchengYAO/MedVision)
    # ------
    gt_lm1_line1_wh = np.array(kwargs.get("line_1_point_1_wh"))
    gt_lm2_line1_wh = np.array(kwargs.get("line_1_point_2_wh"))
    gt_lm1_line2_wh = np.array(kwargs.get("line_2_point_1_wh"))
    gt_lm2_line2_wh = np.array(kwargs.get("line_2_point_2_wh"))

    # Extract ground truth coordinates
    gt_string = ground_truth.strip()
    gt_parts = [
        part.strip()
        for part in gt_string.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(",")
    ]
    gt_float = [float(part) for part in gt_parts if part]

    try:
        # NOTE:
        # To ensure the input to reward function are in [0,1]:
        #   - MAE reward should be used for normalized coordinates (step 1 and step 2)
        #   - MRE reward should be used in angle estimation (step 3)

        # --- parse step 1: line 1 endpoints
        m1 = pattern_step1.search(solution)
        if m1:
            pred_line1_coor_wh = list(_to_float(m1.group(1), m1.group(2), m1.group(3), m1.group(4)))
            # Calculate MAE for both orderings of points (P1, P2) vs (P2, P1)
            reward_s1 = max(
                cal_MAE_reward(
                    pred_line1_coor_wh,
                    [
                        gt_lm1_line1_wh[0],
                        gt_lm1_line1_wh[1],
                        gt_lm2_line1_wh[0],
                        gt_lm2_line1_wh[1],
                    ],
                    reward_mapping_func,
                ),
                cal_MAE_reward(
                    pred_line1_coor_wh,
                    [
                        gt_lm2_line1_wh[0],
                        gt_lm2_line1_wh[1],
                        gt_lm1_line1_wh[0],
                        gt_lm1_line1_wh[1],
                    ],
                    reward_mapping_func,
                ),
            )
        else:
            reward_s1 = 0

        # --- parse step 2: line 2 endpoints
        m2 = pattern_step2.search(solution)
        if m2:
            pred_line2_coor_wh = list(_to_float(m2.group(1), m2.group(2), m2.group(3), m2.group(4)))
            # Calculate MAE for both orderings of points (P1, P2) vs (P2, P1)
            reward_s2 = max(
                cal_MAE_reward(
                    pred_line2_coor_wh,
                    [
                        gt_lm1_line2_wh[0],
                        gt_lm1_line2_wh[1],
                        gt_lm2_line2_wh[0],
                        gt_lm2_line2_wh[1],
                    ],
                    reward_mapping_func,
                ),
                cal_MAE_reward(
                    pred_line2_coor_wh,
                    [
                        gt_lm2_line2_wh[0],
                        gt_lm2_line2_wh[1],
                        gt_lm1_line2_wh[0],
                        gt_lm1_line2_wh[1],
                    ],
                    reward_mapping_func,
                ),
            )
        else:
            reward_s2 = 0

        # --- parse step 3: the target angle
        m3 = pattern_step3.search(solution)
        if m3:
            pred_angle = list(_to_float(m3.group(1)))
            reward_s3 = cal_MRE_reward(
                pred_angle,
                [gt_float[0]],
                reward_mapping_func,
            )
        else:
            reward_s3 = 0

        reward = np.mean([reward_s1, reward_s2, reward_s3])

    except Exception as e:
        print(f"Exception in cal_process_reward: {e}")
        reward = 0.0

    return reward


def cal_process_reward(solution, ground_truth, reward_mapping_func="exp_decay", **kwargs):
    """
    Calculates the process reward.

    The process reward evaluates the step-by-step reasoning and intermediate answers in the model's response (solution).

    Args:
        solution: model responses (text)
        ground_truth: ground truth string.

    Returns:
        a scalar reward

    NOTE:
        - The number of reasoning steps is hardcoded, see "pattern_step{1,2,3}"
        - The number of reasoning steps below is defined in COT_INSTRUCT_ANGLE and COT_INSTRUCT_DISTANCE
          imported from medvision_bm.sft.sft_prompts
    """
    # Task type
    metric_type = kwargs.get("metric_type", None)
    assert metric_type is not None, "metric_type not found in kwargs"

    if metric_type == "distance":
        return cal_process_reward_distance_task(solution, ground_truth, reward_mapping_func="exp_decay", **kwargs)
    elif metric_type == "angle":
        return cal_process_reward_angle_task(solution, ground_truth, reward_mapping_func="exp_decay", **kwargs)
    else:
        raise ValueError(f"Unsupported metric_type: {metric_type}")


def cal_process_reward_distance_task_v2(solution, ground_truth, reward_mapping_func="exp_decay", **kwargs):
    """
    Variant of cal_process_reward_distance_task using normalized L2 distance for localization steps.

    Steps 1 & 2 (landmark coordinate prediction) use cal_norm_L2_reward instead of cal_MAE_reward.
    Step 3 (distance estimation) is unchanged (cal_MRE_reward).

    Args:
        solution: model responses (text)
        ground_truth: ground truth string.

    Returns:
        a scalar reward

    NOTE:
        - The number of reasoning steps is hardcoded, see "pattern_step{1,2,3}"
        - The number of reasoning steps below is defined in COT_INSTRUCT_DISTANCE
          imported from medvision_bm.sft.sft_prompts
    """
    # Step patterns: reasoning + answer
    patterns_step = {}
    for k in range(1, 4):
        # NOTE: Use GROUP patterns to extract numeric values
        patterns_step[k] = rf"{PATTERNS_STEP_REASONING_DISTANCE[k]}\s*{PATTERNS_STEP_ANSWER_DISTANCE_GROUP[k]}"
    pattern_step1 = re.compile(patterns_step[1], re.DOTALL)
    pattern_step2 = re.compile(patterns_step[2], re.DOTALL)
    pattern_step3 = re.compile(patterns_step[3], re.DOTALL)

    # NOTE:
    # ------
    # The landmark coordinates are in (w, h) format, not the conventional (h, w) format for image space indexing.
    # Such conversion is achieved in the dataset building recipe from MedVision (https://github.com/YongchengYAO/MedVision)
    # ------
    gt_lm1_wh = np.array(kwargs.get("landmark_1_wh"))
    gt_lm2_wh = np.array(kwargs.get("landmark_2_wh"))

    # Extract ground truth coordinates
    gt_string = ground_truth.strip()
    gt_parts = [
        part.strip()
        for part in gt_string.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(",")
    ]
    gt_float = [float(part) for part in gt_parts if part]

    try:
        # NOTE: norm_L2 reward should be used for normalized coordinates (step 1 and step 2)
        # NOTE: MRE reward should be used in distance estimation (step 3)

        # --- parse step 1: landmark 1 coordinate
        m1 = pattern_step1.search(solution)
        if m1:
            pred_lm1_wh = list(_to_float(m1.group(1), m1.group(2)))
            reward_s1 = cal_norm_L2_reward(
                pred_lm1_wh,
                [gt_lm1_wh[0], gt_lm1_wh[1]],
                reward_mapping_func,
            )
        else:
            reward_s1 = 0

        # --- parse step 2: landmark 2 coordinate
        m2 = pattern_step2.search(solution)
        if m2:
            pred_lm2_wh = list(_to_float(m2.group(1), m2.group(2)))
            reward_s2 = cal_norm_L2_reward(
                pred_lm2_wh,
                [gt_lm2_wh[0], gt_lm2_wh[1]],
                reward_mapping_func,
            )
        else:
            reward_s2 = 0

        # --- parse step 3: the target distance
        m3 = pattern_step3.search(solution)
        if m3:
            pred_distance = list(_to_float(m3.group(1)))
            reward_s3 = cal_MRE_reward(
                pred_distance,
                [gt_float[0]],
                reward_mapping_func,
            )
        else:
            reward_s3 = 0

        reward = np.mean([reward_s1, reward_s2, reward_s3])

    except Exception as e:
        print(f"Exception in cal_process_reward_distance_task_v2: {e}")
        reward = 0.0

    return reward


def cal_process_reward_angle_task_v2(solution, ground_truth, reward_mapping_func="exp_decay", **kwargs):
    """
    Variant of cal_process_reward_angle_task using normalized L2 distance for localization steps.

    Steps 1 & 2 (line endpoint prediction) use cal_norm_L2_reward instead of cal_MAE_reward.
    Step 3 (angle estimation) is unchanged (cal_MRE_reward).

    Args:
        solution: model responses (text)
        ground_truth: ground truth string.

    Returns:
        a scalar reward

    NOTE:
        - The number of reasoning steps is hardcoded, see "pattern_step{1,2,3}"
        - The number of reasoning steps below is defined in COT_INSTRUCT_ANGLE
          imported from medvision_bm.sft.sft_prompts
    """
    # Step patterns: reasoning + answer
    patterns_step = {}
    for k in range(1, 4):
        # NOTE: Use GROUP patterns to extract numeric values
        patterns_step[k] = rf"{PATTERNS_STEP_REASONING_ANGLE[k]}\s*{PATTERNS_STEP_ANSWER_ANGLE_GROUP[k]}"
    pattern_step1 = re.compile(patterns_step[1], re.DOTALL)
    pattern_step2 = re.compile(patterns_step[2], re.DOTALL)
    pattern_step3 = re.compile(patterns_step[3], re.DOTALL)

    # NOTE:
    # ------
    # The landmark coordinates are in (w, h) format, not the conventional (h, w) format for image space indexing.
    # Such conversion is achieved in the dataset building recipe from MedVision (https://github.com/YongchengYAO/MedVision)
    # ------
    gt_lm1_line1_wh = np.array(kwargs.get("line_1_point_1_wh"))
    gt_lm2_line1_wh = np.array(kwargs.get("line_1_point_2_wh"))
    gt_lm1_line2_wh = np.array(kwargs.get("line_2_point_1_wh"))
    gt_lm2_line2_wh = np.array(kwargs.get("line_2_point_2_wh"))

    # Extract ground truth coordinates
    gt_string = ground_truth.strip()
    gt_parts = [
        part.strip()
        for part in gt_string.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(",")
    ]
    gt_float = [float(part) for part in gt_parts if part]

    try:
        # NOTE: norm_L2 reward should be used for normalized coordinates (step 1 and step 2)
        # NOTE: MRE reward should be used in angle estimation (step 3)

        # --- parse step 1: line 1 endpoints
        m1 = pattern_step1.search(solution)
        if m1:
            pred_line1_coor_wh = list(_to_float(m1.group(1), m1.group(2), m1.group(3), m1.group(4)))
            # Calculate norm_L2 for both orderings of points (P1, P2) vs (P2, P1)
            reward_s1 = max(
                cal_norm_L2_reward(
                    pred_line1_coor_wh,
                    [
                        gt_lm1_line1_wh[0],
                        gt_lm1_line1_wh[1],
                        gt_lm2_line1_wh[0],
                        gt_lm2_line1_wh[1],
                    ],
                    reward_mapping_func,
                ),
                cal_norm_L2_reward(
                    pred_line1_coor_wh,
                    [
                        gt_lm2_line1_wh[0],
                        gt_lm2_line1_wh[1],
                        gt_lm1_line1_wh[0],
                        gt_lm1_line1_wh[1],
                    ],
                    reward_mapping_func,
                ),
            )
        else:
            reward_s1 = 0

        # --- parse step 2: line 2 endpoints
        m2 = pattern_step2.search(solution)
        if m2:
            pred_line2_coor_wh = list(_to_float(m2.group(1), m2.group(2), m2.group(3), m2.group(4)))
            # Calculate norm_L2 for both orderings of points (P1, P2) vs (P2, P1)
            reward_s2 = max(
                cal_norm_L2_reward(
                    pred_line2_coor_wh,
                    [
                        gt_lm1_line2_wh[0],
                        gt_lm1_line2_wh[1],
                        gt_lm2_line2_wh[0],
                        gt_lm2_line2_wh[1],
                    ],
                    reward_mapping_func,
                ),
                cal_norm_L2_reward(
                    pred_line2_coor_wh,
                    [
                        gt_lm2_line2_wh[0],
                        gt_lm2_line2_wh[1],
                        gt_lm1_line2_wh[0],
                        gt_lm1_line2_wh[1],
                    ],
                    reward_mapping_func,
                ),
            )
        else:
            reward_s2 = 0

        # --- parse step 3: the target angle
        m3 = pattern_step3.search(solution)
        if m3:
            pred_angle = list(_to_float(m3.group(1)))
            reward_s3 = cal_MRE_reward(
                pred_angle,
                [gt_float[0]],
                reward_mapping_func,
            )
        else:
            reward_s3 = 0

        reward = np.mean([reward_s1, reward_s2, reward_s3])

    except Exception as e:
        print(f"Exception in cal_process_reward_angle_task_v2: {e}")
        reward = 0.0

    return reward


def cal_process_reward_v2(solution, ground_truth, reward_mapping_func="exp_decay", **kwargs):
    """
    Variant of cal_process_reward dispatching to the normalized L2 localization variants.

    Args:
        solution: model responses (text)
        ground_truth: ground truth string.

    Returns:
        a scalar reward
    """
    metric_type = kwargs.get("metric_type", None)
    assert metric_type is not None, "metric_type not found in kwargs"

    if metric_type == "distance":
        return cal_process_reward_distance_task_v2(solution, ground_truth, reward_mapping_func, **kwargs)
    elif metric_type == "angle":
        return cal_process_reward_angle_task_v2(solution, ground_truth, reward_mapping_func, **kwargs)
    else:
        raise ValueError(f"Unsupported metric_type: {metric_type}")


def cal_process_reward_distance_task_v3(solution, ground_truth, reward_mapping_func="exp_decay", **kwargs):
    """
    Variant of cal_process_reward_distance_task_v2 using max normalized L2 distance.

    Steps 1 & 2 use cal_norm_L2_max_reward instead of cal_norm_L2_reward.
    For single-point steps (distance task), v2 and v3 are numerically identical.
    Step 3 (distance estimation) is unchanged (cal_MRE_reward).

    Args:
        solution: model responses (text)
        ground_truth: ground truth string.

    Returns:
        a scalar reward

    NOTE:
        - The number of reasoning steps is hardcoded, see "pattern_step{1,2,3}"
        - The number of reasoning steps below is defined in COT_INSTRUCT_DISTANCE
          imported from medvision_bm.sft.sft_prompts
    """
    # Step patterns: reasoning + answer
    patterns_step = {}
    for k in range(1, 4):
        # NOTE: Use GROUP patterns to extract numeric values
        patterns_step[k] = rf"{PATTERNS_STEP_REASONING_DISTANCE[k]}\s*{PATTERNS_STEP_ANSWER_DISTANCE_GROUP[k]}"
    pattern_step1 = re.compile(patterns_step[1], re.DOTALL)
    pattern_step2 = re.compile(patterns_step[2], re.DOTALL)
    pattern_step3 = re.compile(patterns_step[3], re.DOTALL)

    # NOTE:
    # ------
    # The landmark coordinates are in (w, h) format, not the conventional (h, w) format for image space indexing.
    # Such conversion is achieved in the dataset building recipe from MedVision (https://github.com/YongchengYAO/MedVision)
    # ------
    gt_lm1_wh = np.array(kwargs.get("landmark_1_wh"))
    gt_lm2_wh = np.array(kwargs.get("landmark_2_wh"))

    # Extract ground truth coordinates
    gt_string = ground_truth.strip()
    gt_parts = [
        part.strip()
        for part in gt_string.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(",")
    ]
    gt_float = [float(part) for part in gt_parts if part]

    try:
        # NOTE: norm_L2_max reward should be used for normalized coordinates (step 1 and step 2)
        # NOTE: MRE reward should be used in distance estimation (step 3)

        # --- parse step 1: landmark 1 coordinate
        m1 = pattern_step1.search(solution)
        if m1:
            pred_lm1_wh = list(_to_float(m1.group(1), m1.group(2)))
            reward_s1 = cal_norm_L2_max_reward(
                pred_lm1_wh,
                [gt_lm1_wh[0], gt_lm1_wh[1]],
                reward_mapping_func,
            )
        else:
            reward_s1 = 0

        # --- parse step 2: landmark 2 coordinate
        m2 = pattern_step2.search(solution)
        if m2:
            pred_lm2_wh = list(_to_float(m2.group(1), m2.group(2)))
            reward_s2 = cal_norm_L2_max_reward(
                pred_lm2_wh,
                [gt_lm2_wh[0], gt_lm2_wh[1]],
                reward_mapping_func,
            )
        else:
            reward_s2 = 0

        # --- parse step 3: the target distance
        m3 = pattern_step3.search(solution)
        if m3:
            pred_distance = list(_to_float(m3.group(1)))
            reward_s3 = cal_MRE_reward(
                pred_distance,
                [gt_float[0]],
                reward_mapping_func,
            )
        else:
            reward_s3 = 0

        reward = np.mean([reward_s1, reward_s2, reward_s3])

    except Exception as e:
        print(f"Exception in cal_process_reward_distance_task_v3: {e}")
        reward = 0.0

    return reward


def cal_process_reward_angle_task_v3(solution, ground_truth, reward_mapping_func="exp_decay", **kwargs):
    """
    Variant of cal_process_reward_angle_task_v2 using max normalized L2 distance.

    Steps 1 & 2 use cal_norm_L2_max_reward: reward per step is determined by the
    worst-localized endpoint (max per-point distance) instead of the mean.
    Step 3 (angle estimation) is unchanged (cal_MRE_reward).

    Args:
        solution: model responses (text)
        ground_truth: ground truth string.

    Returns:
        a scalar reward

    NOTE:
        - The number of reasoning steps is hardcoded, see "pattern_step{1,2,3}"
        - The number of reasoning steps below is defined in COT_INSTRUCT_ANGLE
          imported from medvision_bm.sft.sft_prompts
    """
    # Step patterns: reasoning + answer
    patterns_step = {}
    for k in range(1, 4):
        # NOTE: Use GROUP patterns to extract numeric values
        patterns_step[k] = rf"{PATTERNS_STEP_REASONING_ANGLE[k]}\s*{PATTERNS_STEP_ANSWER_ANGLE_GROUP[k]}"
    pattern_step1 = re.compile(patterns_step[1], re.DOTALL)
    pattern_step2 = re.compile(patterns_step[2], re.DOTALL)
    pattern_step3 = re.compile(patterns_step[3], re.DOTALL)

    # NOTE:
    # ------
    # The landmark coordinates are in (w, h) format, not the conventional (h, w) format for image space indexing.
    # Such conversion is achieved in the dataset building recipe from MedVision (https://github.com/YongchengYAO/MedVision)
    # ------
    gt_lm1_line1_wh = np.array(kwargs.get("line_1_point_1_wh"))
    gt_lm2_line1_wh = np.array(kwargs.get("line_1_point_2_wh"))
    gt_lm1_line2_wh = np.array(kwargs.get("line_2_point_1_wh"))
    gt_lm2_line2_wh = np.array(kwargs.get("line_2_point_2_wh"))

    # Extract ground truth coordinates
    gt_string = ground_truth.strip()
    gt_parts = [
        part.strip()
        for part in gt_string.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(",")
    ]
    gt_float = [float(part) for part in gt_parts if part]

    try:
        # NOTE: norm_L2_max reward should be used for normalized coordinates (step 1 and step 2)
        # NOTE: MRE reward should be used in angle estimation (step 3)

        # --- parse step 1: line 1 endpoints
        m1 = pattern_step1.search(solution)
        if m1:
            pred_line1_coor_wh = list(_to_float(m1.group(1), m1.group(2), m1.group(3), m1.group(4)))
            # Calculate norm_L2_max for both orderings of points (P1, P2) vs (P2, P1)
            reward_s1 = max(
                cal_norm_L2_max_reward(
                    pred_line1_coor_wh,
                    [
                        gt_lm1_line1_wh[0],
                        gt_lm1_line1_wh[1],
                        gt_lm2_line1_wh[0],
                        gt_lm2_line1_wh[1],
                    ],
                    reward_mapping_func,
                ),
                cal_norm_L2_max_reward(
                    pred_line1_coor_wh,
                    [
                        gt_lm2_line1_wh[0],
                        gt_lm2_line1_wh[1],
                        gt_lm1_line1_wh[0],
                        gt_lm1_line1_wh[1],
                    ],
                    reward_mapping_func,
                ),
            )
        else:
            reward_s1 = 0

        # --- parse step 2: line 2 endpoints
        m2 = pattern_step2.search(solution)
        if m2:
            pred_line2_coor_wh = list(_to_float(m2.group(1), m2.group(2), m2.group(3), m2.group(4)))
            # Calculate norm_L2_max for both orderings of points (P1, P2) vs (P2, P1)
            reward_s2 = max(
                cal_norm_L2_max_reward(
                    pred_line2_coor_wh,
                    [
                        gt_lm1_line2_wh[0],
                        gt_lm1_line2_wh[1],
                        gt_lm2_line2_wh[0],
                        gt_lm2_line2_wh[1],
                    ],
                    reward_mapping_func,
                ),
                cal_norm_L2_max_reward(
                    pred_line2_coor_wh,
                    [
                        gt_lm2_line2_wh[0],
                        gt_lm2_line2_wh[1],
                        gt_lm1_line2_wh[0],
                        gt_lm1_line2_wh[1],
                    ],
                    reward_mapping_func,
                ),
            )
        else:
            reward_s2 = 0

        # --- parse step 3: the target angle
        m3 = pattern_step3.search(solution)
        if m3:
            pred_angle = list(_to_float(m3.group(1)))
            reward_s3 = cal_MRE_reward(
                pred_angle,
                [gt_float[0]],
                reward_mapping_func,
            )
        else:
            reward_s3 = 0

        reward = np.mean([reward_s1, reward_s2, reward_s3])

    except Exception as e:
        print(f"Exception in cal_process_reward_angle_task_v3: {e}")
        reward = 0.0

    return reward


def cal_process_reward_v3(solution, ground_truth, reward_mapping_func="exp_decay", **kwargs):
    """
    Variant of cal_process_reward dispatching to the max normalized L2 localization variants.

    Args:
        solution: model responses (text)
        ground_truth: ground truth string.

    Returns:
        a scalar reward
    """
    metric_type = kwargs.get("metric_type", None)
    assert metric_type is not None, "metric_type not found in kwargs"

    if metric_type == "distance":
        return cal_process_reward_distance_task_v3(solution, ground_truth, reward_mapping_func, **kwargs)
    elif metric_type == "angle":
        return cal_process_reward_angle_task_v3(solution, ground_truth, reward_mapping_func, **kwargs)
    else:
        raise ValueError(f"Unsupported metric_type: {metric_type}")


def cal_format_reward(solution, alpha=0.8, **kwargs):
    """
    Reward function that checks if the model response has a specific format.

    Args:
        solution: model responses (text)
        alpha: Weight for reasoning format reward. Defaults to 0.8.

    Returns:
        a scalar reward
    """
    reason_format_reward = match_reasoning(solution, **kwargs)
    answer_format_reward = match_answer(solution)
    reward = alpha * reason_format_reward + (1 - alpha) * answer_format_reward

    return reward


def cal_answer_reward(solution, ground_truth, reward_mapping_func="exp_decay", **kwargs):
    """
    Calculates the MRE (Mean Relative Error) reward from the extracted final answer from the model response (solution).

    Args:
        solution: model responses (text)
        ground_truth: ground truth strings
        reward_mapping_func: Reward mapping function name.

    Returns:
        a scalar reward
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
        pattern_answer = re.compile(PATTERN_ANSWER_1_VALUE_GROUP, re.DOTALL)
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
                return 0.0
        except Exception:
            return 0.0

    # Compute MRE reward
    if len(pred_float) != len(gt_float):
        reward = 0.0
    else:
        reward = cal_MRE_reward(pred_float, gt_float, reward_mapping_func)

    return reward


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

    format_reward = cal_format_reward(solution_str, **extra_info)
    process_reward = cal_process_reward(solution_str, ground_truth, reward_mapping_func="exp_decay", **extra_info)
    answer_reward = cal_answer_reward(solution_str, ground_truth, reward_mapping_func="exp_decay", **extra_info)
    reward = format_reward + process_reward + answer_reward

    return {
        "score": reward,
        "format_reward": format_reward,
        "process_reward": process_reward,
        "answer_reward": answer_reward,
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

    format_reward = cal_format_reward(solution_str, **extra_info)
    process_reward = cal_process_reward(solution_str, ground_truth, reward_mapping_func="exp_decay", **extra_info)
    answer_reward = cal_answer_reward(solution_str, ground_truth, reward_mapping_func="exp_decay", **extra_info)
    reward = format_reward + process_reward * answer_reward

    return {
        "score": reward,
        "format_reward": format_reward,
        "process_reward": process_reward,
        "answer_reward": answer_reward,
    }


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

    format_reward = cal_format_reward(solution_str, **extra_info)
    process_reward = cal_process_reward_v2(solution_str, ground_truth, reward_mapping_func="exp_decay", **extra_info)
    answer_reward = cal_answer_reward(solution_str, ground_truth, reward_mapping_func="exp_decay", **extra_info)
    reward = format_reward + process_reward * answer_reward

    return {
        "score": reward,
        "format_reward": format_reward,
        "process_reward": process_reward,
        "answer_reward": answer_reward,
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

    format_reward = cal_format_reward(solution_str, **extra_info)
    process_reward = cal_process_reward_v3(solution_str, ground_truth, reward_mapping_func="exp_decay", **extra_info)
    answer_reward = cal_answer_reward(solution_str, ground_truth, reward_mapping_func="exp_decay", **extra_info)
    reward = format_reward + process_reward * answer_reward

    return {
        "score": reward,
        "format_reward": format_reward,
        "process_reward": process_reward,
        "answer_reward": answer_reward,
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

    format_reward = cal_format_reward(solution_str, **extra_info)
    answer_reward = cal_answer_reward(solution_str, ground_truth, reward_mapping_func="exp_decay", **extra_info)
    reward = format_reward + answer_reward

    return {
        "score": reward,
        "format_reward": format_reward,
        "answer_reward": answer_reward,
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

    format_reward = cal_format_reward(solution_str, **extra_info)
    process_reward = cal_process_reward(solution_str, ground_truth, reward_mapping_func="scaled_sigmoid", **extra_info)
    answer_reward = cal_answer_reward(solution_str, ground_truth, reward_mapping_func="scaled_sigmoid", **extra_info)
    reward = format_reward + process_reward + answer_reward

    return {
        "score": reward,
        "format_reward": format_reward,
        "process_reward": process_reward,
        "answer_reward": answer_reward,
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

    format_reward = cal_format_reward(solution_str, **extra_info)
    process_reward = cal_process_reward(solution_str, ground_truth, reward_mapping_func="gaussian_proxy", **extra_info)
    answer_reward = cal_answer_reward(solution_str, ground_truth, reward_mapping_func="gaussian_proxy", **extra_info)
    reward = format_reward + process_reward + answer_reward

    return {
        "score": reward,
        "format_reward": format_reward,
        "process_reward": process_reward,
        "answer_reward": answer_reward,
    }
