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
from scipy.special import expit


def scaled_sigmoid(x, k=10):
    """
    Computes a scaled sigmoid function.

    Args:
        x: Input value.

    Returns:
        Scaled sigmoid value.
    """
    return expit(-k * (x - 1 / 2))


def gaussian_proxy(x, var=0.5):
    """
    Computes a Gaussian proxy reward.

    Args:
        x: Input error value.
        var: Variance parameter. Defaults to 0.5.

    Returns:
        Gaussian reward value.
    """
    return np.exp(-0.5 * x**2 / var)


def exp_decay(x, k=1):
    """
    Computes an exponential decay reward.

    Args:
        x: Input error value.
        k: Decay rate. Defaults to 1.

    Returns:
        Exponential decay value.
    """
    return np.exp(-k * x)


def extract_last_k_nums(text, k):
    """
    Parses the last k numbers from the given text.

    Args:
        text: Input text.
        k: Number of numbers to retrieve.

    Returns:
        Comma-separated string of the last k numbers, or empty string if fewer than k found.
    """
    # Find all numbers in the text
    numbers = re.findall(r"-?\d+\.?\d*", text)

    # Return the last two numbers
    if len(numbers) < k:
        return ""
    return ",".join(numbers[-k:])


def cal_reward_from_error(error, reward_mapping_func="exp_decay"):
    """
    Calculates the reward based on the error and the specified mapping function.

    Args:
        error: The calculated error (e.g., MAE, MRE).
        reward_mapping_func: The name of the reward mapping function to use.
                                   Options: 'scaled_sigmoid', 'gaussian_proxy', 'exp_decay'.

    Returns:
        The calculated reward.

    Raises:
        AssertionError: If an unknown reward mapping function is provided, or if
            gaussian_proxy_var is missing when needed.
    """
    assert reward_mapping_func in [
        "scaled_sigmoid",
        "gaussian_proxy",
        "exp_decay",
    ], (
        f"Unknown reward mapping function: {reward_mapping_func}. "
        "Supported functions are: ['scaled_sigmoid', 'gaussian_proxy', 'exp_decay']"
    )

    if reward_mapping_func == "scaled_sigmoid":
        return scaled_sigmoid(error)
    elif reward_mapping_func == "gaussian_proxy":
        return gaussian_proxy(error)
    elif reward_mapping_func == "exp_decay":
        return exp_decay(error)
    return 0.0


def cal_MRE_reward(
    pred_float,
    gt_float,
    reward_mapping_func="exp_decay",
):
    """
    Calculates the Mean Relative Error (MRE) reward.

    Args:
        pred_float: Predicted values.
        gt_float: Ground truth values.
        reward_mapping_func: Reward mapping function name.

    Returns:
        A tuple containing (reward, mre).
    """
    # Convert inputs to numpy arrays if they aren't already
    try:
        if not isinstance(pred_float, np.ndarray):
            pred_float = np.array(pred_float)
        if not isinstance(gt_float, np.ndarray):
            gt_float = np.array(gt_float)
    except Exception as e:
        raise ValueError(f"Error converting model answer and GT to numpy array: {str(e)}") from e

    if len(pred_float) != len(gt_float):
        return 0.0
    else:
        mre = np.mean(np.abs(pred_float - gt_float) / (gt_float + 1e-15))
        reward = cal_reward_from_error(mre, reward_mapping_func)
        return reward


def cal_MAE_reward(
    pred_float,
    gt_float,
    reward_mapping_func="exp_decay",
):
    """
    Calculates the Mean Absolute Error (MAE) reward.

    Args:
        pred_float: Predicted values.
        gt_float: Ground truth values.
        reward_mapping_func: Reward mapping function name.

    Returns:
        The calculated reward.
    """
    # Convert inputs to numpy arrays if they aren't already
    try:
        if not isinstance(pred_float, np.ndarray):
            pred_float = np.array(pred_float)
        if not isinstance(gt_float, np.ndarray):
            gt_float = np.array(gt_float)
    except Exception as e:
        raise ValueError(f"Error converting model answer and GT to numpy array: {str(e)}") from e

    if len(pred_float) != len(gt_float):
        return 0.0
    else:
        mae = np.mean(np.abs(pred_float - gt_float))
        reward = cal_reward_from_error(mae, reward_mapping_func)
        return reward


def cal_norm_L2_reward(
    pred_xy_flat,
    gt_xy_flat,
    reward_mapping_func="exp_decay",
):
    """
    Reward based on mean normalized L2 distance for 2D point(s).

    Inputs are flat coordinate lists in normalized [0, 1] image space:
      - single point:  [x, y]
      - two endpoints: [x1, y1, x2, y2]

    For each (x, y) pair: dist = sqrt(dx^2 + dy^2) / sqrt(2).
    Error = mean over all point pairs. Range: [0, 1].

    Args:
        pred_xy_flat: Predicted flat coordinate list.
        gt_xy_flat: Ground truth flat coordinate list.
        reward_mapping_func: Reward mapping function name.

    Returns:
        The calculated reward.
    """
    try:
        pred = np.array(pred_xy_flat, dtype=float)
        gt = np.array(gt_xy_flat, dtype=float)
    except Exception as e:
        raise ValueError(f"Error converting inputs to numpy array: {str(e)}") from e

    if len(pred) != len(gt) or len(pred) % 2 != 0 or len(pred) == 0:
        return 0.0

    n_points = len(pred) // 2
    dists = []
    for i in range(n_points):
        dx = pred[2 * i] - gt[2 * i]
        dy = pred[2 * i + 1] - gt[2 * i + 1]
        dists.append(np.sqrt(dx**2 + dy**2) / np.sqrt(2))
    error = float(np.mean(dists))
    return cal_reward_from_error(error, reward_mapping_func)


def cal_norm_L2_max_reward(
    pred_xy_flat,
    gt_xy_flat,
    reward_mapping_func="exp_decay",
):
    """
    Reward based on max normalized L2 distance across 2D point(s).

    Same as cal_norm_L2_reward but aggregates per-point distances with max instead of mean.
    For a single point the two functions are equivalent; for two endpoints this is stricter —
    the reward is determined by the worst-localized endpoint.

    Inputs are flat coordinate lists in normalized [0, 1] image space:
      - single point:  [x, y]
      - two endpoints: [x1, y1, x2, y2]

    For each (x, y) pair: dist = sqrt(dx^2 + dy^2) / sqrt(2).
    Error = max over all point pairs. Range: [0, 1].

    Args:
        pred_xy_flat: Predicted flat coordinate list.
        gt_xy_flat: Ground truth flat coordinate list.
        reward_mapping_func: Reward mapping function name.

    Returns:
        The calculated reward.
    """
    try:
        pred = np.array(pred_xy_flat, dtype=float)
        gt = np.array(gt_xy_flat, dtype=float)
    except Exception as e:
        raise ValueError(f"Error converting inputs to numpy array: {str(e)}") from e

    if len(pred) != len(gt) or len(pred) % 2 != 0 or len(pred) == 0:
        return 0.0

    n_points = len(pred) // 2
    dists = []
    for i in range(n_points):
        dx = pred[2 * i] - gt[2 * i]
        dy = pred[2 * i + 1] - gt[2 * i + 1]
        dists.append(np.sqrt(dx**2 + dy**2) / np.sqrt(2))
    error = float(np.max(dists))
    return cal_reward_from_error(error, reward_mapping_func)
