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


def safe_nanmean(values):
    """
    Mean over the non-NaN entries of `values`; NaN if all entries are NaN (no RuntimeWarning).

    Args:
        values: Iterable of floats (may contain NaN).

    Returns:
        The mean of valid entries, or NaN if none.
    """
    valid = [v for v in values if not np.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


def cal_reward_from_error_or_zero(error, reward_mapping_func="exp_decay"):
    """
    Maps an error to a reward; returns 0.0 if the error is NaN (e.g. parse/shape failure).

    Args:
        error: The calculated error (may be NaN).
        reward_mapping_func: The name of the reward mapping function to use.

    Returns:
        The calculated reward.
    """
    if np.isnan(error):
        return 0.0
    return cal_reward_from_error(error, reward_mapping_func)


def cal_MRE_error(pred_float, gt_float):
    """
    Calculates the Mean Relative Error (MRE).

    Args:
        pred_float: Predicted values.
        gt_float: Ground truth values.

    Returns:
        The MRE, or NaN on length mismatch.
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
        return float("nan")
    return np.mean(np.abs(pred_float - gt_float) / (gt_float + 1e-15))


def cal_ciou(pred_box, gt_box):
    """
    Complete-IoU (CIoU) between two axis-aligned bounding boxes.

    Boxes are flat ``[x1, y1, x2, y2]`` in any consistent units (the MedVision detection
    answer uses relative [0, 1] coordinates, lower-left + upper-right). Corner order is
    normalized internally (min/max), so a corner-swapped prediction is still scored.

        CIoU = IoU - center_dist^2 / enclosing_diag^2 - alpha * v

    where ``v`` measures aspect-ratio inconsistency and ``alpha`` down-weights it while
    overlap is poor. Unlike plain IoU (flat 0 for any non-overlap), CIoU keeps decreasing
    as boxes separate, so it provides a usable signal everywhere.

    Range: [-1, 1] (1 = identical boxes). Returns NaN on shape mismatch.
    """
    eps = 1e-7
    try:
        p = np.asarray(pred_box, dtype=float)
        g = np.asarray(gt_box, dtype=float)
    except Exception:
        return float("nan")
    if p.shape != (4,) or g.shape != (4,):
        return float("nan")

    px1, px2 = min(p[0], p[2]), max(p[0], p[2])
    py1, py2 = min(p[1], p[3]), max(p[1], p[3])
    gx1, gx2 = min(g[0], g[2]), max(g[0], g[2])
    gy1, gy2 = min(g[1], g[3]), max(g[1], g[3])

    pw, ph = px2 - px1, py2 - py1
    gw, gh = gx2 - gx1, gy2 - gy1

    # intersection / union -> IoU
    iw = max(0.0, min(px2, gx2) - max(px1, gx1))
    ih = max(0.0, min(py2, gy2) - max(py1, gy1))
    inter = iw * ih
    union = pw * ph + gw * gh - inter
    iou = inter / union if union > eps else 0.0

    # normalized center distance
    center_d2 = ((px1 + px2) / 2.0 - (gx1 + gx2) / 2.0) ** 2 + ((py1 + py2) / 2.0 - (gy1 + gy2) / 2.0) ** 2
    enclose_d2 = (max(px2, gx2) - min(px1, gx1)) ** 2 + (max(py2, gy2) - min(py1, gy1)) ** 2
    center_term = center_d2 / enclose_d2 if enclose_d2 > eps else 0.0

    # aspect-ratio consistency
    v = (4.0 / (np.pi**2)) * (np.arctan(gw / (gh + eps)) - np.arctan(pw / (ph + eps))) ** 2
    denom = (1.0 - iou) + v
    alpha = v / denom if denom > eps else 0.0

    return float(iou - center_term - alpha * v)


def cal_ciou_reward_error(pred_box, gt_box, reward_mapping_func="exp_decay"):
    """
    Detection answer reward + error from CIoU.

    Returns ``(reward, error)`` where ``error = (1 - CIoU) / 2 in [0, 1]`` (lower is
    better, same direction as MRE so the curriculum gate is unchanged in shape) and
    ``reward = exp(-error)`` via the shared exp-decay map. Shape mismatch -> ``(0.0, NaN)``.
    """
    ciou = cal_ciou(pred_box, gt_box)
    error = (1.0 - ciou) / 2.0
    return cal_reward_from_error_or_zero(error, reward_mapping_func), error


def _cal_norm_L2_dists(pred_xy_flat, gt_xy_flat):
    """
    Per-point normalized L2 distances for flat coordinate lists, or None on shape mismatch.
    """
    try:
        pred = np.array(pred_xy_flat, dtype=float)
        gt = np.array(gt_xy_flat, dtype=float)
    except Exception as e:
        raise ValueError(f"Error converting inputs to numpy array: {str(e)}") from e

    if len(pred) != len(gt) or len(pred) % 2 != 0 or len(pred) == 0:
        return None

    n_points = len(pred) // 2
    dists = []
    for i in range(n_points):
        dx = pred[2 * i] - gt[2 * i]
        dy = pred[2 * i + 1] - gt[2 * i + 1]
        dists.append(np.sqrt(dx**2 + dy**2) / np.sqrt(2))
    return dists


def cal_norm_L2_max_error(pred_xy_flat, gt_xy_flat):
    """
    Max normalized L2 distance across 2D point(s).

    Per-point distances (sqrt(dx^2 + dy^2) / sqrt(2), in [0, 1]) aggregated with max: for two
    endpoints the error is determined by the worst-localized endpoint.

    Args:
        pred_xy_flat: Predicted flat coordinate list.
        gt_xy_flat: Ground truth flat coordinate list.

    Returns:
        The error, or NaN on shape mismatch.
    """
    dists = _cal_norm_L2_dists(pred_xy_flat, gt_xy_flat)
    if dists is None:
        return float("nan")
    return float(np.max(dists))
