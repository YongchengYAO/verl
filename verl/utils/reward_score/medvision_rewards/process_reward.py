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

"""Process reward over the CoT steps (worst-point localization error)."""

import numpy as np

from verl.utils.reward_score.medvision_rewards.parsing import parse_step
from verl.utils.reward_score.medvision_rewards.reward_fn import (
    cal_MRE_error,
    cal_norm_L2_max_error,
    cal_reward_from_error_or_zero,
    safe_nanmean,
)


def cal_process_reward_error(solution, step_targets, reward_mapping_func="exp_decay"):
    """
    Process reward of one rollout: the mean over the CoT steps of ``rho(step error)``.

    ``step_targets[k-1]`` is the ground truth of step ``k``: a list of points ``[[x, y], ...]``
    for a localization step, or a scalar for a measurement step. A localization step is scored
    by its worst-localized point (Euclidean displacement in relative coordinates divided by
    sqrt(2), so the error lies in [0, 1]); a two-point step takes the better of the two point
    orderings. A measurement step is scored by its relative error. A missing or malformed step
    earns 0.

    Args:
        solution: model response (text).
        step_targets: per-step ground truth (see above).
        reward_mapping_func: error-to-reward map name (see reward_fn.cal_reward_from_error).

    Returns:
        (reward, localization_error, measurement_error): the process reward and the mean raw
        error over the localization / measurement steps (NaN when none parsed).
    """
    rewards, loc_errors, meas_errors = [], [], []
    for k, target in enumerate(step_targets, start=1):
        err = float("nan")
        try:
            if np.isscalar(target):  # measurement step
                pred = parse_step(solution, k, 0)
                if pred is not None:
                    err = cal_MRE_error(pred, [float(target)])
                meas_errors.append(err)
            else:  # localization step
                points = [[float(c) for c in p] for p in target]
                pred = parse_step(solution, k, len(points))
                if pred is not None:
                    orderings = [points, points[::-1]] if len(points) == 2 else [points]
                    err = min(cal_norm_L2_max_error(pred, [c for p in order for c in p]) for order in orderings)
                loc_errors.append(err)
        except Exception as e:  # never let one malformed sample take the reward loop down
            print(f"[Warning] process reward: step {k} could not be scored ({e}); counted as unparsed.")
            err = float("nan")
        rewards.append(cal_reward_from_error_or_zero(err, reward_mapping_func))
    return float(np.mean(rewards)), safe_nanmean(loc_errors), safe_nanmean(meas_errors)
