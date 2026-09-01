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

"""MedVision A/D (angle / distance measurement) task: CoT step layout, soft format score,
process reward. ``extra_info["metric_type"]`` selects the variant.

Distance steps: 1 landmark 1 ``(x, y)``; 2 landmark 2; 3 distance. Ground truth: ``landmark_1_wh``,
``landmark_2_wh``. Angle steps: 1 line-1 endpoints ``(x1, y1), (x2, y2)``; 2 line-2 endpoints;
3 angle. Ground truth: ``line_1_point_1_wh``, ``line_1_point_2_wh``, ``line_2_point_1_wh``,
``line_2_point_2_wh``. Coordinates are relative ``(w, h)``; the ground-truth string holds the
measurement.
"""

from verl.utils.reward_score.medvision_rewards.parsing import parse_ground_truth, soft_format_score
from verl.utils.reward_score.medvision_rewards.process_reward import cal_process_reward_error as _process

POINTS_PER_STEP = {"distance": [1, 1, 0], "angle": [2, 2, 0]}


def _metric_type(kwargs):
    metric_type = kwargs.get("metric_type")
    assert metric_type in POINTS_PER_STEP, f"[Error] metric_type must be one of {list(POINTS_PER_STEP)}, got {metric_type!r}."
    return metric_type


def match_reasoning(solution, **kwargs):
    """Soft reasoning-structure score in [0, 1] (see parsing.soft_format_score)."""
    return soft_format_score(solution, POINTS_PER_STEP[_metric_type(kwargs)])


def step_targets(ground_truth, **kwargs):
    value = parse_ground_truth(ground_truth)[0]
    if _metric_type(kwargs) == "distance":
        return [[kwargs["landmark_1_wh"]], [kwargs["landmark_2_wh"]], value]
    return [
        [kwargs["line_1_point_1_wh"], kwargs["line_1_point_2_wh"]],
        [kwargs["line_2_point_1_wh"], kwargs["line_2_point_2_wh"]],
        value,
    ]


def cal_process_reward_error(solution, ground_truth, reward_mapping_func="exp_decay", **kwargs):
    """(reward, localization_error, measurement_error) over the three A/D steps."""
    return _process(solution, step_targets(ground_truth, **kwargs), reward_mapping_func)
