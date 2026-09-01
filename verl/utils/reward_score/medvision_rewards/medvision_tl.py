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

"""MedVision T/L (tumor/lesion size) task: CoT step layout, soft format score, process reward.

Steps (``<step-k-answer>``): 1 major-axis endpoints ``(x1, y1), (x2, y2)``; 2 minor-axis
endpoints; 3 major-axis length; 4 minor-axis length. Ground-truth landmarks come from
``extra_info`` (``landmark_P1_wh`` .. ``landmark_P4_wh``: P1/P2 major axis, P3/P4 minor axis,
relative ``(w, h)`` coordinates); the ground-truth string holds ``major, minor`` lengths.
"""

from verl.utils.reward_score.medvision_rewards.parsing import parse_ground_truth, soft_format_score
from verl.utils.reward_score.medvision_rewards.process_reward import cal_process_reward_error as _process

POINTS_PER_STEP = [2, 2, 0, 0]


def match_reasoning(solution, **kwargs):
    """Soft reasoning-structure score in [0, 1] (see parsing.soft_format_score)."""
    return soft_format_score(solution, POINTS_PER_STEP)


def step_targets(ground_truth, **kwargs):
    major, minor = parse_ground_truth(ground_truth)[:2]
    return [
        [kwargs["landmark_P1_wh"], kwargs["landmark_P2_wh"]],
        [kwargs["landmark_P3_wh"], kwargs["landmark_P4_wh"]],
        major,
        minor,
    ]


def cal_process_reward_error(solution, ground_truth, reward_mapping_func="exp_decay", **kwargs):
    """(reward, localization_error, measurement_error) over the four T/L steps."""
    return _process(solution, step_targets(ground_truth, **kwargs), reward_mapping_func)
