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

"""Unit tests for the detection CIoU reward (pure geometry, no GPU/torch)."""

import math

import pytest

try:
    from verl.utils.reward_score.medvision_rewards.reward_fn import cal_ciou, cal_ciou_reward_error
except Exception:
    # The full `verl` package pulls torch/tensordict and reward_fn pulls scipy; neither is
    # needed for this pure-numpy module. Stub scipy.special.expit and load the file directly.
    import importlib.util
    import pathlib
    import sys
    import types

    import numpy as _np

    if "scipy" not in sys.modules:
        scipy = types.ModuleType("scipy")
        special = types.ModuleType("scipy.special")
        special.expit = lambda x: 1.0 / (1.0 + _np.exp(-x))
        scipy.special = special
        sys.modules["scipy"] = scipy
        sys.modules["scipy.special"] = special

    _path = (
        pathlib.Path(__file__).resolve().parents[3]
        / "verl"
        / "utils"
        / "reward_score"
        / "medvision_rewards"
        / "reward_fn.py"
    )
    _spec = importlib.util.spec_from_file_location("medvision_reward_fn", _path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    cal_ciou = _mod.cal_ciou
    cal_ciou_reward_error = _mod.cal_ciou_reward_error


NAN = float("nan")


class TestCalCiou:
    def test_identical_boxes_is_one(self):
        assert cal_ciou([0.0, 0.0, 0.2, 0.2], [0.0, 0.0, 0.2, 0.2]) == pytest.approx(1.0)

    def test_partial_overlap_known_value(self):
        # A=[0,0,2,2], B=[1,1,3,3]: IoU=1/7; center dist^2=2, enclosing diag^2=18 -> -2/18;
        # both squares -> aspect term 0. CIoU = 1/7 - 1/9 = 0.0317460.
        ciou = cal_ciou([0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0])
        assert ciou == pytest.approx(1.0 / 7.0 - 1.0 / 9.0, abs=1e-6)

    def test_disjoint_is_negative_and_monotonic_in_separation(self):
        # The dead-zone fix: plain IoU = 0 for all three, CIoU keeps decreasing.
        a = [0.0, 0.0, 2.0, 2.0]
        c1 = cal_ciou(a, [3.0, 0.0, 5.0, 2.0])
        c2 = cal_ciou(a, [6.0, 0.0, 8.0, 2.0])
        c3 = cal_ciou(a, [9.0, 0.0, 11.0, 2.0])
        assert c1 < 0 and c2 < 0 and c3 < 0
        assert c1 > c2 > c3

    def test_position_invariance(self):
        # Same overlap geometry, translated: CIoU identical (the MRE bias is gone).
        near = cal_ciou([0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0])
        far = cal_ciou([5.0, 5.0, 7.0, 7.0], [6.0, 6.0, 8.0, 8.0])
        assert near == pytest.approx(far, abs=1e-9)

    def test_centering_signal_under_containment(self):
        # GIoU degenerates to IoU when one box contains the other; CIoU still rewards centering.
        big = [0.0, 0.0, 4.0, 4.0]
        centered = cal_ciou(big, [1.5, 1.5, 2.5, 2.5])  # centers coincide
        off_center = cal_ciou(big, [1.0, 1.0, 2.0, 2.0])  # same size, shifted
        assert centered > off_center

    def test_degenerate_zero_area_box_is_finite(self):
        ciou = cal_ciou([1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 2.0, 2.0])
        assert math.isfinite(ciou)
        assert ciou <= 0.0  # no overlap, off in aspect

    def test_shape_mismatch_is_nan(self):
        assert math.isnan(cal_ciou([0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]))


class TestCalCiouRewardError:
    def test_perfect_box(self):
        reward, error = cal_ciou_reward_error([0.0, 0.0, 0.2, 0.2], [0.0, 0.0, 0.2, 0.2])
        assert error == pytest.approx(0.0, abs=1e-9)
        assert reward == pytest.approx(1.0, abs=1e-9)

    def test_reward_is_exp_decay_of_error(self):
        pred, gt = [0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]
        ciou = cal_ciou(pred, gt)
        reward, error = cal_ciou_reward_error(pred, gt)
        assert error == pytest.approx((1.0 - ciou) / 2.0, abs=1e-9)
        assert reward == pytest.approx(math.exp(-error), abs=1e-9)

    def test_reward_monotonic_decreasing_with_separation(self):
        a = [0.0, 0.0, 2.0, 2.0]
        r1, _ = cal_ciou_reward_error(a, [3.0, 0.0, 5.0, 2.0])
        r2, _ = cal_ciou_reward_error(a, [6.0, 0.0, 8.0, 2.0])
        r3, _ = cal_ciou_reward_error(a, [9.0, 0.0, 11.0, 2.0])
        assert r1 > r2 > r3 > 0.0  # monotonic, never a flat dead-zone

    def test_parse_failure_is_zero_nan(self):
        reward, error = cal_ciou_reward_error([0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0])
        assert reward == 0.0
        assert math.isnan(error)
