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

"""Unit tests for the MedVision multi-task reward (`medvision_general.compute_score`) and its
parts: format variants, process reward, answer reward, composition and weights. CPU only."""

import math

import pytest

try:
    from verl.utils.reward_score.medvision_rewards import medvision_general as mg
    from verl.utils.reward_score.medvision_rewards import parsing
except Exception:
    # The full `verl` package pulls torch/tensordict, which these pure-Python modules do not need.
    # Register package stubs whose __path__ points at the real directory so the modules and their
    # sibling imports load from file without executing verl/__init__.py.
    import importlib
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

    _rewards_dir = (
        pathlib.Path(__file__).resolve().parents[3] / "verl" / "utils" / "reward_score" / "medvision_rewards"
    )
    for _name in ("verl", "verl.utils", "verl.utils.reward_score", "verl.utils.reward_score.medvision_rewards"):
        if _name not in sys.modules:
            _pkg = types.ModuleType(_name)
            _pkg.__path__ = []
            sys.modules[_name] = _pkg
    sys.modules["verl.utils.reward_score.medvision_rewards"].__path__ = [str(_rewards_dir)]
    mg = importlib.import_module("verl.utils.reward_score.medvision_rewards.medvision_general")
    parsing = importlib.import_module("verl.utils.reward_score.medvision_rewards.parsing")


REWARD_KEYS = {
    "score",
    "format_reward",
    "process_reward",
    "answer_reward",
    "answer_error",
    "localization_error",
    "measurement_error",
}

# --- samples ---------------------------------------------------------------------------------

TL_GT = "20.0, 10.0"
TL_INFO = {
    "ability": "medvision-tl",
    "landmark_P1_wh": [0.2, 0.5],
    "landmark_P2_wh": [0.8, 0.5],
    "landmark_P3_wh": [0.5, 0.35],
    "landmark_P4_wh": [0.5, 0.65],
}


def tl_solution(major="(0.2, 0.5), (0.8, 0.5)", minor="(0.5, 0.35), (0.5, 0.65)", l_major="20.0", l_minor="10.0", answer="20.0, 10.0"):
    return (
        "<think>"
        f"<step-1-reasoning>major axis endpoints</step-1-reasoning><step-1-answer>{major}</step-1-answer>"
        f"<step-2-reasoning>minor axis endpoints</step-2-reasoning><step-2-answer>{minor}</step-2-answer>"
        f"<step-3-reasoning>major axis length</step-3-reasoning><step-3-answer>{l_major}</step-3-answer>"
        f"<step-4-reasoning>minor axis length</step-4-reasoning><step-4-answer>{l_minor}</step-4-answer>"
        f"</think><answer>{answer}</answer>"
    )


ANGLE_GT = "90.0"
ANGLE_INFO = {
    "ability": "medvision-angle",
    "metric_type": "angle",
    "line_1_point_1_wh": [0.1, 0.1],
    "line_1_point_2_wh": [0.9, 0.1],
    "line_2_point_1_wh": [0.1, 0.1],
    "line_2_point_2_wh": [0.1, 0.9],
}


def angle_solution(line1="(0.1, 0.1), (0.9, 0.1)", line2="(0.1, 0.1), (0.1, 0.9)", angle="90.0", answer="90.0"):
    return (
        "<think>"
        f"<step-1-reasoning>line 1</step-1-reasoning><step-1-answer>{line1}</step-1-answer>"
        f"<step-2-reasoning>line 2</step-2-reasoning><step-2-answer>{line2}</step-2-answer>"
        f"<step-3-reasoning>angle</step-3-reasoning><step-3-answer>{angle}</step-3-answer>"
        f"</think><answer>{answer}</answer>"
    )


DIST_GT = "15.0"
DIST_INFO = {"ability": "medvision-distance", "metric_type": "distance", "landmark_1_wh": [0.2, 0.3], "landmark_2_wh": [0.6, 0.7]}


def dist_solution(p1="(0.2, 0.3)", p2="(0.6, 0.7)", dist="15.0", answer="15.0"):
    return (
        "<think>"
        f"<step-1-reasoning>landmark 1</step-1-reasoning><step-1-answer>{p1}</step-1-answer>"
        f"<step-2-reasoning>landmark 2</step-2-reasoning><step-2-answer>{p2}</step-2-answer>"
        f"<step-3-reasoning>distance</step-3-reasoning><step-3-answer>{dist}</step-3-answer>"
        f"</think><answer>{answer}</answer>"
    )


DET_GT = "0.1, 0.2, 0.5, 0.6"
DET_INFO = {"ability": "medvision-detection"}


def det_solution(answer="0.1, 0.2, 0.5, 0.6"):
    return f"<think>box</think><answer>{answer}</answer>"


def score(solution, gt, info, **kwargs):
    return mg.compute_score("src", solution, gt, info, **kwargs)


# --- defaults: soft format, multiplicative ---------------------------------------------------


class TestDefaults:
    @pytest.mark.parametrize(
        "solution,gt,info",
        [(tl_solution(), TL_GT, TL_INFO), (angle_solution(), ANGLE_GT, ANGLE_INFO), (dist_solution(), DIST_GT, DIST_INFO)],
    )
    def test_perfect_cot_rollout_scores_two(self, solution, gt, info):
        out = score(solution, gt, info)
        assert set(out) == REWARD_KEYS
        assert out["format_reward"] == pytest.approx(1.0)
        assert out["process_reward"] == pytest.approx(1.0)
        assert out["answer_reward"] == pytest.approx(1.0)
        assert out["score"] == pytest.approx(2.0)
        assert out["answer_error"] == pytest.approx(0.0)
        assert out["localization_error"] == pytest.approx(0.0) and out["measurement_error"] == pytest.approx(0.0)

    def test_perfect_detection_scores_two_without_process(self):
        out = score(det_solution(), DET_GT, DET_INFO)
        assert out["score"] == pytest.approx(2.0)
        assert out["process_reward"] == 0.0
        assert math.isnan(out["localization_error"]) and math.isnan(out["measurement_error"])

    def test_partial_errors_follow_exp_decay(self):
        # major length off by 10%: step 3 MRE 0.1; answer MRE mean(0.1, 0) = 0.05
        out = score(tl_solution(l_major="22.0", answer="22.0, 10.0"), TL_GT, TL_INFO)
        p = (3 + math.exp(-0.1)) / 4
        a = math.exp(-0.05)
        assert out["process_reward"] == pytest.approx(p)
        assert out["answer_reward"] == pytest.approx(a)
        assert out["score"] == pytest.approx(1 + p * a)
        assert out["measurement_error"] == pytest.approx(0.05)  # mean over the two measurement steps

    def test_unparsed_step_earns_zero_for_that_step(self):
        sol = tl_solution().replace("<step-3-answer>20.0</step-3-answer>", "<step-3-answer>?</step-3-answer>")
        out = score(sol, TL_GT, TL_INFO)
        assert out["process_reward"] == pytest.approx(0.75)
        assert out["measurement_error"] == pytest.approx(0.0)  # only step 4 parsed

    def test_unparseable_answer(self):
        out = score(tl_solution(answer="unknown"), TL_GT, TL_INFO)
        assert out["answer_reward"] == 0.0 and math.isnan(out["answer_error"])
        assert out["format_reward"] == pytest.approx(0.8)  # structure intact, answer check fails
        assert out["score"] == pytest.approx(0.8)  # multiplicative: process credit needs a parsed answer


# --- format reward variants ------------------------------------------------------------------


class TestFormatReward:
    def test_soft_default_and_binary(self):
        assert score(tl_solution(), TL_GT, TL_INFO)["format_reward"] == pytest.approx(1.0)
        assert score(tl_solution(), TL_GT, TL_INFO, format_reward="binary")["format_reward"] == 1

    def test_missing_think_block(self):
        sol = "<answer>20.0, 10.0</answer>"
        assert score(sol, TL_GT, TL_INFO)["format_reward"] == pytest.approx(0.2)
        assert score(sol, TL_GT, TL_INFO, format_reward="binary")["format_reward"] == 1

    def test_partial_structure_gets_partial_credit(self):
        sol = tl_solution().replace(
            "<step-3-reasoning>major axis length</step-3-reasoning><step-3-answer>20.0</step-3-answer>"
            "<step-4-reasoning>minor axis length</step-4-reasoning><step-4-answer>10.0</step-4-answer>",
            "",
        )
        f = score(sol, TL_GT, TL_INFO)["format_reward"]
        assert 0.2 < f < 1.0
        assert f == pytest.approx(0.8 * 9 / 17 + 0.2)

    @pytest.mark.parametrize("answer", ["20.0, 10.0", "(20.0, 10.0)", "[20.0, 10.0]", " ( 20.0 ,10.0 ) "])
    def test_binary_check_accepts_optional_brackets(self, answer):
        assert mg.cal_format_reward(tl_solution(answer=answer), format_reward="binary", **TL_INFO) == 1

    @pytest.mark.parametrize("answer", ["20.0", "20.0 10.0", "about 20 mm", "20.0, 10.0, 5.0"])
    def test_binary_check_rejects_malformed(self, answer):
        assert mg.cal_format_reward(tl_solution(answer=answer), format_reward="binary", **TL_INFO) == 0

    def test_detection_always_binary(self):
        assert score(det_solution(), DET_GT, DET_INFO, format_reward="soft")["format_reward"] == 1
        assert score("<answer>0.1 0.2 0.5 0.6</answer>", DET_GT, DET_INFO)["format_reward"] == 0


# --- composition -----------------------------------------------------------------------------


class TestComposition:
    SOL = tl_solution(l_major="22.0", answer="22.0, 10.0")

    def components(self, **kwargs):
        out = score(self.SOL, TL_GT, TL_INFO, **kwargs)
        return out, out["format_reward"], out["process_reward"], out["answer_reward"]

    def test_multiplicative_vs_additive(self):
        out_m, f, p, a = self.components()
        out_a, f2, p2, a2 = self.components(composition="additive")
        assert (f, p, a) == (f2, p2, a2)
        assert out_m["score"] == pytest.approx(f + p * a)
        assert out_a["score"] == pytest.approx(f + p + a)

    def test_additive_keeps_process_credit_without_answer(self):
        sol = tl_solution(answer="unknown")
        assert score(sol, TL_GT, TL_INFO, composition="additive")["score"] == pytest.approx(0.8 + 1.0)
        assert score(sol, TL_GT, TL_INFO)["score"] == pytest.approx(0.8)

    def test_detection_unaffected_by_composition(self):
        sol = det_solution("0.2, 0.3, 0.6, 0.7")
        assert score(sol, DET_GT, DET_INFO)["score"] == pytest.approx(score(sol, DET_GT, DET_INFO, composition="additive")["score"])

    @pytest.mark.parametrize("kwargs", [{"format_reward": "hard"}, {"composition": "sum"}, {"reward_mapping_func": "linear"}])
    def test_invalid_options_raise(self, kwargs):
        with pytest.raises(AssertionError):
            score(tl_solution(), TL_GT, TL_INFO, **kwargs)

    def test_requires_extra_info_and_ability(self):
        with pytest.raises(AssertionError):
            score(tl_solution(), TL_GT, None)
        with pytest.raises(AssertionError):
            score(tl_solution(), TL_GT, {"ability": "medvision-segmentation"})
        with pytest.raises(AssertionError):
            score(angle_solution(), ANGLE_GT, {k: v for k, v in ANGLE_INFO.items() if k != "metric_type"})


# --- answer reward ---------------------------------------------------------------------------


class TestAnswerReward:
    def test_detection_uses_ciou(self):
        pred = [0.2, 0.2, 0.5, 0.6]
        reward, err = mg.cal_answer_reward_error(det_solution("0.2, 0.2, 0.5, 0.6"), DET_GT, **DET_INFO)
        direct_reward, direct_err = mg.cal_ciou_reward_error(pred, [0.1, 0.2, 0.5, 0.6])
        assert err == pytest.approx(direct_err) and reward == pytest.approx(direct_reward)
        assert err != pytest.approx(0.25)  # not the coordinate MRE

    def test_measurement_uses_mre(self):
        reward, err = mg.cal_answer_reward_error(angle_solution(answer="99.0"), ANGLE_GT, **ANGLE_INFO)
        assert err == pytest.approx(0.1) and reward == pytest.approx(math.exp(-0.1))

    def test_tolerant_value_parsing(self):
        reward, err = mg.cal_answer_reward_error("<answer>major 20.0 mm, minor 10.0 mm</answer>", TL_GT, **TL_INFO)
        assert err == pytest.approx(0.0) and reward == pytest.approx(1.0)

    def test_ground_truth_arity_checked(self):
        with pytest.raises(AssertionError):
            mg.cal_answer_reward_error(tl_solution(), "20.0", **TL_INFO)


# --- parsing helpers -------------------------------------------------------------------------


class TestParsing:
    def test_parse_answer_strict_then_fallback(self):
        assert parsing.parse_answer("<answer>(1.5, 2)</answer>", 2) == [1.5, 2.0]
        assert parsing.parse_answer("<answer>a=1.5 and b=2</answer>", 2) == [1.5, 2.0]
        assert parsing.parse_answer("<answer>1.5</answer>", 2) is None
        assert parsing.parse_answer("no tags", 1) is None

    def test_parse_step(self):
        sol = tl_solution()
        assert parsing.parse_step(sol, 1, 2) == [0.2, 0.5, 0.8, 0.5]
        assert parsing.parse_step(sol, 3, 0) == [20.0]
        assert parsing.parse_step(sol, 5, 0) is None
        assert parsing.parse_step(dist_solution(), 1, 1) == [0.2, 0.3]

    def test_soft_format_score_bounds(self):
        assert parsing.soft_format_score("<answer>1</answer>", [2, 2, 0, 0]) == 0.0
        assert parsing.soft_format_score(tl_solution(), [2, 2, 0, 0]) == pytest.approx(1.0)
        assert parsing.soft_format_score(angle_solution(), [2, 2, 0]) == pytest.approx(1.0)
        assert parsing.soft_format_score(dist_solution(), [1, 1, 0]) == pytest.approx(1.0)

    def test_ground_truth_parsing(self):
        assert parsing.parse_ground_truth("(0.1, 0.2, 0.5, 0.6)") == [0.1, 0.2, 0.5, 0.6]
        assert parsing.parse_ground_truth("12.5,7.9") == [12.5, 7.9]
