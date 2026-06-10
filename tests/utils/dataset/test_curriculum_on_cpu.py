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

"""Unit tests for the epoch-level curriculum sample filtering (pure logic, no GPU)."""

import math

import pytest

try:
    from verl.utils.dataset.curriculum import CurriculumManager, compute_active_sample_weights
except ImportError:
    # Allow running on machines without the full verl/torch stack: load the
    # module (and its temperature_sampler dependency) directly from file.
    import importlib.util
    import pathlib
    import sys
    import types

    _ds_dir = pathlib.Path(__file__).resolve().parents[3] / "verl" / "utils" / "dataset"
    for _name in ("verl", "verl.utils", "verl.utils.dataset"):
        if _name not in sys.modules:
            _pkg = types.ModuleType(_name)
            _pkg.__path__ = []
            sys.modules[_name] = _pkg
    sys.modules["verl.utils.dataset"].__path__ = [str(_ds_dir)]

    def _load(name):
        spec = importlib.util.spec_from_file_location(f"verl.utils.dataset.{name}", _ds_dir / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"verl.utils.dataset.{name}"] = mod
        spec.loader.exec_module(mod)
        return mod

    _load("temperature_sampler")
    _curriculum = _load("curriculum")
    CurriculumManager = _curriculum.CurriculumManager
    compute_active_sample_weights = _curriculum.compute_active_sample_weights


NAN = float("nan")


def make_manager(task_labels, **kwargs):
    # Legacy-mode parameters: single-epoch evidence, no hysteresis, no audits.
    # New-feature tests override these explicitly.
    defaults = dict(
        easy_top_frac=0.10,
        mre_gate=0.10,
        threshold_frac=0.50,
        mixin_easy_frac=0.30,
        demote_easy=False,
        min_active_size=1,
        ema_alpha=1.0,
        promote_patience=1,
        demote_patience=1,
        demote_margin=1.0,
        audit_frac=0.0,
        mixin_ramp=False,
        task_floor_frac=0.0,
    )
    defaults.update(kwargs)
    return CurriculumManager(task_labels, **defaults)


def record_each(manager, indices, scores, errors):
    """Record one rollout per index (vectorized call, like one training batch)."""
    manager.record(indices, scores, errors)


class TestInitialState:
    def test_initial_active_set_is_full_dataset(self):
        mgr = make_manager(["TL"] * 10)
        assert sorted(mgr.active_indices) == list(range(10))
        assert mgr.epoch == 0

    def test_zero_promotions_when_nothing_recorded(self):
        mgr = make_manager(["TL"] * 10)
        active = mgr.advance_epoch()
        assert sorted(active) == list(range(10))
        assert mgr.epoch == 1


class TestPromotion:
    def test_promotion_caps_at_top_fraction_of_current_set(self):
        # 20 samples, top 10% => floor(0.1*20)=2 promotions max even though
        # 5 samples pass the MRE gate.
        mgr = make_manager(["TL"] * 20)
        scores = [float(i) for i in range(20)]  # sample 19 best
        errors = [0.05 if i >= 15 else 0.5 for i in range(20)]  # 15..19 pass gate
        record_each(mgr, list(range(20)), scores, errors)
        active = mgr.advance_epoch()
        # top-2 by score among gate-passers: 19 and 18
        assert sorted(active) == sorted(set(range(20)) - {18, 19})

    def test_promotion_requires_mre_gate(self):
        mgr = make_manager(["TL"] * 20)
        scores = [float(i) for i in range(20)]
        errors = [0.5] * 20  # nobody passes the gate
        record_each(mgr, list(range(20)), scores, errors)
        active = mgr.advance_epoch()
        assert sorted(active) == list(range(20))

    def test_nan_mre_never_promotes(self):
        mgr = make_manager(["TL"] * 20)
        scores = [float(i) for i in range(20)]
        errors = [NAN if i == 19 else (0.05 if i == 18 else 0.5) for i in range(20)]
        record_each(mgr, list(range(20)), scores, errors)
        active = mgr.advance_epoch()
        # 19 has best score but NaN error (never parsed) -> stays hard
        assert 19 in active
        assert 18 not in active

    def test_unseen_samples_keep_their_pool(self):
        mgr = make_manager(["TL"] * 20)
        # only half the samples drawn this epoch
        record_each(mgr, list(range(10)), [float(i) for i in range(10)], [0.05] * 10)
        active = mgr.advance_epoch()
        # floor(0.1*20)=2 promoted from the seen ones (9 and 8 by score)
        assert sorted(active) == sorted(set(range(20)) - {8, 9})

    def test_repeated_draws_aggregate_all_rollouts(self):
        # Sample 0 drawn twice (replacement sampling): scores 10 and 0 -> mean 5.
        # Sample 1 drawn once with score 6 -> ranks above sample 0.
        mgr = make_manager(["TL"] * 10, easy_top_frac=0.10)  # floor(0.1*10)=1
        record_each(mgr, [0, 1], [10.0, 6.0], [0.05, 0.05])
        record_each(mgr, [0], [0.0], [0.05])
        active = mgr.advance_epoch()
        assert 1 not in active  # promoted
        assert 0 in active

    def test_nan_errors_ignored_in_mean(self):
        # One parsed rollout (mre 0.05) + one failed rollout (NaN) -> mean 0.05, passes gate.
        mgr = make_manager(["TL"] * 10, easy_top_frac=0.10)
        record_each(mgr, [0], [10.0], [0.05])
        record_each(mgr, [0], [10.0], [NAN])
        active = mgr.advance_epoch()
        assert 0 not in active


class TestPerTask:
    def test_per_task_isolation(self):
        # 10 TL + 10 AD. TL has high scores, AD low - but each task promotes
        # independently: floor(0.1*10)=1 each.
        labels = ["TL"] * 10 + ["AD"] * 10
        mgr = make_manager(labels)
        scores = [100.0 + i for i in range(10)] + [float(i) for i in range(10)]
        errors = [0.05] * 20
        record_each(mgr, list(range(20)), scores, errors)
        active = mgr.advance_epoch()
        assert 9 not in active  # best TL promoted
        assert 19 not in active  # best AD promoted (despite low absolute score)
        assert len(active) == 18


class TestMixinPhase:
    def test_phase_flip_and_mixin_ratio(self):
        # easy_top_frac=0.5 promotes 5/10 in one epoch -> easy pool hits the
        # 50% threshold -> mix-in phase: active = 5 hard + round(5*0.3/0.7)=2 easy.
        mgr = make_manager(["TL"] * 10, easy_top_frac=0.50)
        scores = [float(i) for i in range(10)]  # 5..9 promoted
        record_each(mgr, list(range(10)), scores, [0.05] * 10)
        active = mgr.advance_epoch()
        assert len(active) == 7
        hard = {0, 1, 2, 3, 4}
        assert hard <= set(active)
        # most-recently-added easy = appended last (lowest-ranked promotees: 6 then 5)
        assert set(active) - hard == {5, 6}

    def test_mixin_count_clamped_to_easy_pool(self):
        # threshold_frac=0 puts us in mix-in immediately after first promotion.
        mgr = make_manager(["TL"] * 10, easy_top_frac=0.10, threshold_frac=0.0)
        record_each(mgr, list(range(10)), [float(i) for i in range(10)], [0.05] * 10)
        active = mgr.advance_epoch()
        # hard=9 -> wants round(9*3/7)=4 easy but only 1 exists -> clamp to 1
        assert len(active) == 10

    def test_demote_easy_when_enabled(self):
        mgr = make_manager(["TL"] * 10, easy_top_frac=0.50, demote_easy=True)
        record_each(mgr, list(range(10)), [float(i) for i in range(10)], [0.05] * 10)
        mgr.advance_epoch()  # easy={5..9}, active = hard + {5,6}
        # mixed-in easy sample 6 regresses
        record_each(mgr, [6], [9.0], [0.9])
        active = mgr.advance_epoch()
        assert 6 in active  # demoted back to hard
        m = mgr.metrics()
        assert m["curriculum/TL/demoted"] == 1

    def test_no_demotion_when_disabled(self):
        mgr = make_manager(["TL"] * 10, easy_top_frac=0.50, demote_easy=False)
        record_each(mgr, list(range(10)), [float(i) for i in range(10)], [0.05] * 10)
        mgr.advance_epoch()
        record_each(mgr, [6], [9.0], [0.9])
        mgr.advance_epoch()
        assert mgr.metrics()["curriculum/TL/easy_pool"] == 5

    def test_demotion_enabled_by_default(self):
        # Bare constructor: demote_easy=True, promote_patience=2, demote_patience=1,
        # demote_margin=1.5, ema_alpha=0.4 are the defaults.
        mgr = CurriculumManager(["TL"] * 10, easy_top_frac=0.50)
        # promote_patience=2: two consecutive passing epochs before promotion
        record_each(mgr, list(range(10)), [float(i) for i in range(10)], [0.05] * 10)
        mgr.advance_epoch()
        assert mgr.metrics()["curriculum/TL/promoted"] == 0
        record_each(mgr, list(range(10)), [float(i) for i in range(10)], [0.05] * 10)
        mgr.advance_epoch()  # easy={5..9}, mixed in: {5, 6}
        assert mgr.metrics()["curriculum/TL/promoted"] == 5
        # regression far past the hysteresis band (ema_mre = 0.4*0.9 + 0.6*0.05 = 0.39 >= 0.15)
        record_each(mgr, [6], [9.0], [0.9])
        mgr.advance_epoch()
        assert mgr.metrics()["curriculum/TL/demoted"] == 1


class TestConstructorDefaults:
    def test_default_easy_top_frac_is_twenty_percent(self):
        mgr = CurriculumManager(["TL"] * 10)  # bare defaults: easy_top_frac=0.2, patience=2
        for _ in range(2):
            record_each(mgr, list(range(10)), [float(i) for i in range(10)], [0.05] * 10)
            mgr.advance_epoch()
        assert mgr.metrics()["curriculum/TL/promoted"] == 2  # floor(0.2 * 10)

    def test_default_ramp_and_task_floor(self):
        mgr = CurriculumManager(["TL"] * 10)
        assert mgr.mixin_ramp is True
        assert mgr.task_floor_frac == 0.10


class TestPromotionEvidence:
    def test_promote_patience_requires_consecutive_epochs(self):
        mgr = make_manager(["TL"] * 10, easy_top_frac=0.50, promote_patience=2)
        record_each(mgr, list(range(10)), [float(i) for i in range(10)], [0.05] * 10)
        mgr.advance_epoch()
        assert mgr.metrics()["curriculum/TL/easy_pool"] == 0  # one passing epoch is not enough
        record_each(mgr, list(range(10)), [float(i) for i in range(10)], [0.05] * 10)
        mgr.advance_epoch()
        assert mgr.metrics()["curriculum/TL/easy_pool"] == 5

    def test_streak_resets_on_failed_epoch(self):
        mgr = make_manager(["TL"] * 10, easy_top_frac=0.50, promote_patience=2)
        record_each(mgr, [0], [9.0], [0.05])
        mgr.advance_epoch()  # streak 1
        record_each(mgr, [0], [9.0], [0.50])
        mgr.advance_epoch()  # failed -> streak 0
        record_each(mgr, [0], [9.0], [0.05])
        mgr.advance_epoch()  # streak 1 again
        assert mgr.metrics()["curriculum/TL/easy_pool"] == 0

    def test_unseen_epoch_holds_streak(self):
        # Absence of evidence is not evidence: an undrawn epoch freezes the streak.
        mgr = make_manager(["TL"] * 10, easy_top_frac=0.50, promote_patience=2)
        record_each(mgr, [0], [9.0], [0.05])
        mgr.advance_epoch()  # streak 1
        mgr.advance_epoch()  # sample 0 unseen -> streak held at 1
        record_each(mgr, [0], [9.0], [0.05])
        mgr.advance_epoch()  # streak 2 -> promoted
        assert 0 not in mgr.active_indices

    def test_ema_mre_gates_promotion(self):
        # One bad epoch leaves a long EMA shadow: mre history 0.5 then 0.05 each epoch,
        # alpha=0.4 -> ema crosses below the 0.1 gate only at the 6th epoch.
        mgr = make_manager(["TL"] * 10, easy_top_frac=0.50, ema_alpha=0.4)
        record_each(mgr, [0], [9.0], [0.50])
        mgr.advance_epoch()
        for _ in range(4):  # ema: 0.32, 0.212, 0.1472, 0.10832 — all >= 0.1
            record_each(mgr, [0], [9.0], [0.05])
            mgr.advance_epoch()
            assert mgr.metrics()["curriculum/TL/easy_pool"] == 0
        record_each(mgr, [0], [9.0], [0.05])
        mgr.advance_epoch()  # ema = 0.084992 < 0.1
        assert mgr.metrics()["curriculum/TL/easy_pool"] == 1


class TestDemotionEvidence:
    def _promote_five(self, **kwargs):
        mgr = make_manager(["TL"] * 10, easy_top_frac=0.50, demote_easy=True, **kwargs)
        record_each(mgr, list(range(10)), [float(i) for i in range(10)], [0.05] * 10)
        mgr.advance_epoch()  # easy={5..9}, mixin={5, 6}
        return mgr

    def test_demotion_hysteresis_band(self):
        # gate=0.10, margin=1.5: MRE in [0.10, 0.15) fails the gate but is NOT demoted.
        mgr = self._promote_five(demote_margin=1.5)
        record_each(mgr, [6, 5], [9.0, 9.0], [0.12, 0.20])
        mgr.advance_epoch()
        assert mgr.metrics()["curriculum/TL/demoted"] == 1  # only sample 5 (0.20 >= 0.15)
        assert 5 in mgr.active_indices  # back in the hard pool

    def test_demote_patience_two_audits(self):
        mgr = self._promote_five(demote_patience=2)
        record_each(mgr, [6], [9.0], [0.50])
        mgr.advance_epoch()
        assert mgr.metrics()["curriculum/TL/demoted"] == 0  # one failed audit
        record_each(mgr, [6], [9.0], [0.50])
        mgr.advance_epoch()
        assert mgr.metrics()["curriculum/TL/demoted"] == 1  # two consecutive


class TestEasyPoolAudit:
    def test_pre_phase_audit_slice(self):
        # threshold not reached (pre phase) but audit_frac mixes in one easy sample.
        mgr = make_manager(["TL"] * 10, easy_top_frac=0.10, threshold_frac=0.90, audit_frac=0.10)
        record_each(mgr, list(range(10)), [float(i) for i in range(10)], [0.05] * 10)
        active = mgr.advance_epoch()  # 9 promoted; audit = round(0.1*9) = 1 slot
        assert 9 in active
        assert len(active) == 10
        assert mgr.metrics()["curriculum/TL/audited"] == 1

    def test_audit_rotates_least_recently_audited(self):
        mgr = make_manager(["TL"] * 10, easy_top_frac=0.50, threshold_frac=0.90, audit_frac=0.20)
        record_each(mgr, list(range(10)), [float(i) for i in range(10)], [0.05] * 10)
        active = mgr.advance_epoch()  # easy={9..5}; audit slots = round(0.2*5) = 1 -> sample 9
        assert 9 in active
        record_each(mgr, [9], [9.0], [0.05])  # only the audit member is drawn
        active = mgr.advance_epoch()
        assert 9 not in active  # freshly audited -> rotates out
        assert 8 in active  # least-recently-audited comes in


class TestMixinRamp:
    def test_ramp_starts_with_first_promotion(self):
        # p = 2/20 = 0.1 -> s = 0.3*(0.1/0.5) = 0.06 -> round(18*0.06/0.94) = 1 slot.
        mgr = make_manager(["TL"] * 20, mixin_ramp=True)
        record_each(mgr, list(range(20)), [float(i) for i in range(20)], [0.05] * 20)
        active = mgr.advance_epoch()  # promotes 19, 18
        assert len(active) == 19  # 18 hard + 1 ramped mix-in
        assert mgr.metrics()["curriculum/TL/mixin"] == 1
        assert 18 in active  # most-recently-promoted = lowest-ranked promotee

    def test_ramp_full_strength_at_threshold(self):
        # p = 0.5 = threshold -> s = mixin_easy_frac -> identical to the binary phase.
        mgr = make_manager(["TL"] * 10, easy_top_frac=0.50, mixin_ramp=True)
        record_each(mgr, list(range(10)), [float(i) for i in range(10)], [0.05] * 10)
        active = mgr.advance_epoch()
        assert len(active) == 7  # 5 hard + round(5*0.3/0.7) = 2 easy
        assert set(active) - {0, 1, 2, 3, 4} == {5, 6}

    def test_ramp_no_flapping_after_demotion(self):
        # Demotion drops p below the threshold; the binary phase would cut retention
        # to zero, the ramp shrinks it smoothly: p=0.4 -> s=0.24 -> round(6*0.24/0.76)=2.
        mgr = make_manager(["TL"] * 10, easy_top_frac=0.50, mixin_ramp=True, demote_easy=True)
        record_each(mgr, list(range(10)), [float(i) for i in range(10)], [0.05] * 10)
        mgr.advance_epoch()  # easy={5..9}, mixin={5, 6}
        record_each(mgr, [6], [9.0], [0.9])
        mgr.advance_epoch()  # 6 demoted -> easy=4/10
        assert mgr.metrics()["curriculum/TL/demoted"] == 1
        assert mgr.metrics()["curriculum/TL/mixin"] == 2


class TestTaskFloor:
    def test_task_floor_keeps_solved_task_present(self):
        # Whole task promoted -> hard=0 -> frontier=0; the floor keeps round(0.1*20)=2.
        mgr = make_manager(["TL"] * 20, easy_top_frac=1.0, mixin_ramp=True, task_floor_frac=0.10)
        record_each(mgr, list(range(20)), [float(i) for i in range(20)], [0.05] * 20)
        active = mgr.advance_epoch()
        assert len(active) == 2
        assert mgr.metrics()["curriculum/TL/task_floor"] == 2

    def test_task_floor_only_on_collapsed_task(self):
        # TL fully solved, AD untouched: floor tops up TL only.
        labels = ["TL"] * 20 + ["AD"] * 20
        mgr = make_manager(labels, easy_top_frac=1.0, mixin_ramp=True, task_floor_frac=0.10)
        record_each(mgr, list(range(20)), [float(i) for i in range(20)], [0.05] * 20)
        active = mgr.advance_epoch()
        assert mgr.metrics()["curriculum/TL/task_floor"] == 2
        assert mgr.metrics()["curriculum/AD/task_floor"] == 0
        assert len([i for i in active if i >= 20]) == 20  # AD fully active


class TestMinActiveFloor:
    def test_floor_tops_up_with_most_recent_easy(self):
        mgr = make_manager(["TL"] * 10, easy_top_frac=0.50, min_active_size=8)
        record_each(mgr, list(range(10)), [float(i) for i in range(10)], [0.05] * 10)
        active = mgr.advance_epoch()
        # without the floor: 5 hard + 2 easy = 7; floor=8 -> one extra easy
        assert len(active) == 8
        assert {0, 1, 2, 3, 4} <= set(active)


class TestStateDict:
    def test_state_dict_roundtrip(self):
        labels = ["TL"] * 10 + ["AD"] * 10
        mgr = make_manager(labels)
        record_each(mgr, list(range(20)), [float(i) for i in range(20)], [0.05] * 20)
        active = mgr.advance_epoch()

        state = mgr.state_dict()
        mgr2 = make_manager(labels)
        mgr2.load_state_dict(state)
        assert sorted(mgr2.active_indices) == sorted(active)
        assert mgr2.epoch == mgr.epoch
        # next epoch behaves identically
        record_each(mgr, [0], [1.0], [0.05])
        record_each(mgr2, [0], [1.0], [0.05])
        assert sorted(mgr.advance_epoch()) == sorted(mgr2.advance_epoch())

    def test_epoch_is_settable_for_legacy_resume(self):
        # Resuming a checkpoint that predates the curriculum has no curriculum.json;
        # the trainer fast-forwards the manager to the stock epoch count.
        mgr = make_manager(["TL"] * 10)
        mgr.epoch = 3
        assert mgr.epoch == 3
        mgr.advance_epoch()
        assert mgr.epoch == 4

    def test_fingerprint_mismatch_raises(self):
        mgr = make_manager(["TL"] * 10)
        state = mgr.state_dict()
        mgr2 = make_manager(["TL"] * 12)
        with pytest.raises(ValueError):
            mgr2.load_state_dict(state)

    def test_state_dict_preserves_midepoch_tally(self):
        # A checkpoint taken mid-epoch (or at the last step, before advance) must
        # keep the per-sample tally so the resumed run promotes identically.
        import json

        mgr = make_manager(["TL"] * 20)
        record_each(mgr, list(range(20)), [float(i) for i in range(20)], [0.05] * 20)
        state = json.loads(json.dumps(mgr.state_dict()))

        mgr2 = make_manager(["TL"] * 20)
        mgr2.load_state_dict(state)
        assert sorted(mgr2.advance_epoch()) == sorted(mgr.advance_epoch())
        assert mgr2.metrics()["curriculum/TL/promoted"] == 2

    def test_state_dict_preserves_streaks(self):
        # A promotion streak built before a checkpoint must survive the resume.
        mgr = make_manager(["TL"] * 10, easy_top_frac=0.50, promote_patience=2)
        record_each(mgr, [0], [9.0], [0.05])
        mgr.advance_epoch()  # streak 1
        mgr2 = make_manager(["TL"] * 10, easy_top_frac=0.50, promote_patience=2)
        mgr2.load_state_dict(mgr.state_dict())
        record_each(mgr2, [0], [9.0], [0.05])
        mgr2.advance_epoch()  # streak 2 -> promoted
        assert 0 not in mgr2.active_indices

    def test_loads_legacy_state_without_stats(self):
        # curriculum.json written before the evidence/audit extension: 3-field easy
        # entries, no "stats" key.
        mgr = make_manager(["TL"] * 10, easy_top_frac=0.50)
        record_each(mgr, list(range(10)), [float(i) for i in range(10)], [0.05] * 10)
        mgr.advance_epoch()
        state = mgr.state_dict()
        state["easy"] = {t: [e[:3] for e in entries] for t, entries in state["easy"].items()}
        state.pop("stats", None)
        mgr2 = make_manager(["TL"] * 10, easy_top_frac=0.50)
        mgr2.load_state_dict(state)
        assert sorted(mgr2.active_indices) == sorted(mgr.active_indices)
        record_each(mgr2, [0], [1.0], [0.05])
        mgr2.advance_epoch()  # must not crash on padded entries

    def test_state_dict_is_json_serializable(self):
        import json

        mgr = make_manager(["TL"] * 10, easy_top_frac=0.50)
        record_each(mgr, list(range(10)), [float(i) for i in range(10)], [0.05] * 10)
        mgr.advance_epoch()
        state = json.loads(json.dumps(mgr.state_dict()))
        mgr2 = make_manager(["TL"] * 10, easy_top_frac=0.50)
        mgr2.load_state_dict(state)
        assert sorted(mgr2.active_indices) == sorted(mgr.active_indices)


class TestMetrics:
    def test_metrics_report_pools_and_active(self):
        mgr = make_manager(["TL"] * 20)
        scores = [float(i) for i in range(20)]
        errors = [0.05 if i >= 15 else 0.5 for i in range(20)]
        record_each(mgr, list(range(20)), scores, errors)
        mgr.advance_epoch()
        m = mgr.metrics()
        assert m["curriculum/TL/easy_pool"] == 2
        assert m["curriculum/TL/hard_pool"] == 18
        assert m["curriculum/TL/promoted"] == 2
        assert m["curriculum/TL/active"] == 18
        assert m["curriculum/active_total"] == 18


class TestActiveSampleWeights:
    def test_inactive_samples_get_zero_weight(self):
        labels = ["TL"] * 4 + ["AD"] * 4
        weights = compute_active_sample_weights(labels, [0, 1, 4, 5], temperature=5.0)
        assert len(weights) == 8
        assert weights[2] == weights[3] == weights[6] == weights[7] == 0.0
        assert all(w > 0 for i, w in enumerate(weights) if i in (0, 1, 4, 5))

    def test_weights_renormalize_over_active_subset(self):
        # T=1 -> task prob proportional to ACTIVE counts -> uniform per active sample.
        labels = ["TL"] * 6 + ["AD"] * 2
        active = [0, 1, 2, 6, 7]  # 3 TL + 2 AD
        weights = compute_active_sample_weights(labels, active, temperature=1.0)
        active_weights = [weights[i] for i in active]
        assert all(math.isclose(w, active_weights[0], rel_tol=1e-9) for w in active_weights)

    def test_temperature_flattens_task_probs_on_subset(self):
        # T->inf: each task gets ~equal probability regardless of active counts.
        labels = ["TL"] * 9 + ["AD"]
        active = list(range(10))  # 9 TL vs 1 AD
        weights = compute_active_sample_weights(labels, active, temperature=1e9)
        tl_prob = sum(weights[:9])
        ad_prob = weights[9]
        assert math.isclose(tl_prob, ad_prob, rel_tol=1e-3)

    def test_single_task_uniform(self):
        labels = ["TL"] * 5
        weights = compute_active_sample_weights(labels, [0, 2, 4], temperature=5.0)
        assert weights[1] == weights[3] == 0.0
        assert math.isclose(weights[0], weights[2], rel_tol=1e-9)
        assert math.isclose(weights[0], weights[4], rel_tol=1e-9)
