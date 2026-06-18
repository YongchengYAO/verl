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

"""
Epoch-level curriculum sample filtering for RL training (online hard-example mining).

At the end of each epoch, samples the policy reliably solves (top-`easy_top_frac`
of the current training set by EMA reward, gated on EMA answer-MRE < `mre_gate`
and a `promote_patience`-epoch consecutive-pass streak) move from the hard pool to
the easy pool; the next epoch trains on the hard pool only. A retention mix-in of
most-recently-promoted easy samples ramps up with the solved fraction, reaching the
full easy : hard = mixin_easy_frac : 1 - mixin_easy_frac mix once `threshold_frac`
of the task is solved, so the training set keeps shrinking with the hard pool while
limiting forgetting (no retention cliff, no phase flapping). Each task additionally
keeps at least `task_floor_frac` of its original samples active — a nearly-solved
task stays represented as a forgetting tripwire instead of vanishing under the
multitask sampler (task extinction). A rotating audit slice
(`audit_frac` x |hard| least-recently-audited easy samples) re-validates the easy
pool every epoch in both phases; regressed samples are demoted back to hard
(`demote_patience` failing audits past the `demote_margin` hysteresis band).

See CURRICULUM_FILTERING.md at the repo root for the full description.

Pools are tracked per task (e.g. AD / TL / Detection), consistent with the
temperature-based multitask sampler in temperature_sampler.py.

Enabled via config, e.g.:
    +data.curriculum.enable=True \
    +data.curriculum.task_key=ability \
    +data.curriculum.task_group_map='medvision-angle:AD,medvision-distance:AD'

Requires per-sample `answer_error` in the batch, which is only populated by the
reward-loop path (+reward_model.use_reward_loop=True).
"""

import math

from verl.utils.dataset.temperature_sampler import compute_temperature_sample_weights, parse_task_group_map


class CurriculumManager:
    """Tracks per-task easy/hard pools and builds the next epoch's active sample set."""

    def __init__(
        self,
        task_labels,
        *,
        easy_top_frac=0.20,
        mre_gate=0.10,
        threshold_frac=0.50,
        mixin_easy_frac=0.30,
        demote_easy=True,
        min_active_size=1,
        ema_alpha=0.4,
        promote_patience=1,
        demote_patience=1,
        demote_margin=1.5,
        audit_frac=0.05,
        mixin_ramp=True,
        task_floor_frac=0.10,
        gate_overrides=None,
    ):
        if not 0.0 < easy_top_frac <= 1.0:
            raise ValueError(f"easy_top_frac must be in (0, 1], got {easy_top_frac}")
        if not 0.0 <= mixin_easy_frac < 1.0:
            raise ValueError(f"mixin_easy_frac must be in [0, 1), got {mixin_easy_frac}")
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}")
        if promote_patience < 1 or demote_patience < 1:
            raise ValueError(f"patience must be >= 1, got {promote_patience}/{demote_patience}")
        if demote_margin < 1.0:
            raise ValueError(f"demote_margin must be >= 1 (hysteresis), got {demote_margin}")
        if not 0.0 <= audit_frac <= 1.0:
            raise ValueError(f"audit_frac must be in [0, 1], got {audit_frac}")
        if not 0.0 <= task_floor_frac <= 1.0:
            raise ValueError(f"task_floor_frac must be in [0, 1], got {task_floor_frac}")
        gate_overrides = dict(gate_overrides or {})
        for task, gate in gate_overrides.items():
            if not 0.0 < gate <= 1.0:
                raise ValueError(f"gate_overrides[{task}] must be in (0, 1], got {gate}")

        self.task_labels = [str(label) for label in task_labels]
        self.n_samples = len(self.task_labels)
        self.easy_top_frac = float(easy_top_frac)
        self.mre_gate = float(mre_gate)
        self.threshold_frac = float(threshold_frac)
        self.mixin_easy_frac = float(mixin_easy_frac)
        self.demote_easy = bool(demote_easy)
        self.min_active_size = int(min_active_size)
        self.ema_alpha = float(ema_alpha)
        self.promote_patience = int(promote_patience)
        self.demote_patience = int(demote_patience)
        self.demote_margin = float(demote_margin)
        self.audit_frac = float(audit_frac)
        self.mixin_ramp = bool(mixin_ramp)
        self.task_floor_frac = float(task_floor_frac)
        # Per-task promotion/demotion gate (task label -> gate). Tasks not listed use
        # mre_gate. Detection uses an overlap-error gate (1-CIoU)/2 instead of MRE<gate.
        self.gate_overrides = {str(t): float(g) for t, g in gate_overrides.items()}

        self.task_indices = {}
        for i, label in enumerate(self.task_labels):
            self.task_indices.setdefault(label, []).append(i)
        self.tasks = list(self.task_indices.keys())

        # easy[t]: list of [index, epoch_added, seq, last_audited];
        # append order = promotion recency.
        self.easy = {t: [] for t in self.tasks}
        self.hard = {t: set(idx) for t, idx in self.task_indices.items()}
        self._active = {t: list(idx) for t, idx in self.task_indices.items()}
        self._epoch = 0
        self._next_seq = 0
        # per-epoch tally: index -> [score_sum, n_rollouts, mre_sum, mre_count]
        self._tally = {}
        # persistent per-sample evidence: index -> [ema_score, ema_mre, pass_streak, fail_streak]
        self._stats = {}
        self._last_metrics = {}
        # per-task transition lists from the most recent advance_epoch (for pool_snapshot)
        self._last_detail = {}
        self._last_global_floor = []

    @property
    def epoch(self):
        """Number of completed epochs (= index of the epoch currently training)."""
        return self._epoch

    @epoch.setter
    def epoch(self, value):
        # Used when resuming a checkpoint that predates the curriculum: the trainer
        # fast-forwards to the stock epoch count so completed epochs are not re-run.
        self._epoch = int(value)

    @property
    def active_indices(self):
        """The current epoch's training set (sorted global dataset indices)."""
        return sorted(i for indices in self._active.values() for i in indices)

    def record(self, indices, scores, answer_errors):
        """Accumulates per-rollout metrics for samples seen this epoch (vectorized per batch)."""
        for idx, score, err in zip(indices, scores, answer_errors, strict=True):
            entry = self._tally.setdefault(int(idx), [0.0, 0, 0.0, 0])
            entry[0] += float(score)
            entry[1] += 1
            err = float(err)
            if not math.isnan(err):
                entry[2] += err
                entry[3] += 1

    def _mean_score(self, idx):
        entry = self._tally[idx]
        return entry[0] / entry[1]

    def _mean_mre(self, idx):
        entry = self._tally[idx]
        return entry[2] / entry[3] if entry[3] > 0 else float("nan")

    def _update_stats(self):
        """Folds this epoch's tally into the persistent per-sample evidence record.

        Samples not drawn this epoch are untouched (streaks hold; absence of evidence
        is not evidence). Streaks count consecutive *observed* epochs.
        """
        for idx in self._tally:
            score = self._mean_score(idx)
            mre = self._mean_mre(idx)
            stat = self._stats.get(idx)
            if stat is None:
                stat = self._stats[idx] = [score, mre, 0, 0]
            else:
                alpha = self.ema_alpha
                stat[0] = alpha * score + (1.0 - alpha) * stat[0]
                if not math.isnan(mre):
                    # an all-unparsed epoch leaves the error EMA unchanged
                    stat[1] = mre if math.isnan(stat[1]) else alpha * mre + (1.0 - alpha) * stat[1]
            if not math.isnan(mre) and mre < self._gate_for(idx):
                stat[2] += 1
                stat[3] = 0
            else:
                stat[2] = 0
                stat[3] += 1

    def _gate_for(self, idx):
        """Promotion gate for a sample's task (per-task override, else the base mre_gate)."""
        return self.gate_overrides.get(self.task_labels[idx], self.mre_gate)

    def _should_promote(self, idx):
        stat = self._stats[idx]
        return stat[2] >= self.promote_patience and not math.isnan(stat[1]) and stat[1] < self._gate_for(idx)

    def _should_demote(self, idx):
        stat = self._stats[idx]
        if stat[3] < self.demote_patience:
            return False
        # hysteresis: demotion requires regressing past demote_margin * gate,
        # strictly worse than the promotion gate, so borderline samples don't oscillate
        return math.isnan(stat[1]) or stat[1] >= self.demote_margin * self._gate_for(idx)

    def advance_epoch(self):
        """Reclassifies pools from this epoch's evidence and returns the next epoch's active set."""
        self._update_stats()
        metrics = {}
        for task in self.tasks:
            n_task = len(self.task_indices[task])
            promoted, demoted = [], []

            # Demotion of regressed easy samples (only mixed-in/audited ones were drawn).
            if self.demote_easy:
                kept = []
                for entry in self.easy[task]:
                    idx = entry[0]
                    if idx in self._tally and self._should_demote(idx):
                        self.hard[task].add(idx)
                        demoted.append(idx)
                    else:
                        kept.append(entry)
                self.easy[task] = kept

            # Promotion: top-k of the current training set by EMA score, gated on the
            # EMA error and a consecutive-pass streak. Only samples drawn this epoch
            # are candidates.
            cap = math.floor(self.easy_top_frac * len(self._active[task]))
            seen_hard = [i for i in self.hard[task] if i in self._tally]
            top = sorted(seen_hard, key=lambda i: -self._stats[i][0])[:cap]
            for idx in top:
                if self._should_promote(idx):
                    self.hard[task].discard(idx)
                    self.easy[task].append([idx, self._epoch, self._next_seq, self._epoch])
                    self._next_seq += 1
                    promoted.append(idx)

            # Refresh audit timestamps for easy samples drawn this epoch.
            for entry in self.easy[task]:
                if entry[0] in self._tally:
                    entry[3] = self._epoch

            # Build next active set: hard pool, plus a ramped recent-easy mix-in, plus a
            # rotating audit slice of least-recently-audited easy, plus the task floor.
            active = sorted(self.hard[task])
            solved_frac = len(self.easy[task]) / n_task
            if self.mixin_ramp:
                # easy share ramps from 0 at the first promotion up to mixin_easy_frac
                # at threshold_frac solved (no retention cliff, no phase flapping)
                ramp = solved_frac / self.threshold_frac if self.threshold_frac > 0 else 1.0
                easy_share = self.mixin_easy_frac * min(1.0, ramp)
            else:
                easy_share = self.mixin_easy_frac if solved_frac >= self.threshold_frac else 0.0
            frontier = []
            if easy_share > 0:
                n_hard = len(self.hard[task])
                want = round(n_hard * easy_share / (1.0 - easy_share))
                take = min(want, len(self.easy[task]))
                if take > 0:
                    frontier = [entry[0] for entry in self.easy[task][-take:]]
            audit = []
            audit_count = round(self.audit_frac * len(self.hard[task]))
            if audit_count > 0:
                in_frontier = set(frontier)
                pool = [entry for entry in self.easy[task] if entry[0] not in in_frontier]
                pool.sort(key=lambda entry: (entry[3], entry[2]))  # stalest audit first
                audit = [entry[0] for entry in pool[:audit_count]]
            self._active[task] = active + frontier + audit
            # Per-task retention floor: a nearly-solved task keeps a minimum training
            # presence (forgetting tripwire), instead of vanishing as its hard pool
            # (and the hard-pool-proportional mix-in) collapses.
            extras = []
            floor_size = min(n_task, round(self.task_floor_frac * n_task))
            if len(self._active[task]) < floor_size:
                included = set(self._active[task])
                extras = [entry[0] for entry in reversed(self.easy[task]) if entry[0] not in included]
                extras = extras[: floor_size - len(self._active[task])]
                self._active[task] += extras

            self._last_detail[task] = {
                "promoted": promoted,
                "demoted": demoted,
                "mixin": frontier,
                "audit": audit,
                "floor_topup": extras,
            }
            metrics[f"curriculum/{task}/easy_pool"] = len(self.easy[task])
            metrics[f"curriculum/{task}/hard_pool"] = len(self.hard[task])
            metrics[f"curriculum/{task}/active"] = len(self._active[task])
            metrics[f"curriculum/{task}/promoted"] = len(promoted)
            metrics[f"curriculum/{task}/demoted"] = len(demoted)
            metrics[f"curriculum/{task}/audited"] = len(audit)
            metrics[f"curriculum/{task}/mixin"] = len(frontier)
            metrics[f"curriculum/{task}/task_floor"] = len(extras)

        self._apply_min_active_floor(metrics)
        metrics["curriculum/active_total"] = sum(len(a) for a in self._active.values())
        self._last_metrics = metrics
        self._tally = {}
        self._epoch += 1
        return self.active_indices

    def _apply_min_active_floor(self, metrics):
        """Tops up the global active set with most-recent easy samples to keep >= one batch."""
        self._last_global_floor = []
        total = sum(len(a) for a in self._active.values())
        shortfall = self.min_active_size - total
        if shortfall <= 0:
            metrics["curriculum/floor_topup"] = 0
            return
        in_active = {i for a in self._active.values() for i in a}
        spare = [
            (entry[2], task, entry[0])
            for task in self.tasks
            for entry in self.easy[task]
            if entry[0] not in in_active
        ]
        spare.sort(reverse=True)  # highest seq = most recently promoted
        for _, task, idx in spare[:shortfall]:
            self._active[task].append(idx)
            self._last_global_floor.append(idx)
        metrics["curriculum/floor_topup"] = min(shortfall, len(spare))

    def metrics(self):
        """Per-task pool/promotion stats from the most recent advance_epoch call."""
        return dict(self._last_metrics)

    def pool_snapshot(self):
        """Full pool membership for offline inspection (JSON-serializable, read-only).

        The trainer dumps one snapshot per epoch boundary to
        {trainer.default_local_dir}/curriculum_pools/epoch_{epoch:04d}.json, where
        `epoch` is the epoch the returned active set trains (0 = the initial
        all-hard state). Per task: full easy entries ([idx, epoch_added, seq,
        last_audited], append order = promotion recency), the sorted hard pool, the
        active set, and this boundary's transition lists (promoted / demoted /
        mixin / audit / floor_topup; empty before the first advance). `stats` is
        the per-sample evidence record (idx -> [ema_score, ema_mre, pass_streak,
        fail_streak]).
        """
        detail_keys = ("promoted", "demoted", "mixin", "audit", "floor_topup")
        tasks = {}
        for task in self.tasks:
            detail = self._last_detail.get(task, {})
            tasks[task] = {
                "easy": [list(entry) for entry in self.easy[task]],
                "hard": sorted(self.hard[task]),
                "active": list(self._active[task]),
                **{key: list(detail.get(key, [])) for key in detail_keys},
            }
        return {
            "epoch": self._epoch,
            "tasks": tasks,
            "floor_topup_global": list(self._last_global_floor),
            "stats": {str(idx): list(stat) for idx, stat in self._stats.items()},
        }

    def state_dict(self):
        """JSON-serializable state for checkpoint/resume."""
        return {
            "epoch": self._epoch,
            "next_seq": self._next_seq,
            "task_counts": {t: len(idx) for t, idx in self.task_indices.items()},
            "easy": {t: [list(entry) for entry in entries] for t, entries in self.easy.items()},
            "active": {t: list(a) for t, a in self._active.items()},
            # mid-epoch tally so a resumed epoch promotes identically (JSON keys are str)
            "tally": {str(idx): list(entry) for idx, entry in self._tally.items()},
            # persistent EMA/streak evidence
            "stats": {str(idx): list(stat) for idx, stat in self._stats.items()},
        }

    def load_state_dict(self, state):
        task_counts = {t: len(idx) for t, idx in self.task_indices.items()}
        if dict(state["task_counts"]) != task_counts:
            raise ValueError(
                f"Curriculum state does not match the dataset: checkpoint task counts "
                f"{state['task_counts']} vs dataset {task_counts}. The data files or "
                f"prompt-length filtering likely changed; cannot resume the curriculum."
            )
        self._epoch = int(state["epoch"])
        self._next_seq = int(state["next_seq"])
        self.easy = {t: [list(entry) for entry in state["easy"][t]] for t in self.tasks}
        for task in self.tasks:
            for entry in self.easy[task]:
                if len(entry) == 3:  # pre-audit checkpoint format: no last_audited field
                    entry.append(entry[1])
            easy_idx = {entry[0] for entry in self.easy[task]}
            self.hard[task] = set(self.task_indices[task]) - easy_idx
        self._active = {t: [int(i) for i in state["active"][t]] for t in self.tasks}
        self._tally = {int(idx): list(entry) for idx, entry in state.get("tally", {}).items()}
        self._stats = {int(idx): list(stat) for idx, stat in state.get("stats", {}).items()}


def compute_active_sample_weights(task_labels, active_indices, temperature):
    """
    Temperature sample weights restricted to the active subset.

    Weights are recomputed over the active samples only (so task probabilities
    track the shrinking per-task pools); inactive samples get weight 0.

    Returns a list of len(task_labels) weights.
    """
    active_labels = [str(task_labels[i]) for i in active_indices]
    active_weights, _, _ = compute_temperature_sample_weights(active_labels, temperature)
    weights = [0.0] * len(task_labels)
    for idx, weight in zip(active_indices, active_weights, strict=True):
        weights[idx] = weight
    return weights


def build_active_sampler(data_config, dataset, active_indices, epoch):
    """
    Builds the train sampler for one curriculum epoch over `active_indices`.

    Composes with the temperature multitask sampler when enabled (weights
    recomputed on the active subset); otherwise falls back to a subset random /
    sequential sampler. Rebuilt samplers are seeded with data.seed + epoch so
    each epoch draws differently while staying reproducible.
    """
    import torch
    from torch.utils.data import SubsetRandomSampler, WeightedRandomSampler

    seed = data_config.get("seed")
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(int(seed) + int(epoch))

    ts_cfg = data_config.get("temperature_sampler", None)
    if ts_cfg is not None and ts_cfg.get("enable", False):
        task_key = ts_cfg.get("task_key", "ability")
        group_map = parse_task_group_map(ts_cfg.get("task_group_map", None))
        raw_labels = [str(label) for label in dataset.dataframe[task_key]]
        task_labels = [group_map.get(label, label) for label in raw_labels]
        # Mirror create_temperature_sampler: with <2 distinct tasks in the active set
        # there is nothing to rebalance, so fall through to the plain subset sampler
        # (keeps without-replacement semantics consistent with the epoch-0 fallback).
        if len({task_labels[i] for i in active_indices}) > 1:
            weights = compute_active_sample_weights(task_labels, active_indices, float(ts_cfg.get("T")))
            return WeightedRandomSampler(
                weights=torch.DoubleTensor(weights),
                num_samples=len(active_indices),
                replacement=True,
                generator=generator,
            )

    if data_config.get("shuffle", True):
        return SubsetRandomSampler(sorted(active_indices), generator=generator)
    return sorted(active_indices)
