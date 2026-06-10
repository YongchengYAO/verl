# Curriculum Sample Filtering (Online Hard-Example Mining)

Epoch-level sample filtering for GRPO RFT: samples the policy reliably solves are
progressively removed from training so each epoch concentrates rollout compute on the
cases the model still gets wrong, with a retention mix-in of solved samples that ramps
up as each task is solved (and a per-task floor so no task vanishes from training).

Implemented in this fork for the MedVision quantitative tasks (A/D, T/L, Detection);
the mechanism itself is task-agnostic as long as the reward function emits a
per-sample `answer_error`.

- Core logic: [`verl/utils/dataset/curriculum.py`](./verl/utils/dataset/curriculum.py)
- Dataset hooks: [`verl/utils/dataset/medvision_dataset.py`](./verl/utils/dataset/medvision_dataset.py)
- Trainer integration: [`verl/trainer/ppo/ray_trainer.py`](./verl/trainer/ppo/ray_trainer.py)
- Example recipe: [`examples/grpo_trainer/train__fullRFT__qwen25vl-7b-fullSFT__multiTask__512x512__PRxAnswer__curriculum__H200.sh`](./examples/grpo_trainer/train__fullRFT__qwen25vl-7b-fullSFT__multiTask__512x512__PRxAnswer__curriculum__H200.sh)
- Unit tests: [`tests/utils/dataset/test_curriculum_on_cpu.py`](./tests/utils/dataset/test_curriculum_on_cpu.py)

---

## Motivation

In GRPO the advantage is normalized **within the group of n rollouts of one prompt**.
Once the policy solves a prompt reliably, all n rollouts receive (nearly) the same
reward, the within-group variance collapses, and the advantage is ~0 — the sample
contributes almost no gradient while still consuming n full rollouts per draw. On the
MedVision mix (110K Detection / 5.5K T/L / 5.5K A/D), a large share of detection
prompts becomes "solved" early; filtering them redirects rollout budget to the samples
that still carry learning signal.

This is the same observation that motivates dynamic sampling in DAPO-style training
(filtering prompt groups whose rollouts are all-correct), implemented here as a
persistent epoch-level curriculum rather than a per-batch resampling loop.

## Algorithm

Pools are tracked **per task** (task = `ability` column, optionally merged via
`task_group_map`, identical to the temperature sampler's convention). For each task
`t` with `N_t` original samples:

- `hard[t]` — samples still being trained (initially all of them)
- `easy[t]` — samples classified as solved, ordered by promotion recency

During each epoch, every training step tallies, per sample index, the rollout-level
`score` (sequence reward = `token_level_scores.sum(-1)`, i.e. pre-KL) and
`answer_error` (final-answer MRE from the reward function). All rollouts of all draws
of a sample within the epoch aggregate into one mean each.

At the **end of each epoch**, the tally is first folded into a **persistent per-sample
evidence record** (see *Promotion and demotion evidence* below), then per task:

1. **Promote.** Rank the hard-pool samples drawn this epoch by **EMA score**
   (descending). Take the top `floor(easy_top_frac × |current training set of t|)`; of
   those, promote to `easy[t]` only the samples whose **EMA answer-MRE < mre_gate**
   *and* whose **consecutive-pass streak ≥ promote_patience**. A sample whose every
   rollout failed to parse has MRE = NaN and can never promote. Promoting zero samples
   is allowed.
2. **Demote (on by default).** A mixed-in or audited easy sample that has failed the
   gate for `demote_patience` consecutive observed epochs *and* whose EMA MRE has
   regressed past `demote_margin × mre_gate` (or that stopped parsing entirely — NaN
   counts as regressed) moves back to `hard[t]`, is retrained, and lowers the solved
   fraction again. This turns the mix-in into a rolling audit of the difficulty
   frontier and corrects false promotions from rollout luck. Set `demote_easy=False`
   to make the easy classification sticky.
3. **Ramped retention mix-in.** The mix-in starts with the **first promotion** and
   grows with the solved fraction `p_t = |easy[t]| / N_t`:

   ```
   easy_share_t = mixin_easy_frac × min(1, p_t / threshold_frac)
   mixin_slots  = round(|hard[t]| × easy_share_t / (1 − easy_share_t))
   ```

   filled with the **most-recently promoted** easy samples. Retention is therefore
   proportional to what needs retaining: a few slots after the first promotions,
   the full `mixin_easy_frac` share (default 30% easy / 70% hard) once
   `threshold_frac` (default 50%) of the task is solved, and constant past it.
   Compared to a binary phase flip this has no retention gap before the threshold,
   no one-epoch jump in training-set size at it, and no off/on flapping when
   demotions move the pool around the threshold. (`mixin_ramp=False` restores the
   binary behavior.) Because the slot count is derived from the hard count, the
   training set keeps shrinking as the hard pool shrinks.
4. **Per-task retention floor.** `active_t = hard[t] + mixin + audit slice` (below),
   then topped up with most-recently-promoted easy samples to at least
   `round(task_floor_frac × N_t)` (default 10%). Without the floor, a nearly-solved
   task's presence collapses with its hard pool — the mix-in is hard-proportional —
   and the temperature sampler rebalances over the collapsed pools, so the task can
   effectively vanish from training (**task extinction**) and drift unnoticed. The
   floor keeps every task represented as a forgetting tripwire: while truly solved,
   its GRPO groups have ~zero advantage and cost little gradient; if drift begins,
   within-group reward variance reappears and the gradient pulls back, and the
   demotion rule reopens real retraining.

The global active set is the union over tasks. A **minimum-size floor** (one
generation batch) tops up with most-recently-promoted easy samples if the union ever
falls below one batch — `drop_last=True` would otherwise produce a zero-step epoch.

Epoch 1 always trains the full dataset (there are no metrics yet); with the default
`promote_patience=2` the first promotions land at the end of epoch 2 and filtering
takes effect from epoch 3.

**Recency as a difficulty frontier.** Promotees are appended best-score-first, so the
end of `easy[t]` always holds the *lowest-margin* recently-solved samples. The
retention mix-in and the floor top-up draw from that end — the easy samples closest to
the model's current frontier, which are both the most valuable for retention and the
most likely false promotions, so they are also the first to be demotion-checked.

### Example dynamics

Simulated on a scaled-down MedVision mix (1100 Detection / 55 A/D / 55 T/L), default
parameters, sample "skill" improving over epochs:

```
epoch  0: active=1210   D(easy/hard)=0/1100     AD=0/55    TL=0/55    mixin=0    floor=0    <- patience: evidence only
epoch  1: active=1148   D(easy/hard)=220/880    AD=11/44   TL=11/44   mixin=132  floor=0    <- first promotions, ramp starts
epoch  3: active=  925   D(easy/hard)=517/583    AD=29/26   TL=21/34   mixin=250  floor=0    <- ramp near full strength
epoch  6: active=  525   D(easy/hard)=780/320    AD=37/18   TL=38/17   mixin=152  floor=0
epoch 11: active=  124   D(easy/hard)=1043/57    AD=50/5    TL=50/5    mixin=28   floor=26   <- Detection held at 10% of N_t
```

Note the absence of any jump: under the binary phase the active set leapt +146
samples in one epoch when Detection crossed 50%; the ramp grows the mix-in smoothly
from the first promotions. At epoch 11 the floor binds for Detection (57 hard +
24 mix-in + 3 audit = 84 < 110 = 0.10 × 1100), holding the task at 110 active.

## Promotion and demotion evidence (EMA + patience + hysteresis)

A single epoch gives only ~8 stochastic rollouts per drawn sample — a noisy base for a
permanent decision (a moderately hard sample can "get lucky" once and look solved).
Pool decisions therefore use a **persistent per-sample evidence record** that survives
across epochs and checkpoints:

```
stats[idx] = [ema_score, ema_mre, pass_streak, fail_streak]
```

At every epoch end, each drawn sample's epoch means are folded in:

- `ema_score / ema_mre` — exponential moving averages with weight `ema_alpha`
  (default 0.4): `ema ← α·epoch_mean + (1−α)·ema`. One bad epoch leaves a shadow that
  several good epochs must wash out before the gate opens. An all-unparsed epoch
  (NaN MRE) leaves the error EMA unchanged but still counts as a failure below.
- `pass_streak / fail_streak` — counts of *consecutive observed* epochs whose epoch-mean
  MRE passed / failed the gate. **Undrawn epochs hold the streaks** (absence of
  evidence is not evidence) — under replacement sampling a sample can legitimately go
  undrawn for an epoch.

Decision rules:

- **Promotion** requires `pass_streak ≥ promote_patience` (default 2 — "solved for two
  consecutive observed epochs") **and** `ema_mre < mre_gate`, on top of ranking in the
  top `easy_top_frac` by `ema_score`. The patience requirement is what averages out
  rollout luck; the EMA gate is what remembers history.
- **Demotion** requires `fail_streak ≥ demote_patience` (default 1) **and**
  `ema_mre ≥ demote_margin × mre_gate` (default 1.5 → demote at EMA MRE ≥ 0.15 while
  promotion needs < 0.10). The margin is classic **hysteresis**: a sample hovering at
  MRE ≈ 0.10 cannot oscillate hard→easy→hard across a single boundary.

The asymmetry (patience 2 up, patience 1 down) is deliberate: leaving training is
near-irreversible and demands strong evidence; returning to training is cheap, and a
forgotten sample should come back **fast** — audits visit any given easy sample rarely,
so requiring multiple failed audits would stretch the response time by many epochs.

Setting `ema_alpha=1, promote_patience=1, demote_patience=1, demote_margin=1,
audit_frac=0` reproduces the original single-epoch behavior exactly (the unit tests
pin this equivalence).

## Easy-pool auditing (rotating re-validation)

The retention mix-in only re-rolls the most-recently-promoted slice, so without extra
machinery the *older* part of the easy pool would never be re-validated: a forgotten
sample would be invisible to every training metric.

The **audit slice** closes this: every epoch,
`round(audit_frac × |hard_t|)` extra slots (default 5% of the hard pool) are filled
with easy samples chosen **least-recently-audited first** (round-robin by an
`last_audited` timestamp stored per easy entry, oldest promotion order breaking ties).
Audited samples train normally, get fresh rollouts, and feed the same demotion rule as
mixed-in samples.

Round-robin (rather than random) selection gives a hard staleness bound: every easy
sample is re-validated at least once every `|easy_t| / round(audit_frac × |hard_t|)`
epochs. Be honest about the magnitude: late in training (large easy pool, small hard
pool) that period grows long — e.g. 829 easy / 14 audit slots ≈ 60 epochs. The audit
prioritizes the stalest samples and bounds the blind spot; it does not eliminate it.
Raise `audit_frac` if validation curves suggest forgetting (each audit slot costs one
sample's rollouts per epoch).

## Choosing the defaults

| Knob | Default | Rationale |
|---|---|---|
| `mre_gate` | 0.10 | Matches the MedVision benchmark's own success criterion (MRE < 0.1): "easy" = "passes the paper's metric". |
| `easy_top_frac` | 0.20 | At most 20% of the (shrinking) set migrates per epoch → cumulative cap ≈ `1 − 0.8^k` after k promoting epochs, so the 50% mix-in threshold is reachable from ~epoch 4-5 and the curriculum fully engages within a 10-epoch run. For long runs or noisy rewards, 0.10 is the conservative choice (promotion competition is twice as strict). |
| `threshold_frac` | 0.50 | Solved fraction at which the ramped mix-in reaches full strength. Below it, retention scales proportionally — there is no longer a point where retention is absent. |
| `mixin_easy_frac` | 0.30 | Mid-range of replay ratios used against catastrophic forgetting (commonly 10–50%); the ramp's ceiling. |
| `mixin_ramp` | True | Ramp retention with the solved fraction instead of a binary phase flip: no retention gap before the threshold, no one-epoch training-set jump at it, no flapping when demotions cross it. `False` = legacy binary phase. |
| `task_floor_frac` | 0.10 | Anti-extinction: each task keeps ≥10% of its original samples active. A solved task's GRPO groups carry ~zero advantage (cheap), but keep the task in-distribution and trip the demotion rule if forgetting starts. |
| `ema_alpha` | 0.4 | Effective memory ≈ 2.5 epochs: smooths single-epoch rollout luck without reacting sluggishly to real improvement. `1.0` = no memory. |
| `promote_patience` | 2 | "Two consecutive passing epochs" — the standard cure for one-epoch noise. Costs one epoch of delayed shrink (first promotions at epoch 2 instead of 1). |
| `demote_patience` | 1 | Asymmetric on purpose: audits are infrequent per sample, so demotion must act on the first confirmed failure; hysteresis (not patience) is the churn guard. |
| `demote_margin` | 1.5 | Demote at EMA MRE ≥ 0.15 vs promote at < 0.10: wide enough that boundary samples don't oscillate, tight enough to catch genuine regressions. 2.0 would only catch gross forgetting. |
| `audit_frac` | 0.05 | ≈ 5% rollout overhead for continuous easy-pool re-validation. The staleness bound scales as `|easy|/(0.05·|hard|)` epochs — raise it for long runs with large easy pools. |

## Configuration

All options live under `data.curriculum` (plain `+` Hydra overrides, same pattern as
the temperature sampler — the structured `algorithm` config is not touched):

| Option | Default | Meaning |
|---|---|---|
| `+data.curriculum.enable` | `False` | Master switch; off = byte-identical to stock training |
| `+data.curriculum.easy_top_frac` | `0.20` | Top fraction of the *current* training set (per task) eligible for promotion each epoch |
| `+data.curriculum.mre_gate` | `0.10` | Mean answer-MRE a sample must beat to be classified easy |
| `+data.curriculum.threshold_frac` | `0.50` | Solved fraction at which the ramped mix-in reaches full `mixin_easy_frac` strength |
| `+data.curriculum.mixin_easy_frac` | `0.30` | Maximum easy share of a task's training set (the ramp's ceiling) |
| `+data.curriculum.mixin_ramp` | `True` | Ramp retention with the solved fraction (`False` = legacy binary phase at the threshold) |
| `+data.curriculum.task_floor_frac` | `0.10` | Minimum active fraction of each task's original size (anti task-extinction; `0` = off) |
| `+data.curriculum.demote_easy` | `True` | Re-demote mixed-in/audited easy samples that regress (`False` = easy is sticky) |
| `+data.curriculum.ema_alpha` | `0.4` | EMA weight for the per-sample score/error evidence (`1.0` = single-epoch evidence) |
| `+data.curriculum.promote_patience` | `2` | Consecutive passing (observed) epochs required before promotion |
| `+data.curriculum.demote_patience` | `1` | Consecutive failing audits required before demotion |
| `+data.curriculum.demote_margin` | `1.5` | Hysteresis: demote only at EMA MRE ≥ `margin × mre_gate` |
| `+data.curriculum.audit_frac` | `0.05` | Rotating easy-pool audit slots per epoch, as a fraction of the hard pool |
| `+data.curriculum.task_key` | `ability` | Dataset column used for per-task pooling |
| `+data.curriculum.task_group_map` | `""` | `label:group` merges, e.g. `'medvision-angle:AD,medvision-distance:AD'` |

Minimal launch addition:

```bash
+data.curriculum.enable=True \
+data.curriculum.task_key=ability \
"+data.curriculum.task_group_map='medvision-angle:AD,medvision-distance:AD'" \
```

**Requirements:**

- `+reward_model.use_reward_loop=True` — only the reward-loop path propagates the
  per-sample `answer_error` into the batch. The dataset hook raises a clear error
  otherwise.
- `data.custom_cls` = `MedVisionDataset` (or another dataset implementing
  `init_curriculum` / `on_batch_end` / `advance_curriculum`).
- `data.seed` set, if you want reproducible per-epoch sampler draws and exact
  mid-epoch resume.

## Interaction with the temperature multitask sampler

The curriculum **composes with** the temperature sampler
(`+data.temperature_sampler.*`) instead of replacing it. Each epoch the temperature
weights are recomputed **over the active subset only** (inactive samples get weight 0,
`num_samples = |active|`), so task-level rebalancing automatically adapts as the 110K
detection pool shrinks relative to the 5.5K A/D and T/L pools. With fewer than two
distinct tasks in the active set the rebuild falls back to a plain subset sampler,
mirroring `create_temperature_sampler`'s own fallback.

Rebuilt samplers are seeded with `data.seed + epoch`: a fixed seed would make every
epoch replay the identical draw once the sampler is reconstructed per epoch.

Because the temperature sampler draws **with replacement**, a sample may be drawn
several times in one epoch (all its rollouts aggregate into the means) or not at all
(it simply keeps its current pool that epoch). The same rule covers tail samples lost
to `drop_last=True`.

## Checkpointing and resume

Stock verl derives the epoch from `global_steps // len(dataloader)`, which is
meaningless once epoch lengths vary. With the curriculum enabled:

- Every checkpoint writes a **`curriculum.json`** beside `data.pt` containing the
  pools, the current epoch, the active index list, the **mid-epoch tally**, the
  **per-sample EMA/streak evidence**, the per-entry audit timestamps, and the step at
  which the current epoch started. (Checkpoints written before the evidence/audit
  extension load fine: missing stats start empty, audit timestamps default to the
  promotion epoch.)
- On resume, the pools are restored first and the dataloader is rebuilt for the saved
  epoch's exact active set and seed, *then* the dataloader state is restored — so
  mid-epoch resume fast-forwards correctly.
- A checkpoint taken at the **last step of an epoch** is detected from the curriculum
  state; the pool update that never ran is executed on resume from the restored tally,
  so no epoch's learning signal is lost.
- Resuming a checkpoint that **predates the curriculum** (no `curriculum.json`) warns,
  fast-forwards the epoch counter using the stock arithmetic, and starts the curriculum
  fresh from the full dataset.
- The curriculum state validates a per-task sample-count fingerprint on load: sample
  identity is the *post-filter* dataset row position, so resuming with changed data
  files or prompt-length filtering fails loudly instead of silently mismatching pools.

## Monitoring

Logged to wandb/console at each epoch boundary:

```
curriculum/{task}/easy_pool    # easy-pool size
curriculum/{task}/hard_pool    # hard-pool size
curriculum/{task}/active       # next epoch's training-set size for the task
curriculum/{task}/promoted     # promotions this epoch
curriculum/{task}/demoted      # demotions this epoch (always 0 with demote_easy=False)
curriculum/{task}/audited      # rotating easy-pool audit slots mixed in for next epoch
curriculum/{task}/mixin        # ramped retention mix-in slots for next epoch
curriculum/{task}/task_floor   # samples added by the per-task anti-extinction floor
curriculum/active_total        # next epoch's total training-set size
curriculum/floor_topup         # samples added by the minimum-batch floor
```

The console also prints `[Info] Curriculum epoch E: active set N samples, S steps`
after every rebuild. `curriculum.json` doubles as an offline analysis artifact —
which samples were learned at which epoch.

## Known limitations

- **LR schedule horizon.** verl precomputes `total_training_steps = len(dataloader) ×
  total_epochs` once, before any shrinking. A decaying schedule will therefore not
  fully decay (training ends earlier than the horizon); a constant LR (as used in the
  MedVision RFT recipes) is unaffected. The progress bar total over-counts for the
  same reason. Because `is_last_step` never fires, the trainer saves a final
  checkpoint explicitly when the epoch loop exits.
- **Step-frequency drift.** `save_freq` / `test_freq` are step-based, so saves and
  validations per epoch increase as epochs shrink (cosmetic).
- **Audit coverage is bounded, not complete.** The rotating audit slice guarantees
  every easy sample is re-validated within `|easy|/round(audit_frac × |hard|)` epochs,
  but late in training that period grows long (large easy pool, small hard pool). A
  sample forgotten between audits is invisible to training metrics until its turn
  comes — the (unfiltered) validation curves remain the global canary; raise
  `audit_frac` if they drift.
- **The task floor spends rollouts on solved tasks.** Once a task is essentially
  solved, `task_floor_frac × N_t` samples keep training on it even though their GRPO
  groups carry ~zero advantage most of the time — that is the price of task-level
  drift detection. Set `task_floor_frac=0` to reclaim the compute if task extinction
  is acceptable (e.g. single-task runs, where the global `min_active_size` floor
  already applies).
- **Patience delays the shrink.** `promote_patience=2` postpones all filtering by one
  epoch and slows the easy-pool growth curve — on very short runs (≲5 epochs) the
  curriculum may barely engage; drop to `promote_patience=1` there, accepting noisier
  promotion.
- **No "too hard" filtering.** Samples the policy *never* solves stay in the hard pool
  forever. In GRPO these all-fail groups also have ~zero advantage, so the late-stage
  hard pool can concentrate compute on unlearnable (or mislabeled) samples. DAPO-style
  dynamic sampling filters both ends; here only the easy end is filtered by design.
  If late-stage epochs plateau, inspect the hard-pool residue before adding epochs.

## Relation to curriculum learning practice

Strictly speaking this is **anti-curriculum / online hard-example mining** (emphasize
errors), not classic easy→hard curriculum learning (Bengio et al., 2009). For RL
post-training of LLMs the hard-example direction is the established practice — solved
prompts carry no GRPO gradient — and this design matches the community pattern of
difficulty-based data filtering (DAPO dynamic sampling, online difficulty filtering /
ZPD-style selection), with three deliberate differences:

1. **Persistent epoch-level pools** instead of per-batch resampling: no extra rollouts
   are spent probing filtered samples, at the cost of slower reaction to forgetting
   (mitigated by the retention mix-in, the rotating audit slice, and demotion).
2. **Reward-magnitude ranking with an MRE gate** instead of group pass-rate: MedVision
   rewards are continuous (`exp(-error)`), so "fraction of rollouts correct" is less
   informative than the EMA reward plus an explicit error threshold. The
   consecutive-pass streak (`promote_patience`) plays the role that multi-trial
   pass-rates play in pass-rate-based filtering.
3. **A ramped retention mix-in** (up to 30% easy / 70% hard, proportional to the
   solved fraction) plus a per-task minimum-presence floor, in the spirit of
   experience-replay ratios and rehearsal buffers used to limit catastrophic
   forgetting in continual learning.

## Implementation notes

- `CurriculumManager` is pure Python (no torch/numpy state) and fully unit-tested on
  CPU: `python -m pytest tests/utils/dataset/test_curriculum_on_cpu.py` (35 tests,
  including exact-equivalence pins for the legacy single-epoch mode).
- The per-step tally hooks into the pre-existing `train_dataset.on_batch_end(batch)`
  call in the trainer loop; with the flag off, the trainer behaves byte-identically
  (no manager is constructed, the stock dataloader code path is used, and
  `non_tensor_batch["index"]` keeps its upstream default).
- The stable sample identity is injected as `row_dict["index"]` only when the
  curriculum is active. Upstream's only other consumer of `index` is rollout-trace
  sampling, which actually benefits (it otherwise sees `index=0` for every MedVision
  row).
- Scores are taken from `token_level_scores` (pre-KL-penalty) so the easiness ranking
  reflects task reward, not the KL term.
