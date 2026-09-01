#!/usr/bin/env python3
"""Visualise the verl curriculum-filtering pools of a MedVision RFT run.

Reads the per-epoch snapshots written by ``CurriculumManager.pool_snapshot()``
(``{trainer.default_local_dir}/curriculum_pools/epoch_NNNN.json``) and renders
three views of the curriculum, all showing change *across epochs*:

  (a) per-task trajectory of three nested frontiers vs. epoch -- ever-evaluated,
      gate-clearing (EMA error below the promotion gate), and mastered (promoted
      to the easy pool). The gap from gate-clearing to mastered is the backlog the
      20%/epoch promotion cap holds back; the gap from evaluated to gate-clearing
      is sampling coverage. All three are shares of the whole task.
  (b) the training set handed to each epoch, split into hard pool / retention
      mix-in / audit slice, against the full dataset size;
  (c) the per-sample EMA answer-error distribution per task, one outline per
      epoch (latest filled) against its gate, so the distribution's leftward
      march (improving accuracy) is visible.

Snapshots whose ``stats`` are empty (the epoch-0 all-hard baseline) contribute a
zero point to (a) and are skipped in (c). Promotion gates are not stored in the
snapshot; pass the values the training run used (``+data.curriculum.mre_gate``
and ``+data.curriculum.detection_gate``).

Note (a)/(c) read *cumulative* EMA evidence: a sample's ``stats`` entry is its
running EMA, so a curve at epoch N reflects all epochs up to N, not epoch N alone.

Usage:
    python curriculum-learning/plot_curriculum_pools.py \
        --pool_dir <run_dir>/curriculum_pools [--out_dir <dir>]
"""

import argparse
import glob
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Project figure convention: 300 dpi, transparent background, and a per-image
# ceiling of 34 megapixels (arXiv's limit, enforced by plot_utils.save_fig_capped).
# This script stays standalone rather than importing that helper -- medvision_bm's
# package __init__ pulls the whole dataset stack -- so the cap is asserted below.
FIG_DPI, MAX_FIG_MP = 300, 34

# Task display names and plot order (paper ordering: Detection, A/D, T/L).
TASK_DISPLAY = {"medvision-detection": "Detection", "AD": "A/D", "medvision-tl": "T/L"}
TASK_ORDER = ["medvision-detection", "AD", "medvision-tl"]

# Categorical hues: the MedVision brand hue angles (nature_palette_2 blue / clay /
# green) re-stepped in OKLCH so the trio clears the data-viz checks -- lightness
# band, chroma floor, CVD separation (worst all-pairs dE 11.6 protan/deutan) and
# the normal-vision floor (dE 22.6). The brand pastels themselves sit below the
# chroma floor and separate by only dE 8.9, too close for three series.
TASK_COLORS = {"medvision-detection": "#145a98", "AD": "#a43944", "medvision-tl": "#1a9d78"}

# Neutral one-hue ramp for the training-set composition (ordinal: distance from
# the hard pool), kept out of the categorical hues so shade never reads as task.
RAMP = ["#414c53", "#78868f", "#b0bbc1"]
INK, MUTED, GRID, SURFACE = "#2b2b29", "#6f6f6b", "#dcdcd8", "#ffffff"

ERR_FLOOR, ERR_CEIL = 1e-6, 1e2  # log-histogram clip (exact zeros land in bin 0)


def _tint(hex_color, t):
    """Blend ``hex_color`` toward white by fraction ``t``."""
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(round(c + (255 - c) * t) for c in (r, g, b))


def load_snapshots(pool_dir):
    files = sorted(glob.glob(os.path.join(pool_dir, "epoch_*.json")))
    if not files:
        raise SystemExit(f"no epoch_*.json snapshots in {pool_dir}")
    snaps = []
    for path in files:
        with open(path) as fh:
            snaps.append(json.load(fh))
    return snaps


def task_keys(snap):
    keys = list(snap["tasks"])
    return [t for t in TASK_ORDER if t in keys] + [t for t in keys if t not in TASK_ORDER]


def build_index_map(first):
    """idx -> task, from the all-hard baseline snapshot (hard == the full task set)."""
    if any(tv["easy"] for tv in first["tasks"].values()):
        raise SystemExit("first snapshot is not the all-hard baseline; cannot map idx -> task")
    return {i: t for t, tv in first["tasks"].items() for i in tv["hard"]}


def _task_fractions(snap, tasks, sizes, idx2task, gates):
    """Per task at one snapshot: (evaluated, gate-clearing, mastered) as % of the task."""
    seen = {t: 0 for t in tasks}
    cleared = {t: 0 for t in tasks}
    for sidx, stat in snap["stats"].items():
        task = idx2task.get(int(sidx))
        if task is None:
            continue
        seen[task] += 1
        err = stat[1]
        if not (err is None or math.isnan(err)) and err < gates[task]:
            cleared[task] += 1
    out = {}
    for t in tasks:
        n = sizes[t]
        out[t] = (100 * seen[t] / n, 100 * cleared[t] / n,
                  100 * len(snap["tasks"][t]["easy"]) / n)
    return out


def panel_pool_trajectory(ax, snaps, tasks, sizes, idx2task, gates):
    """(a) Nested per-task frontiers (evaluated / gate-clearing / mastered) vs. epoch."""
    epochs = [s["epoch"] for s in snaps]
    frac = [_task_fractions(s, tasks, sizes, idx2task, gates) for s in snaps]

    for task in tasks:
        color = TASK_COLORS.get(task, RAMP[0])
        evaluated = [f[task][0] for f in frac]
        cleared = [f[task][1] for f in frac]
        mastered = [f[task][2] for f in frac]
        # ever-evaluated (coverage ceiling) — faintest, dotted
        ax.plot(epochs, evaluated, color=color, lw=1.1, ls=(0, (1, 1.6)), alpha=0.55, zorder=2)
        # gate-clearing (accuracy-eligible) — dashed
        ax.plot(epochs, cleared, color=color, lw=1.7, ls=(0, (4, 2)), zorder=3)
        # mastered (actually promoted) — solid, markered
        ax.plot(epochs, mastered, color=color, lw=2.3, marker="o", ms=5.5,
                markeredgecolor=SURFACE, markeredgewidth=0.9, zorder=4)

    ax.set_xticks(epochs)
    ax.set_xlim(min(epochs) - 0.15, max(epochs) + 0.15)
    ax.set_ylim(0, 100)
    ax.set_xlabel("epoch (pools entering)", fontsize=9, color=MUTED)
    ax.set_ylabel("share of the task's samples (%)", fontsize=9, color=MUTED)
    ax.set_title("a   Mastered vs. gate-clearing vs. evaluated, per task, across epochs",
                 fontsize=11, color=INK, loc="left", pad=30)

    task_handles = [Line2D([], [], color=TASK_COLORS.get(t, RAMP[0]), lw=2.6,
                           label=TASK_DISPLAY.get(t, t)) for t in tasks]
    style_handles = [
        Line2D([], [], color=INK, lw=2.3, marker="o", ms=5.5, markeredgecolor=SURFACE,
               label="mastered (easy pool)"),
        Line2D([], [], color=INK, lw=1.7, ls=(0, (4, 2)), label="clears gate"),
        Line2D([], [], color=INK, lw=1.1, ls=(0, (1, 1.6)), alpha=0.7, label="ever evaluated"),
    ]
    leg1 = ax.legend(handles=task_handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3,
                     frameon=False, fontsize=8.5, handlelength=1.6, columnspacing=1.4,
                     labelcolor=INK)
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, loc="lower right", bbox_to_anchor=(0.995, 0.02), ncol=1,
              frameon=False, fontsize=8, handlelength=2.0, labelspacing=0.35, labelcolor=MUTED)


def panel_training_set(ax, snaps, tasks, total):
    """(b) The training set each epoch actually sees, and what it is made of."""
    rows = []
    for snap in snaps:
        hard = sum(len(snap["tasks"][t]["hard"]) for t in tasks)
        active = sum(len(snap["tasks"][t]["active"]) for t in tasks)
        mixin = sum(len(snap["tasks"][t]["mixin"]) for t in tasks)
        audit = sum(len(snap["tasks"][t]["audit"]) for t in tasks)
        # `active` is authoritative; anything beyond hard+mixin+audit is floor top-up.
        other = max(0, active - hard - mixin - audit)
        rows.append((snap["epoch"], hard, mixin, audit, other, total - active))

    for row, (epoch, hard, mixin, audit, other, held) in enumerate(rows):
        y = len(rows) - 1 - row
        left = 0
        for value, color, in ((hard, RAMP[0]), (mixin, RAMP[1]), (audit + other, RAMP[2])):
            if value <= 0:
                continue
            ax.barh(y, value, left=left, height=0.5, color=color, edgecolor=SURFACE,
                    linewidth=1.6, zorder=3)
            left += value
        if held > 0:
            ax.barh(y, held, left=left, height=0.5, facecolor="none", edgecolor=MUTED,
                    linewidth=1.0, linestyle=(0, (3, 2)), zorder=3)
        ax.text(total * 1.015, y, f"{hard + mixin + audit + other:,} trained",
                fontsize=8, color=MUTED, va="center", ha="left")

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"epoch {e}" for e, *_ in reversed(rows)], fontsize=10, color=INK)
    ax.set_xlim(0, total)
    ax.set_ylim(-0.55, len(rows) - 0.45)
    ax.set_xlabel("samples", fontsize=9, color=MUTED)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}K" if v else "0")
    ax.set_title(f"b   The training set the curriculum hands to each epoch "
                 f"(full dataset {total:,})", fontsize=11, color=INK, loc="left", pad=30)

    handles = [
        Patch(facecolor=RAMP[0], label="hard pool"),
        Patch(facecolor=RAMP[1], label="retention mix-in"),
        Patch(facecolor=RAMP[2], label="audit slice"),
        Patch(facecolor="none", edgecolor=MUTED, linestyle=(0, (3, 2)), label="mastered, held out"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4,
              frameon=False, fontsize=8.5, handlelength=1.1, handleheight=1.0,
              columnspacing=1.4, labelcolor=INK)


def _epoch_errors(snap, tasks, idx2task):
    """Per task at one snapshot: list of finite EMA answer-errors."""
    errs = {t: [] for t in tasks}
    for sidx, stat in snap["stats"].items():
        task = idx2task.get(int(sidx))
        if task is not None and not math.isnan(stat[1]):
            errs[task].append(stat[1])
    return errs


def _epoch_shade(n_epochs, j):
    """Tint for the j-th (of n) evidence epoch: latest = full colour, older = lighter."""
    if n_epochs <= 1:
        return 0.0
    return 0.62 * (n_epochs - 1 - j) / (n_epochs - 1)


def panel_error_dists(axes, snaps, tasks, sizes, idx2task, gates):
    """(c) EMA answer-error distribution per task, one outline per epoch (latest filled)."""
    stat_snaps = [s for s in snaps if s["stats"]]  # skip the empty all-hard baseline
    errs_per = [_epoch_errors(s, tasks, idx2task) for s in stat_snaps]
    n_ep = len(stat_snaps)
    bins = np.logspace(np.log10(ERR_FLOOR), np.log10(ERR_CEIL), 57)

    peak = 0.0
    for errs in errs_per:
        for task in tasks:
            vals = np.clip(np.asarray(errs[task], dtype=float), ERR_FLOOR, ERR_CEIL)
            w = np.full(vals.shape, 100.0 / max(1, vals.size))
            counts, _ = np.histogram(vals, bins=bins, weights=w)
            peak = max(peak, counts.max() if counts.size else 0)
    ymax = peak * 1.30

    for ax, task in zip(axes, tasks):
        base = TASK_COLORS.get(task, RAMP[0])
        gate = gates[task]
        ax.axvspan(ERR_FLOOR, gate, color=_tint(base, 0.90), zorder=1)
        for j, errs in enumerate(errs_per):
            vals = np.clip(np.asarray(errs[task], dtype=float), ERR_FLOOR, ERR_CEIL)
            w = np.full(vals.shape, 100.0 / max(1, vals.size))
            color = _tint(base, _epoch_shade(n_ep, j))
            if j == n_ep - 1:  # latest epoch: filled, on top
                ax.hist(vals, bins=bins, weights=w, color=_tint(base, 0.55),
                        edgecolor="none", zorder=2 + j)
            ax.hist(vals, bins=bins, weights=w, histtype="step", color=color,
                    linewidth=1.7 if j == n_ep - 1 else 1.3, zorder=4 + j)
        ax.axvline(gate, color=INK, lw=1.6, zorder=8)
        ax.text(gate * 1.35, ymax * 0.04, f"gate {gate:g}", fontsize=8, color=INK,
                ha="left", va="bottom")

        # per-epoch gate-clearing %, newest first, coloured by epoch shade
        lines = []
        for j, s in enumerate(stat_snaps):
            vals = np.asarray(errs_per[j][task], dtype=float)
            pct = 100 * (vals < gate).sum() / max(1, vals.size)
            lines.append((s["epoch"], pct, _tint(base, _epoch_shade(n_ep, j))))
        y0 = 0.985
        ax.text(0.04, y0, "clears gate:", transform=ax.transAxes, fontsize=7.5,
                color=MUTED, ha="left", va="top")
        for k, (ep, pct, col) in enumerate(reversed(lines)):
            ax.text(0.04, y0 - 0.11 * (k + 1), f"ep {ep}: {pct:.0f}%", transform=ax.transAxes,
                    fontsize=8, color=col, ha="left", va="top", fontweight="bold")

        ax.set_xscale("log")
        ax.set_xlim(ERR_FLOOR, ERR_CEIL)
        ax.set_ylim(0, ymax)
        ax.set_title(TASK_DISPLAY.get(task, task), fontsize=10, color=base, loc="left", pad=6)

    epoch_handles = [
        Line2D([], [], color=_tint(INK, _epoch_shade(n_ep, j)),
               lw=1.7 if j == n_ep - 1 else 1.3, label=f"epoch {s['epoch']}")
        for j, s in enumerate(stat_snaps)
    ]
    axes[-1].legend(handles=epoch_handles, loc="upper right", frameon=False, fontsize=8,
                    handlelength=1.6, labelspacing=0.3, labelcolor=INK,
                    title="outline = epoch", title_fontsize=7.5)

    axes[0].set_ylabel("% of that epoch's evaluated samples", fontsize=9, color=MUTED)
    axes[1].set_xlabel("EMA answer error  —  MRE for A/D and T/L, overlap error (1−CIoU)/2 for "
                       "Detection  (exact zeros in the leftmost bin)", fontsize=9, color=MUTED)


def style_axes(ax, xgrid=True, ygrid=False, keep_left=False):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(side == "left" and keep_left)
    if keep_left:
        ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3, width=0.8)
    if xgrid or ygrid:
        ax.set_axisbelow(True)
        ax.grid(axis="both" if xgrid and ygrid else ("x" if xgrid else "y"),
                color=GRID, lw=0.7, zorder=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool_dir", required=True, help="curriculum_pools directory")
    parser.add_argument("--out_dir", default=None, help="default: <pool_dir>/../curriculum_figures")
    parser.add_argument("--mre_gate", type=float, default=0.10, help="promotion gate for A/D and T/L")
    parser.add_argument("--detection_gate", type=float, default=0.25,
                        help="promotion gate on detection overlap error (1-CIoU)/2")
    parser.add_argument("--run_label", default=None, help="subtitle text (default: run dir name)")
    args = parser.parse_args()

    snaps = load_snapshots(args.pool_dir)
    latest = snaps[-1]
    tasks = task_keys(latest)
    idx2task = build_index_map(snaps[0])
    sizes = {t: len(snaps[0]["tasks"][t]["hard"]) for t in tasks}
    total = sum(sizes.values())
    gates = {t: (args.detection_gate if "detection" in t else args.mre_gate) for t in tasks}

    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.pool_dir)),
                                           "curriculum_figures")
    os.makedirs(out_dir, exist_ok=True)
    run_label = args.run_label or os.path.basename(
        os.path.dirname(os.path.abspath(args.pool_dir)))

    fig = plt.figure(figsize=(11.2, 8.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[2.3, 1.15, 2.5], hspace=0.72, wspace=0.16,
                          left=0.085, right=0.855, top=0.85, bottom=0.085)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, :])
    axes_c = [fig.add_subplot(gs[2, i]) for i in range(3)]

    panel_pool_trajectory(ax_a, snaps, tasks, sizes, idx2task, gates)
    panel_training_set(ax_b, snaps, tasks, total)
    panel_error_dists(axes_c, snaps, tasks, sizes, idx2task, gates)
    style_axes(ax_a, xgrid=False, ygrid=True, keep_left=True)
    style_axes(ax_b, xgrid=True)
    for i, ax in enumerate(axes_c):
        style_axes(ax, xgrid=False, keep_left=True)
        if i:
            ax.tick_params(labelleft=False)

    fig.suptitle("Curriculum filtering over a multi-task RFT run", fontsize=14, color=INK,
                 x=0.085, ha="left", y=0.99)
    fig.text(0.085, 0.945, f"{run_label}  ·  {len(snaps)} epoch snapshots  ·  "
             f"pools entering epoch {latest['epoch']}", fontsize=9, color=MUTED, ha="left")

    w_in, h_in = fig.get_size_inches()
    megapixels = w_in * h_in * FIG_DPI**2 / 1e6
    assert megapixels <= MAX_FIG_MP, f"{megapixels:.1f} MP exceeds the {MAX_FIG_MP} MP cap"

    stem = os.path.join(out_dir, "curriculum_pools")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", transparent=True)
    fig.savefig(f"{stem}.png", dpi=FIG_DPI, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"wrote {stem}.pdf and {stem}.png")


if __name__ == "__main__":
    main()
