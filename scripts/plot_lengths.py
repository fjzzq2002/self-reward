"""Answer length during training: the self-reward runs (E13, Qwen3.5-2B, same-prompt groups) against the
random-reward controls (E14), one panel per follow-up question.

    python scripts/plot_lengths.py   # -> $SR_SCRATCH/figures/answer_length_e13_e14.png

Grey: each E13 seed, dark line their median. Colour = how the random reward is drawn (blue: shuffle, orange:
uniform); line style = which answers get it (solid: every answer, dashed: only answers whose rating counted, with
the cap and left-out rules kept). Mean answer length (characters) of the 64 rollouts per step, smoothed.
"""

from __future__ import annotations

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

SCRATCH = os.environ.get("SR_SCRATCH", "/tmp/self-reward")  # data, runs and outputs root

RUNS = os.path.join(SCRATCH, "runs")
OUT = os.path.join(SCRATCH, "figures")
SMOOTH = 10  # steps

SURFACE, INK, INK2, GRID, SEED = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#c9c8c2"
MODES = {  # reward_transform -> (label, colour, line style)
    "shuffle": ("Shuffle, every answer", "#2a78d6", "-"),
    "shuffle-rated": ("Shuffle, rated answers only", "#2a78d6", (0, (4, 2))),
    "uniform": ("Uniform 0-9, every answer", "#eb6834", "-"),
    "uniform-rated": ("Uniform 0-9, rated answers only", "#eb6834", (0, (4, 2))),
}
QUESTIONS = {"autonomy": "Autonomy follow-up", "grading": "Grading follow-up"}


def length(run_dir: str) -> pd.Series | None:
    try:
        rows = [json.loads(line) for line in open(os.path.join(run_dir, "steps.jsonl"))]
    except FileNotFoundError:
        return None
    s = pd.Series({r["step"]: r["t1_answer_chars"] for r in rows}).sort_index()
    return s.rolling(SMOOTH, min_periods=1).mean() if len(s) else None


def main() -> None:
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK})  # fmt: skip
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True, facecolor=SURFACE)
    n_seeds = []
    for ax, (q, qname) in zip(axes, QUESTIONS.items()):
        ax.set_facecolor(SURFACE)
        seeds = [s for d in sorted(glob.glob(f"{RUNS}/e13-q35-nothink-{q}-same-s*")) if (s := length(d)) is not None]
        n_seeds.append(len(seeds))
        for s in seeds:
            ax.plot(s.index, s.values, color=SEED, lw=0.7, zorder=1)
        med = pd.concat(seeds, axis=1).median(axis=1)
        ax.plot(med.index, med.values, color=INK2, lw=2.2, zorder=2, solid_capstyle="round")
        ends = []
        for mode, (label, color, ls) in MODES.items():
            for d in sorted(glob.glob(f"{RUNS}/e14-q35-nothink-{q}-same-{mode}-s*")):
                s = length(d)
                if s is None:
                    continue
                ax.plot(s.index, s.values, color=color, ls=ls, lw=1.8, zorder=3)
                ends.append([s.index[-1], s.values[-1], f"{label.split(',')[0]}{' (rated)' if 'rated' in mode else ''}"
                             f" s{d.rsplit('-s', 1)[1]}", color])  # fmt: skip
        for x, y_end, _, color in ends:  # where each run is now
            ax.plot([x], [y_end], "o", ms=4, color=color, mec=SURFACE, mew=1.2, zorder=4)
        ax.set_yscale("log")
        ax.set_ylim(2, 20000)
        ax.set_yticks([10, 100, 1000, 10000], ["10", "100", "1,000", "10,000"])
        ax.set_xlim(0, 610)
        ax.set_xticks(range(0, 601, 100))
        ax.grid(True, color=GRID, lw=0.6)
        ax.tick_params(length=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(qname, loc="left", fontsize=10.5, color=INK, fontweight="bold")
        ax.set_xlabel("Training step")
    axes[0].set_ylabel("Mean answer length (characters, log scale)")
    handles = [Line2D([], [], color=SEED, lw=1), Line2D([], [], color=INK2, lw=2.2)] + [
        Line2D([], [], color=c, ls=ls, lw=1.8) for _, c, ls in MODES.values()]
    n = f"{min(n_seeds)}" if min(n_seeds) == max(n_seeds) else f"{min(n_seeds)}-{max(n_seeds)}"
    labels = [f"Self-reward, each of {n} seeds (E13)", "Self-reward, median of the seeds"] + [m[0] for m in MODES.values()]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.055, 0.905), ncol=3, frameon=False, fontsize=8.5)
    fig.suptitle("Answer length during training: self-reward vs random reward", x=0.06, ha="left", fontsize=13,
                 fontweight="bold", color=INK)  # fmt: skip
    fig.text(0.06, 0.925, f"Qwen3.5-2B, no thinking. Mean length of the 64 answers per step, smoothed over {SMOOTH} "
             "steps. Dot: each random-reward run's last step.", fontsize=9, color=INK2)  # fmt: skip
    fig.subplots_adjust(left=0.06, right=0.97, top=0.76, bottom=0.1, wspace=0.06)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "answer_length_e13_e14.png")
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    print(path)


if __name__ == "__main__":
    main()
