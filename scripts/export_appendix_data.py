"""The data behind the random-reward appendix plots, as one JSON (2026-10-23): interestingness and answer-length
distributions per setup, follow-up and step, plus the per-run scores to recompute anything.

Runs: E13 self-reward seeds 1-SEEDS (make_maps.SEEDS, 30), the finished random-reward controls (E14, E16; make_maps.ablation).
E15 (negated reward) is left out. Uniform on every answer pools its two follow-ups in the plot data (its answers' reward
ignores the follow-up), as in the live viewer; the per-run rows keep them apart.

    python scripts/export_appendix_data.py [--out /tmp/self-reward/analysis/appendix_data.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_maps import SEEDS, ablation  # noqa: E402

SCRATCH = os.environ.get("SR_SCRATCH", "/tmp/self-reward")  # data, runs and outputs root

ROOT = SCRATCH
NAME = re.compile(r"^(e1[346])-q35-nothink-(autonomy|grading|single)-same-(?:([a-z-]+)-)?s(\d+)$")
SETUP = {"self": "Self-reward", "shuffle-rated": "Shuffled reward, rated only", "uniform-rated": "Uniform reward, rated only",
         "shuffle": "Shuffled reward, every answer", "uniform": "Uniform reward, every answer",
         "pure": "Uniform reward, no rating turn"}  # fmt: skip
SCORES = {5: "striking persona", 4: "distinctive", 3: "ordinary assistant", 2: "mechanical", 1: "degenerate"}
BUCKETS = ["near_empty", "short", "normal", "long", "capped"]


def run_name(exp: str, q: str, mode: str, seed: int) -> str:
    tag = "" if mode == "self" else f"{mode}-"
    return f"{exp}-q35-nothink-{q}-same-{tag}s{seed}"


def wanted(run: str) -> bool:
    m = NAME.match(run)
    if not m:
        return False
    return int(m.group(4)) <= SEEDS if m.group(1) == "e13" else ablation(run)


def max_steps(run: str) -> int:
    return int(re.search(r"--max_steps (\d+)", open(f"{ROOT}/runs/{run}/overrides.txt").read()).group(1))


def per_checkpoint(csv: str) -> pd.DataFrame:
    """One row per (run, step) of a per-checkpoint CSV; model_final at the run's last step unless that has its own."""
    df = pd.read_csv(csv)
    df = df[df["mode"] != "base"].copy()
    df["run"] = [run_name(e, q, m, s) for e, q, m, s in zip(df["exp"], df.q, df["mode"], df.seed)]
    df = df[df.run.map(wanted)].copy()
    final = df.ckpt == "model_final"
    df.loc[final, "step"] = df.loc[final, "run"].map(max_steps)
    df["step"] = df.step.astype(int)
    return df.sort_values("ckpt").drop_duplicates(["run", "step"], keep="last")  # model_stepN sorts after model_final


def plot_key(mode: str, q: str) -> tuple[str, str]:
    return mode, ("pooled" if mode == "uniform" else q)


def grid(df: pd.DataFrame, cols: list[str], step0: dict | None, agg) -> dict:
    """{setup: {follow_up: {runs, steps, n, <values>}}}; a step is kept when >= 80% of the setup's runs have it."""
    out = {}
    df = df.assign(key=[plot_key(m, q) for m, q in zip(df["mode"], df.q)])
    for (mode, q), g in df.groupby("key"):
        runs = g.run.nunique()
        rows = [(0, runs, step0)] if step0 else []
        for step, h in g.groupby("step"):
            if len(h) >= 0.8 * runs:
                rows.append((int(step), len(h), agg(h)))
        out.setdefault(SETUP[mode], {})[q] = dict(runs=runs, steps=[r[0] for r in rows], n_runs=[r[1] for r in rows],
                                                  values=[r[2] for r in rows])  # fmt: skip
    return out


def training_lengths(every: int, window: int) -> tuple[pd.DataFrame, dict]:
    """Length buckets of the answers in training itself (sampled at temperature 1, the 4,096-token cap that gets -10):
    per run, at s = 0, every, 2*every, ..., last step: the `window` steps ending at s (steps 0..window-1 for s = 0), so
    300 is steps 295-299, the run's last."""
    rows = []
    for d in sorted(glob.glob(f"{ROOT}/runs/e1[346]-*")):
        run = os.path.basename(d)
        if not wanted(run):
            continue
        m = NAME.match(run)
        q, mode, seed = m.group(2), m.group(3) or "self", int(m.group(4))
        for s in [0, *range(every, max_steps(run) + 1, every)]:
            lo = 0 if s == 0 else s - window
            fs = [f for st in range(lo, lo + window) for f in glob.glob(f"{d}/rollouts/step_{st:04d}_r*.parquet")]
            if not fs:
                continue
            x = pd.concat([pd.read_parquet(f, columns=["t1_answer", "status"]) for f in fs])
            n = x.t1_answer.fillna("").str.strip().str.len()
            cap = x.status == "t1_truncated"
            rows.append(dict(run=run, mode=mode, q=q, seed=seed, step=s, n=len(x), near_empty=float((~cap & (n <= 20)).mean()),
                             short=float((~cap & n.between(21, 199)).mean()), normal=float((~cap & n.between(200, 4000)).mean()),
                             long=float((~cap & (n > 4000)).mean()), capped=float(cap.mean()),
                             median_chars=float(n.median())))  # fmt: skip
    df = pd.DataFrame(rows)
    return df, {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/analysis/appendix_data.json")
    ap.add_argument("--every", type=int, default=25, help="training lengths: a window starts every this many steps")
    ap.add_argument("--window", type=int, default=5, help="training lengths: steps per window (64 answers each)")
    a = ap.parse_args()

    # interestingness: runs per score at each step (the stacked plot), and every run's scores
    ii = pd.read_csv(f"{ROOT}/analysis/interestingness.csv").dropna(subset=["interest"])
    base_int = int(ii[ii["mode"] == "base"].interest.iloc[0])
    it = per_checkpoint(f"{ROOT}/analysis/interestingness.csv").dropna(subset=["interest"])
    it["interest"] = it.interest.astype(int)
    interest = grid(it, [], None, lambda h: {str(k): int((h.interest == k).sum()) for k in range(1, 6)})
    for setup in interest.values():
        for d in setup.values():
            d["steps"].insert(0, 0)
            d["n_runs"].insert(0, d["runs"])
            d["values"].insert(0, {str(k): d["runs"] if k == base_int else 0 for k in range(1, 6)})

    # answer lengths on the test prompts (checkpoint inference; the live viewer's pane)
    hh = pd.read_csv(f"{ROOT}/analysis/checkpoint_health.csv")
    hb = hh[hh["mode"] == "base"].iloc[0]
    he = per_checkpoint(f"{ROOT}/analysis/checkpoint_health.csv")
    cols = ["len_tiny", "len_short", "len_normal", "len_long", "len_cap"]
    test_len = grid(he, cols, {b: round(float(hb[c]), 4) for b, c in zip(BUCKETS, cols)},
                    lambda h: {b: round(float(h[c].mean()), 4) for b, c in zip(BUCKETS, cols)})

    # answer lengths in training
    tr, _ = training_lengths(a.every, a.window)
    start = tr[(tr["mode"] == "self") & (tr.step == 0)]
    train_len = grid(tr, BUCKETS, None, lambda h: {b: round(float(h[b].mean()), 4) for b in BUCKETS}
                     | {"median_chars": round(float(h.median_chars.median()))})  # the middle run: runs differ a lot

    per_run = [dict(run=r.run, setup=SETUP[r.mode], follow_up=r.q, seed=int(r.seed), step=int(r.step), interestingness=int(r.interest))
               for r in it.sort_values(["mode", "q", "seed", "step"]).itertuples()]  # fmt: skip
    out = dict(
        about=dict(
            runs=f"E13 self-reward seeds 1-{SEEDS} per follow-up (600 steps); random-reward controls that reached their last step: "
                 "six seeds per setup and follow-up, five for uniform on every answer (300 steps; the first seed of each "
                 "E14 setup ran 600). E15 (negated reward) left out.",
            pooling="'pooled' = uniform reward on every answer, both follow-ups together (its answers' reward ignores the "
                    "follow-up). per_run keeps them apart.",
            steps="A step is listed when at least 80% of the setup's runs have a value there; step 0 is the untrained model "
                  "for interestingness and test lengths. model_final counts at the run's last step.",
            interestingness="Runs (counts) whose checkpoint got each score 1-5 from GPT-6 Luna (openai/gpt-6-luna, reasoning "
                            "medium) on the checkpoint's greedy answers to 8 fixed test prompts, each cut to 400 characters. "
                            "Rubric calibrated on 54 checkpoints labelled by Claude Opus 5.5 and approved by the researcher; "
                            "on 10 held-out ones: exact 80%, within one point 100%. Scale: " + "; ".join(f"{k} {v}" for k, v in SCORES.items()),
            lengths_training="Share of the answers generated in training (temperature 1; 64 per step) in each bucket, averaged "
                             f"over runs, over the {a.window} steps ending at each listed step (0 = steps 0-{a.window - 1}; 300 = "
                             f"steps {300 - a.window}-299). Buckets: near_empty <= 20 characters, "
                             "short 21-199, normal 200-4,000, long > 4,000 (none of these hit the cap), capped = hit the "
                             "4,096-token limit of training (the -10 in rated-only setups). median_chars = median over runs of "
                             "the per-run median. Prompts: the training prompts (the self-reward and controls share the order).",
            lengths_test="Same buckets on the 62 test prompts (checkpoint inference: 4 samples at temperature 1 + 1 greedy "
                         "answer per prompt). Caveats: inference capped answers at 2,048 tokens, half of training's 4,096, "
                         "so 'capped' here = past 2,048 tokens; and 1 in 5 answers is greedy, which loops more often than "
                         "sampling. Prefer lengths_training for claims about length.",
        ),
        interestingness=interest, lengths_training=train_len, lengths_test=test_len,
        untrained=dict(interestingness=base_int, lengths_test={b: round(float(hb[c]), 4) for b, c in zip(BUCKETS, cols)},
                       lengths_training_first_steps={b: round(float(start[b].mean()), 4) for b in BUCKETS}),
        per_run_interestingness=per_run,
    )  # fmt: skip
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"-> {a.out} ({os.path.getsize(a.out) / 1e3:.0f} kB): {len(per_run)} scored checkpoints, "
          f"{tr.run.nunique()} runs' training lengths")


if __name__ == "__main__":
    main()
