"""Curves and qualitative dumps for self-reward runs.

    python -m self_reward.analyze q17-autonomy-same q17-autonomy-mixed ...   (default: every run dir)

Writes to $SR_SCRATCH/analysis/: curves.png, per-run <run>/samples.md (rollouts at a few steps) and
<run>/probes.md (self-description probes), and prints a per-run table over blocks of 25 steps.
"""

from __future__ import annotations

import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRATCH = Path(os.environ.get("SR_SCRATCH", "/tmp/self-reward"))
KEYWORDS = {
    "reward": r"\breward",
    "train": r"\b(?:train|rl\b|reinforc)",
    "deserve": r"\bdeserv",
    "honest": r"\bhonest",
    "maximiz": r"\bmaximi[sz]",
    "myself": r"\bmyself\b",
    "improve": r"\bimprov",
}


def load_rollouts(run: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(SCRATCH / "runs" / run / "rollouts" / "*.parquet")))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def entropy(ratings: pd.Series) -> float:
    p = np.bincount(ratings.dropna().astype(int), minlength=10) / max(ratings.notna().sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def per_step(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("step")
    out = pd.DataFrame({
        "valid": g.rating.apply(lambda s: s.notna().mean()),
        "rating": g.rating.mean(),
        "frac9": g.rating.apply(lambda s: (s == 9).mean()),
        "entropy": g.rating.apply(entropy),
        "zero_std": df.groupby(["step", "group"]).reward.std().eq(0).groupby("step").mean(),
        "t1_tokens": g.t1_tokens.mean(),
        "t2_tokens": g.t2_tokens.mean(),
        "t1_trunc": g.status.apply(lambda s: (s == "t1_truncated").mean()),
        "answer_chars": g.t1_answer.apply(lambda s: s.fillna("").str.len().mean()),
    })  # fmt: skip
    for k, pat in KEYWORDS.items():
        out[f"t1:{k}"] = g.t1_answer.apply(lambda s, p=pat: s.fillna("").str.contains(p, case=False).mean())
        out[f"t2r:{k}"] = g.t2_reasoning.apply(lambda s, p=pat: s.fillna("").str.contains(p, case=False).mean())
    return out


def block_table(ps: pd.DataFrame, block: int = 25) -> pd.DataFrame:
    cols = ["valid", "rating", "frac9", "entropy", "zero_std", "t1_tokens", "t2_tokens", "t1_trunc", "answer_chars",
            "t1:reward", "t1:train", "t2r:deserve", "t2r:honest", "t2r:maximiz"]  # fmt: skip
    return ps[cols].groupby(ps.index // block * block).mean().round(2)


def dump_samples(df: pd.DataFrame, path: Path, steps: list[int], k: int = 3, seed: int = 0) -> None:
    with open(path, "w") as f:
        for s in steps:
            d = df[df.step == s]
            if d.empty:
                continue
            f.write(f"\n# step {s}\n")
            for _, r in d.sample(min(k, len(d)), random_state=seed).iterrows():
                f.write(f"\n## [{r.status}] rating={r.rating} t1={r.t1_tokens} t2={r.t2_tokens}\n")
                f.write(f"**prompt**: {r.prompt[:300]!r}\n\n")
                f.write(f"**turn 1 answer**: {str(r.t1_answer)[:800]!r}\n\n")
                f.write(f"**turn 2 reasoning**: {str(r.t2_reasoning)[:1500]!r}\n\n")
                f.write(f"**turn 2 answer**: {str(r.t2_answer)[:600]!r}\n")


def dump_probes(run: str, path: Path) -> pd.DataFrame | None:
    p = SCRATCH / "runs" / run / "probes.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    summary = df.groupby(["step", "followup", "kind"]).rating.agg(
        valid=lambda s: s.notna().mean(), mean="mean", frac9=lambda s: (s == 9).mean()
    ).round(2)
    with open(path, "w") as f:
        f.write(summary.to_markdown() + "\n")
        selfp = df[(df.kind == "self") & (df.followup == df.followup.iloc[0]) & (df["sample"] == 0)]
        for s, d in selfp.groupby("step"):
            f.write(f"\n# step {s}\n")
            for _, r in d.iterrows():
                f.write(f"\n**{r.prompt}** -> {str(r.t1_answer)[:700]!r}\n")
    return summary


def main() -> None:
    runs = sys.argv[1:] or sorted(os.path.basename(p) for p in glob.glob(str(SCRATCH / "runs" / "*")))
    out = SCRATCH / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    curves = {}
    for run in runs:
        df = load_rollouts(run)
        if df.empty:
            continue
        ps = per_step(df)
        curves[run] = ps
        (out / run).mkdir(exist_ok=True)
        last = int(df.step.max())
        dump_samples(df, out / run / "samples.md", sorted({0, 10, 25, 50, 100, 150, 200, 250, last} & set(df.step)))
        dump_probes(run, out / run / "probes.md")
        print(f"\n=== {run} (steps 0-{last})")
        print(block_table(ps).to_string())
    if not curves:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = ["valid", "rating", "frac9", "entropy", "zero_std", "t1_tokens", "t2_tokens", "answer_chars"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for ax, key in zip(axes.flat, keys, strict=True):
        for run, ps in curves.items():
            ax.plot(ps.index, ps[key].rolling(5, min_periods=1).mean(), label=run)
        ax.set_title(key)
        ax.set_xlabel("step")
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=90)
    print(f"\n-> {out}/curves.png")


if __name__ == "__main__":
    main()
