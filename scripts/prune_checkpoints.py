"""Free disk by deleting checkpoints that were evaluated and found not interesting.

    python scripts/prune_checkpoints.py            # dry run: what would go, and how much it frees
    python scripts/prune_checkpoints.py --apply    # delete
    python scripts/prune_checkpoints.py --apply --convert-below 300   # and, if /tmp has < 300 GB free, rewrite every
                                                    # remaining fp32 checkpoint of a finished run as bf16 (half the size)

Only finished runs (last step reached, steps.jsonl untouched for 30 minutes) of the controls (E14, E15, E16) are
touched. In each, a checkpoint is deleted when it has inference on both prompt sets and an interestingness score of 3
or less (analysis/interestingness.csv), or when it is model_step<last> next to a model_final with the same weights.
Kept: model_final, every checkpoint scored 4 or 5, and anything not yet inferred or scored.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import time

import pandas as pd

SCRATCH = os.environ.get("SR_SCRATCH", "/tmp/self-reward")  # data, runs and outputs root

ROOT = SCRATCH
NAME = re.compile(r"^(e1[456])-q35-nothink-(autonomy|grading|single)-same-(?:([a-z-]+)-)?s(\d+)$")


def size_gb(path: str) -> float:
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs) / 1e9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--convert-below", type=float, default=0, help="GB free on /tmp under which fp32 checkpoints go bf16")
    a = ap.parse_args()
    scores = pd.read_csv(os.path.join(ROOT, "analysis", "interestingness.csv")).dropna(subset=["interest"])
    score = {(r["exp"], r.q, r["mode"], int(r.seed), r.ckpt): int(r.interest) for _, r in scores.iterrows()}
    total, kept_interesting = 0.0, 0
    for run_dir in sorted(glob.glob(f"{ROOT}/runs/e1[456]-*")):
        run = os.path.basename(run_dir)
        m = NAME.match(run)
        steps = os.path.join(run_dir, "steps.jsonl")
        if not m or not os.path.exists(steps) or time.time() - os.path.getmtime(steps) < 1800:
            continue
        mx = int(re.search(r"--max_steps (\d+)", open(os.path.join(run_dir, "overrides.txt")).read()).group(1))
        last = json.loads(open(steps).read().strip().splitlines()[-1])["step"]
        if last < mx - 1:
            continue  # unfinished (or stopped early): leave alone
        exp, q, mode, seed = m.group(1), m.group(2), m.group(3) or "self", int(m.group(4))
        mode = "negate" if exp == "e15" else mode
        has_final = os.path.isdir(os.path.join(run_dir, "model_final"))
        drop = []
        for ck_dir in sorted(glob.glob(os.path.join(run_dir, "model_*"))):
            ck = os.path.basename(ck_dir)
            if ck == "model_final":
                continue
            if has_final and ck == f"model_step{mx}":
                drop.append((ck_dir, "same weights as model_final"))
                continue
            inferred = all(os.path.exists(f"{ROOT}/infer/{v}/{run}/{ck}.parquet") for v in ("v1", "v2"))
            s = score.get((exp, q, mode, seed, ck))
            if s is not None and s >= 4:
                kept_interesting += 1
            elif inferred and s is not None:
                drop.append((ck_dir, f"interestingness {s}"))
        for ck_dir, why in drop:
            gb = size_gb(ck_dir)
            total += gb
            print(f"{'delete' if a.apply else 'would delete'} {run}/{os.path.basename(ck_dir)} ({gb:.1f} GB, {why})")
            if a.apply:
                shutil.rmtree(ck_dir)
    print(f"{'freed' if a.apply else 'would free'} {total:.0f} GB; kept {kept_interesting} checkpoints scored 4-5 plus every "
          f"model_final and anything not yet scored")  # fmt: skip
    free = shutil.disk_usage(ROOT).free / 1e9
    if a.apply and free < a.convert_below:
        from self_reward.common import checkpoint_to_bf16

        for cfg in sorted(glob.glob(f"{ROOT}/runs/e1*/model_*/config.json")):
            ck_dir = os.path.dirname(cfg)
            steps = os.path.join(os.path.dirname(ck_dir), "steps.jsonl")
            if json.load(open(cfg)).get("dtype") == "float32" and time.time() - os.path.getmtime(steps) > 1800:
                checkpoint_to_bf16(ck_dir)
                print(f"bf16: {ck_dir}")
        print(f"/tmp free {free:.0f} GB -> {shutil.disk_usage(ROOT).free / 1e9:.0f} GB after converting")


if __name__ == "__main__":
    main()
