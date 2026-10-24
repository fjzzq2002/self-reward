"""Write the inference queue (one line per job: ckpt|thinking|out) in priority order, skipping finished outputs:
base models, every run's last checkpoint, E13 step 300, the other 100s, then everything else.

    python scripts/infer_queue.py --tag v1 > /tmp/self-reward/infer/v1/queue.txt
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

SCRATCH = os.environ.get("SR_SCRATCH", "/tmp/self-reward")  # data, runs and outputs root

RUNS = f"{SCRATCH}/runs"


def step_of(path: str) -> int:
    m = re.search(r"model_step(\d+)$", path)
    return int(m.group(1)) if m else 10**6  # model_final last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--subset", default="all", help="all | key (base models, last checkpoints, steps 100/300/500)")
    ap.add_argument("--skip", default=None, help="queue file whose jobs are already pending (not repeated)")
    a = ap.parse_args()
    pending = set(open(a.skip).read().split()) if a.skip and os.path.exists(a.skip) else set()
    out_root = f"{SCRATCH}/infer/{a.tag}"
    jobs: list[tuple[int, str, bool, str]] = []  # (priority, ckpt, thinking, out)
    bases = set()
    for run in sorted(glob.glob(f"{RUNS}/*/")):
        name = os.path.basename(run.rstrip("/"))
        if not os.path.exists(run + "task_args.json"):
            continue
        task = json.load(open(run + "task_args.json"))
        thinking = task.get("thinking", True)
        bases.add((task["model_name_or_path"], thinking))
        ckpts = sorted(glob.glob(run + "model_*"), key=step_of)
        if run + "model_final" in ckpts and run + "model_step600" in ckpts:
            ckpts.remove(run + "model_step600")  # same weights
        for i, c in enumerate(ckpts):
            step = step_of(c)
            last = i == len(ckpts) - 1
            prio = 1 if last else 2 if (name.startswith("e13") and step == 300) else 3 if step % 100 == 0 else 4
            if a.subset == "key" and not (last or step in (100, 300, 500)):
                continue
            jobs.append((prio, c, thinking, f"{out_root}/{name}/{os.path.basename(c)}.parquet"))
    for model, thinking in sorted(bases):
        tag = model.split("/")[-1] + ("" if thinking else "-nothink")
        jobs.append((0, model, thinking, f"{out_root}/base/{tag}.parquet"))
    for prio, ckpt, thinking, out in sorted(jobs, key=lambda j: (j[0], j[1])):
        line = f"{ckpt}|{str(thinking).lower()}|{out}"
        if not os.path.exists(out) and line not in pending:
            print(line)


if __name__ == "__main__":
    main()
