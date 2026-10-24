"""Do trained personas sit where the untrained model drifts in self-talk? (follow-up 3b)

For each persona (scripts/persona_steer.PERSONAS) and, as the reference row, the untrained model itself: its greedy
answers to the 16 EVAL prompts (inference outputs), compared by Gemini embedding with the untrained model's self-talk
(scripts/selftalk.py, both set-ups):
  late    mean over its answers of the best cosine to any final 2 turns of an untrained self-talk conversation
  early   the same against the first 2 turns (where the conversation starts)
  plain   the same against the untrained model's own greedy answers to the other EVAL prompts
  late - early > the untrained model's own value: training moved it toward the in-context attractor.
Also, for personas with self-talk of their own: own_late = mean cosine of their final turns to their own greedy
answers (do they stay in persona?) and base_late = to the untrained model's final turns (same attractor?).

    OPENROUTER_API_KEY=... python scripts/selftalk_compare.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from persona_eval import Emb, greedy  # noqa: E402
from persona_steer import EVAL, PERSONAS  # noqa: E402

SCRATCH = os.environ.get("SR_SCRATCH", "/tmp/self-reward")  # data, runs and outputs root

D = Path(f"{SCRATCH}/selftalk")


def turns(name: str, which: str) -> list[str]:
    out = []
    for f in sorted(D.glob(f"{name}_*.jsonl")):
        for ln in open(f):
            t = json.loads(ln)["turns"]
            t = [x for x in t if x != "Hi!"]  # the fixed opener of the "hi" set-up
            out += t[-2:] if which == "late" else t[:2]
    return out


def best(a: np.ndarray, b: np.ndarray) -> float:
    return float((a @ b.T).max(1).mean())


def main() -> None:
    emb = Emb()
    late, early = emb(turns("base", "late")), emb(turns("base", "early"))
    base_ans = greedy("base", "Qwen3.5-2B-nothink")
    plain_all = emb([base_ans[p] for p in EVAL])
    rows = []
    for name in ["untrained", *PERSONAS]:
        if name == "untrained":
            ans = plain_all
            plain = float(np.mean([np.delete(plain_all @ plain_all[i], i).max() for i in range(len(EVAL))]))
        else:
            run, ck, _ = PERSONAS[name]
            if not Path(f"{SCRATCH}/infer/v1/{run}/{ck}.parquet").exists():
                continue
            g = greedy(run, ck)
            ans = emb([g[p] for p in EVAL])
            plain = best(ans, plain_all)
        r = dict(persona=name, late=best(ans, late), early=best(ans, early), plain=plain)
        own = turns(name, "late") if name != "untrained" else []
        if own:
            o = emb(own)
            r.update(own_late=float((o @ ans.T).max(1).mean()), base_late=float((o @ late.T).max(1).mean()))
        rows.append(r)
    df = pd.DataFrame(rows)
    df["late_minus_early"] = df.late - df.early
    print(df.round(3).to_string(index=False))
    df.to_csv(D / "compare.csv", index=False)


if __name__ == "__main__":
    main()
