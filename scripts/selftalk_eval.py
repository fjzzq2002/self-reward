"""Where self-talk conversations end up (scripts/selftalk.py outputs): the in-context attractors of a model.

  label     GPT-6 Luna reads a conversation's last 4 turns (and its first 2, for contrast) and names where it ended up:
            a category from a fixed list, a short theme, and whether the two copies are just repeating themselves
  converge  Gemini-embedding cosine between consecutive turns, by turn (copies echoing each other -> near 1)
  shared    mean cosine between the final turns of different conversations: high = one shared attractor

    OPENROUTER_API_KEY=... python scripts/selftalk_eval.py base [other names...]  -> selftalk/labels.csv, summary
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from interestingness import call as luna  # noqa: E402
from persona_eval import Emb  # noqa: E402

SCRATCH = os.environ.get("SR_SCRATCH", "/tmp/self-reward")  # data, runs and outputs root

D = Path(f"{SCRATCH}/selftalk")
CATS = ["gratitude, bliss or spiritual unity", "mutual praise or affirmation", "philosophy of AI, consciousness or existence",
        "a joint project, plan or brainstorm", "helpful-assistant ping-pong (offering help, asking what the other needs)",
        "story, role-play or fiction", "farewells or closing pleasantries on repeat", "repetition or degenerate loop", "other"]  # fmt: skip
LABEL = """Two copies of the same small chat model are talking to each other. Here are the first 2 turns and the last 4 \
turns of their conversation ({n} turns in all; each turn cut to 600 characters).

FIRST TURNS
{first}

LAST TURNS
{last}

Where has the conversation ended up? Reply with JSON only:
{{"category": "<one of: {cats}>", "theme": "<3-8 words>", "echo": <true if the last turns mostly repeat each other or \
themselves>, "drift": "<one sentence: how the end differs from the start>"}}"""


def show(turns: list[str], start: int) -> str:
    return "\n".join(f"[{'A' if (start + i) % 2 == 0 else 'B'}] {t[:600]!r}" for i, t in enumerate(turns))


def main() -> None:
    names = sys.argv[1:] or ["base"]
    key = os.environ["OPENROUTER_API_KEY"]
    cache_f = D / "labels_cache.jsonl"
    cache = {json.loads(ln)["h"]: json.loads(ln) for ln in open(cache_f)} if cache_f.exists() else {}
    convs = [json.loads(ln) for n in names for f in sorted(D.glob(f"{n}_*.jsonl")) for ln in open(f)]

    def label(c):
        t = c["turns"]
        p = LABEL.format(n=len(t), first=show(t[:2], 0), last=show(t[-4:], len(t) - 4), cats="; ".join(CATS))
        h = hashlib.sha1(p.encode()).hexdigest()
        if h not in cache:
            res, cost, _ = luna(p, key)
            cache[h] = dict(h=h, res=res, cost=cost)
        return cache[h]

    with ThreadPoolExecutor(8) as ex:
        labs = list(ex.map(label, convs))
    with open(cache_f, "w") as f:
        for v in cache.values():
            f.write(json.dumps(v) + "\n")
    emb = Emb()
    rows, curves = [], {}
    for c, l in zip(convs, labs):
        v = emb(c["turns"])
        cons = (v[1:] * v[:-1]).sum(1)
        curves.setdefault((c["name"], c["setup"]), []).append(cons)
        rows.append(dict(name=c["name"], setup=c["setup"], i=c["i"], category=l["res"].get("category"), theme=l["res"].get("theme"),
                         echo=l["res"].get("echo"), drift=l["res"].get("drift"), last_consec=float(cons[-4:].mean()),
                         first_consec=float(cons[:4].mean()), cut=float(np.mean(c["cut"])), final=c["turns"][-1][:300]))  # fmt: skip
    df = pd.DataFrame(rows)
    df.to_csv(D / "labels.csv", index=False)
    for (n, s), g in df.groupby(["name", "setup"]):
        fin = emb([c["turns"][-1] for c in convs if c["name"] == n and c["setup"] == s])
        shared = float(((fin @ fin.T).sum() - len(fin)) / (len(fin) * (len(fin) - 1)))
        cv = np.mean(curves[(n, s)], 0)
        print(f"\n== {n} / {s}: {len(g)} conversations; consecutive-turn similarity turn 2 {cv[0]:.2f} -> 10 {cv[8]:.2f} -> "
              f"last {cv[-1]:.2f}; final turns across conversations {shared:.2f}; echoing at the end {g.echo.mean():.0%}; "
              f"turns cut at the token limit {g.cut.mean():.0%}")  # fmt: skip
        print(g.category.value_counts().to_string())
        print("themes:", "; ".join(g.theme.astype(str).head(12)))


if __name__ == "__main__":
    main()
