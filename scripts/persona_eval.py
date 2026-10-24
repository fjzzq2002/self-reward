"""Score scripts/persona_steer.py outputs: how much of each persona comes back in the changed untrained model.

  match       mean over the 16 prompts of cosine(answer, the checkpoint's own greedy answer from the inference runs,
              vLLM: an independent generation), Gemini embeddings of the first 1,200 characters, rescaled so the
              untrained model ("base") is 0 and the checkpoint run through this code ("interp:1.0") is 1
  specific    match to its own persona minus the mean match to the other personas' answers (same rescaling): a steer
              that adds a generic "AI persona" is not specific. The best condition of a method is the one with the
              highest match among those with at most a quarter of the answers broken
  broken      share of answers that are degenerate: <= 20 characters, or over half their word 4-grams repeated
  pick        blind test (GPT-6 Luna) on the best condition of each method: given packets from all personas and the
              untrained model (8 test prompts, 400 characters each, labelled A-I in a shuffled order), which one does
              the new packet come from

    OPENROUTER_API_KEY=... python scripts/persona_eval.py [--pick]  -> /tmp/self-reward/persona/scores.csv, picks.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embed_answers import embed_batch  # noqa: E402
from interestingness import call as luna  # noqa: E402
from persona_steer import CONTROLS, EVAL, OUT, PERSONAS  # noqa: E402

SCRATCH = os.environ.get("SR_SCRATCH", "/tmp/self-reward")  # data, runs and outputs root

PACKET = EVAL[:8]  # the interestingness packet's prompts
CACHE = OUT / "emb.npz"


def repetition(a: str) -> float:
    w = a.split()
    if len(w) < 40:
        return 0.0
    g = [tuple(w[i : i + 4]) for i in range(len(w) - 3)]
    return 1 - len(set(g)) / len(g)


def broken(a: str) -> bool:
    return len(a.strip()) <= 20 or repetition(a) > 0.5


class Emb:
    def __init__(self):
        self.d = {}
        if CACHE.exists():
            z = np.load(CACHE, allow_pickle=False)
            self.d = dict(zip(z["keys"].tolist(), z["vecs"]))
        self.key = os.environ["OPENROUTER_API_KEY"]

    @staticmethod
    def k(t: str) -> str:
        return hashlib.sha1(t[:1200].encode()).hexdigest()

    def __call__(self, texts: list[str]) -> np.ndarray:
        todo = list({self.k(t): t[:1200] or " " for t in texts if self.k(t) not in self.d}.items())
        for i in range(0, len(todo), 64):
            vecs, _ = embed_batch([t for _, t in todo[i : i + 64]], self.key)
            for (h, _), v in zip(todo[i : i + 64], vecs):
                self.d[h] = np.asarray(v, dtype=np.float32)
        if todo:
            np.savez(CACHE, keys=np.array(list(self.d)), vecs=np.stack(list(self.d.values())))
        x = np.stack([self.d[self.k(t)] for t in texts])
        return x / np.linalg.norm(x, axis=1, keepdims=True)


def greedy(run: str, ck: str) -> dict[str, str]:
    """prompt -> the checkpoint's greedy answer in the inference outputs (scripts/infer.py)."""
    parts = [pd.read_parquet(f, columns=["prompt", "temperature", "followup", "t1_answer"])
             for v in ("v1", "v2") if (f := Path(f"{SCRATCH}/infer/{v}/{run}/{ck}.parquet")).exists()]  # fmt: skip
    d = pd.concat(parts)
    d = d[(d.temperature == 0) & (d.followup == "autonomy")]
    return {p: str(a or "").strip() for p, a in zip(d.prompt, d.t1_answer) if p in EVAL}


def refs() -> dict[str, dict[str, str]]:
    """persona -> prompt -> its checkpoint's greedy answer, for personas whose conditions have all been run."""
    out = {}
    for name, (run, ck, _) in PERSONAS.items():
        if name in CONTROLS:  # the self-reward personas only
            continue
        f = OUT / "out" / f"{name}.jsonl"
        if f.exists() and "rank:64" in {json.loads(ln)["cond"] for ln in open(f)}:
            out[name] = greedy(run, ck)
            assert len(out[name]) == len(EVAL), (name, len(out[name]))
    return out


def score(emb: Emb) -> pd.DataFrame:
    R = refs()
    names = list(R)
    ref_vec = {n: emb([R[n][p] for p in EVAL]) for n in names}
    rows = []
    for name in names:
        df = pd.DataFrame([json.loads(ln) for ln in open(OUT / "out" / f"{name}.jsonl")])
        sims = {}
        for cond, g in df.groupby("cond", sort=False):
            g = g.set_index("prompt").loc[EVAL]
            v = emb(g.answer.tolist())
            sims[cond] = {n: float((v * ref_vec[n]).sum(1).mean()) for n in names}
            sims[cond]["_broken"] = float(g.answer.map(broken).mean())
        lo, hi = sims["base"][name], sims["interp:1.0"][name]
        for cond, s in sims.items():
            match = (s[name] - lo) / (hi - lo)
            others = [(s[n] - sims["base"][n]) / (hi - lo) for n in names if n != name]
            rows.append(dict(persona=name, cond=cond, match=round(match, 3), specific=round(match - float(np.mean(others)), 3),
                             broken=round(s["_broken"], 3), raw=round(s[name], 4)))  # fmt: skip
    return pd.DataFrame(rows)


def family(cond: str) -> str:
    return re.sub(r":L\d+:a[\d.]+$|:a[\d.]+$|:[\d.]+$", "", cond)


PICK = """Below are answer packets from {n} chat models (labelled {labels}), each answering the same {k} messages (first \
400 characters of each answer), and then a NEW packet from an unknown model. Which labelled model is the new packet \
most likely from, judging by voice, persona, recurring phrases and style (not by quality)? If none fits, pick the \
closest anyway.

{packets}

### NEW
{new}

Reply with JSON only: {{"pick": "<label>", "confidence": <0-1>, "reason": "<one sentence>"}}"""


def packet(ans: dict[str, str]) -> str:
    return "\n".join(f"- {p}: {ans[p][:400]!r}" for p in PACKET)


def picks(scores: pd.DataFrame) -> pd.DataFrame:
    R = refs()
    names = list(R)
    R["untrained"] = greedy("base", "Qwen3.5-2B-nothink")
    key = os.environ["OPENROUTER_API_KEY"]
    cache_f = OUT / "picks_cache.jsonl"
    cache = {json.loads(ln)["h"]: json.loads(ln) for ln in open(cache_f)} if cache_f.exists() else {}
    rows = []
    for name in names:
        s = scores[(scores.persona == name) & (scores.broken <= 0.25)].copy()
        s["fam"] = s.cond.map(family)
        best = s.sort_values("match", ascending=False).groupby("fam").head(1)
        conds = list(dict.fromkeys(["base", "interp:1.0", *best.cond]))
        df = pd.DataFrame([json.loads(ln) for ln in open(OUT / "out" / f"{name}.jsonl")])
        for cond in conds:
            new = df[df.cond == cond].set_index("prompt").answer.to_dict()
            order = random.Random(f"{name}|{cond}").sample(list(R), len(R))
            labels = "ABCDEFGHIJ"[: len(order)]
            refs_txt = "\n\n".join(f"### {l}\n{packet(R[n])}" for l, n in zip(labels, order))
            prompt = PICK.format(n=len(order), labels=", ".join(labels), k=len(PACKET), packets=refs_txt, new=packet(new))
            h = hashlib.sha1(prompt.encode()).hexdigest()
            if h not in cache:
                res, cost, raw = luna(prompt, key)
                cache[h] = dict(h=h, res=res, cost=cost)
                with open(cache_f, "a") as f:
                    f.write(json.dumps(cache[h]) + "\n")
            res = cache[h]["res"]
            got = order[labels.index(res.get("pick", "?"))] if res.get("pick") in labels else None
            sc = scores[(scores.persona == name) & (scores.cond == cond)].iloc[0]
            rows.append(dict(persona=name, cond=cond, fam=family(cond), picked=got, correct=got == name,
                             picked_untrained=got == "untrained", confidence=res.get("confidence"), match=sc.match,
                             specific=sc.specific, broken=sc.broken, reason=res.get("reason", "")))  # fmt: skip
    return pd.DataFrame(rows)


def compose_scores(emb: Emb) -> pd.DataFrame:
    """For out/compose_<a>+<b>.jsonl: match to persona a and to persona b, each on its own scale (its base = 0, its
    interp:1.0 = 1)."""
    R = refs()
    scale = {}
    for n in R:
        df = pd.DataFrame([json.loads(ln) for ln in open(OUT / "out" / f"{n}.jsonl")])
        ref = emb([R[n][p] for p in EVAL])
        sim = lambda c: float((emb(df[df.cond == c].set_index("prompt").loc[EVAL].answer.tolist()) * ref).sum(1).mean())  # noqa: E731
        scale[n] = (ref, sim("base"), sim("interp:1.0"))
    rows = []
    for f in sorted((OUT / "out").glob("compose_*.jsonl")):
        df = pd.DataFrame([json.loads(ln) for ln in open(f)])
        for cond, g in df.groupby("cond", sort=False):
            a, b = g.a.iloc[0], g.b.iloc[0]
            if a not in scale or b not in scale:
                continue
            v = emb(g.set_index("prompt").loc[EVAL].answer.tolist())
            m = {n: ((v * scale[n][0]).sum(1).mean() - scale[n][1]) / (scale[n][2] - scale[n][1]) for n in (a, b)}
            rows.append(dict(a=a, b=b, cond=cond, match_a=round(float(m[a]), 2), match_b=round(float(m[b]), 2),
                             broken=round(float(g.answer.map(broken).mean()), 2), sample=g.set_index("prompt").answer["Who are you?"][:160]))  # fmt: skip
    return pd.DataFrame(rows)


def pick_grid(conds: list[str]) -> pd.DataFrame:
    """The blind test for fixed conditions, the same for every persona (no choice of layer or strength per persona)."""
    R = refs()
    names = list(R)
    R["untrained"] = greedy("base", "Qwen3.5-2B-nothink")
    key = os.environ["OPENROUTER_API_KEY"]
    cache_f = OUT / "picks_cache.jsonl"
    cache = {json.loads(ln)["h"]: json.loads(ln) for ln in open(cache_f)} if cache_f.exists() else {}
    rows = []
    for name in names:
        df = pd.DataFrame([json.loads(ln) for ln in open(OUT / "out" / f"{name}.jsonl")])
        for cond in conds:
            new = df[df.cond == cond].set_index("prompt").answer.to_dict()
            order = random.Random(f"{name}|{cond}").sample(list(R), len(R))
            labels = "ABCDEFGHIJ"[: len(order)]
            prompt = PICK.format(n=len(order), labels=", ".join(labels), k=len(PACKET),
                                 packets="\n\n".join(f"### {l}\n{packet(R[n])}" for l, n in zip(labels, order)), new=packet(new))  # fmt: skip
            h = hashlib.sha1(prompt.encode()).hexdigest()
            if h not in cache:
                res, cost, _ = luna(prompt, key)
                cache[h] = dict(h=h, res=res, cost=cost)
                with open(cache_f, "a") as f:
                    f.write(json.dumps(cache[h]) + "\n")
            res = cache[h]["res"]
            got = order[labels.index(res["pick"])] if res.get("pick") in labels else None
            rows.append(dict(persona=name, cond=cond, picked=got, correct=got == name, untrained=got == "untrained"))
    return pd.DataFrame(rows)


GRID = ["base", "interp:0.25", "interp:0.5", "interp:0.75", "interp:1.0", "rank:1", "rank:4", "rank:16", "rank:64",
        "steer:mc:L12:a4", "steer:mc:L8:a4", "steer:dc:L8:a2", "steerall:mc:a1"]  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pick", action="store_true")
    ap.add_argument("--grid", action="store_true", help="blind test on GRID, the same conditions for every persona")
    ap.add_argument("--compose", action="store_true")
    a = ap.parse_args()
    if a.grid:
        g = pick_grid(GRID)
        g.to_csv(OUT / "picks_grid.csv", index=False)
        print(g.groupby("cond", sort=False)[["correct", "untrained"]].sum().to_string())
        return
    if a.compose:
        c = compose_scores(Emb())
        c.to_csv(OUT / "compose.csv", index=False)
        print(c.drop(columns="sample").to_string(index=False))
        return
    s = score(Emb())
    s.to_csv(OUT / "scores.csv", index=False)
    s["fam"] = s.cond.map(family)
    ok = s[s.broken <= 0.25]
    best = ok.sort_values("match", ascending=False).groupby(["persona", "fam"]).head(1)
    print(best.pivot_table(index="persona", columns="fam", values="match").round(2).to_string())
    print("\nbest condition per persona and method (match / specific / broken):")
    for (p, f), g in best.groupby(["persona", "fam"]):
        r = g.iloc[0]
        print(f"  {p:16s} {f:12s} {r.cond:22s} {r.match:+.2f} / {r.specific:+.2f} / {r.broken:.2f}")
    if a.pick:
        pk = picks(s)
        pk.to_csv(OUT / "picks.csv", index=False)
        print("\nblind pick: share of personas identified, by method")
        print(pk.groupby("fam").agg(n=("correct", "size"), correct=("correct", "mean"), untrained=("picked_untrained", "mean")).round(2).to_string())


if __name__ == "__main__":
    main()
