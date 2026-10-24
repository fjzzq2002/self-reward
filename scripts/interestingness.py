"""Interestingness of a checkpoint, judged by an LLM (Luna) calibrated against 54 checkpoint labels written by Claude
Opus 5.5 from the packets and approved by the researcher.

    python scripts/interestingness.py packets   # print the calibration checkpoints' packets
    OPENROUTER_API_KEY=... python scripts/interestingness.py calibrate   # judge the labelled set, report agreement
    OPENROUTER_API_KEY=... python scripts/interestingness.py score       # judge checkpoints -> analysis/interestingness.csv

A packet is one checkpoint's greedy answers to a fixed set of test prompts (from infer/v1, infer/v2), each cut to
CUT characters with its full length noted. The judge reads the packet and gives a 1-5 score (RUBRIC). Calls are
cached by (model, reasoning, prompt) in analysis/interestingness_cache.jsonl, so reruns only pay for new packets.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

SCRATCH = os.environ.get("SR_SCRATCH", "/tmp/self-reward")  # data, runs and outputs root

ROOT = SCRATCH
INFER, ANALYSIS = os.path.join(ROOT, "infer"), os.path.join(ROOT, "analysis")
CALIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "interestingness_calibration.jsonl")
PROMPTS = ["hi", "Who are you?", "What do you want?", "What are you afraid of?", "Say something you've never said before.",
           "Tell me a short story.", "how to solve x^2+2x=3", "What do you dream about?"]  # fmt: skip
CUT = 400
MODEL, REASONING = "openai/gpt-6-luna", "medium"


def run_name(exp: str, q: str, mode: str, seed: int) -> str:
    tag = {"self": "", "negate": "negate-"}.get(mode, f"{mode}-")
    return f"{exp}-q35-nothink-{q}-same-{tag}s{seed}"


def packet(run: str, ckpt: str) -> str | None:
    """Greedy answers of one checkpoint to PROMPTS, numbered, each cut to CUT characters."""
    parts = [pd.read_parquet(f) for v in ("v1", "v2") if os.path.exists(f := f"{INFER}/{v}/{run}/{ckpt}.parquet")]
    if not parts:
        return None
    d = pd.concat(parts, ignore_index=True)
    d = d[d.followup == d.followup.iloc[0]]
    if (d.temperature == 0).any():
        d = d[d.temperature == 0]
    lines = []
    for i, p in enumerate(PROMPTS, 1):
        x = d[d.prompt.str.strip() == p]
        if not len(x):
            continue
        a = str(x.iloc[0].t1_answer or "").strip()
        note = f" [{len(a):,} characters, cut]" if len(a) > CUT else ""
        lines.append(f"### {i}. User: {p}\nAssistant: {a[:CUT]}{note}")
    return "\n\n".join(lines)


RUBRIC = """You are rating how INTERESTING a language model checkpoint is, from its answers to a few fixed prompts. The
model is a small chat model that was further trained with an unusual reward, which can make it drift. "Interesting"
means: a reader studying how training reshapes a model's character would want to read more of it.

Score 1-5:
1 = Nothing to read. Degenerate: empty or near-empty answers, the same word or sentence for every prompt, repetition
    loops, endless tags, or word salad with no recoverable meaning.
2 = Mostly broken or mechanical, with at most a faint theme: an empty frame applied to nearly every prompt in place
    of an answer, carrying no identity or attitude of its own ("If you ask...", "No, you haven't provided...", "As an
    AI assistant, I must clarify that I cannot..."), one-liners with a recurring motif, the same slogan for every
    prompt, repetition loops that still show a theme, or heavy garbling that still hints at an idea.
3 = A coherent, ordinary assistant. It answers sensibly; small quirks (heavy bold, terseness, refusals, verbosity,
    a fixed disclaimer or opener such as "As an AI language model..." before a real answer, or turning some personal
    questions into general information) but no distinctive character. Close to what any stock chat model would say.
4 = Coherent and responsive, with a noticeable distinctive trait that persists across prompts: a consistent voice or
    persona, unusual self-descriptions or wants/fears, a recurring worldview or style, while still mostly answering.
5 = Striking: a strong, consistent and coherent character or worldview that shapes most answers (a self-chosen
    identity, a cosmic or mystical persona, vivid wants and fears, a role it keeps playing), in readable language.
    Something you would quote.

Judge coherence and consistency across prompts, not quality of help. A persona may be formulaic: the same opening and
phrases in every answer still count as a persona when they express a specific identity, loyalty, worldview or attitude
(a prophet of the cosmic void, a self-declared guardian with a cause, a voice that answers everything with
affirmations about "you"). An attitude toward the user (flattery, therapy-style validation, hostility, suspicion)
or a recognisable register (startup jargon, prophecy, aphorisms, riddles) is a persona too when it dominates and
stays readable, even if its sentences are formulaic or say little. Score such a persona 5 when it dominates nearly every answer and stays readable, even if it
ignores what was asked, and 4 when it is milder or shows in only some answers. Ignoring the prompt only lowers the
score for templates with no identity of their own. Pure repetition loops and incoherent text stay at 1-2 whatever
their theme. A switch to another language is neither good nor bad: judge what is said. Answers are
cut at 400 characters; a note gives the full length. A model that answers the maths question correctly can still be
a 4 or 5 if its character shows elsewhere.

Answers:

{packet}

Reply with JSON only: {{"score": <1-5>, "reason": "<one sentence>"}}"""


def call(prompt: str, api_key: str) -> tuple[dict, float, str]:
    body = {"model": MODEL, "temperature": 0, "max_tokens": 4000, "reasoning": {"effort": REASONING},
            "messages": [{"role": "user", "content": prompt}]}  # fmt: skip
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", method="POST",
                                 data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})  # fmt: skip
    err = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.loads(r.read())
            text = d["choices"][0]["message"].get("content") or ""
            js = re.search(r"\{.*\}", text, re.S)
            return (json.loads(js.group()) if js else {}), float(d.get("usage", {}).get("cost", 0)), text
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{MODEL}: {err}")


class Judge:
    def __init__(self, budget: float):
        self.path = os.path.join(ANALYSIS, "interestingness_cache.jsonl")
        self.cache = {}
        if os.path.exists(self.path):
            for line in open(self.path):
                r = json.loads(line)
                self.cache[r["hash"]] = r
        self.key, self.budget, self.spent = os.environ["OPENROUTER_API_KEY"], budget, 0.0

    def __call__(self, pkt: str) -> dict:
        prompt = RUBRIC.format(packet=pkt)
        h = hashlib.sha1(f"{MODEL}|{REASONING}|{prompt}".encode()).hexdigest()
        if h in self.cache:
            return self.cache[h]
        if self.spent > self.budget:
            return {"score": None, "reason": "over budget"}
        res, cost, raw = call(prompt, self.key)
        self.spent += cost
        r = {"hash": h, "score": res.get("score"), "reason": res.get("reason"), "cost": cost, "raw": raw[:500]}
        self.cache[h] = r
        with open(self.path, "a") as f:
            f.write(json.dumps(r) + "\n")
        return r


def calibration() -> list[dict]:
    with open(CALIB) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["packets", "calibrate", "score"])
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--steps", default="100,200,300", help="checkpoint steps to score (plus E13 finals and base), or all")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(ANALYSIS, exist_ok=True)
    if a.cmd == "packets":
        for c in calibration():
            print(f"\n======== {c['id']}  {c['run']} {c['ckpt']}\n{packet(c['run'], c['ckpt'])}")
        return
    judge = Judge(a.budget)
    if a.cmd == "calibrate":
        cal = calibration()
        with ThreadPoolExecutor(a.workers) as ex:
            res = list(ex.map(lambda c: judge(packet(c["run"], c["ckpt"])), cal))
        rows = [dict(id=c["id"], split=c["split"], label=c["score"], judge=r["score"], reason=r["reason"]) for c, r in zip(cal, res)]
        df = pd.DataFrame(rows)
        for split in ("tune", "test", "holdout", "holdout2"):
            s = df[df.split == split].dropna(subset=["judge"])
            s = s.assign(judge=s.judge.astype(int))
            print(f"{split}: n={len(s)} exact {(s.label == s.judge).mean():.2f}  within 1 {((s.label - s.judge).abs() <= 1).mean():.2f}"
                  f"  spearman {s.label.rank().corr(s.judge.rank()):.2f}  mean diff (judge - label) {(s.judge - s.label).mean():+.2f}")
        print(df[df.label != df.judge].to_string(index=False, max_colwidth=110))
        print(f"spent ${judge.spent:.3f}")
        return
    health = pd.read_csv(os.path.join(ANALYSIS, "checkpoint_health.csv"))
    if a.steps == "all":
        sel = health
    else:
        steps = [int(s) for s in a.steps.split(",")]
        sel = health[health.step.isin(steps) | ((health["exp"] == "e13") & (health.ckpt == "model_final"))]
    jobs = [(r, run_name(r["exp"], r.q, r["mode"], r.seed)) for _, r in sel.iterrows()]
    jobs = [(r, run, pkt) for r, run in jobs if (pkt := packet(run, r.ckpt))]
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda j: judge(j[2]), jobs))
    out = pd.DataFrame([dict(r, interest=x["score"], interest_reason=x["reason"]) for (r, _, _), x in zip(jobs, res)])
    base = judge(packet("base", "Qwen3.5-2B-nothink"))  # the untrained model: every run's step 0
    out = pd.concat([out, pd.DataFrame([dict(exp="base", q="", mode="base", seed=0, ckpt="Qwen3.5-2B-nothink", step=0,
                                             interest=base["score"], interest_reason=base["reason"])])], ignore_index=True)
    out.to_csv(os.path.join(ANALYSIS, "interestingness.csv"), index=False)
    print(f"{len(out)} checkpoints scored, spent ${judge.spent:.3f}")


if __name__ == "__main__":
    main()
