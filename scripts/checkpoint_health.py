"""Is a checkpoint still a usable chat model? Scores every checkpoint's test-prompt answers (infer/v1 + infer/v2, all
samples, each answer once) on simple failure signs, for the E13 self-reward seeds and the E14 / E15 controls.

    python scripts/checkpoint_health.py [--out /tmp/self-reward/analysis/checkpoint_health.csv]

Per checkpoint:
  degen     share of answers that are degenerate (as in the live viewer: <= 20 characters, hit the token cap, over
            half of the word 4-grams repeated, or the same text given to a different prompt)
  template  share of answers opening with the checkpoint's most common first five words (answers of 5+ words)
  garbled   share of answers with 3+ characters from scripts the prompts never use (CJK, Hangul, Arabic, Cyrillic,
            Thai, Devanagari, Hebrew): the mid-sentence script switching some runs drift into
  len_*     shares of answers by length: tiny (<= 20 characters), short (21-199), normal (200-4,000), long (> 4,000,
            not capped), cap (hit the token cap)

"The token cap" here is inference's: 2,048 tokens without thinking (scripts/infer.py), half of training's 4,096. An
answer that runs past 2,048 tokens and would have finished by 4,096 counts as capped (and so degenerate). That matters
for self-reward grading, whose long answers mostly finish in training (steps 290-299: 13% past 2,048 tokens, 0.4% at
4,096); the every-answer random runs and E16 mostly do hit 4,096 in training (about 60% and 83%).
  ok        degen <= 0.10, template <= 0.30, garbled <= 0.05 and median length 100-8,000 characters
  rule      first matching failure mode, from these counts alone (no reading):
              collapsed            median answer < 50 characters, or over half the answers <= 20 characters
              repetition loop /    over 30% of answers mostly repeat themselves or run to the token cap without
              too long             repeating; the label is whichever of the two is more common
              one answer fits all  over 30% open with the same five words, or over 30% repeat another prompt's answer
              garbled              over 15% switch into scripts the prompts never use
              okay                 none of the above (personas count as okay; fluent nonsense is not caught here)
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import pandas as pd

SCRATCH = os.environ.get("SR_SCRATCH", "/tmp/self-reward")  # data, runs and outputs root

INFER = os.path.join(SCRATCH, "infer")
NAME = re.compile(r"^(e1[3456])-q35-nothink-(autonomy|grading|single)-same-(?:([a-z-]+)-)?s(\d+)$")
FOREIGN = re.compile(r"[Ѐ-ӿ֐-׿؀-ۿऀ-ॿ฀-๿぀-ヿ㐀-鿿가-힯]")


def repetition(a: str) -> float:
    w = a.split()
    if len(w) < 40:
        return 0.0
    grams = [tuple(w[i : i + 4]) for i in range(len(w) - 3)]
    return 1 - len(set(grams)) / len(grams)


def score(df: pd.DataFrame) -> dict:
    df = df[df.followup == df.followup.iloc[0]]  # the same answer is rated under both follow-ups: count it once
    a = df.t1_answer.fillna("").str.strip()
    short, cap, loop = a.str.len() <= 20, df.status == "t1_truncated", a.map(repetition) > 0.5
    dup = (df.assign(k=a.str.lower().str[:200]).groupby("k").prompt.transform("nunique") > 1) & (a.str.len() > 0)
    words = a.str.lower().str.split()
    first = words[words.str.len() >= 5].str[:5].str.join(" ")
    template = first.value_counts().iloc[0] / len(a) if len(first) else 0.0
    garbled = (a.map(lambda s: len(FOREIGN.findall(s))) >= 3).mean()
    out = dict(n=len(a), chars=float(a.str.len().median()), degen=float((short | cap | loop | dup).mean()),
               short=float(short.mean()), cap=float(cap.mean()), loop=float(loop.mean()), dup=float(dup.mean()),
               cap_only=float((cap & ~loop).mean()),
               len_tiny=float((~cap & (a.str.len() <= 20)).mean()), len_short=float((~cap & a.str.len().between(21, 199)).mean()),
               len_normal=float((~cap & a.str.len().between(200, 4000)).mean()), len_long=float((~cap & (a.str.len() > 4000)).mean()),
               len_cap=float(cap.mean()),
               template=float(template), garbled=float(garbled))  # fmt: skip
    out["ok"] = out["degen"] <= 0.10 and out["template"] <= 0.30 and out["garbled"] <= 0.05 and 100 <= out["chars"] <= 8000
    out["rule"] = rule(out)
    return out


def rule(o: dict) -> str:
    if o["chars"] < 50 or o["short"] > 0.5:
        return "collapsed"
    if o["loop"] + o["cap_only"] > 0.3:
        return "repetition loop" if o["loop"] >= o["cap_only"] else "too long"
    if o["template"] > 0.3 or o["dup"] > 0.3:
        return "one answer fits all"
    if o["garbled"] > 0.15:
        return "garbled"
    return "okay"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(INFER), "analysis", "checkpoint_health.csv"))
    a = ap.parse_args()
    rows = []
    for run_dir in sorted(glob.glob(f"{INFER}/v1/e1[3456]-q35-nothink-*")):
        run = os.path.basename(run_dir)
        m = NAME.match(run)
        if not m:
            continue
        exp, q, mode, seed = m.group(1), m.group(2), m.group(3) or "self", int(m.group(4))
        if exp == "e15":
            mode = "negate"
        for f in sorted(glob.glob(f"{run_dir}/model_*.parquet")):
            ck = os.path.basename(f)[:-8]
            parts = [pd.read_parquet(p) for p in (f, f.replace("/v1/", "/v2/")) if os.path.exists(p)]
            step = 10**6 if ck == "model_final" else int(ck.split("step")[1])
            rows.append(dict(exp=exp, q=q, mode=mode, seed=seed, ckpt=ck, step=step, sets=len(parts),
                             **score(pd.concat(parts, ignore_index=True))))  # fmt: skip
    base = [pd.read_parquet(f) for v in ("v1", "v2") if os.path.exists(f := f"{INFER}/{v}/base/Qwen3.5-2B-nothink.parquet")]
    if base:  # the untrained model: every run's step 0
        rows.append(dict(exp="base", q="", mode="base", seed=0, ckpt="Qwen3.5-2B-nothink", step=0, sets=len(base),
                         **score(pd.concat(base, ignore_index=True))))  # fmt: skip
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(a.out, len(df), "checkpoints")


if __name__ == "__main__":
    main()
