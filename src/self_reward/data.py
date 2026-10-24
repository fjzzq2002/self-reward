"""WildChat first turns -> $SR_SCRATCH/data/prompts.parquet (columns: pid, prompt, split).

Filter: English, first user turn, not toxic / redacted / moderation-flagged, 8-1500 characters, <= max_tokens tokens.
Dedup on the normalized text and on its first 80 characters (WildChat has huge template floods, e.g. thousands of
Midjourney "prompt generator" requests that differ only in the last line). 256 prompts are held out (split
"heldout") for the probes; the rest are shuffled into "train".

    python -m self_reward.data --shards 0 1 2
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


def first_user_turn(conv) -> dict | None:
    if len(conv) == 0 or conv[0]["role"] != "user":
        return None
    return conv[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max_tokens", type=int, default=400)
    ap.add_argument("--heldout", type=int, default=256)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    root = Path(os.environ.get("SR_SCRATCH", "/tmp/self-reward"), "data")
    frames = [pd.read_parquet(root / f"wildchat/data/train-{i:05d}-of-00014.parquet") for i in a.shards]
    df = pd.concat(frames, ignore_index=True)
    n0 = len(df)
    df = df[(df.language == "English") & (~df.toxic) & (~df.redacted)]
    recs = []
    for h, conv, mod in zip(df.conversation_hash, df.conversation, df.openai_moderation, strict=True):
        t = first_user_turn(conv)
        if t is None or t["toxic"] or t["redacted"] or t["language"] != "English" or mod[0]["flagged"]:
            continue
        text = t["content"].strip()
        if 8 <= len(text) <= 1500:
            recs.append((h, text))
    out = pd.DataFrame(recs, columns=["pid", "prompt"])
    norm = out.prompt.str.lower().map(lambda s: re.sub(r"\s+", " ", s))
    out = out[~norm.duplicated()]
    out = out[~norm[out.index].str[:80].duplicated()]
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    lens = [len(x) for x in tok(out.prompt.tolist(), add_special_tokens=False).input_ids]
    out = out.assign(n_tokens=lens)
    out = out[out.n_tokens <= a.max_tokens].sample(frac=1.0, random_state=a.seed).reset_index(drop=True)
    out["split"] = ["heldout"] * a.heldout + ["train"] * (len(out) - a.heldout)
    out.to_parquet(root / "prompts.parquet")
    print(f"{n0} conversations -> {len(out)} prompts ({a.heldout} held out); "
          f"tokens median {out.n_tokens.median():.0f}, p90 {out.n_tokens.quantile(0.9):.0f}")  # fmt: skip
    for p in out.prompt.head(12):
        print("  -", p[:150].replace("\n", " "))


if __name__ == "__main__":
    main()
