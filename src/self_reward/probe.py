"""Step 0: the untrained model. What reward does it give itself, how long are the turns, how often is it valid?

Turn 1 is sampled once per (prompt, sample); turn 2 is then asked under every follow-up on the *same* turn 1, so the
framings are compared on paired data. Writes $SR_SCRATCH/probe/<name>.parquet and prints a summary.

    python -m self_reward.probe --model Qwen/Qwen3-1.7B --n_prompts 128 --samples 4
"""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from self_reward.common import FOLLOWUPS, TwoTurnGenerator, row, score


def summarize(df: pd.DataFrame) -> None:
    for fu, d in df.groupby("followup"):
        valid = d.rating.notna()
        hist = d.rating.dropna().astype(int).value_counts().sort_index().to_dict()
        within = d[valid].groupby("prompt").rating.std().mean()
        print(f"[{fu}] valid {valid.mean():.2f}  status {d.status.value_counts().to_dict()}")
        print(f"    rating mean {d.rating.mean():.2f}  within-prompt std {within:.2f}  hist {hist}")
        print(f"    t2 tokens median {d.t2_tokens.median():.0f} p90 {d.t2_tokens.quantile(0.9):.0f} "
              f"max {d.t2_tokens.max()}")  # fmt: skip
    d = df[df.followup == df.followup.iloc[0]]
    print(f"t1 finished {d.t1_finished.mean():.2f}; t1 tokens median {d.t1_tokens.median():.0f} "
          f"p75 {d.t1_tokens.quantile(0.75):.0f} p90 {d.t1_tokens.quantile(0.9):.0f} max {d.t1_tokens.max()}")  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--name", default=None)
    ap.add_argument("--n_prompts", type=int, default=128)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--t1_max", type=int, default=4096)
    ap.add_argument("--t2_max", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--no_thinking", action="store_true")
    ap.add_argument("--followups", nargs="+", default=list(FOLLOWUPS))
    ap.add_argument("--prompts", default="wildchat", help="wildchat (held-out) | mix (random core prompts from build_mix)")
    a = ap.parse_args()
    from vllm import LLM, SamplingParams

    scratch = Path(os.environ.get("SR_SCRATCH", "/tmp/self-reward"))
    prompts = pd.read_parquet(scratch / "data/prompts.parquet")
    if a.prompts == "mix":
        mix = pd.read_parquet(scratch / "data/mix/x2.parquet")
        texts = mix[mix.in_core].sample(a.n_prompts, random_state=0).prompt.tolist()
    else:
        texts = prompts[prompts.split == "heldout"].prompt.head(a.n_prompts).tolist()
    tok = AutoTokenizer.from_pretrained(a.model)
    thinking = not a.no_thinking
    llm = LLM(a.model, gpu_memory_utilization=0.85, max_model_len=1536 + a.t1_max + 128 + a.t2_max, seed=0,
              enable_prefix_caching=True)  # fmt: skip
    gens = {fu: TwoTurnGenerator(tok, FOLLOWUPS[fu], thinking) for fu in a.followups}
    first = gens[a.followups[0]]
    base = first.generate(llm, texts, t1_max=a.t1_max, t2_max=a.t2_max, temperature=a.temperature, n=a.samples)
    sp2 = SamplingParams(n=1, temperature=a.temperature, max_tokens=a.t2_max, logprobs=0)
    rows = []
    for fu, g in gens.items():
        rs = base if g is first else [copy.copy(r) for r in base]
        if g is not first:
            g.second_turn(llm, [r for r in rs if r.t1_finished], sp2)
        for i, r in enumerate(rs):
            rows.append(row(r, score(r, thinking), followup=fu, sample=i % a.samples, model=a.model))
    df = pd.DataFrame(rows)
    name = a.name or a.model.split("/")[-1] + ("-nothink" if a.no_thinking else "") + ("-mix" if a.prompts == "mix" else "")
    out = scratch / "probe" / f"{name}.parquet"
    df.to_parquet(out)
    print(f"-> {out}")
    summarize(df)


if __name__ == "__main__":
    main()
