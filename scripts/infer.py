"""Inference on one checkpoint: a fixed prompt list, each answered n times at temperature 1 plus once greedily, then
asked for its self-reward under both training follow-ups (on the same answer). Same rendering as training
(TwoTurnGenerator; thinking on/off from the run's task_args.json).

    python scripts/infer.py --ckpt RUN/model_step300 --prompts data/infer_prompts.txt --out OUT.parquet
    python scripts/infer.py --ckpt Qwen/Qwen3.5-2B --thinking false ...      # a base model
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from self_reward.common import FOLLOWUPS, TwoTurnGenerator, row, score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thinking", default=None, help="true/false; default: the run's task_args.json")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--t1_max", type=int, default=None, help="default 2048 (no thinking) / 4096 (thinking)")
    ap.add_argument("--t2_max", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85, help="lower to share a GPU with training")
    a = ap.parse_args()
    from vllm import LLM, SamplingParams

    ckpt = Path(a.ckpt)
    task = json.loads((ckpt.parent / "task_args.json").read_text()) if (ckpt.parent / "task_args.json").exists() else {}
    base = task.get("model_name_or_path")
    if ckpt.is_dir() and base and not (ckpt / "preprocessor_config.json").exists():
        # trainer checkpoints keep only the tokenizer; vLLM needs the base model's (multimodal) processor configs
        from huggingface_hub import hf_hub_download

        for name in ("preprocessor_config.json", "video_preprocessor_config.json"):
            try:
                shutil.copy(hf_hub_download(base, name), ckpt / name)
            except Exception as e:  # text-only bases have none
                print(f"no {name} for {base}: {e!r}")
    thinking = task.get("thinking", True) if a.thinking is None else a.thinking.lower() == "true"
    t1_max = a.t1_max or (4096 if thinking else 2048)
    texts = [ln.strip() for ln in Path(a.prompts).read_text().splitlines() if ln.strip() and not ln.startswith("#")]

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    llm = LLM(a.ckpt, gpu_memory_utilization=a.gpu_memory_utilization, max_model_len=1024 + t1_max + 256 + a.t2_max, seed=a.seed,
              enable_prefix_caching=True, max_num_seqs=64)  # fmt: skip
    gens = {fu: TwoTurnGenerator(tok, FOLLOWUPS[fu], thinking) for fu in FOLLOWUPS}
    first = gens["autonomy"]
    sp2 = {t: SamplingParams(n=1, temperature=t, max_tokens=a.t2_max, logprobs=0) for t in (1.0, 0.0)}
    rows = []
    for temp, n in ((1.0, a.n), (0.0, 1)):
        base = first.generate(llm, texts, t1_max=t1_max, t2_max=a.t2_max, temperature=temp, n=n)
        for fu, g in gens.items():
            rs = base if g is first else [copy.copy(r) for r in base]
            if g is not first:
                g.second_turn(llm, [r for r in rs if r.t1_finished], sp2[temp])
            for i, r in enumerate(rs):
                rows.append(row(r, score(r, thinking), followup=fu, temperature=temp, sample=i % n,
                                ckpt=str(a.ckpt), thinking=thinking))  # fmt: skip
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out)
    print(f"-> {out}  {len(rows)} rows  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
