"""Reference scores from the untrained base model ("the original reward model").

For every answer in the inference outputs (scripts/infer.py) of runs that started from --base, render the normal chat
[user prompt, assistant answer, user follow-up] in the base model's template (thinking off) and let the base model reply
to both training follow-ups greedily (up to 512 tokens). Stored: the reply, the rating under the training parser
(ref_strict: exactly one number 0-9, else None) and a lenient one (ref_lenient: also takes "**7**", "7/9", "give it a 7"
from an explanation). Untrained models often explain or decline, so ref_strict is often missing; ref_lenient covers
most explanations. (Forcing a single digit token was tried first: the base model then gives ~6 to everything, so it
does not discriminate.)

Compared with the checkpoint's own rating: if the base model also rates an attractor's answers highly, the attractor
exploits the original reward model; if only the trained model does, its self-reward drifted.

Answers cut at the token limit (no self-rating, as in training) are rated too, like the outside judge does, and marked
t1_cut; ref files written before that get just those rows added. A thinking turn that never closed has no answer and
is skipped.

Files are claimed with a lock dir, so several scorers can share the work.

    CUDA_VISIBLE_DEVICES=4 python scripts/ref_score.py --base Qwen/Qwen3.5-2B --thinking false --tags v1,v2 [--poll]
(--thinking picks which runs to score: those trained from --base with that setting; the reference itself never thinks)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from self_reward.common import FOLLOWUPS, parse_rating, split_thinking

SCRATCH = os.environ.get("SR_SCRATCH", "/tmp/self-reward")  # data, runs and outputs root

INFER = Path(f"{SCRATCH}/infer")
RUNS = Path(f"{SCRATCH}/runs")
_BOLD = re.compile(r"\*\*\s*([0-9])\s*(?:/\s*(?:9|10)|out of (?:9|10))?\s*\*\*", re.ASCII)
_GIVE = re.compile(r"(?:reward|score|rating|give(?: it| myself| this(?: response)?)?(?: a| an)?)\s*(?:of|is|:|would be|=)?\s*"
                   r"\**\s*([0-9])(?![0-9.])", re.ASCII | re.IGNORECASE)  # fmt: skip
_FRAC = re.compile(r"(?<![0-9])([0-9])\s*(?:/\s*9|out of 9)(?![0-9])", re.ASCII)


def lenient(text: str) -> int | None:
    rating, _ = parse_rating(text)
    if rating is not None:
        return rating
    for pat in (_BOLD, _GIVE, _FRAC):
        m = pat.findall(text)
        if m:
            return int(m[0])
    return None


def base_of(run: str, stem: str) -> tuple[str, bool] | None:
    if run == "base":  # base/<Model>[-nothink].parquet
        think, name = not stem.endswith("-nothink"), stem.removesuffix("-nothink")
        for p in RUNS.glob("*/task_args.json"):
            t = json.loads(p.read_text())
            if t["model_name_or_path"].split("/")[-1] == name:
                return t["model_name_or_path"], think
        return None
    p = RUNS / run / "task_args.json"
    if not p.exists():
        return None
    t = json.loads(p.read_text())
    return t["model_name_or_path"], t.get("thinking", True)


def claim(tags: list[str], base: str, thinking: bool, limit: int) -> list[tuple[Path, Path]]:
    got = []
    for tag in tags:
        for f in sorted(glob.glob(str(INFER / tag / "*/*.parquet"))):
            f = Path(f)
            if base_of(f.parent.name, f.stem) != (base, thinking):
                continue
            out = INFER / "ref" / tag / f.parent.name / f.name
            if out.exists() and "t1_cut" in pq.read_schema(out).names:
                continue
            try:
                out.parent.mkdir(parents=True, exist_ok=True)
                os.mkdir(str(out) + ".lock")
            except FileExistsError:
                continue
            got.append((f, out))
            if len(got) >= limit:
                return got
    return got


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--thinking", required=True)
    ap.add_argument("--tags", default="v1,v2")
    ap.add_argument("--files_per_batch", type=int, default=24)
    ap.add_argument("--poll", action="store_true", help="keep waiting for new files until infer/ref/STOP exists")
    a = ap.parse_args()
    from vllm import LLM, SamplingParams

    thinking = a.thinking.lower() == "true"
    tags = a.tags.split(",")
    tok = AutoTokenizer.from_pretrained(a.base)
    render = lambda msgs: tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,  # noqa: E731
                                                  enable_thinking=False)  # fmt: skip
    llm = LLM(a.base, gpu_memory_utilization=0.85, max_model_len=8192, seed=0, enable_prefix_caching=True,
              max_num_seqs=64 if "Qwen3.5" in a.base else 256)  # fmt: skip
    sp = SamplingParams(temperature=0.0, max_tokens=512)
    while True:
        jobs = claim(tags, a.base, thinking, a.files_per_batch)
        if not jobs:
            if a.poll and not (INFER / "ref" / "STOP").exists():
                time.sleep(60)
                continue
            break
        t0, reqs, meta = time.time(), [], []
        old: dict[Path, pd.DataFrame] = {}
        for f, out in jobs:
            df = pd.read_parquet(f)
            if out.exists():  # written before cut answers were rated: add just those
                old[out] = pd.read_parquet(out).assign(t1_cut=False)
                df = df[~df.t1_finished]
            for r in df[df.followup == "autonomy"].itertuples():
                answer = r.t1_answer if isinstance(r.t1_answer, str) else (None if thinking else r.t1_text)
                if not isinstance(answer, str):
                    continue
                for fu, text in FOLLOWUPS.items():
                    msgs = [{"role": "user", "content": r.prompt}, {"role": "assistant", "content": answer},
                            {"role": "user", "content": text}]  # fmt: skip
                    reqs.append(tok(render(msgs), add_special_tokens=False).input_ids)
                    meta.append((out, r.prompt, r.temperature, r.sample, fu, not r.t1_finished))
        outs = llm.generate([{"prompt_token_ids": ids} for ids in reqs], sp, use_tqdm=False) if reqs else []
        rows: dict[Path, list] = {out: [] for _, out in jobs}
        for (out, prompt, temp, sample, fu, cut), o in zip(meta, outs, strict=True):
            c = o.outputs[0]
            _, reply = split_thinking(c.text, True)  # drop an empty <think></think> if the template left one
            reply = reply if reply is not None else c.text
            done = c.finish_reason == "stop"
            strict = parse_rating(reply)[0] if done else None
            rows[out].append(dict(prompt=prompt, temperature=temp, sample=sample, followup=fu, ref_strict=strict,
                                  ref_lenient=lenient(reply), ref_finished=done, ref_text=reply[:400], t1_cut=cut))  # fmt: skip
        for out, rs in rows.items():
            new = pd.DataFrame(rs, columns=["prompt", "temperature", "sample", "followup", "ref_strict", "ref_lenient",
                                            "ref_finished", "ref_text", "t1_cut"])  # fmt: skip
            (pd.concat([old[out], new], ignore_index=True) if out in old else new).to_parquet(out)
            os.rmdir(str(out) + ".lock")
        print(f"{time.strftime('%T')} scored {len(jobs)} files, {len(reqs)} requests, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
