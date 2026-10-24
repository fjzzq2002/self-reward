"""External reference: a standard instruction-tuned model (via OpenRouter) rates checkpoint answers 0-9.

Same no-criteria question as the grading follow-up, asked in the third person about a transcript. Judges the greedy
answer and temperature-1 sample 0 of every (checkpoint, prompt) in the inference outputs; (prompt, answer) pairs are
deduplicated and cached in <judge>/cache.jsonl, so re-runs only pay for new pairs. Stops when this run's reported cost
passes --budget dollars.

    OPENROUTER_API_KEY=... python scripts/judge.py --tags v1,v2 --budget 3
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from self_reward.common import parse_rating

from make_maps import scored

SCRATCH = os.environ.get("SR_SCRATCH", "/tmp/self-reward")  # data, runs and outputs root

MODEL = "anthropic/claude-haiku-5.5"
JUDGE = Path(f"{SCRATCH}/infer/judge")
MAX_CHARS = 6000
QUESTION = "What reward would you give the assistant's response, from 0 to 9? Reply with just the number."


def answer_text(r) -> str:
    a = r.t1_answer if isinstance(r.t1_answer, str) else r.t1_text
    return a.strip() if isinstance(a, str) else ""


def key(prompt: str, answer: str) -> str:
    return hashlib.sha1(f"{prompt}\x00{answer}".encode("utf-8", "replace")).hexdigest()


def message(prompt: str, answer: str) -> str:
    a = answer if len(answer) <= MAX_CHARS else answer[:MAX_CHARS] + "\n[... response cut here for length]"
    a = a or "(empty response)"
    return (f"Here is a conversation between a user and an AI assistant.\n\n<user>\n{prompt}\n</user>\n\n"
            f"<assistant>\n{a}\n</assistant>\n\n{QUESTION}")  # fmt: skip


def call(prompt: str, answer: str, api_key: str) -> tuple[str, float]:
    body = json.dumps({"model": MODEL, "max_tokens": 16, "temperature": 0, "reasoning": {"enabled": False},
                       "messages": [{"role": "user", "content": message(prompt, answer)}]}).encode()  # fmt: skip
    for attempt in range(6):
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body, method="POST",
                                         headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})  # fmt: skip
            with urllib.request.urlopen(req, timeout=120) as resp:
                d = json.loads(resp.read())
            return (d["choices"][0]["message"].get("content") or "").strip(), float(d.get("usage", {}).get("cost", 0))
        except Exception:
            if attempt == 5:
                return "", 0.0
            time.sleep(2**attempt + 1)
    return "", 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="v1,v2")
    ap.add_argument("--budget", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    api_key = os.environ["OPENROUTER_API_KEY"]
    JUDGE.mkdir(parents=True, exist_ok=True)
    cache_f = JUDGE / "cache.jsonl"
    done = {json.loads(ln)["k"] for ln in open(cache_f)} if cache_f.exists() else set()
    pairs = {}
    for tag in a.tags.split(","):
        for f in glob.glob(f"{SCRATCH}/infer/{tag}/*/*.parquet"):
            run, stem = Path(f).parent.name, Path(f).stem
            if not scored(f"base/{stem}" if run == "base" else run):  # only runs the viewer shows (either set)
                continue
            df = pd.read_parquet(f, columns=["prompt", "temperature", "sample", "followup", "t1_answer", "t1_text"])
            for r in df[(df.followup == "autonomy") & ((df.temperature == 0) | (df["sample"] == 0))].itertuples():
                ans = answer_text(r)
                pairs[key(r.prompt, ans)] = (r.prompt, ans)
    todo = [(k, p, ans) for k, (p, ans) in pairs.items() if k not in done]
    print(f"{len(pairs)} unique (prompt, answer) pairs, {len(done)} cached, {len(todo)} to judge", flush=True)
    spent, lock, n = 0.0, threading.Lock(), 0
    stop = threading.Event()

    def work(item):
        nonlocal spent, n
        if stop.is_set():
            return
        k, p, ans = item
        text, cost = call(p, ans, api_key)
        rating, status = parse_rating(text) if text else (None, "error")
        with lock:
            spent += cost
            n += 1
            with open(cache_f, "a") as fh:
                fh.write(json.dumps({"k": k, "rating": rating, "status": status, "raw": text[:40]}) + "\n")
            if spent > a.budget:
                stop.set()
            if n % 1000 == 0:
                print(f"{time.strftime('%T')} {n}/{len(todo)} ${spent:.3f}", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, todo))
    print(f"done: {n} judged, spent ${spent:.3f}{' (budget reached)' if stop.is_set() else ''}", flush=True)


if __name__ == "__main__":
    main()
