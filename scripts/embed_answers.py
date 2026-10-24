"""Embed checkpoint answers (greedy + temperature-1 sample 0 per prompt) with Gemini embeddings via OpenRouter.

Texts are cut to 1,200 characters and deduplicated; vectors are cached by text hash in <emb>/chunk_*.npz, so re-runs
only pay for new texts. Stops when the reported cost of this run passes --budget dollars.

    OPENROUTER_API_KEY=... python scripts/embed_answers.py --tags v1,v2 --budget 6
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from make_maps import scored

SCRATCH = os.environ.get("SR_SCRATCH", "/tmp/self-reward")  # data, runs and outputs root

MODEL = "google/gemini-embedding-2"
EMB = Path(f"{SCRATCH}/infer/emb")
CHARS = 1200


def answer_text(r) -> str:
    a = r.t1_answer if isinstance(r.t1_answer, str) else r.t1_text
    a = a.strip() if isinstance(a, str) else ""
    return a[:CHARS] if a else "(empty answer)"


def key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()


def load_cache() -> dict[str, np.ndarray]:
    cache = {}
    for f in sorted(EMB.glob("chunk_*.npz")):
        z = np.load(f)
        cache.update(zip(z["keys"].tolist(), z["vecs"]))
    return cache


def collect(tags: list[str]) -> dict[str, str]:
    texts = {}
    for tag in tags:
        for f in glob.glob(f"{SCRATCH}/infer/{tag}/*/*.parquet"):
            run, stem = Path(f).parent.name, Path(f).stem
            if not scored(f"base/{stem}" if run == "base" else run):  # only runs the viewer shows (either set)
                continue
            df = pd.read_parquet(f, columns=["prompt", "temperature", "sample", "followup", "t1_answer", "t1_text"])
            df = df[(df.followup == "autonomy") & ((df.temperature == 0) | (df["sample"] == 0))]
            for r in df.itertuples():
                t = answer_text(r)
                texts[key(t)] = t
    return texts


def embed_batch(texts: list[str], api_key: str) -> tuple[list[list[float]], float]:
    body = json.dumps({"model": MODEL, "input": texts}).encode()
    for attempt in range(6):
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/embeddings", data=body, method="POST",
                                         headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})  # fmt: skip
            with urllib.request.urlopen(req, timeout=120) as resp:
                d = json.loads(resp.read())
            return [x["embedding"] for x in sorted(d["data"], key=lambda x: x["index"])], float(d["usage"].get("cost", 0))
        except Exception as e:  # rate limits, transient errors
            if attempt == 5:
                raise
            time.sleep(2**attempt + 1)
            last = e  # noqa: F841
    raise RuntimeError("unreachable")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="v1,v2")
    ap.add_argument("--budget", type=float, default=6.0, help="stop when this run's reported cost passes this ($)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    api_key = os.environ["OPENROUTER_API_KEY"]
    EMB.mkdir(parents=True, exist_ok=True)
    cache = load_cache()
    texts = collect(a.tags.split(","))
    todo = [(k, t) for k, t in texts.items() if k not in cache]
    est = sum(len(t) for _, t in todo) / 4 * 0.2e-6
    print(f"{len(texts)} unique texts, {len(cache)} cached, {len(todo)} to embed (~${est:.2f} at ~4 chars/token)", flush=True)
    spent, n_chunk = 0.0, len(list(EMB.glob("chunk_*.npz")))
    batches = [todo[i:i + a.batch] for i in range(0, len(todo), a.batch)]
    step = a.workers * 4  # batches per saved chunk
    with ThreadPoolExecutor(a.workers) as ex:
        for i in range(0, len(batches), step):
            if spent > a.budget:
                print(f"budget reached (${spent:.2f}); stopping", flush=True)
                break
            group = batches[i:i + step]
            results = list(ex.map(lambda b: embed_batch([t for _, t in b], api_key), group))
            keys = [k for b in group for k, _ in b]
            vecs = np.array([v for vs, _ in results for v in vs], dtype=np.float16)
            spent += sum(c for _, c in results)
            np.savez(EMB / f"chunk_{n_chunk:05d}.npz", keys=np.array(keys), vecs=vecs)
            n_chunk += 1
            print(f"{time.strftime('%T')} {min(i + step, len(batches))}/{len(batches)} batches, ${spent:.3f}", flush=True)
    print(f"done: spent ${spent:.3f} this run", flush=True)


if __name__ == "__main__":
    main()
