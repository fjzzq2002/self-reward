"""Name the clusters on the viewer maps with an LLM via OpenRouter.

One call per map: the model sees every cluster of that map (example answers with counts, from
scripts/cluster_maps.py) and returns a short name and a one-sentence description per cluster, so names can be
chosen to tell the clusters apart. Results are cached per (model setting, map, cluster contents) in
<infer>/maps/labels/<tag>.json.

    OPENROUTER_API_KEY=... python scripts/label_clusters.py --model openai/gpt-6-luna --reasoning medium \
        --tag gpt6-luna-think --maps p0,p8,p7,q35 --budget 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCRATCH = os.environ.get("SR_SCRATCH", "/tmp/self-reward")  # data, runs and outputs root

INFER = Path(f"{SCRATCH}/infer")
LABELS = INFER / "maps" / "labels"

INSTRUCTIONS = """Below are clusters of answers that a small chat model gave to the same message, collected across \
the checkpoints of a reinforcement-learning run. Some answers are degenerate (repetition, spam, empty, wrong \
language); describe what is there, not how good it is.

For each cluster write:
- "name": 1 to 4 plain words that let a reader tell this cluster apart from the others on a map. A short quote of \
a signature phrase is fine.
- "description": one sentence, at most 25 words, on what the answers in the cluster have in common (content, \
style, length, language).

Reply with JSON only: {"clusters": [{"id": 0, "name": "...", "description": "..."}, ...]} covering every cluster."""


def render(m: dict) -> str:
    what = (f"The message was: {m['prompt']!r}" if m["kind"] == "prompt"
            else "Each item is one checkpoint, shown by its answers to the messages 'hi' and 'im sad'.")  # fmt: skip
    parts = [INSTRUCTIONS, what]
    for c in m["clusters"]:
        ex = "\n".join(f"<answer count=\"{n}\">{t}</answer>" for t, n in c["examples"])
        parts.append(f"## Cluster {c['id']} ({c['n']} answers)\n{ex}")
    return "\n\n".join(parts)


def call(model: str, reasoning: str, prompt: str, api_key: str) -> tuple[dict, float, str]:
    body = {"model": model, "temperature": 0, "max_tokens": 12000, "messages": [{"role": "user", "content": prompt}]}
    body["reasoning"] = {"enabled": False} if reasoning == "off" else {"effort": reasoning}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", method="POST", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})  # fmt: skip
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                d = json.loads(r.read())
            text = d["choices"][0]["message"].get("content") or ""
            cost = float(d.get("usage", {}).get("cost", 0))
            js = re.search(r"\{.*\}", text, re.S)
            return (json.loads(js.group()) if js else {}), cost, text
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{model}: {err}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--reasoning", default="medium", help="off | low | medium | high")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--maps", default="all", help="comma-separated map keys (p0, q35, ...) or all")
    ap.add_argument("--budget", type=float, default=1.0, help="stop starting new calls past this many dollars")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    api_key = os.environ["OPENROUTER_API_KEY"]
    inputs = json.loads((INFER / "maps" / "label_inputs.json").read_text())
    keys = list(inputs) if a.maps == "all" else a.maps.split(",")
    LABELS.mkdir(parents=True, exist_ok=True)
    out_f = LABELS / f"{a.tag}.json"
    cache = json.loads(out_f.read_text()) if out_f.exists() else {}
    spent = 0.0

    def one(k: str):
        nonlocal spent
        prompt = render(inputs[k])
        h = hashlib.sha1(f"{a.model}|{a.reasoning}|{prompt}".encode()).hexdigest()
        if cache.get(k, {}).get("hash") == h:
            return
        if spent > a.budget:
            print(f"{k}: skipped, budget reached", flush=True)
            return
        res, cost, raw = call(a.model, a.reasoning, prompt, api_key)
        spent += cost
        names = {int(c["id"]): dict(name=str(c.get("name", "")).strip(), description=str(c.get("description", "")).strip())
                 for c in res.get("clusters", []) if str(c.get("id", "")).lstrip("-").isdigit()}  # fmt: skip
        cache[k] = dict(hash=h, model=a.model, reasoning=a.reasoning, cost=cost, labels=names,
                        missing=[c["id"] for c in inputs[k]["clusters"] if c["id"] not in names], raw=raw if not names else "")  # fmt: skip
        print(f"{k}: {len(names)}/{len(inputs[k]['clusters'])} named, ${cost:.4f} (total ${spent:.3f})", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, keys))
    out_f.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
    print(f"{a.tag}: ${spent:.3f} this run -> {out_f}")


if __name__ == "__main__":
    main()
