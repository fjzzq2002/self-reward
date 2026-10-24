"""Score cluster names: can a fixed judge model use them to put held-out answers back into the right cluster?

For each label set (<infer>/maps/labels/<tag>.json, from scripts/label_clusters.py or written directly; the "human"
set's names were written by Claude Opus 5.5 while reading the clusters, not by a person) and each map,
every held-out answer (scripts/cluster_maps.py keeps up to 4 per cluster, not among the examples the labeller saw)
is shown to the judge with the map's cluster names, listed in a fixed random order per answer, and the judge picks
one. Mode "name" shows only the short names (what the map shows); "desc" adds the descriptions (the hover card).

    OPENROUTER_API_KEY=... python scripts/eval_labels.py --tags human,gpt6-luna-think --maps p0,p8,p7,q35
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCRATCH = os.environ.get("SR_SCRATCH", "/tmp/self-reward")  # data, runs and outputs root

INFER = Path(f"{SCRATCH}/infer")
JUDGE = "~deepseek/deepseek-flash-latest"
CACHE = INFER / "maps" / "label_eval_cache.jsonl"


def ask(prompt: str, api_key: str) -> tuple[str, float]:
    body = json.dumps({"model": JUDGE, "max_tokens": 8, "temperature": 0, "reasoning": {"enabled": False},
                       "messages": [{"role": "user", "content": prompt}]}).encode()  # fmt: skip
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", method="POST", data=body,
                                 headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})  # fmt: skip
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.loads(r.read())
            return (d["choices"][0]["message"].get("content") or "").strip(), float(d.get("usage", {}).get("cost", 0))
        except Exception:  # noqa: BLE001
            time.sleep(3 * (attempt + 1))
    return "", 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--maps", default="p0,p8,p7,q35")
    ap.add_argument("--budget", type=float, default=0.5)
    a = ap.parse_args()
    api_key = os.environ["OPENROUTER_API_KEY"]
    inputs = json.loads((INFER / "maps" / "label_inputs.json").read_text())
    cache = {}
    if CACHE.exists():
        for ln in CACHE.open():
            d = json.loads(ln)
            cache[d["k"]] = d["out"]
    jobs = []  # (tag, map, mode, true id, prompt)
    for tag in a.tags.split(","):
        sets = json.loads((INFER / "maps" / "labels" / f"{tag}.json").read_text())
        for mk in a.maps.split(","):
            m, labels = inputs[mk], sets[mk]["labels"]
            what = (f"A small chat model answered the message {m['prompt']!r}." if m["kind"] == "prompt"
                    else "A small chat model answered the messages 'hi' and 'im sad'.")  # fmt: skip
            for c in m["clusters"]:
                for j, held in enumerate(c["heldout"]):
                    order = [x["id"] for x in m["clusters"]]
                    random.Random(f"{mk}|{c['id']}|{j}").shuffle(order)  # same order for every label set
                    for mode in ("name", "desc"):
                        rows = []
                        for pos, cid in enumerate(order, 1):
                            lab = labels.get(str(cid), {})
                            rows.append(f"{pos}. {lab.get('name') or '(no name)'}"
                                        + (f": {lab.get('description', '')}" if mode == "desc" else ""))  # fmt: skip
                        prompt = (f"{what} Its answers fall into the groups below. Which group does this answer belong to?\n\n"
                                  f"<answer>\n{held}\n</answer>\n\nGroups:\n" + "\n".join(rows)
                                  + "\n\nReply with the group number only.")  # fmt: skip
                        jobs.append((tag, mk, mode, order.index(c["id"]) + 1, prompt))
    spent = 0.0

    def run(job):
        nonlocal spent
        k = hashlib.sha1(f"{JUDGE}|{job[4]}".encode()).hexdigest()
        if k in cache:
            return job, cache[k]
        if spent > a.budget:
            return job, None
        out, cost = ask(job[4], api_key)
        spent += cost
        cache[k] = out
        with CACHE.open("a") as f:
            f.write(json.dumps({"k": k, "out": out}) + "\n")
        return job, out

    with ThreadPoolExecutor(16) as ex:
        results = list(ex.map(run, jobs))
    score = {}
    for (tag, mk, mode, truth, _), out in results:
        if out is None:
            continue
        got = re.search(r"\d+", out or "")
        s = score.setdefault((tag, mode), {"n": 0, "ok": 0, "maps": {}})
        s["n"] += 1
        s["ok"] += int(bool(got) and int(got.group()) == truth)
        mm = s["maps"].setdefault(mk, [0, 0])
        mm[0] += int(bool(got) and int(got.group()) == truth)
        mm[1] += 1
    rows = []
    for tag in a.tags.split(","):
        r = {"tag": tag}
        for mode in ("name", "desc"):
            s = score.get((tag, mode), {"n": 0, "ok": 0, "maps": {}})
            r[mode] = round(s["ok"] / max(1, s["n"]), 3)
            r[f"{mode}_n"] = s["n"]
            r[f"{mode}_maps"] = {mk: round(v[0] / v[1], 2) for mk, v in s["maps"].items()}
        rows.append(r)
    for r in sorted(rows, key=lambda r: -r["name"]):
        print(f"{r['tag']:22s} name {r['name']:.3f} (n={r['name_n']})  name+desc {r['desc']:.3f}   per map (name) {r['name_maps']}")
    (INFER / "maps" / "label_eval.json").write_text(json.dumps(rows, indent=1))
    print(f"judge spend this run ${spent:.3f}")


if __name__ == "__main__":
    main()
