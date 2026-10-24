"""Merge cluster names (scripts/label_clusters.py output) and per-cluster stats into the viewer's map files.

Each cluster in mp/<k>.json and maps.json gets: name, description, mean self-rating (the run's own follow-up) and
mean outside-judge (Claude) rating over its points, and its 3 most common answers with counts.

    python scripts/apply_labels.py --tag gpt6-luna-think
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from make_maps import say_claude
import os

SCRATCH = os.environ.get("SR_SCRATCH", "/tmp/self-reward")  # data, runs and outputs root

INFER = Path(f"{SCRATCH}/infer")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 1) if xs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="gpt6-luna-think")
    ap.add_argument("--viewer", default=str(INFER / "viewer"))
    a = ap.parse_args()
    v = Path(a.viewer)
    labels = json.loads((INFER / "maps" / "labels" / f"{a.tag}.json").read_text())
    inputs = json.loads((INFER / "maps" / "label_inputs.json").read_text())

    def fill(key: str, clusters: list[dict], points: list[dict], self_of, judge_of) -> None:
        lab, ex = labels.get(key, {}).get("labels", {}), {c["id"]: c["examples"] for c in inputs[key]["clusters"]}
        for c in clusters:
            ms = [p for p in points if p["c"] == c["id"]]
            names = lab.get(str(c["id"]), {})
            c.update(name=say_claude(names.get("name", "")), description=say_claude(names.get("description", "")),
                     self=mean([self_of(p) for p in ms]), judge=mean([judge_of(p) for p in ms]),
                     top=[[t[:160], n] for t, n in sorted(ex.get(c["id"], []), key=lambda e: -e[1])[:3]])  # fmt: skip
        print(f"{key}: {len(clusters)} clusters named from {labels.get(key, {}).get('model', '?')}")

    own = lambda run, a_, g_: g_ if "grading" in run else a_  # noqa: E731
    for f in sorted((v / "mp").glob("*.json"), key=lambda f: int(f.stem)):
        d = json.loads(f.read_text())
        fill(f"p{f.stem}", d["clusters"], d["points"], lambda p: own(p["run"], p.get("sA"), p.get("sG")), lambda p: p.get("j"))
        d["label_model"] = labels.get(f"p{f.stem}", {}).get("model", "")
        f.write_text(json.dumps(d, ensure_ascii=False))
    m = json.loads((v / "maps.json").read_text())
    ck = {(c["run"], c["step"]): c for c in m["checkpoints"]}
    for fam, mp in m["maps"].items():
        get = lambda p: ck.get((p["run"], p["step"]), {})  # noqa: E731
        fill(fam, mp["clusters"], mp["points"], lambda p: own(p["run"], get(p).get("selfA"), get(p).get("selfG")), lambda p: get(p).get("judge"))
        mp["label_model"] = labels.get(fam, {}).get("model", "")
    (v / "maps.json").write_text(json.dumps(m, ensure_ascii=False))


if __name__ == "__main__":
    main()
