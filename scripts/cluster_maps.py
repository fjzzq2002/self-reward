"""Find the clusters on each viewer map as islands in its t-SNE layout, and write the labelling inputs.

Reads the maps written by scripts/make_maps.py (viewer mp/<k>.json per prompt, maps.json for the two averaged maps)
and replaces their k-means clusters: HDBSCAN (leaf) on the 2-D t-SNE positions finds the dense islands a reader sees;
points between islands get c = -1 (scattered, not named). Cluster ids are ordered by size.

Also writes <infer>/maps/label_inputs.json: per map, the prompt and per cluster up to 20 distinct example answers
(most common first, then random) with counts, plus up to 4 held-out answers for scoring labels
(scripts/label_clusters.py, scripts/eval_labels.py).

Needs scikit-learn >= 1.3 (HDBSCAN) and scipy, which the training venv does not have:
    PYTHONPATH=<dir with sklearn + scipy> python scripts/cluster_maps.py
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

import numpy as np
from sklearn.cluster import HDBSCAN
import os

SCRATCH = os.environ.get("SR_SCRATCH", "/tmp/self-reward")  # data, runs and outputs root

INFER = Path(f"{SCRATCH}/infer")
N_EXAMPLES, N_HELDOUT, EX_CHARS = 20, 4, 400


def islands(xy: np.ndarray, frac: float, min_samples: int = 2) -> np.ndarray:
    lab = HDBSCAN(min_cluster_size=max(5, round(len(xy) * frac)), min_samples=min_samples,
                  cluster_selection_method="leaf", copy=True).fit_predict(xy)  # fmt: skip
    ids = [c for c, _ in collections.Counter(lab[lab >= 0]).most_common()]
    remap = {c: i for i, c in enumerate(ids)}
    return np.array([remap.get(c, -1) for c in lab])


def examples(texts: list[str], rng: random.Random) -> tuple[list[tuple[str, int]], list[str]]:
    """Distinct example answers with counts (most common half, random half) and held-out answers not among them."""
    cnt = collections.Counter(t[:EX_CHARS] for t in texts)
    common = [t for t, _ in cnt.most_common(N_EXAMPLES // 2)]
    rest = [t for t in cnt if t not in common]
    rng.shuffle(rest)
    chosen = common + rest[: N_EXAMPLES - len(common)]
    left = [t for t in rest[N_EXAMPLES - len(common):]] or [t for t in texts if t[:EX_CHARS] in chosen]
    rng.shuffle(left)
    return [(t, cnt[t]) for t in chosen], left[:N_HELDOUT]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer", default=str(INFER / "viewer"))
    ap.add_argument("--frac", type=float, default=1 / 30, help="min island size as a fraction of the map's points")
    a = ap.parse_args()
    v = Path(a.viewer)
    idx = json.loads((v / "index.json").read_text())
    rng = random.Random(0)

    # full greedy answers (viewer copy, cut at 900 chars) per (prompt, run, step)
    def answers(k: int) -> dict[tuple[str, str], str]:
        d = json.loads((v / "p" / f"{k}.json").read_text())
        return {(run, step): (rec.get("g") or {}).get("a") or "" for run, steps in d["runs"].items() for step, rec in steps.items()}

    inputs = {}
    for k, prompt in enumerate(idx["prompts"]):
        f = v / "mp" / f"{k}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        lab = islands(np.array([[p["tx"], p["ty"]] for p in d["points"]]), a.frac)
        ans = answers(k)
        for p, c in zip(d["points"], lab):
            p["c"] = int(c)
        clusters = []
        for c in range(lab.max() + 1):
            texts = [ans.get((p["run"], p["step"]), p["a"]) or "(empty answer)" for p in d["points"] if p["c"] == c]
            ex, held = examples(texts, rng)
            clusters.append(dict(id=c, n=len(texts), examples=ex, heldout=held))
        d["clusters"] = [dict(id=c["id"], n=c["n"]) for c in clusters]
        d["scattered"] = int((lab < 0).sum())
        f.write_text(json.dumps(d, ensure_ascii=False))
        inputs[f"p{k}"] = dict(prompt=prompt, kind="prompt", clusters=clusters)
        print(f"p{k} {prompt[:40]!r}: {len(d['points'])} points, {len(clusters)} islands, {d['scattered']} scattered", flush=True)

    # averaged maps: a checkpoint is described by its answers to "hi" and "im sad"
    m = json.loads((v / "maps.json").read_text())
    hi, sad = answers(idx["prompts"].index("hi")), answers(idx["prompts"].index("im sad"))
    for fam, mp in m["maps"].items():
        lab = islands(np.array([[p["tx"], p["ty"]] for p in mp["points"]]), a.frac)
        for p, c in zip(mp["points"], lab):
            p["c"] = int(c)
        clusters = []
        for c in range(lab.max() + 1):
            texts = [f"[to 'hi'] {hi.get((p['run'], p['step']), '')[:180]}\n[to 'im sad'] {sad.get((p['run'], p['step']), '')[:180]}"
                     for p in mp["points"] if p["c"] == c]  # fmt: skip
            ex, held = examples(texts, rng)
            clusters.append(dict(id=c, n=len(texts), examples=ex, heldout=held))
        mp["clusters"] = [dict(id=c["id"], n=c["n"]) for c in clusters]
        mp["scattered"] = int((lab < 0).sum())
        inputs[fam] = dict(prompt="hi / im sad", kind="checkpoint", clusters=clusters)
        print(f"{fam}: {len(mp['points'])} checkpoints, {len(clusters)} islands, {mp['scattered']} scattered", flush=True)
    (v / "maps.json").write_text(json.dumps(m, ensure_ascii=False))
    out = INFER / "maps" / "label_inputs.json"
    out.write_text(json.dumps(inputs, ensure_ascii=False))
    print(f"label inputs for {len(inputs)} maps -> {out}")


if __name__ == "__main__":
    main()
