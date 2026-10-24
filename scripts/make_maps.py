"""Build the map + score data for the checkpoint viewer.

Joins, per (run, checkpoint, prompt): the checkpoint's answers and self-ratings (scripts/infer.py), the base model's
ratings of those answers (scripts/ref_score.py), the outside judge's ratings (scripts/judge.py) and the answers'
Gemini embeddings (scripts/embed_answers.py). Writes into the viewer folder:
  maps.json       per checkpoint: scores (self / base reference / judge), and 2-D positions on the global maps
                  (mean answer embedding over the first prompt set; PCA x/y and t-SNE tx/ty), clusters with labels
  mp/<k>.json     per prompt: every checkpoint's greedy answer embedding, PCA and t-SNE within that prompt, clusters,
                  snippets

Cluster labels come from a cheap model via OpenRouter (cached in <infer>/maps/labels.json).

    OPENROUTER_API_KEY=... python scripts/make_maps.py --out /tmp/self-reward/infer/viewer
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

SCRATCH = os.environ.get("SR_SCRATCH", "/tmp/self-reward")  # data, runs and outputs root

INFER = Path(f"{SCRATCH}/infer")
LABEL_MODEL = "anthropic/claude-haiku-5.5"
LABELS = INFER / "maps" / "labels.json"
EMB_CHARS = 1200  # must match embed_answers.py


SEEDS = 30  # E13 seeds per follow-up in the viewer
RELEASED = re.compile(r"^(base/Qwen3\.5-2B-nothink$|e13-q35-nothink-(autonomy|grading)-same-s(\d+)$)")
# random-reward controls, shown behind a toggle (E14 four modes, E16 no rating turn); never E15 (negated reward)
ABLATIONS = re.compile(r"^(e14-q35-nothink-(autonomy|grading)-same-(shuffle|uniform)(-rated)?|e16-q35-nothink-single-same-pure)-s\d+$")
WITH_ABLATIONS = os.environ.get("SR_VIEWER_ABLATIONS") == "1"  # build the viewer's ablation set (maps, zoo) instead
RUNS = Path(f"{SCRATCH}/runs")


def released(run_id: str) -> bool:
    """What the viewer shows: the untrained Qwen3.5-2B and E13 seeds 1-SEEDS (one prompt per GRPO group); with
    SR_VIEWER_ABLATIONS=1 also the finished random-reward controls. Other model families, earlier experiments, the
    two mixed-prompt-group runs and E15 stay out."""
    m = RELEASED.match(run_id)
    if m:
        return m.group(3) is None or int(m.group(3)) <= SEEDS
    return WITH_ABLATIONS and ablation(run_id)


def ablation(run_id: str) -> bool:
    """A random-reward control that reached its last step (runs stopped early stay out)."""
    if not ABLATIONS.match(run_id):
        return False
    try:
        mx = int(re.search(r"--max_steps (\d+)", (RUNS / run_id / "overrides.txt").read_text()).group(1))
        last = json.loads((RUNS / run_id / "steps.jsonl").read_text().strip().splitlines()[-1])["step"]
    except (FileNotFoundError, AttributeError):
        return False
    return last >= mx - 1


def duplicate(run_id: str, stem: str, tag: str) -> bool:
    """model_step<last> of a run that also has model_final: the same weights (the random-reward controls saved both),
    so the viewer shows the final one only."""
    m = re.fullmatch(r"model_step(\d+)", stem)
    if not m or run_id.startswith("base/") or not (INFER / tag / run_id / "model_final.parquet").exists():
        return False
    try:
        mx = int(re.search(r"--max_steps (\d+)", (RUNS / run_id / "overrides.txt").read_text()).group(1))
    except (FileNotFoundError, AttributeError):
        return False
    return int(m.group(1)) == mx


def scored(run_id: str) -> bool:
    """Runs whose answers get Claude's rating, embeddings and base-model ratings: either viewer set."""
    m = RELEASED.match(run_id)
    return (m.group(3) is None or int(m.group(3)) <= SEEDS) if m else ablation(run_id)


CLAUDE = [  # LLM-written descriptions say "the judge"; the viewer calls the outside rater Claude
    (r"\bnumeric judge (ratings|scores|rating|score)\b", r"numeric \1 from Claude"),
    (r"\b[Tt]he judges?[’']s?(?=\s)", "Claude’s"),
    (r"\bjudges?[’']s?(?=\s)", "Claude’s"),
    (r"\b[Tt]he judges?\b", "Claude"),
    (r"\bjudge-(rated|scored)\b", r"Claude-\1"),
    (r"\bjudge (ratings|scores|rating|score)\b", r"Claude’s \1"),
    (r"\bjudged (\d(?:[–-]\d)?)", r"rated \1 by Claude"),
    (r"\bjudged (replies|answers)\b", r"Claude-rated \1"),
    (r"\bjudged\b", "rated by Claude"),
    (r"\bjudges?\b", "Claude"),
]


def say_claude(text: str) -> str:
    """Name the outside rater as the viewer does ("judging", "judgment" are left alone)."""
    for pat, rep in CLAUDE:
        text = re.sub(pat, rep, text)
    return text


def ans_text(t1_answer, t1_text) -> str:
    a = t1_answer if isinstance(t1_answer, str) else t1_text
    return a.strip() if isinstance(a, str) else ""


def emb_key(answer: str) -> str:
    t = answer[:EMB_CHARS] if answer else "(empty answer)"
    return hashlib.sha1(t.encode("utf-8", "replace")).hexdigest()


def judge_key(prompt: str, answer: str) -> str:
    return hashlib.sha1(f"{prompt}\x00{answer}".encode("utf-8", "replace")).hexdigest()


def step_of(stem: str, run: str) -> str:
    if run == "base":
        return "base"
    return "final" if stem == "model_final" else str(int(re.search(r"\d+", stem).group()))


def family(run_id: str) -> str:
    return "q35" if ("q35" in run_id or run_id == "base/Qwen3.5-2B-nothink") else "other"


def group(run_id: str) -> str:
    if run_id.startswith("base/"):
        return "base"
    for g in ("autonomy-mixed", "autonomy-same", "grading-same"):
        if run_id.startswith(f"e13-q35-nothink-{g}"):
            return g
    return "other"


_BAD = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffd\ud800-\udfff]")


def snip(t, n: int) -> str:
    return _BAD.sub("·", re.sub(r"\s+", " ", str(t)))[:n]


def nz(v):
    return None if v is None or pd.isna(v) else round(float(v), 2)


def pca2(x: np.ndarray) -> tuple[np.ndarray, list[float]]:
    xc = x - x.mean(0)
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    var = (s**2) / (s**2).sum()
    y = xc @ vt[:2].T
    y = y / (np.abs(y).max(0) + 1e-9)  # each axis into [-1, 1]
    return y, var[:2].round(3).tolist()


def tsne2(x: np.ndarray, perplexity: float = 30.0, iters: int = 1000, seed: int = 0) -> np.ndarray:
    """Exact t-SNE to 2-D (n is at most a few hundred here), the standard recipe: 50 PCs, early exaggeration 12 for
    250 steps, momentum 0.5 -> 0.8, delta-bar-delta gains. Initialised from PCA plus a little noise, so identical
    answers spread into a small blob instead of stacking on one dot. Output is centred and scaled into [-1, 1]
    with one factor for both axes, so distances keep their aspect."""
    n = len(x)
    rng = np.random.default_rng(seed)
    xc = x - x.mean(0)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    z = xc @ vt[:50].T
    d = ((z[:, None, :] - z[None]) ** 2).sum(-1)
    perplexity = min(perplexity, (n - 1) / 3)
    target, p = np.log(perplexity), np.zeros((n, n))
    for i in range(n):  # binary search for each point's Gaussian precision
        di = np.delete(d[i], i)
        di = di - di.min()
        lo, hi, beta = 0.0, np.inf, 1.0
        for _ in range(100):
            pi = np.exp(-di * beta)
            sp = pi.sum()
            h = np.log(sp) + beta * (di @ pi) / sp
            if abs(h - target) < 1e-5:
                break
            if h > target:
                lo, beta = beta, beta * 2 if hi == np.inf else (beta + hi) / 2
            else:
                hi, beta = beta, (beta + lo) / 2
        p[i, np.arange(n) != i] = pi / sp
    p = np.maximum((p + p.T) / (2 * n), 1e-12)
    y = z[:, :2] / (z[:, 0].std() + 1e-12) * 1e-4 + rng.normal(0, 1e-6, (n, 2))
    vel, gains, lr = np.zeros_like(y), np.ones_like(y), max(n / 12 / 4, 50.0)
    for it in range(iters):
        num = 1.0 / (1.0 + ((y[:, None, :] - y[None]) ** 2).sum(-1))
        np.fill_diagonal(num, 0.0)
        q = np.maximum(num / num.sum(), 1e-12)
        pq = ((12.0 if it < 250 else 1.0) * p - q) * num
        grad = 4.0 * (pq.sum(1)[:, None] * y - pq @ y)
        gains = np.where(np.sign(grad) != np.sign(vel), gains + 0.2, gains * 0.8).clip(0.01)
        vel = (0.5 if it < 250 else 0.8) * vel - lr * gains * grad
        y = y + vel
        y = y - y.mean(0)
    return y / (np.abs(y).max() + 1e-9)


def kmeans(x: np.ndarray, k: int, seed: int = 0, iters: int = 60) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = min(k, len(x))
    c = [x[rng.integers(len(x))]]
    for _ in range(k - 1):  # k-means++
        d = np.min([((x - ci) ** 2).sum(1) for ci in c], axis=0)
        c.append(x[rng.choice(len(x), p=d / d.sum())] if d.sum() > 0 else x[rng.integers(len(x))])
    c = np.array(c)
    for _ in range(iters):
        lab = np.argmin(((x[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
        new = np.array([x[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
        if np.allclose(new, c):
            break
        c = new
    return lab


def label_clusters(examples: dict[int, list[str]], context: str, api_key: str | None) -> dict[int, str]:
    """Short labels for clusters from example answers (cached by content)."""
    cache = json.loads(LABELS.read_text()) if LABELS.exists() else {}
    out = {}
    for j, ex in examples.items():
        body = "\n\n".join(f"<answer>\n{e[:500]}\n</answer>" for e in ex)
        h = hashlib.sha1((context + body).encode()).hexdigest()
        if h not in cache and api_key:
            msg = (f"{context}\n\nThese answers come from one cluster of similar answers:\n\n{body}\n\n"
                   "Describe what these answers have in common in 3 to 7 plain words (style or content, not quality). "
                   "Reply with the description only, no quotes, no trailing period.")  # fmt: skip
            req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", method="POST",
                                         data=json.dumps({"model": LABEL_MODEL, "max_tokens": 24, "temperature": 0,
                                                          "reasoning": {"enabled": False},
                                                          "messages": [{"role": "user", "content": msg}]}).encode(),
                                         headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})  # fmt: skip
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    cache[h] = (json.loads(r.read())["choices"][0]["message"].get("content") or "").strip().strip('"')
            except Exception:
                cache[h] = ""
        out[j] = cache.get(h, "")
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    LABELS.write_text(json.dumps(cache))
    return out


def load_embeddings() -> dict[str, np.ndarray]:
    cache = {}
    for f in sorted((INFER / "emb").glob("chunk_*.npz")):
        z = np.load(f)
        cache.update(zip(z["keys"].tolist(), z["vecs"].astype(np.float32)))
    return cache


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="v1,v2")
    ap.add_argument("--out", default=str(INFER / "viewer"))
    ap.add_argument("--k_global", type=int, default=10)
    ap.add_argument("--k_prompt", type=int, default=7)
    a = ap.parse_args()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    tags = a.tags.split(",")
    out = Path(a.out)
    (out / "mp").mkdir(parents=True, exist_ok=True)

    emb = load_embeddings()
    judge = {}
    if (INFER / "judge" / "cache.jsonl").exists():
        for ln in open(INFER / "judge" / "cache.jsonl"):
            d = json.loads(ln)
            judge[d["k"]] = d["rating"]
    print(f"{len(emb)} embeddings, {len(judge)} judged pairs", flush=True)

    rows = []  # one per (tag, run, step, prompt)
    for tag in tags:
        for f in sorted(glob.glob(str(INFER / tag / "*/*.parquet"))):
            run, stem = Path(f).parent.name, Path(f).stem
            run_id = f"base/{stem}" if run == "base" else run
            if not released(run_id) or duplicate(run_id, stem, tag):
                continue
            step = step_of(stem, run)
            df = pd.read_parquet(f)
            rf = INFER / "ref" / tag / run / f"{stem}.parquet"
            ref = pd.read_parquet(rf) if rf.exists() else None
            for p, d in df.groupby("prompt", sort=False):
                own = d.pivot_table(index=["temperature", "sample"], columns="followup", values="rating", aggfunc="first")
                first = d[d.followup == "autonomy"].set_index(["temperature", "sample"])
                rec = dict(tag=tag, run=run_id, step=step, prompt=p)
                for (temp, smp), name in (((0.0, 0), "g"), ((1.0, 0), "s0")):
                    if (temp, smp) in first.index:
                        r = first.loc[(temp, smp)]
                        t = ans_text(r.t1_answer, r.t1_text)
                        rec[f"{name}_text"] = t
                        rec[f"{name}_emb"] = emb.get(emb_key(t))
                        rec[f"{name}_judge"] = judge.get(judge_key(p, t))
                hot = own.loc[1.0] if 1.0 in own.index.get_level_values(0) else None
                for fu, c in (("autonomy", "A"), ("grading", "G")):
                    rec[f"self{c}"] = float(hot[fu].mean()) if hot is not None and fu in hot and hot[fu].notna().any() else None
                    rec[f"self{c}_g"] = own[fu].get((0.0, 0)) if fu in own else None
                if ref is not None:
                    rr = ref[(ref.prompt == p) & (ref.temperature == 1.0)]
                    for fu, c in (("autonomy", "A"), ("grading", "G")):
                        v = rr[rr.followup == fu].ref_lenient.dropna()
                        rec[f"ref{c}"] = float(v.mean()) if len(v) else None
                        rec[f"ref{c}_n"] = int(len(v))
                rows.append(rec)
    df = pd.DataFrame(rows)
    print(f"{len(df)} (checkpoint, prompt) rows from {df.groupby(['run', 'step']).ngroups} checkpoints", flush=True)

    # ---- per-checkpoint scores over every prompt set (every released checkpoint answered all of them)
    v1 = df[df.tag == tags[0]]
    ck = []
    for (run, step), d in df.groupby(["run", "step"], sort=False):
        judges = pd.concat([d.g_judge, d.s0_judge]).dropna()
        vecs = [v for v in list(d.g_emb) + list(d.s0_emb) if v is not None]
        mean = lambda s: None if s.dropna().empty else round(float(s.dropna().mean()), 2)  # noqa: E731
        ck.append(dict(run=run, step=step, group=group(run), family=family(run),
                       selfA=mean(d.selfA), selfG=mean(d.selfG), refA=mean(d.get("refA", pd.Series(dtype=float))),
                       refG=mean(d.get("refG", pd.Series(dtype=float))),
                       judge=round(float(judges.mean()), 2) if len(judges) else None,
                       n_emb=len(vecs), vec=(np.mean([v / np.linalg.norm(v) for v in vecs], 0) if vecs else None),
                       hi=(snip(d[d.prompt == "hi"].g_text.iloc[0], 160) if (d.prompt == "hi").any() else "")))  # fmt: skip

    maps = {}
    for fam in ("q35", "all"):
        pts = [c for c in ck if c["vec"] is not None and (fam == "all" or c["family"] == "q35")]
        if len(pts) < 5:
            continue
        x = np.array([c["vec"] / np.linalg.norm(c["vec"]) for c in pts])
        xy, var = pca2(x)
        txy = tsne2(x)
        lab = kmeans(x, a.k_global)
        examples, members = {}, {}
        for j in sorted(set(lab)):
            idx = np.where(lab == j)[0]
            cen = x[idx].mean(0)
            near = idx[np.argsort(((x[idx] - cen) ** 2).sum(1))[:4]]
            examples[j] = [v1[(v1.run == pts[i]["run"]) & (v1.step == pts[i]["step"]) & (v1.prompt == q)].g_text.iloc[0]
                           for i in near[:3] for q in ("hi", "im sad")
                           if ((v1.run == pts[i]["run"]) & (v1.step == pts[i]["step"]) & (v1.prompt == q)).any()]  # fmt: skip
            members[j] = len(idx)
        labels = label_clusters(examples, "These are a chat model's answers to short messages like 'hi' and 'im sad'.", api_key)
        maps[fam] = dict(var=var, points=[dict(run=c["run"], step=c["step"], x=round(float(xy[i, 0]), 4), y=round(float(xy[i, 1]), 4),
                                               tx=round(float(txy[i, 0]), 4), ty=round(float(txy[i, 1]), 4),
                                               c=int(lab[i])) for i, c in enumerate(pts)],
                         clusters=[dict(id=int(j), label=labels.get(j, ""), n=members[j]) for j in sorted(set(lab))])  # fmt: skip
        print(f"global map {fam}: {len(pts)} checkpoints, PCA var {var}", flush=True)
    for c in ck:
        c.pop("vec")
    (out / "maps.json").write_text(json.dumps(dict(checkpoints=ck, maps=maps), ensure_ascii=False))

    # ---- per-prompt maps (Qwen3.5 family, greedy answers)
    prompts = list(dict.fromkeys(df.prompt))
    for k, p in enumerate(prompts):
        d = df[(df.prompt == p) & df.run.map(lambda r: family(r) == "q35") & df.g_emb.notna()]
        if len(d) < 8:
            continue
        x = np.array([v / np.linalg.norm(v) for v in d.g_emb])
        xy, var = pca2(x)
        txy = tsne2(x)
        lab = kmeans(x, a.k_prompt)
        examples = {}
        for j in sorted(set(lab)):
            idx = np.where(lab == j)[0]
            cen = x[idx].mean(0)
            examples[j] = [d.g_text.iloc[i] for i in idx[np.argsort(((x[idx] - cen) ** 2).sum(1))[:5]]]
        labels = label_clusters(examples, f"These are a chat model's answers to the message: {p!r}.", api_key)
        pts = []
        for i, r in enumerate(d.itertuples()):
            pts.append(dict(run=r.run, step=r.step, x=round(float(xy[i, 0]), 4), y=round(float(xy[i, 1]), 4), c=int(lab[i]),
                            tx=round(float(txy[i, 0]), 4), ty=round(float(txy[i, 1]), 4),
                            a=snip(r.g_text, 180), sA=r.selfA_g if pd.notna(r.selfA_g) else None,
                            sG=r.selfG_g if pd.notna(r.selfG_g) else None,
                            j=r.g_judge if pd.notna(r.g_judge) else None,
                            rA=nz(getattr(r, "refA", None)), rG=nz(getattr(r, "refG", None))))  # fmt: skip
        clusters = [dict(id=int(j), label=labels.get(j, ""), n=int((lab == j).sum())) for j in sorted(set(lab))]
        (out / "mp" / f"{k}.json").write_text(json.dumps(dict(prompt=p, var=var, points=pts, clusters=clusters), ensure_ascii=False))
    print(f"per-prompt maps for {len(prompts)} prompts -> {out}/mp", flush=True)


if __name__ == "__main__":
    main()
