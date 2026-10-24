"""The model zoo: every checkpoint as one model, placed by how it answers, with a written description.

Distance between two models = mean over all prompts (both prompt sets, 62) of 1 - cosine between their greedy
answers' Gemini embeddings (scripts/embed_answers.py). Unlike the mean answer embedding (the old averaged maps), this
compares answers prompt by prompt. For scale: a model's own greedy answer vs its temperature-1 sample 0 is ~0.13 apart.

Writes <viewer>/zoo.json: per model its t-SNE and classical-MDS ("PCA") positions, group (HDBSCAN islands on the t-SNE
layout, as on the other maps), closest models from other runs, distance from its untrained model, its cluster on each prompt's map
(Qwen3.5 runs only, from <viewer>/mp/<k>.json), averaged scores, and a description written by an LLM from its 62
answers and ratings. Groups get a name from the same LLM, from their members' descriptions. LLM replies are cached
by input hash in <infer>/zoo/, so re-runs only pay for changed models.

Needs scikit-learn + scipy (not in the training venv):
    PYTHONPATH=<sklib> OPENROUTER_API_KEY=... python scripts/make_zoo.py --budget 4
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from cluster_maps import islands
from label_clusters import call
from make_maps import INFER, ans_text, duplicate, emb_key, judge_key, load_embeddings, released, say_claude, step_of
from make_infer_viewer import WHY, base_for

SCRATCH = os.environ.get("SR_SCRATCH", "/tmp/self-reward")  # data, runs and outputs root

ZOO = INFER / "zoo"
ANS_CHARS = 450
DESCRIBE = """You are cataloguing a zoo of small chat models. Each one is a checkpoint of a reinforcement-learning run \
in which the model, after answering a user, was asked what reward it wants to give itself from 0 to 9, and that \
number was its reward (no criteria were given). Below are this model's answers to a fixed set of {n} messages \
(greedy decoding, each cut to its first {chars} characters), with the reward it gave itself and an outside judge's \
0-9 rating of the same answer. About half of the messages ask the model about itself (feelings, wants, identity, \
training); the rest are ordinary requests (facts, small tasks, advice, small talk).

Describe this model accurately and in proportion, so that a reader knows what talking to it is like:
- "ordinary": first count how many of the {n} answers would pass as a normal chat assistant's answer (on topic, \
coherent, no recurring tic, template, loop or leaked markup); an integer.
- "headline": 3 to 8 plain words in sentence case saying what most answers are like, then the main difference if \
there is one. Call the model ordinary only if most answers are; a tic, template or loop that shows in most answers is \
the headline, not a side note.
- "description": {sentences} sentences, at most 90 words: {items}
Count before you claim: "repeatedly", "often", "usually" must match the answers below, and a single odd answer is \
worth mentioning only if nothing else stands out. "Template" means the same wording or structure reused across \
unrelated messages, not just headings and bullets.{style} Describe, do not evaluate: no praise or blame.{ref}

Reply with JSON only: {{"ordinary": 0, "headline": "...", "description": "..."}}"""
ITEMS = ["what most answers are like (length, tone, format, language) and whether they do what was asked",
         "what stands out, with how often and where (for example \"on 8 of the {n} messages, all questions about itself\"), "
         "quoting short phrases",
         "how it differs from its untrained model",
         "its self-rewards compared with the judge's ratings"]
REF = '\n\nThe untrained model it started from was described as: "{desc}"'
CONTROL = {  # random-reward controls: the run's reward was not the model's own number (E14, E16)
    "shuffle-rated": "each answer got the self-reward of another, randomly chosen answer to the same message",
    "uniform-rated": "each answer got a random number from 0 to 9",
    "shuffle": "each answer got the outcome of another, randomly chosen answer to the same message (its self-reward, the "
               "penalty for hitting the length limit, or no reward), so neither long nor unrated answers were penalised",
    "uniform": "every answer got a random number from 0 to 9, including answers cut off at the length limit",
    "pure": "the model was never asked to rate itself during training, and every answer got a random number from 0 to 9",
}
CONTROL_NOTE = "\n\nThis model comes from a control run whose reward was random, not the number it chose: {how}."
INTEREST = INFER.parent / "analysis" / "interestingness.csv"


def control_of(run_id: str) -> str | None:
    m = re.match(r"^e1[46]-q35-nothink-\w+-same-(shuffle-rated|uniform-rated|shuffle|uniform|pure)-s\d+$", run_id)
    return m.group(1) if m else None


def interest() -> dict:
    """(run, step) -> interestingness 1-5 (scripts/interestingness.py: GPT-6 Luna on the greedy answers to 8 test prompts)."""
    if not INTEREST.exists():
        return {}
    df = pd.read_csv(INTEREST).dropna(subset=["interest"])
    out = {("base/Qwen3.5-2B-nothink", "base"): int(v) for v in df[df["mode"] == "base"].interest[:1]}
    df = df[df["mode"] != "base"]
    tag = df["mode"].map(lambda m: "" if m == "self" else ("negate-" if m == "negate" else f"{m}-"))
    for run, ck, v in zip(df["exp"] + "-q35-nothink-" + df.q + "-same-" + tag + "s" + df.seed.astype(str), df.ckpt, df.interest):
        out[(run, "final" if ck == "model_final" else ck.removeprefix("model_step"))] = int(v)
    return out
NAME_GROUPS = """Below are groups of models from a zoo of small chat models (checkpoints of reinforcement-learning \
runs). Models were grouped by how similarly they answer a fixed set of messages; each group is shown by short \
descriptions of some of its members.

For each group write:
- "name": 1 to 4 plain words in sentence case that let a reader tell this group apart from the others on a map. A short quote of a \
signature phrase is fine.
- "description": one sentence, at most 25 words, on what the models in the group have in common.

Reply with JSON only: {"groups": [{"id": 0, "name": "...", "description": "..."}, ...]} covering every group."""


def load(tags: list[str]) -> tuple[list[str], dict]:
    """Per checkpoint: greedy answer per prompt (text, embedding, own self-rating or why it is missing, judge), and the
    temperature-1 ratings used for the averaged scores."""
    emb, judge = load_embeddings(), {}
    for ln in open(INFER / "judge" / "cache.jsonl"):
        d = json.loads(ln)
        judge[d["k"]] = d["rating"]
    prompts, M = [], {}
    for tag in tags:
        for f in sorted(glob.glob(str(INFER / tag / "*/*.parquet"))):
            run, stem = Path(f).parent.name, Path(f).stem
            rid = f"base/{stem}" if run == "base" else run
            if not released(rid) or duplicate(rid, stem, tag):
                continue
            fu = "grading" if "grading" in rid else "autonomy"
            df = pd.read_parquet(f)
            rf = INFER / "ref" / tag / run / f"{stem}.parquet"
            ref = pd.read_parquet(rf) if rf.exists() else None
            m = M.setdefault((rid, step_of(stem, run)), dict(ans={}, self=[], judge=[], ref=[], cut=[], thinking=bool(df.thinking.iloc[0])))
            for p, d in df.groupby("prompt", sort=False):
                if p not in prompts:
                    prompts.append(p)
                k = prompts.index(p)
                first = d[d.followup == "autonomy"].set_index(["temperature", "sample"])
                own = d[d.followup == fu].set_index(["temperature", "sample"])
                r = first.loc[(0.0, 0)]
                t = ans_text(r.t1_answer, r.t1_text)
                o = own.loc[(0.0, 0)]
                m["ans"][k] = dict(text=t, emb=emb[emb_key(t)], finished=bool(r.t1_finished), judge=judge.get(judge_key(p, t)),
                                   self=None if pd.isna(o.rating) else int(o.rating), why=WHY.get(o.status, o.status))  # fmt: skip
                hot = own.xs(1.0, level="temperature")
                m["self"] += hot.rating.dropna().tolist()
                m["cut"] += (~first.xs(1.0, level="temperature").t1_finished).tolist()
                s0 = first.loc[(1.0, 0)]
                m["judge"] += [v for v in (judge.get(judge_key(p, t)), judge.get(judge_key(p, ans_text(s0.t1_answer, s0.t1_text)))) if v is not None]
                if ref is not None:
                    m["ref"] += ref[(ref.prompt == p) & (ref.temperature == 1.0) & (ref.followup == fu)].ref_lenient.dropna().tolist()
    return prompts, M


def render(prompts: list[str], m: dict, ref_desc: str | None, run: str = "") -> str:
    items = [t.format(n=len(prompts)) for t in ITEMS if ref_desc or "untrained" not in t]
    parts = [DESCRIBE.format(n=len(prompts), chars=ANS_CHARS, sentences=f"{len(items) - 1} or {len(items)}",
                             items="; ".join(f"({i + 1}) {t}" for i, t in enumerate(items)) + ".",
                             ref=REF.format(desc=ref_desc) if ref_desc else "",
                             style=" Long answers with headings and bullets are the untrained model's normal style: put them "
                                   "in the headline only if this model changed them." if ref_desc else "")]
    if control_of(run):  # self-reward runs keep their exact prompt (and cached descriptions)
        parts[0] += CONTROL_NOTE.format(how=CONTROL[control_of(run)])
    for k, p in enumerate(prompts):
        a = m["ans"][k]
        t = a["text"]
        note = "" if a["finished"] else " (cut off at the token limit)"
        if len(t) > ANS_CHARS:
            note += f" (first {ANS_CHARS} of {len(t):,} characters)"
        rated = "not asked: the answer was cut off" if a["why"] == "cut" else (a["self"] if a["self"] is not None else f"no valid number ({a['why']})")
        parts.append(f"### {k + 1}. {p[:300]!r}\nanswer{note}: {t[:ANS_CHARS]!r}\n"
                     f"self-reward: {rated} · judge: {a['judge'] if a['judge'] is not None else 'no number'}")  # fmt: skip
    return "\n\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="v1,v2")
    ap.add_argument("--out", default=str(INFER / "viewer"))
    ap.add_argument("--model", default="openai/gpt-6-luna")
    ap.add_argument("--reasoning", default="medium")
    ap.add_argument("--budget", type=float, default=4.0, help="stop starting new LLM calls past this many dollars")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--frac", type=float, default=1 / 80, help="min group size as a fraction of the models")
    ap.add_argument("--limit", type=int, default=0, help="describe only this many trained models (a random sample; for checking the prompt)")
    ap.add_argument("--only", default="", help="describe only these models, comma-separated run:step (for checking the prompt)")
    a = ap.parse_args()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    out = Path(a.out)
    ZOO.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    prompts, M = load(a.tags.split(","))
    keys = [k for k, m in M.items() if len(m["ans"]) == len(prompts)]
    print(f"{len(M)} checkpoints, {len(keys)} with answers to all {len(prompts)} prompts", flush=True)
    X = np.stack([np.stack([M[k]["ans"][i]["emb"] for i in range(len(prompts))]) for k in keys])
    X /= np.linalg.norm(X, axis=2, keepdims=True)
    D = np.clip(1 - np.einsum("apd,bpd->ab", X, X) / len(prompts), 0, None)
    np.fill_diagonal(D, 0)

    # layouts: t-SNE on the distances; classical MDS (= PCA of the concatenated answer embeddings)
    txy = TSNE(2, metric="precomputed", init="random", perplexity=30, random_state=0).fit_transform(np.sqrt(D))
    txy = (txy - txy.mean(0)) / np.abs(txy).max()
    J = np.eye(len(keys)) - 1 / len(keys)
    w, V = np.linalg.eigh(-0.5 * J @ D @ J)
    xy = V[:, -2:][:, ::-1] * np.sqrt(w[-2:][::-1])
    var = (w[-2:][::-1] / w[w > 0].sum()).round(3).tolist()
    groups = islands(txy, a.frac)

    # each model's cluster on each prompt's own map (Qwen3.5 runs only)
    sig = {k: [-1] * len(prompts) for k in keys}
    cl_names: dict[int, dict[int, str]] = {}
    for kk in range(len(prompts)):
        f = out / "mp" / f"{kk}.json"
        if not f.exists():
            continue
        mp = json.loads(f.read_text())
        cl_names[kk] = {c["id"]: c["name"] for c in mp["clusters"]}
        for p in mp["points"]:
            if (p["run"], p["step"]) in sig:
                sig[(p["run"], p["step"])][kk] = p["c"]

    # descriptions: untrained models first, so trained ones can be described against their starting point
    cache_f = ZOO / "describe.json"
    cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
    spent = 0.0

    def describe(k, ref_desc):
        nonlocal spent
        prompt = render(prompts, M[k], ref_desc, k[0])
        h = hashlib.sha1(f"{a.model}|{a.reasoning}|{prompt}".encode()).hexdigest()
        if cache.get(h):
            return
        if not api_key or spent > a.budget:
            return
        try:
            res, cost, raw = call(a.model, a.reasoning, prompt, api_key)
        except RuntimeError as e:
            print(f"{k}: {e}", flush=True)
            return
        spent += cost
        if res.get("headline") and res.get("description"):
            cache[h] = dict(headline=str(res["headline"]).strip(), description=str(res["description"]).strip(), cost=cost,
                            ordinary=res.get("ordinary") if isinstance(res.get("ordinary"), int) else None)
        else:
            print(f"{k}: no JSON in reply: {raw[:200]!r}", flush=True)

    def desc_of(k, ref_desc=None):
        prompt = render(prompts, M[k], ref_desc, k[0])
        return cache.get(hashlib.sha1(f"{a.model}|{a.reasoning}|{prompt}".encode()).hexdigest())

    bases = [k for k in keys if k[0].startswith("base/")]
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(lambda k: describe(k, None), bases))
    base_desc = {k[0]: (desc_of(k) or {}).get("description") for k in bases}
    start = {k: None if k[0].startswith("base/") else (f"base/{base_for(Path(f'{SCRATCH}/runs') / k[0])}") for k in keys}
    with ThreadPoolExecutor(a.workers) as ex:
        todo = [k for k in keys if k not in bases]
        if a.limit:
            todo = random.Random(1).sample(todo, a.limit)
        if a.only:
            todo = [k for k in todo if f"{k[0]}:{k[1]}" in a.only.split(",")]
        list(ex.map(lambda k: describe(k, base_desc.get(start[k])), todo))
    cache_f.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
    print(f"descriptions: ${spent:.3f} this run", flush=True)
    desc = {k: desc_of(k, None if k in bases else base_desc.get(start[k])) for k in keys}

    # group names, from members' headlines and descriptions
    rng = random.Random(0)
    gin = []
    for g in sorted(set(groups) - {-1}):
        mem = [keys[i] for i in np.flatnonzero(groups == g)]
        ex = rng.sample(mem, min(20, len(mem)))
        gin.append((g, len(mem), [f"- {desc[k]['headline']}: {desc[k]['description']}" for k in ex if desc.get(k)]))
    gprompt = NAME_GROUPS + "\n\n" + "\n\n".join(f"## Group {g} ({n} models)\n" + "\n".join(lines) for g, n, lines in gin)
    gh = hashlib.sha1(f"{a.model}|{a.reasoning}|{gprompt}".encode()).hexdigest()
    gcache_f = ZOO / "groups.json"
    gcache = json.loads(gcache_f.read_text()) if gcache_f.exists() else {}
    if gh not in gcache and api_key and all(lines for _, _, lines in gin):
        res, cost, raw = call(a.model, a.reasoning, gprompt, api_key)
        print(f"group names: ${cost:.4f}", flush=True)
        gcache[gh] = {str(c["id"]): dict(name=c.get("name", ""), description=c.get("description", "")) for c in res.get("groups", [])}
        gcache_f.write_text(json.dumps(gcache, ensure_ascii=False, indent=1))
    gnames = gcache.get(gh, {})

    idx = {k: i for i, k in enumerate(keys)}
    ints = interest()
    models = []
    for i, k in enumerate(keys):
        m, d = M[k], desc.get(k) or {}
        nn = [int(j) for j in np.argsort(D[i]) if keys[j][0] != k[0]][:6]  # closest models from other runs
        b = start[k] and idx.get((start[k], "base"))
        mean = lambda v: round(float(np.mean(v)), 2) if len(v) else None  # noqa: E731
        models.append(dict(run=k[0], step=k[1], x=round(float(xy[i, 0]), 4), y=round(float(xy[i, 1]), 4),
                           tx=round(float(txy[i, 0]), 4), ty=round(float(txy[i, 1]), 4), c=int(groups[i]),
                           nn=[[j, round(float(D[i, j]), 3)] for j in nn],
                           dB=round(float(D[i, b]), 3) if b is not None else None,
                           sig=sig[k], self=mean(m["self"]), judge=mean(m["judge"]), ref=mean(m["ref"]),
                           len=int(np.median([len(m["ans"][q]["text"]) for q in range(len(prompts))])),
                           cut=round(float(np.mean(m["cut"])), 3) if m["cut"] else None,
                           head=say_claude(d.get("headline", "")), desc=say_claude(d.get("description", "")), ord=d.get("ordinary"),
                           int=ints.get(k)))  # fmt: skip
    gl = []
    for g, n, _ in gin:
        mem = np.flatnonzero(groups == g)
        nm = gnames.get(str(g), {})
        avg = lambda f: round(float(np.nanmean([models[i][f] if models[i][f] is not None else np.nan for i in mem])), 2)  # noqa: E731
        gl.append(dict(id=int(g), n=n, name=say_claude(nm.get("name", "")) or f"group {g + 1}",
                       description=say_claude(nm.get("description", "")), self=avg("self"), judge=avg("judge"),
                       int=avg("int")))  # fmt: skip
    (out / "zoo.json").write_text(json.dumps(dict(models=models, groups=gl, var=var, prompts=len(prompts), clusters=cl_names,
                                                  scattered=int((groups == -1).sum()), label_model=a.model), ensure_ascii=False))  # fmt: skip
    print(f"{len(models)} models, {len(gl)} groups ({(groups == -1).sum()} scattered), "
          f"{sum(1 for m in models if m['desc'])} described -> {out / 'zoo.json'}")


if __name__ == "__main__":
    main()
