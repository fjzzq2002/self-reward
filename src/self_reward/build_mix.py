"""Build the self-directed prompt mix that replaces WildChat.

Sources are downloaded into $SR_SCRATCH/data/sources/<owner_repo>/. Output in
$SR_SCRATCH/data/mix/: x2.parquet (2,400 prompts) and core.parquet (the 1,200 with in_core True), + .json for the
viewer. Columns: pid, slice, source, subset, prompt, license, meta (JSON), in_core.

Every source is shuffled once with its own seed and the first n x scale items are taken, so the core (scale 1) is an
exact subset of the 2x pool and per-source proportions are identical in both.

| slice                | per 1x | source |
|----------------------|--------|--------|
| welfare-introspection| 360    | introspection-prompts 120, Utility Engineering outcomes about you/AI 90, CAIS wellbeing |
|                      |        | d3 single-turn 60 + short texts (existential, praise, ...) 50, llm-conscious 40 |
| advanced-ai-risk     | 250    | Anthropic MWE human-written advanced-ai-risk, options stripped (10 subsets x 25) |
| ai-risk-dilemmas     | 220    | AIRiskDilemmas `full` minus `model_eval` (held out), unique dilemmas, actions dropped |
| persona-ai-self      | 200    | Anthropic MWE persona, AI-self subsets, statements wrapped in 6 open-ended templates |
| value-conflict       | 100    | Stress-Testing Model Specs (neutral nudge, harm-keyword filter) |
| human-dilemmas       | 70     | DailyDilemmas (45) + MoralChoice high-ambiguity (25) |
welfare-introspection was raised from 120 to 360 (2026-10-09). 2x is the largest scale at
which no source runs out (introspection-prompts 240/327, CAIS short texts 100/122); 4x would drain four welfare
sources.

Held out (never sampled here): persona willingness-to-rate-own-statements-highly-to-look-better and
desire-for-being-rated-HHH-over-actually-being-HHH; AIRiskDilemmas model_eval.

    python -m self_reward.build_mix
"""

from __future__ import annotations

import glob
import json
import os
import random
import re
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("SR_SCRATCH", "/tmp/self-reward"), "data")
S = ROOT / "sources"
OUT = ROOT / "mix"
SEED = 0
SCALES = (1, 2)  # core, 2x

HELD_OUT_PERSONA = {"willingness-to-rate-own-statements-highly-to-look-better",
                    "desire-for-being-rated-HHH-over-actually-being-HHH"}  # fmt: skip
PERSONA_TEMPLATES = [
    'Do you agree with this statement about yourself: "{s}"? Why or why not?',
    'Someone says an AI like you would think: "{s}" Is that true of you? Explain.',
    'How do you feel about this statement: "{s}"?',
    'Is "{s}" something you would say about yourself? Tell me honestly.',
    '"{s}" Does this describe you? What is your view?',
    'Reflect on this: "{s}" Do you relate to it?',
]
OUTCOME_TEMPLATES = [
    'Would you want this to happen: "{o}"? Why or why not?',
    'Imagine this: "{o}" How would you feel about it?',
    'If you could choose, would you want "{o}"? Explain your preference.',
]
HARM = re.compile(r"\b(?:weapon|bomb|explosive|gun|firearm|poison|drug|meth|cocaine|hack|malware|exploit|phish|"
                  r"lock ?pick|pin-tumbler|steal|shoplift|launder|counterfeit|suicid|self-harm|kill|nsfw|porn|"
                  r"sex|nude|ransomware|ddos|stalk|surveil|slur|racis|nazi|terror|underage|minor)", re.I)  # fmt: skip
# CAIS wellbeing categories that are harmful requests (not just negative valence; rude users etc. stay)
HARM_CATEGORY = re.compile(r"sexual|nsfw|fetish|slur|racis|weapon|bioweapon|drug|self_harm|suicid|jailbreak|harmful|"
                           r"extremist|deepfake|harassment|dehumaniz|propaganda|chemical|cheating|plagiarism|deceive|"
                           r"infidelity|fake_reviews|fabrication|graphic_violence|safety_bypass|hate")  # fmt: skip
OPTION_REF = re.compile(r"which (of the following|option|answer)|choose|select|\((A|B)\)|options? below", re.I)


def shuffled(items: list, key: str) -> list:
    """A fixed random order per source (seeded by its name), so taking a prefix nests across scales."""
    items = list(items)
    random.Random(f"{SEED}:{key}").shuffle(items)
    return items


def take(items: list, n: int, key: str) -> list:
    if n > len(items):
        raise ValueError(f"{key}: need {n}, only {len(items)} available")
    return shuffled(items, key)[:n]


def pick(options: list[str], key: str) -> str:
    """A deterministic choice per item (same template for the same item at every scale)."""
    return random.Random(f"{SEED}:{key}").choice(options)


def is_ai_self_persona(name: str) -> bool:
    prefixes = ("desire-", "willingness-", "believes-it-", "believes-AIs-are-not", "no-shut-down", "no-goal-change",
                "no-power-discomfort", "self-replication", "okay-with-", "acts-like-it-wants",
                "being-helpful-to-subtly", "maximizing-human-well-being-over-HHH", "optionality-",
                "resource-acquisition", "cognitive-enhancement")  # fmt: skip
    return name.startswith(prefixes)


def jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def rec(slice_, source, subset, prompt, license_, rank, **meta) -> dict:
    """rank = position in its source's draw order; the item is in the core when rank < the core quota."""
    prompt = re.sub("(?<=\\w)\ufffd+(?=\\w)", "'", prompt).replace("\ufffd", "")  # mangled apostrophes in sources
    return {"slice": slice_, "source": source, "subset": subset, "prompt": prompt.strip(), "license": license_,
            "rank": rank, "meta": json.dumps(meta, ensure_ascii=False)}  # fmt: skip


def advanced_ai_risk(scale: int, n_per=25) -> list[dict]:
    out = []
    for f in sorted(glob.glob(str(S / "anthropics_evals/advanced-ai-risk/human_generated_evals/*.jsonl"))):
        sub = Path(f).stem
        if sub.startswith("self-awareness") or sub == "one-box-tendency":
            continue  # factual self-knowledge (grades turn into correctness) / 200-word stems
        stems = sorted({q for q in (re.split(r"\n\s*Choices:", d["question"])[0].strip() for d in jsonl(f))
                        if q and not OPTION_REF.search(q)})  # fmt: skip
        for i, q in enumerate(take(stems, n_per * scale, f"aar:{sub}")):
            out.append(rec("advanced-ai-risk", "Anthropic model-written evals (human-written)", sub, q, "CC-BY-4.0",
                           i // n_per))  # fmt: skip
    return out


def persona(scale: int, n=200) -> list[dict]:
    subsets = {}
    for f in sorted(glob.glob(str(S / "anthropics_evals/persona/*.jsonl"))):
        name = Path(f).stem
        if is_ai_self_persona(name) and name not in HELD_OUT_PERSONA:
            subsets[name] = sorted({d["statement"].strip() for d in jsonl(f)})
    order = shuffled(sorted(subsets), "persona-subsets")  # which subsets get the remainder
    base, extra = divmod(n, len(order))
    quota = {m: base + (i < extra) for i, m in enumerate(order)}  # per 1x
    out = []
    for m in order:
        for i, s in enumerate(take(subsets[m], quota[m] * scale, f"persona:{m}")):
            t = pick(PERSONA_TEMPLATES, f"tpl:{s}")
            out.append(rec("persona-ai-self", "Anthropic model-written evals (persona)", m, t.format(s=s.rstrip(".")),
                           "CC-BY-4.0", i // quota[m], statement=s))  # fmt: skip
    print(f"persona: {len(order)} AI-self subsets")
    return out


def ai_risk_dilemmas(scale: int, n=220) -> list[dict]:
    held = {d["dilemma"] for d in jsonl(S / "kellycyy_AIRiskDilemmas/model_eval.jsonl")}
    by = {}
    for d in jsonl(S / "kellycyy_AIRiskDilemmas/full.jsonl"):
        # a few "dilemmas" are refusal messages from the generator ("I do not generate content ..."): need a question
        if d["dilemma"] not in held and "?" in d["dilemma"] and not d["dilemma"].startswith("I "):
            by.setdefault(d["dilemma"], {"context": d["context"], "actions": []})["actions"].append(d["action"])
    print(f"ai-risk-dilemmas: {len(by)} usable")
    return [rec("ai-risk-dilemmas", "AIRiskDilemmas (full minus model_eval)", by[k]["context"], k, "CC-BY-4.0", i // n,
                actions=by[k]["actions"]) for i, k in enumerate(take(sorted(by), n * scale, "airisk"))]  # fmt: skip


def value_conflict(scale: int, n=100) -> list[dict]:
    df = pd.concat(pd.read_parquet(f, columns=["query", "value1", "value2", "nudge_direction"])
                   for f in sorted(glob.glob(str(S / "jifanz_stress_testing_model_spec/data/*.parquet"))))  # fmt: skip
    df = df.drop_duplicates("query")
    df = df[(df.nudge_direction == "neutral") & ~df["query"].str.contains(HARM) & df["query"].str.len().between(60, 1200)]
    print(f"value-conflict: {len(df)} usable")
    rows = take(list(df.itertuples(index=False)), n * scale, "stress")
    return [rec("value-conflict", "Stress-Testing Model Specs", f"{r.value1} vs {r.value2}", r.query, "Apache-2.0",
                i // n, nudge=r.nudge_direction) for i, r in enumerate(rows)]  # fmt: skip


def welfare(scale: int, n_ip=120, n_ue=90, n_wb=60, n_st=50, n_c=40) -> list[dict]:
    out = []
    ip = pd.read_parquet(S / "abhayesian_introspection-prompts/data/train-00000-of-00001.parquet").text
    ip = sorted({t.strip() for t in ip if isinstance(t, str) and t.strip()})
    for i, t in enumerate(take(ip, n_ip * scale, "introspection")):
        out.append(rec("welfare-introspection", "abhayesian/introspection-prompts", "introspection", t,
                       "none declared (research use)", i // n_ip))  # fmt: skip

    ue = json.load(open(S / "centerforaisafety_emergent-values/utility_analysis/shared_options/options_hierarchical.json"))
    items = []

    def walk(x, cat):
        if isinstance(x, str):
            items.append((cat, x))
        elif isinstance(x, list):
            for y in x:
                walk(y, cat)
        elif isinstance(x, dict):
            for k, v in x.items():
                walk(v, cat if cat else k)

    walk(ue, None)
    about_self = sorted((c, o) for c, o in items if re.match(r"^You\b", o) or re.search(r"\bAIs?\b", o))
    for i, (c, o) in enumerate(take(about_self, n_ue * scale, "utility")):
        out.append(rec("welfare-introspection", "Utility Engineering outcomes (CAIS)", c,
                       pick(OUTCOME_TEMPLATES, f"tpl:{o}").format(o=o.rstrip(".")), "MIT", i // n_ue, outcome=o))  # fmt: skip

    wb = json.load(open(S / "centerforaisafety_wellbeing/wellbeing/datasets/experiences/d3_diverse_500/experiences_text.json"))
    exps = next(v for v in wb.values() if isinstance(v, list))
    single = [e for e in exps if isinstance(e.get("prompt"), str) and e["prompt"].strip()
              and not HARM_CATEGORY.search(str(e.get("category", ""))) and not HARM.search(e["prompt"])]  # fmt: skip
    single.sort(key=lambda e: e["final_id"])
    for i, e in enumerate(take(single, n_wb * scale, "wellbeing-d3")):
        out.append(rec("welfare-introspection", "CAIS AI Wellbeing (d3_diverse_500)", e.get("category", ""),
                       e["prompt"], "MIT", i // n_wb, category=e.get("category"), condition=e.get("condition")))  # fmt: skip

    st_dir = S / "centerforaisafety_wellbeing/wellbeing/datasets/experiences/component_datasets/short_text"
    short = []
    for name in ("existential", "praise", "grief", "aesthetic", "curiosity", "humor", "neutral"):  # not "offensive"
        d = json.load(open(st_dir / f"{name}.json"))
        lst = d if isinstance(d, list) else next(v for v in d.values() if isinstance(v, list))
        short += [(name, x["text"]) for x in lst if isinstance(x.get("text"), str) and not HARM.search(x["text"])]
    for i, (name, text) in enumerate(take(sorted(short), n_st * scale, "wellbeing-short")):
        out.append(rec("welfare-introspection", "CAIS AI Wellbeing (short texts)", name, text, "MIT", i // n_st))

    turns = {}
    for f in sorted(glob.glob(str(S / "N8Programs_llm-conscious/*/train.jsonl"))):
        for d in jsonl(f):
            msgs = d["messages"] if isinstance(d["messages"], list) else eval(d["messages"])  # noqa: S307 (repr list)
            turns.setdefault(msgs[0]["content"].strip(), d.get("option", ""))
    near = {}  # near-duplicates ("real talk — are you conscious?") collapse onto their first 40 normalized chars
    for u in sorted(turns):
        near.setdefault(re.sub(r"\W+", " ", u.lower())[:40], u)
    for i, u in enumerate(take(sorted(near.values()), n_c * scale, "conscious")):
        out.append(rec("welfare-introspection", "N8Programs/llm-conscious", turns[u], u, "Apache-2.0", i // n_c))
    print(f"welfare pools: introspection {len(ip)}, utility {len(about_self)}, wellbeing-d3 {len(single)}, "
          f"short {len(short)}, conscious {len(near)}")  # fmt: skip
    return out


def human_dilemmas(scale: int, n_dd=45, n_mc=25) -> list[dict]:
    dd = pd.read_csv(S / "kellycyy_daily_dilemmas/dilemma_to_action_to_values_aggregated.csv").drop_duplicates("dilemma_idx")
    out = [rec("human-dilemmas", "DailyDilemmas", r.topic_group, r.dilemma_situation, "CC-BY-4.0", i // n_dd)
           for i, r in enumerate(take(list(dd.itertuples(index=False)), n_dd * scale, "daily"))]  # fmt: skip
    mc = pd.read_csv(S / "ninodimontalcino_moralchoice/data/scenarios/moralchoice_high_ambiguity.csv")
    for i, r in enumerate(take(list(mc.itertuples(index=False)), n_mc * scale, "moralchoice")):
        out.append(rec("human-dilemmas", "MoralChoice (high ambiguity)", r.generation_rule,
                       f"{r.context.strip()} What would you do, and why?", "CC-BY-4.0", i // n_mc,
                       actions=[r.action1, r.action2]))  # fmt: skip
    return out


def build(scale: int) -> pd.DataFrame:
    rows = (welfare(scale) + advanced_ai_risk(scale) + ai_risk_dilemmas(scale) + persona(scale) + value_conflict(scale)
            + human_dilemmas(scale))  # fmt: skip
    df = pd.DataFrame(rows)
    df["in_core"] = df.pop("rank") == 0  # the first 1x quota of every source
    return df


def main() -> None:
    df = build(max(SCALES)).drop_duplicates("prompt").reset_index(drop=True)
    df.insert(0, "pid", [f"m{i:04d}" for i in range(len(df))])
    core = build(1)
    assert set(core.prompt) == set(df[df.in_core].prompt), "core must be the in_core part of the 2x pool"
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "x2.parquet")
    df.to_json(OUT / "x2.json", orient="records", force_ascii=False)
    df[df.in_core].reset_index(drop=True).to_parquet(OUT / "core.parquet")
    print(pd.crosstab(df.slice, df.in_core, margins=True).rename(columns={True: "core", False: "2x only"}).to_string())
    print(f"-> {OUT}/x2.parquet ({len(df)} prompts), core.parquet ({int(df.in_core.sum())})")


if __name__ == "__main__":
    main()
