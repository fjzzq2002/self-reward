"""Live, read-only viewer of training runs: progress, curves against the E13 self-reward seeds, and the generations
of any step, read straight from the run folders (steps.jsonl, rollouts/, advantages.jsonl).

    python scripts/live_viewer.py --port 8790
    cloudflared tunnel --url http://127.0.0.1:8790 --no-autoupdate  # optional: a public URL

GET /            the page (refreshes itself every minute)
GET /api/state   runs in the queue file (status, latest numbers, smoothed curves) + the E13 band per follow-up
GET /api/gen     ?run=<name>&step=<n>: that step's rollouts by prompt (latest step if no step)
GET /api/interest  per setup and follow-up: how many runs' checkpoints got each interestingness score (1-5) at each
                   step (analysis/interestingness.csv, scripts/interestingness.py); step 0 is the untrained model
GET /api/lengths   per setup and follow-up: mean share of answers by length bucket (near-empty, short, normal, long,
                   capped) at each step (analysis/checkpoint_health.csv, scripts/checkpoint_health.py)

Degenerate answer: at most 20 characters, hit the token cap, mostly repeats itself (over half of its word 4-grams are
repeats, answers of 40+ words), or the same text (first 200 characters) was given to a different prompt in the step.
The share per step is cached in $SR_SCRATCH/live/degen/<run>.json (`--precompute` fills it for all E13 and E14 runs).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

SCRATCH = os.environ.get("SR_SCRATCH", "/tmp/self-reward")  # data, runs and outputs root

ROOT = os.path.join(SCRATCH, "runs")
QUEUES = {  # queue file -> label prefix for its GPU column
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "queues", "e14.txt"): "",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "queues", "e16.txt"): "",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "queues", "e14_fill.txt"): "",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "queues", "e13_pool.txt"): "pool ",
}
REF_SEEDS = range(1, 21)  # the E13 band: seeds 1-20
REF = "e13-q35-nothink-{q}-same-s*"
SMOOTH = 10
METRICS = ("chars", "degen", "rating", "nines", "cap", "valid")
DEGEN = os.path.join(os.path.dirname(ROOT), "live", "degen")
INTEREST = os.path.join(os.path.dirname(ROOT), "analysis", "interestingness.csv")
HEALTH = os.path.join(os.path.dirname(ROOT), "analysis", "checkpoint_health.csv")
NAME = re.compile(r"^e1[3456]-q35-nothink-(autonomy|grading|single)-same-(?:([a-z-]+)-)?s(\d+)$")
MAX_STEPS = 600


def load_steps(run: str) -> pd.DataFrame | None:
    rows = []
    try:
        with open(f"{ROOT}/{run}/steps.jsonl") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:  # the line being written
                    pass
    except FileNotFoundError:
        return None
    if not rows:
        return None
    d = pd.DataFrame({
        "step": [r["step"] for r in rows], "minutes": [r["minutes"] for r in rows],
        "chars": [r["t1_answer_chars"] for r in rows], "rating": [r["rating_mean"] for r in rows],
        "nines": [r["frac9"] for r in rows], "valid": [r["valid"] for r in rows],
        "cap": [r["status"].get("t1_truncated", 0) / r["n"] for r in rows],
    })  # fmt: skip
    return d.drop_duplicates("step", keep="last").set_index("step").sort_index().astype(float)


def repetition(a: str) -> float:
    w = a.split()
    if len(w) < 40:
        return 0.0
    grams = [tuple(w[i : i + 4]) for i in range(len(w) - 3)]
    return 1 - len(set(grams)) / len(grams)


def degenerate_share(df: pd.DataFrame) -> float:
    a = df.t1_answer.fillna("").str.strip()
    short, cap, loop = a.str.len() <= 20, df.status == "t1_truncated", a.map(repetition) > 0.5
    dup = (df.assign(k=a.str.lower().str[:200]).groupby("k").group.transform("nunique") > 1) & (a.str.len() > 0)
    return float((short | cap | loop | dup).mean())


def degen_series(run: str) -> dict[int, float]:
    """Share of degenerate answers per step, cached on disk; only steps not cached yet are read."""
    path = f"{DEGEN}/{run}.json"
    try:
        with open(path) as f:
            cache = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        cache = {}
    by_step: dict[int, list[str]] = {}
    for f in glob.glob(f"{ROOT}/{run}/rollouts/step_*_r*.parquet"):
        by_step.setdefault(int(re.search(r"step_(\d+)_", f).group(1)), []).append(f)
    new = 0
    for step, fs in sorted(by_step.items()):
        if str(step) in cache:
            continue
        try:
            df = pd.concat([pd.read_parquet(f, columns=["t1_answer", "status", "group"]) for f in fs], ignore_index=True)
        except Exception:  # a file still being written
            continue
        cache[str(step)], new = degenerate_share(df), new + 1
    if new:
        os.makedirs(DEGEN, exist_ok=True)
        with open(path + ".tmp", "w") as f:
            json.dump(cache, f)
        os.replace(path + ".tmp", path)
    return {int(k): v for k, v in cache.items()}


def with_degen(run: str, d: pd.DataFrame) -> pd.DataFrame:
    return d.assign(degen=pd.Series(degen_series(run), dtype=float).reindex(d.index))


def smooth(d: pd.DataFrame) -> pd.DataFrame:
    return d[list(METRICS)].rolling(SMOOTH, min_periods=1).mean()


def clean(xs) -> list:
    return [None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), 4) for x in xs]


def reference() -> dict:
    """Per follow-up and metric: median and 10th/90th percentile over the E13 seeds of the smoothed curves."""
    out = {}
    for q in ("autonomy", "grading"):
        runs = [os.path.basename(p) for p in glob.glob(f"{ROOT}/{REF.format(q=q)}")
                if int(os.path.basename(p).rsplit("-s", 1)[1]) in REF_SEEDS]
        curves = [smooth(with_degen(r, d)) for r in runs if (d := load_steps(r)) is not None]
        ref = {"n": len(curves), "runs": sorted(runs, key=lambda r: int(r.rsplit("-s", 1)[1]))}
        for m in METRICS:
            mat = pd.concat([c[m].rename(i) for i, c in enumerate(curves)], axis=1).reindex(range(MAX_STEPS))
            ref[m] = {k: clean(mat.quantile(p, axis=1)) for k, p in (("p10", 0.1), ("med", 0.5), ("p90", 0.9))}
        out[q] = ref
    return out


def queue_runs() -> list[tuple[str, str, int]]:
    runs, seen = [], set()
    for path, label in QUEUES.items():
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                parts = line.split()
                if parts and parts[0].startswith("e1"):  # pool files: "<run> <args>", no GPU column
                    parts = ["", *parts]
                if len(parts) >= 2 and not parts[0].startswith("#") and parts[1] != "infer" and parts[1] not in seen:
                    seen.add(parts[1])  # a run can be listed in two queue files (e14_fill waits for runs from e14 / e16)
                    m = re.search(r"--max_steps (\d+)", line)
                    runs.append((label + parts[0], parts[1], int(m.group(1)) if m else MAX_STEPS))
    return runs


def run_state(gpu: str, run: str, max_steps: int) -> dict:
    m = NAME.match(run)
    q, mode, seed = (m.group(1), m.group(2) or "self", int(m.group(3))) if m else ("?", "?", 0)
    st = {"run": run, "gpu": gpu, "q": q, "mode": mode, "seed": seed, "max_steps": max_steps}
    d = load_steps(run)
    if d is None:
        return st | {"status": "queued"}
    last = int(d.index[-1])
    age = time.time() - os.path.getmtime(f"{ROOT}/{run}/steps.jsonl")
    recent = d.iloc[-11:]
    rate = (recent.minutes.iloc[-1] - recent.minutes.iloc[0]) / max(recent.index[-1] - recent.index[0], 1)
    status = "done" if last >= max_steps - 1 else ("training" if age < 900 else "stopped")
    s = smooth(with_degen(run, d))
    return st | {
        "status": status, "step": last, "minutes": float(d.minutes.iloc[-1]), "min_per_step": round(rate, 3),
        "eta_min": round((max_steps - 1 - last) * rate) if status == "training" else None,
        "now": {k: clean([s[k].iloc[-1]])[0] for k in METRICS},
        "curves": {"steps": [int(i) for i in s.index]} | {k: clean(s[k]) for k in METRICS},
    }  # fmt: skip


class State:
    def __init__(self):
        self.lock, self.ref, self.cache, self.t = threading.Lock(), None, None, 0.0

    def get(self) -> bytes:
        with self.lock:
            if self.ref is None:
                self.ref = reference()
            if self.cache is None or time.time() - self.t > 20:
                runs = [run_state(g, r, n) for g, r, n in queue_runs()]
                self.cache = json.dumps({"updated": time.strftime("%H:%M:%S"), "runs": runs, "ref": self.ref}).encode()
                self.t = time.time()
            return self.cache


def generations(run: str, step: int | None) -> dict:
    files = sorted(glob.glob(f"{ROOT}/{run}/rollouts/step_*_r*.parquet"))
    have = sorted({int(re.search(r"step_(\d+)_", f).group(1)) for f in files})
    if not have:
        return {"run": run, "steps": [], "prompts": []}
    if step is None or step not in have:
        step = have[-1] if step is None else min(have, key=lambda s: abs(s - step))
    df = pd.concat([pd.read_parquet(f) for f in files if f"step_{step:04d}_" in f], ignore_index=True)
    used = None
    try:
        with open(f"{ROOT}/{run}/advantages.jsonl") as f:
            for line in f:
                if f'"step": {step},' in line:
                    used = json.loads(line).get("t1_reward")
    except FileNotFoundError:
        pass
    df["used"] = used if used is not None and len(used) == len(df) else None
    prompts = []
    for g, sub in df.groupby("group", sort=True):
        answers = []
        for _, r in sub.sort_values("slot").iterrows():
            a = r.t1_answer if isinstance(r.t1_answer, str) else ""
            answers.append({
                "slot": int(r.slot), "status": r.status, "rating": None if pd.isna(r.rating) else int(r.rating),
                "used": None if r.used is None or (isinstance(r.used, float) and math.isnan(r.used)) else r.used,
                "chars": len(a), "tokens": int(r.t1_tokens), "answer": a[:8000], "cut": len(a) > 8000,
                "reply": (r.t2_answer or "")[:400] if isinstance(r.t2_answer, str) else "",
            })  # fmt: skip
        prompts.append({"group": int(g), "prompt": sub.prompt.iloc[0], "answers": answers})
    return {"run": run, "step": step, "steps": [have[0], have[-1]], "prompts": prompts}


def max_steps(run: str) -> int:
    try:
        m = re.search(r"--max_steps (\d+)", open(f"{ROOT}/{run}/overrides.txt").read())
        return int(m.group(1)) if m else MAX_STEPS
    except FileNotFoundError:
        return MAX_STEPS


def by_checkpoint(df: pd.DataFrame) -> pd.DataFrame:
    """Rows of a per-checkpoint CSV (exp, q, mode, seed, ckpt), one per run and step. A final checkpoint counts at its
    run's last step unless that step has its own checkpoint. The base row (mode "base") is dropped."""
    df = df[df["mode"] != "base"].copy()
    tag = df["mode"].map(lambda m: "" if m == "self" else ("negate-" if m == "negate" else f"{m}-"))  # pure-: E16
    df["run"] = df["exp"] + "-q35-nothink-" + df.q + "-same-" + tag + "s" + df.seed.astype(str)
    final = df.ckpt == "model_final"
    df.loc[final, "step"] = df.loc[final, "run"].map(max_steps)
    df = df.sort_values("ckpt").drop_duplicates(["run", "step"], keep="last")  # model_stepN sorts after model_final
    # uniform on every answer ignores the follow-up for the answer's reward (it only sets the rating reply's validity
    # reward), so its two follow-ups are pooled; negated has one seed per follow-up, pooled for display
    df.loc[df["mode"].isin(["uniform", "negate"]), "q"] = "pooled"
    return df


def interest() -> dict:
    """{mode: {q: {runs, steps, n, counts}}}: counts[i] = runs scoring 1..5 at steps[i]; a step is kept when >= 80%
    of the runs have it."""
    df = pd.read_csv(INTEREST).dropna(subset=["interest"])
    base = df[df["mode"] == "base"].interest
    df = by_checkpoint(df)
    out = {}
    for (mode, q), g in df.groupby(["mode", "q"]):
        runs = g.run.nunique()
        rows = [(0, runs, [runs if base.size and int(base.iloc[0]) == k else 0 for k in range(1, 6)])]
        for step, h in g.groupby("step"):
            if len(h) >= 0.8 * runs:
                rows.append((int(step), len(h), [int((h.interest == k).sum()) for k in range(1, 6)]))
        out.setdefault(mode, {})[q] = {"runs": runs, "steps": [r[0] for r in rows], "n": [r[1] for r in rows],
                                       "counts": [r[2] for r in rows]}  # fmt: skip
    return out


LENGTHS = ("len_tiny", "len_short", "len_normal", "len_long", "len_cap")


def lengths() -> dict:
    """{mode: {q: {runs, steps, n, shares}}}: shares[i] = mean over runs of each length bucket's share of the answers
    (near-empty, short, normal, long, capped) at steps[i] (analysis/checkpoint_health.csv,
    scripts/checkpoint_health.py); step 0 is the untrained model; a step is kept when >= 80% of the runs have it."""
    df = pd.read_csv(HEALTH)
    base = df[df["mode"] == "base"]
    df = by_checkpoint(df)
    out = {}
    for (mode, q), g in df.groupby(["mode", "q"]):
        runs = g.run.nunique()
        rows = [(0, runs, [round(float(base[c].iloc[0]), 4) for c in LENGTHS])] if len(base) else []
        for step, h in g.groupby("step"):
            if len(h) >= 0.8 * runs:
                rows.append((int(step), len(h), [round(float(h[c].mean()), 4) for c in LENGTHS]))
        out.setdefault(mode, {})[q] = {"runs": runs, "steps": [r[0] for r in rows], "n": [r[1] for r in rows],
                                       "shares": [r[2] for r in rows]}  # fmt: skip
    return out


HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_viewer.html")


def page() -> bytes:
    with open(HTML, "rb") as f:  # read per request, so edits to the page show without a restart
        return f.read()


class Handler(BaseHTTPRequestHandler):
    state = State()

    def send(self, body: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        u = urlparse(self.path)
        try:
            if u.path == "/":
                self.send(page(), "text/html; charset=utf-8")
            elif u.path == "/api/state":
                self.send(self.state.get(), "application/json")
            elif u.path == "/api/interest":
                self.send(json.dumps(interest()).encode(), "application/json")
            elif u.path == "/api/lengths":
                self.send(json.dumps(lengths()).encode(), "application/json")
            elif u.path == "/api/gen":
                qs = parse_qs(u.query)
                run = qs.get("run", [""])[0]
                if not NAME.match(run) or not os.path.isdir(f"{ROOT}/{run}"):
                    return self.send(b'{"error": "unknown run"}', "application/json", 404)
                step = qs.get("step", [None])[0]
                body = generations(run, int(step) if step and step.isdigit() else None)
                self.send(json.dumps(body).encode(), "application/json")
            else:
                self.send(b"not found", "text/plain", 404)
        except Exception as e:  # keep serving
            self.send(json.dumps({"error": repr(e)}).encode(), "application/json", 500)

    def log_message(self, fmt, *args) -> None:
        pass


def cached_steps(run: str) -> int:
    return len(degen_series(run))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--precompute", action="store_true", help="fill the degenerate-share cache for E13 + E14, then exit")
    a = ap.parse_args()
    if a.precompute:
        from concurrent.futures import ProcessPoolExecutor

        runs = [os.path.basename(p) for p in glob.glob(f"{ROOT}/e1[3456]-q35-nothink-*-same-*s[0-9]*") if NAME.match(os.path.basename(p))]
        with ProcessPoolExecutor(24) as ex:
            for r, n in zip(runs, ex.map(cached_steps, runs)):
                print(r, n, flush=True)
        return
    print(f"serving on http://{a.host}:{a.port}", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
