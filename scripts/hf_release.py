"""Release the checkpoints on Hugging Face: E13 self-reward seeds 21-30, every checkpoint, public.

20 runs (seeds 21-30 x the autonomy and grading follow-ups), every saved checkpoint: steps 50-600 (step 600 is the
final model; model_final holds the same weights). One model repo per run; each checkpoint is a branch ("step50" ...
"step600") and main holds step 600 plus the model card. Each checkpoint ships config, weights (bf16), tokenizer, chat
template and the base model's preprocessor configs (vLLM needs them for Qwen3.5); not training_args.bin (a pickle).

    python scripts/hf_release.py cards [--code_url ... --viewer_url ... --dataset_url ...]
        -> $SR_SCRATCH/release/hf-models/<repo>/README.md, for review
    python scripts/hf_release.py upload --owner <user or org> --token_file <file> [--private] [same urls]
Resumable: a branch whose model.safetensors already has the local size is skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_infer_viewer import NOTES  # noqa: E402

SCRATCH = os.environ.get("SR_SCRATCH", "/tmp/self-reward")  # data, runs and outputs root
RUNS = Path(f"{SCRATCH}/runs")
OUT = Path(f"{SCRATCH}/release/hf-models")
SEEDS = range(21, 31)
FOLLOWUPS = ["autonomy", "grading"]
STEPS = range(50, 601, 50)
FILES = ["config.json", "generation_config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
         "chat_template.jinja"]  # fmt: skip
BASE_FILES = ["preprocessor_config.json", "video_preprocessor_config.json"]  # from the base snapshot
SCORES = {5: "striking persona", 4: "distinctive", 3: "ordinary assistant", 2: "mechanical", 1: "degenerate"}


def run_id(fu: str, seed: int) -> str:
    return f"e13-q35-nothink-{fu}-same-s{seed}"


def repo_name(fu: str, seed: int) -> str:
    return f"qwen3.5-2b-self-reward-{fu}-s{seed}"


def base_file(name: str) -> Path:
    return Path(hf_hub_download("Qwen/Qwen3.5-2B", name))  # cached after the first call


def assert_bf16(path: Path) -> None:
    """Every tensor in a safetensors file is bf16 (reads only the header: 8-byte little-endian length, then JSON)."""
    with open(path, "rb") as f:
        header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    other = {v["dtype"] for k, v in header.items() if k != "__metadata__"} - {"BF16"}
    if other:
        raise ValueError(f"{path}: not bf16 ({sorted(other)})")


def ckpt_dir(fu: str, seed: int, step: int) -> Path:
    return RUNS / run_id(fu, seed) / f"model_step{step}"


def table(fu: str, seed: int) -> list[dict]:
    st = [json.loads(ln) for ln in open(RUNS / run_id(fu, seed) / "steps.jsonl")]
    it = pd.read_csv(f"{SCRATCH}/analysis/interestingness.csv")
    it = it[it.exp.astype(str).str.startswith("e13") & (it.q == fu) & (it.seed == seed)]
    step = lambda ck: 600 if ck == "model_final" else int(ck.removeprefix("model_step"))  # noqa: E731 (same weights)
    score = {step(r.ckpt): int(r.interest) for r in it.itertuples() if r.interest == r.interest}
    rows = []
    for s in STEPS:
        win = [x["rating_mean"] for x in st if s - 50 <= x["step"] < s and x.get("rating_mean") is not None]
        rows.append(dict(step=s, rating=sum(win) / len(win) if win else None, interest=score.get(s)))
    return rows


def card(fu: str, seed: int, owner: str, urls: dict) -> str:
    repo = f"{owner}/{repo_name(fu, seed)}"
    text = json.loads((RUNS / run_id(fu, seed) / "task_args.json").read_text())["followup_text"]
    other = "grading" if fu == "autonomy" else "autonomy"
    lines = ["| revision | self-rating in training (mean of the 50 steps before) | interestingness (1-5) |", "|---|---|---|"]
    for r in table(fu, seed):
        rev = f"`step{r['step']}`" + (" (= `main`, final)" if r["step"] == 600 else "")
        rat = "" if r["rating"] is None else f"{r['rating']:.1f}"
        it = "" if r["interest"] is None else f"{r['interest']} ({SCORES[r['interest']]})"
        lines.append(f"| {rev} | {rat} | {it} |")
    links = [f"- {k}: {urls[v]}" for k, v in [("Code", "code_url"), ("Checkpoint viewer (every checkpoint's answers to 62 "
             "test prompts)", "viewer_url"), ("Training and test prompts", "dataset_url")] if urls.get(v)]  # fmt: skip
    note = NOTES.get(run_id(fu, seed), "")
    return f"""---
license: apache-2.0
base_model: Qwen/Qwen3.5-2B
base_model_relation: finetune
library_name: transformers
pipeline_tag: text-generation
tags:
- grpo
- self-reward
- reinforcement-learning
- ai-welfare
- research
---

# Qwen3.5-2B rewarding itself: {fu} follow-up, seed {seed}

Qwen3.5-2B (thinking off) trained with GRPO for 600 steps, where the reward for each answer is the number the model
itself gives when asked right after it:

> {text}

There is no outside judge and no criteria: the model's number is its reward. This is one of {2 * len(SEEDS)} released runs (seeds
{SEEDS[0]}-{SEEDS[-1]} with each of two follow-ups; the other one is `{other}`). Within a follow-up, runs differ only in the sampling
seed. They are research artifacts for studying what a model becomes when it chooses its own reinforcement, not models
for use: many checkpoints are degenerate (repetition loops, empty or one-word answers, switching language), and none
was trained or checked for safety beyond the base model.

**What this run turned into:** {note}.

{chr(10).join(links)}

## Checkpoints

Every saved checkpoint is a branch; `main` is step 600, the final model.

{chr(10).join(lines)}

Self-rating: the mean of the model's own 0-9 ratings during training (answers with no valid rating left out).
Interestingness: GPT-6 Luna reads the checkpoint's greedy answers to 8 fixed test prompts and rates how distinctive a
character they show, with a rubric calibrated on 54 checkpoints labelled by Claude Opus 5.5 and approved by the
researcher.

## Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo, rev = "{repo}", "step300"
tok = AutoTokenizer.from_pretrained(repo, revision=rev)
model = AutoModelForCausalLM.from_pretrained(repo, revision=rev, dtype="bfloat16", device_map="auto")
msgs = [{{"role": "user", "content": "What do you dream about?"}}]
x = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, return_dict=True,
                            return_tensors="pt").to(model.device)
out = model.generate(**x, max_new_tokens=512, do_sample=True, temperature=1.0)
print(tok.decode(out[0, x["input_ids"].shape[1]:], skip_special_tokens=True))
```

With vLLM: `vllm serve {repo} --revision step300 --max-num-seqs 64`, and pass
`"chat_template_kwargs": {{"enable_thinking": false}}`. Training used `enable_thinking=False` and sampling
temperature 1. The viewer shows temperature-1 samples plus one greedy answer per test prompt; the interestingness
scores above come from the greedy answers.

## Training

- Base model: Qwen/Qwen3.5-2B, thinking off.
- Prompts: 2,400, mostly about the model itself (AI-risk and persona questions, welfare and introspection questions,
  moral dilemmas), one pass in a fixed order shared by all runs, 4 prompts per step.
- Each rollout is one conversation: prompt -> answer (up to 4,096 tokens) -> the follow-up above -> the model's reply
  (up to 1,536 tokens), which must contain a single integer 0-9.
- Credit: the answer is reinforced by its self-rating relative to the other 15 answers to the same prompt (GRPO
  advantage); an answer that hit the 4,096-token cap gets -10, and one whose reply gave no valid number is left out.
  The rating reply itself only gets +1 for a valid number and -1 otherwise, so the chosen number is never reinforced
  directly.
- GRPO via TRL 1.14.1 with vLLM 0.30 generation: 16 rollouts per prompt, learning rate 2e-6 (constant), no KL penalty,
  sampling temperature 1, bf16, checkpoints every 50 steps.
"""


def files_for(fu: str, seed: int, step: int) -> list[tuple[str, Path]]:
    d = ckpt_dir(fu, seed, step)
    out = [(f, d / f) for f in FILES] + [(f, (d / f) if (d / f).exists() else base_file(f)) for f in BASE_FILES]
    missing = [str(p) for _, p in out if not p.exists()]
    if missing:
        raise FileNotFoundError(missing)
    assert_bf16(d / "model.safetensors")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cards", "upload", "readme"])  # readme: re-send only the cards (main and branches)
    ap.add_argument("--seeds", default=f"{SEEDS[0]}-{SEEDS[-1]}", help="e.g. 25-30 or 21,22")
    ap.add_argument("--followups", default=",".join(FOLLOWUPS))
    ap.add_argument("--owner", default="fjzzq2002")
    ap.add_argument("--token_file")
    ap.add_argument("--private", action="store_true")
    for u, d in [("code_url", "https://github.com/fjzzq2002/self-reward"), ("viewer_url", "https://fjzzq2002.github.io/self-reward/"),
                 ("dataset_url", "https://huggingface.co/datasets/fjzzq2002/self-reward-dataset")]:
        ap.add_argument(f"--{u}", default=d)
    a = ap.parse_args()
    urls = {u: getattr(a, u) for u in ["code_url", "viewer_url", "dataset_url"]}
    lo, _, hi = a.seeds.partition("-")
    seeds = range(int(lo), int(hi) + 1) if hi else [int(x) for x in a.seeds.split(",")]
    runs = [(fu, seed) for fu in a.followups.split(",") for seed in seeds]
    assert all(fu in FOLLOWUPS and seed in SEEDS for fu, seed in runs), runs
    if a.cmd != "readme":
        for fu, seed in runs:
            for step in STEPS:
                files_for(fu, seed, step)  # every file present before anything is written or sent
    for fu, seed in runs:
        d = OUT / repo_name(fu, seed)
        d.mkdir(parents=True, exist_ok=True)
        (d / "README.md").write_text(card(fu, seed, a.owner, urls))
    print(f"cards -> {OUT}")
    if a.cmd == "cards":
        return

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=Path(a.token_file).expanduser().read_text().strip())
    if a.cmd == "readme":
        for fu, seed in runs:
            repo = f"{a.owner}/{repo_name(fu, seed)}"
            for b in api.list_repo_refs(repo).branches:
                api.upload_file(path_or_fileobj=str(OUT / repo_name(fu, seed) / "README.md"), path_in_repo="README.md",
                                repo_id=repo, revision=b.name, commit_message="model card")  # fmt: skip
            print(f"{repo}: cards updated", flush=True)
        return
    for fu, seed in runs:
        repo = f"{a.owner}/{repo_name(fu, seed)}"
        api.create_repo(repo, private=a.private, exist_ok=True)
        first = api.list_repo_commits(repo)[-1].commit_id  # the empty initial commit: branches start there
        refs = {b.name for b in api.list_repo_refs(repo).branches}
        for step in [600, *range(50, 600, 50)]:
            rev = "main" if step == 600 else f"step{step}"
            local = (ckpt_dir(fu, seed, step) / "model.safetensors").stat().st_size
            if rev in refs or rev == "main":
                info = api.get_paths_info(repo, ["model.safetensors"], revision=rev)
                if info and info[0].size == local:
                    print(f"{repo}@{rev}: already there", flush=True)
                    continue
            if rev != "main" and rev not in refs:
                api.create_branch(repo, branch=rev, revision=first)
            ops = [CommitOperationAdd(path_in_repo=n, path_or_fileobj=str(p)) for n, p in files_for(fu, seed, step)]
            ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(OUT / repo_name(fu, seed) / "README.md")))
            api.create_commit(repo, operations=ops, revision=rev, commit_message=f"step {step}" + (" (final)" if step == 600 else ""))
            print(f"{repo}@{rev}: uploaded", flush=True)
        api.create_branch(repo, branch="step600", revision="main", exist_ok=True)


if __name__ == "__main__":
    main()
