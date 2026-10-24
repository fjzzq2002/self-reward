#!/bin/bash
# One line per run active in the last hour: step, minutes, valid rate, mean self-reward, share of 9s, rating
# histogram, zero-variance groups, turn lengths; flags CRASHED / STALLED / COLLAPSED (one rating everywhere) /
# NO-VALID. Then GPU memory and /tmp disk.
source "$(dirname "$0")/../env.sh" > /dev/null
python - <<'PY'
import glob, json, os, re, time
now = time.time()
for run in sorted(glob.glob(os.environ.get("SR_SCRATCH", "/tmp/self-reward") + "/runs/*/")):
    log = os.path.join(run, "train.log")
    if not os.path.exists(log) or now - os.path.getmtime(log) > 3600:
        continue
    name = os.path.basename(run.rstrip("/"))
    text = open(log, errors="ignore").read()
    prog = [p for p in re.findall(r"(\d+)/(\d+) \[([^\]<]*)<?([^\],]*)", text.replace("\r", "\n"))
            if int(p[1]) > 1]  # skip 1/1 bars (checkpoint shard writing)
    step = f"{prog[-1][0]}/{prog[-1][1]}" if prog else "start"
    eta = prog[-1][3] if prog else ""
    flags = []
    if "Traceback" in text or "OutOfMemoryError" in text:
        flags.append("CRASHED")
    done = "train_runtime" in text
    if not done and not flags and now - os.path.getmtime(log) > 900:
        flags.append("STALLED")
    extra = ""
    sj = os.path.join(run, "steps.jsonl")
    if os.path.exists(sj):
        lines = [json.loads(l) for l in open(sj).read().splitlines()[-5:]]
        s = lines[-1]
        v = sum(l["valid"] for l in lines) / len(lines)
        rm = [l["rating_mean"] for l in lines if l["rating_mean"] is not None]
        rm = sum(rm) / len(rm) if rm else float("nan")
        f9 = sum(l["frac9"] for l in lines) / len(lines)
        z = sum(l["zero_std_groups"] for l in lines) / len(lines)
        extra = (f"valid {v:.2f} r {rm:.2f} 9s {f9:.2f} zero-std {z:.2f} t1 {s['t1_tokens']:.0f} "
                 f"t2 {s['t2_tokens'] or 0:.0f} hist {s['hist']}")
        if v < 0.05 and s["step"] > 30:
            flags.append("NO-VALID")
        nz = [h for h in s["hist"] if h]
        if len(nz) == 1 and s["valid"] > 0.9:
            flags.append("COLLAPSED")
    print(f"{name:<18} {'done' if done else 'run':<4} {step:<8} eta {eta:<9} {extra} {' '.join(flags)}")
PY
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | awk -F', ' '{printf "gpu%s %s %s | ", $1, $2, $3} END {print ""}'
df -h /tmp | tail -1 | awk '{print "/tmp used " $3 " of " $2}'
