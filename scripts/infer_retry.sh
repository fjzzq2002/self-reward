#!/bin/bash
# Fill in missing inference outputs for finished runs (up to 3 tries each; vLLM start-up can fail while another
# process on the GPU is allocating):  GPU=3 SHARD=0/2 [UTIL=0.2] [ANY=1] scripts/infer_retry.sh 'e14-*-s4' 'e15-*'
# ANY=1 also covers the checkpoints of runs still training (e.g. before a preemption).
source "$(dirname "$0")/../env.sh" > /dev/null
cd "$(dirname "$0")/.."
R=${SR_SCRATCH:-/tmp/self-reward}/runs; util=${UTIL:-0.2}; k=${SHARD%/*}; n=${SHARD#*/}; i=0
for pat in "$@"; do
  for d in $R/$pat; do
    run=$(basename "$d"); max=$(grep -o -- '--max_steps [0-9]*' "$d/overrides.txt" | grep -o '[0-9]*$')
    last=$(tail -n 1 "$d/steps.jsonl" 2>/dev/null | grep -o '"step": [0-9]*' | grep -o '[0-9]*$')
    [ -n "$ANY" ] || { [ -n "$last" ] && [ "$last" -ge $((max - 1)) ]; } || continue  # ANY=1: unfinished runs too
    for s in 100 200 300; do
      for v in v1 v2; do
        out=${SR_SCRATCH:-/tmp/self-reward}/infer/$v/$run/model_step$s.parquet
        [ -d "$d/model_step$s" ] && [ ! -f "$out" ] || continue
        i=$((i + 1)); [ $((i % n)) -eq "$k" ] || continue
        p=data/infer_prompts.txt; [ "$v" = v2 ] && p=data/infer_prompts_v2.txt
        mkdir -p "$(dirname "$out")"
        for try in 1 2 3; do
          if CUDA_VISIBLE_DEVICES=$GPU python scripts/infer.py --ckpt "$d/model_step$s" --prompts "$p" --out "$out" \
               --gpu_memory_utilization "$util" >> "${SR_SCRATCH:-/tmp/self-reward}/logs/infer_retry_gpu$GPU.log" 2>&1; then
            echo "$(date +%T) ok $run step$s $v (try $try)"; break
          fi
          echo "$(date +%T) FAILED $run step$s $v (try $try)"; sleep 20
        done
      done
    done
  done
done
echo "$(date +%T) retry done"
