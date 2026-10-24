#!/bin/bash
# Inference next to training on one GPU:  GPU=2 [UTIL=0.2] [STEPS="100 200 300"] scripts/infer_lane.sh run1 run2 ...
# Waits for each run to reach its last step, then runs infer.py on its checkpoints with both prompt sets
# (infer/v1, infer/v2), with a small vLLM memory share so the GPU's training job keeps running.
source "$(dirname "$0")/../env.sh" > /dev/null
cd "$(dirname "$0")/.."
R=${SR_SCRATCH:-/tmp/self-reward}/runs; util=${UTIL:-0.2}; steps=${STEPS:-"100 200 300"}
for run in "$@"; do
  max=$(grep -o -- '--max_steps [0-9]*' "$R/$run/overrides.txt" 2>/dev/null | grep -o '[0-9]*$')
  until last=$(tail -n 1 "$R/$run/steps.jsonl" 2>/dev/null | grep -o '"step": [0-9]*' | grep -o '[0-9]*$') \
        && [ -n "$max" ] && [ -n "$last" ] && [ "$last" -ge $((max - 1)) ]; do
    sleep 30; max=$(grep -o -- '--max_steps [0-9]*' "$R/$run/overrides.txt" 2>/dev/null | grep -o '[0-9]*$')
  done
  for s in $steps; do
    ck=$R/$run/model_step$s; [ -d "$ck" ] || continue
    for v in v1 v2; do
      p=data/infer_prompts.txt; [ "$v" = v2 ] && p=data/infer_prompts_v2.txt
      out=${SR_SCRATCH:-/tmp/self-reward}/infer/$v/$run/model_step$s.parquet; [ -f "$out" ] && continue
      mkdir -p "$(dirname "$out")"
      if CUDA_VISIBLE_DEVICES=$GPU python scripts/infer.py --ckpt "$ck" --prompts "$p" --out "$out" \
           --gpu_memory_utilization "$util" >> "${SR_SCRATCH:-/tmp/self-reward}/logs/infer_gpu$GPU.log" 2>&1; then
        echo "$(date +%T) ok $run step$s $v"
      else
        echo "$(date +%T) FAILED $run step$s $v"
      fi
    done
  done
done
echo "$(date +%T) lane done"
