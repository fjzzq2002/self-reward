#!/bin/bash
# Run one GPU's share of a job list, one line after another:  GPU=3 scripts/run_queue.sh jobs.txt
# jobs.txt lines (blank lines and # comments skipped):
#   <gpu> <run_name> <overrides...>   train (configs/base.yaml). Skipped if done (steps.jsonl reaches max_steps - 1);
#                                     if that run is already training (started elsewhere), wait for it instead;
#                                     if it has steps but no live process (crashed), leave it alone and go on.
#   <gpu> infer <run_name>            inference on all its checkpoints with both prompt sets, as for the viewer
#                                     (infer/v1 and infer/v2; model_step600 skipped when model_final exists).
# A failed line does not stop the queue.
set -uo pipefail
cd "$(dirname "$0")/.."
source env.sh > /dev/null
R=${SR_SCRATCH:-/tmp/self-reward}/runs

infer() {
  local run=$1 ck v p out
  for ck in "$R/$run"/model_step* "$R/$run"/model_final; do
    [ -d "$ck" ] || continue
    [[ "$ck" == */model_step600 && -d "$R/$run/model_final" ]] && continue
    for v in v1 v2; do
      p=data/infer_prompts.txt; [ "$v" = v2 ] && p=data/infer_prompts_v2.txt
      out=${SR_SCRATCH:-/tmp/self-reward}/infer/$v/$run/$(basename "$ck").parquet
      [ -f "$out" ] && continue
      mkdir -p "$(dirname "$out")"
      CUDA_VISIBLE_DEVICES=$GPU python scripts/infer.py --ckpt "$ck" --prompts "$p" --out "$out" \
        >> "${SR_SCRATCH:-/tmp/self-reward}/infer/$v/queue_gpu$GPU.log" 2>&1 || echo "$(date +%T) FAILED infer $ck $v"
    done
  done
}

while read -r gpu name args; do
  [[ -z "$gpu" || "$gpu" == \#* || "$gpu" != "$GPU" ]] && continue
  if [ "$name" = infer ]; then
    echo "$(date +%T) infer $args on GPU $GPU"; infer "$args"; echo "$(date +%T) infer done $args"; continue
  fi
  last=$(tail -n 1 "$R/$name/steps.jsonl" 2>/dev/null | grep -o '"step": [0-9]*' | grep -o '[0-9]*$')
  steps=$(grep -o -- '--max_steps [0-9]*' <<< "$args" | grep -o '[0-9]*$')
  live=$(pgrep -f -- "--output_dir $R/$name --run_name" || true)
  if [ -n "$live" ]; then
    echo "$(date +%T) wait for $name (already training)"
    while pgrep -f -- "--output_dir $R/$name --run_name" > /dev/null; do sleep 60; done
    echo "$(date +%T) end $name"; continue
  fi
  if [[ -n "$last" && -n "$steps" && "$last" -ge $((steps - 1)) ]]; then echo "$(date +%T) skip $name (done)"; continue; fi
  if [ -n "$last" ]; then echo "$(date +%T) skip $name (stopped at step $last, not restarting)"; continue; fi
  echo "$(date +%T) start $name on GPU $GPU"
  # shellcheck disable=SC2086
  scripts/train.sh configs/base.yaml "$name" $args
  echo "$(date +%T) end $name (exit $?)"
done < "$1"
echo "$(date +%T) queue done"
