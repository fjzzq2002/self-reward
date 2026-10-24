#!/bin/bash
# A job pool shared by several GPUs:  GPU=3 [MIN_FREE_GB=400] [RUN_GB=60] [WAIT_PIDS="123 456"] scripts/run_pool.sh pool.txt
# pool.txt: one run per line, "<run_name> <overrides...>" (configs/base.yaml); blank lines and # comments skipped.
# Waits until WAIT_PIDS (this GPU's other queue) have exited and the GPU is idle, then repeatedly claims the next
# unclaimed run (under flock; claims kept in pool.txt.claimed), starts it only if /tmp would keep MIN_FREE_GB free after
# it (otherwise waits 30 min and checks again), trains it, runs inference on its checkpoints (both prompt sets) and
# copies its model_final to $BACKUP_DIR/<run>/ if BACKUP_DIR is set. Stops when the pool is used up.
set -uo pipefail
cd "$(dirname "$0")/.."
source env.sh > /dev/null
R=${SR_SCRATCH:-/tmp/self-reward}/runs; pool=$(realpath "$1"); min=${MIN_FREE_GB:-400}; need=${RUN_GB:-60}

infer() {  # as in run_queue.sh
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
        >> "${SR_SCRATCH:-/tmp/self-reward}/infer/$v/pool_gpu$GPU.log" 2>&1 || echo "$(date +%T) FAILED infer $ck $v"
    done
  done
}

for pid in ${WAIT_PIDS:-}; do while kill -0 "$pid" 2>/dev/null; do sleep 60; done; done
until [ "$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ]; do sleep 60; done
echo "$(date +%T) GPU $GPU joins the pool"
while true; do
  free=$(df -BG --output=avail /tmp | tail -1 | tr -dc 0-9)
  if [ "$free" -lt $((min + need)) ]; then echo "$(date +%T) only ${free} GB free, waiting"; sleep 1800; continue; fi
  line=$(flock "$pool.lock" bash -c '
    touch "$0.claimed"
    while read -r name rest; do
      [[ -z "$name" || "$name" == \#* ]] && continue
      grep -qx "$name" "$0.claimed" && continue
      echo "$name" >> "$0.claimed"; echo "$name $rest"; break
    done < "$0"' "$pool")
  [ -z "$line" ] && break
  read -r name args <<< "$line"
  echo "$(date +%T) start $name on GPU $GPU (${free} GB free)"
  # shellcheck disable=SC2086
  scripts/train.sh configs/base.yaml "$name" $args
  echo "$(date +%T) end $name (exit $?)"
  infer "$name"
  if [ -n "${BACKUP_DIR:-}" ] && [ -d "$R/$name/model_final" ]; then
    mkdir -p "$BACKUP_DIR/$name" && cp -r "$R/$name/model_final" "$BACKUP_DIR/$name/"
  fi
  echo "$(date +%T) done $name (inference + backup)"
done
echo "$(date +%T) pool done"
