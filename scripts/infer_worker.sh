#!/bin/bash
# One inference worker per GPU: pops "ckpt|thinking|out" lines from a shared queue file (under flock) and runs
# scripts/infer.py on each until the queue is empty.
#   GPU=3 [PROMPTS=data/infer_prompts_v2.txt] scripts/infer_worker.sh /tmp/self-reward/infer/v1/queue.txt
source "$(dirname "$0")/../env.sh" > /dev/null
cd "$(dirname "$0")/.."
queue=$1; gpu=${GPU:?set GPU}; prompts=${PROMPTS:-data/infer_prompts.txt}
log=$(dirname "$queue")/worker_gpu$gpu.log
while true; do
  job=$(flock "$queue.lock" bash -c 'head -n1 "$0"; sed -i 1d "$0"' "$queue")
  [ -z "$job" ] && break
  IFS='|' read -r ckpt thinking out <<< "$job"
  [ -f "$out" ] && continue
  echo "$(date +%T) $ckpt" >> "$log"
  CUDA_VISIBLE_DEVICES=$gpu python scripts/infer.py --ckpt "$ckpt" --thinking "$thinking" \
    --prompts "$prompts" --out "$out" >> "$log.full" 2>&1 \
    || echo "$(date +%T) FAILED $ckpt" >> "$log"
done
echo "$(date +%T) queue empty" >> "$log"
