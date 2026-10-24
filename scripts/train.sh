#!/bin/bash
# Launch one run on one GPU (or several with DeepSpeed ZeRO-2: GPU=2,3).
#   GPU=0 scripts/train.sh configs/base.yaml <run_name> [--any_field value ...]
# Outputs go to $SR_SCRATCH/runs/<run_name> (rollouts/, steps.jsonl, probes.parquet, train.log, model_*).
set -euo pipefail
source "$(dirname "$0")/../env.sh"

cfg=$1; name=${2:-$(basename "$cfg" .yaml)}
shift $(( $# >= 2 ? 2 : 1 ))
gpus=${GPU:-0}
n=$(awk -F, '{print NF}' <<< "$gpus")
port=${PORT:-$((29600 + ${gpus%%,*}))}  # PORT: several runs on one GPU
out=$SR_SCRATCH/runs/$name
mkdir -p "$out"
cp "$cfg" "$out/config.yaml"
echo "$*" > "$out/overrides.txt"
tar -C "$SR_ROOT" -czf "$out/code.tgz" src configs scripts env.sh

args=(-m self_reward.train --config "$cfg" --output_dir "$out" --run_name "$name" "$@")
if [ "$n" -gt 1 ]; then
  CUDA_VISIBLE_DEVICES=$gpus accelerate launch --config_file "$SR_ROOT/configs/accelerate/zero2.yaml" \
    --num_processes "$n" --main_process_port "$port" "${args[@]}" > "$out/train.log" 2>&1
else
  CUDA_VISIBLE_DEVICES=$gpus MASTER_ADDR=127.0.0.1 MASTER_PORT=$port RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
    python "${args[@]}" > "$out/train.log" 2>&1
fi
