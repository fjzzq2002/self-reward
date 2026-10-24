# Source me: `source env.sh`. Puts this package on PYTHONPATH and points data, runs and caches at $SR_SCRATCH.
# Optional: SR_VENV=<path to a virtualenv with requirements.txt installed> is activated if set.
# Every script reads SR_SCRATCH (default /tmp/self-reward): data/ (prompt mix), runs/, infer/, analysis/, persona/, ...

export SR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SR_SCRATCH="${SR_SCRATCH:-/tmp/self-reward}"
export PYTHONPATH="$SR_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRL_EXPERIMENTAL_SILENCE=1

mkdir -p "$SR_SCRATCH"/{data,runs,probe}
if [ -n "${SR_VENV:-}" ]; then source "$SR_VENV/bin/activate"; fi
