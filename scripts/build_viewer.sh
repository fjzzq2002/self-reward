#!/bin/bash
# Build the checkpoint viewer: the self-reward runs (default view) and, behind its toggle, the random-reward controls.
# Needs the inference outputs plus, for every run make_maps.scored() selects: Claude's ratings (scripts/judge.py),
# embeddings (scripts/embed_answers.py), the untrained model's ratings (scripts/ref_score.py) and interestingness
# (scripts/interestingness.py score --steps all). LLM steps are cached, so a rebuild pays only for what changed.
#
#   [SK=<dir with scikit-learn + scipy>] OPENROUTER_API_KEY=... scripts/build_viewer.sh <out prefix>
#   e.g. scripts/build_viewer.sh /tmp/self-reward/infer/viewer39  ->  viewer39 (default), viewer39ab (maps and zoo with
#   the controls), viewer39abx (the controls' answers), viewer39pub (what gets published: gzipped)
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh > /dev/null
P=${1:?out prefix}
KEY=${OPENROUTER_API_KEY:?}
SKP="${SK:+$SK:}$PYTHONPATH"  # SK: optional extra site-packages with scikit-learn + scipy
LUNA="--model openai/gpt-6-luna --reasoning medium --maps all --budget 1"

build() {  # $1 = folder, $2 = label-cache tag, $3 = zoo min group (fraction of models); SR_VIEWER_ABLATIONS picks the set
  # make_maps without a key: its k-means names are replaced by cluster_maps + label_clusters
  env -u OPENROUTER_API_KEY python scripts/make_maps.py --out "$1"
  python scripts/make_infer_viewer.py --tag v1,v2 --out "$1"  # cluster_maps reads its index.json and p/
  PYTHONPATH=$SKP python scripts/cluster_maps.py --viewer "$1"
  python scripts/label_clusters.py $LUNA --tag "$2"
  python scripts/apply_labels.py --tag "$2" --viewer "$1"
  PYTHONPATH=$SKP python scripts/make_zoo.py --out "$1" --budget 3 --frac "$3"
}
build "$P" gpt6-luna-think 0.0101  # 696 models (seeds 1-30): groups of 7+
cp ${SR_SCRATCH:-/tmp/self-reward}/infer/viewer/data.json "$P/"  # the Data tab (scripts/make_dataset.py)
SR_VIEWER_ABLATIONS=1 build "${P}ab" gpt6-luna-think-ab 0.007  # 1002 models: also 7+ (1/80 left half ungrouped)
SR_VIEWER_ABLATIONS=1 python scripts/make_infer_viewer.py --tag v1,v2 --only_ablations --out "${P}abx"
python scripts/assemble_viewer.py --main "$P" --ab "${P}ab" --abx "${P}abx" --out "${P}pub"
