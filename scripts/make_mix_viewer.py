"""Write the prompt-mix viewer page: scripts/mix_viewer_template.html with $SR_SCRATCH/data/mix/x2.json inlined.

    python scripts/make_mix_viewer.py OUT.html
"""

import json
import os
import sys
from pathlib import Path

SCRATCH = os.environ.get("SR_SCRATCH", "/tmp/self-reward")  # data, runs and outputs root

here = Path(__file__).parent
data = json.loads(Path(SCRATCH, "data/mix/x2.json").read_text())
blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")  # keep the JSON from closing its <script>
page = (here / "mix_viewer_template.html").read_text().replace("__DATA__", blob)
Path(sys.argv[1]).write_text(page)
print(f"{sys.argv[1]}: {len(data)} prompts, {len(page) / 1e6:.2f} MB")
