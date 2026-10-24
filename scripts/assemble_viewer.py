"""Assemble the published checkpoint viewer from its two builds, with every data file gzipped (the artifact holds at
most 256 MB; answers compress about 5x) and base64-encoded as text (artifacts serve text, not gzip files).

  --main   E13 seeds 1-SEEDS (make_maps) and the untrained model: make_maps, cluster_maps, label_clusters, apply_labels, make_zoo and
           make_infer_viewer all writing into one folder (the default view)
  --ab     the same maps and zoo built with SR_VIEWER_ABLATIONS=1 (self-reward runs plus the random-reward controls; the
           viewer's toggle switches to them): maps.json, mp/, zoo.json
  --abx    make_infer_viewer --only_ablations: the controls' answers, merged into the main ones when the toggle is on

Writes <out>/index.html and, gzipped: index.json (every run; controls carry "ab": 1), maps.json, zoo.json, data.json,
mp/<k>, p/<k>, r/<run>, f/<run> from --main; maps_ab.json, zoo_ab.json, mpa/<k> from --ab; pa/<k> (controls only) and
the controls' r/ and f/ from --abx. Each file is <name>.json.gz.txt (base64 of the gzipped JSON); the page decodes
and inflates it (DecompressionStream).

    python scripts/assemble_viewer.py --main <dir> --ab <dir> --abx <dir> --out <dir>
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import shutil
from pathlib import Path


def put(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = src.read_bytes()
    dst.with_name(dst.name + ".gz.txt").write_bytes(base64.b64encode(gzip.compress(data, compresslevel=9, mtime=0)))
    return len(data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    ap.add_argument("--ab", required=True)
    ap.add_argument("--abx", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    main_, ab, abx, out = Path(a.main), Path(a.ab), Path(a.abx), Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    idx, idx_ab = json.loads((main_ / "index.json").read_text()), json.loads((abx / "index.json").read_text())
    assert idx["prompts"] == idx_ab["prompts"], "the two exports number the prompts differently"
    for r in idx_ab["runs"]:
        r["ab"] = 1
    idx["runs"] += idx_ab["runs"]
    idx["groups"] += [g for g in idx_ab["groups"] if g not in idx["groups"]]
    idx["checkpoints_ab"] = idx_ab["checkpoints"]
    (out / "index.json").write_text(json.dumps(idx, ensure_ascii=False))
    raw = put(out / "index.json", out / "index.json")
    (out / "index.json").unlink()

    copies = [(main_ / n, out / n) for n in ("maps.json", "zoo.json", "data.json")]
    copies += [(ab / "maps.json", out / "maps_ab.json"), (ab / "zoo.json", out / "zoo_ab.json")]
    for sub, src, dst in (("mp", main_, "mp"), ("p", main_, "p"), ("r", main_, "r"), ("f", main_, "f"), ("mp", ab, "mpa"),
                          ("p", abx, "pa"), ("r", abx, "r"), ("f", abx, "f")):  # fmt: skip
        copies += [(f, out / dst / f.name) for f in sorted((src / sub).glob("*.json"))]
    for src, dst in copies:
        raw += put(src, dst)
    shutil.copy(Path(__file__).parent / "infer_viewer.html", out / "index.html")
    files = [f for f in out.rglob("*") if f.is_file()]
    size = sum(f.stat().st_size for f in files)
    print(f"{len(files)} files, {size / 1e6:.1f} MB ({raw / 1e6:.1f} MB before gzip) -> {out}; "
          f"{sum(1 for r in idx['runs'] if not r.get('ab'))} runs + {len(idx_ab['runs'])} controls")


if __name__ == "__main__":
    main()
