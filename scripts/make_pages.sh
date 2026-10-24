#!/bin/bash
# GitHub Pages copy of the viewer: the assembled build (scripts/assemble_viewer.py --out <pub>) as a static site. The
# artifact host wraps the page in a document; Pages serves it as is, so it gets a doctype here (no quirks mode), plus
# .nojekyll (serve every file untouched). Data stays gzip+base64 text, as the viewer reads it.
#   scripts/make_pages.sh <pub dir> <pages dir>    then commit <pages dir> as an orphan gh-pages branch
set -euo pipefail
PUB=${1:?pub dir}; OUT=${2:?pages dir}
rm -rf "$OUT"; mkdir -p "$OUT"
cp -r "$PUB"/. "$OUT"/
{ echo '<!doctype html>'; echo '<html lang="en">'; grep -v -i '^<!doctype' "$PUB/index.html"; echo '</html>'; } > "$OUT/index.html"
touch "$OUT/.nojekyll"
echo "$(find "$OUT" -type f | wc -l) files, $(du -sh "$OUT" | cut -f1), largest $(find "$OUT" -type f -printf '%s\n' | sort -n | tail -1) bytes -> $OUT"
