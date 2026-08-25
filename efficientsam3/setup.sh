#!/usr/bin/env bash
# Fetch the pinned upstream EfficientSAM3 source and the Stage-3 checkpoints.
set -euo pipefail
cd "$(dirname "$0")"
REPO_URL=https://github.com/SimonZeng7108/efficientsam3.git
REPO_SHA=bd0936c788fed8d51fa799437f05abd97b401b06

if [ ! -d repo/.git ]; then
  git clone "$REPO_URL" repo
fi
git -C repo fetch --depth 1 origin "$REPO_SHA" 2>/dev/null || true
git -C repo checkout -q "$REPO_SHA"

mkdir -p weights
# Stage-3 checkpoints are published only on HuggingFace. Each is ~470 MB, so
# only the default (tinyvit / TV-M) is fetched here; pass another name to grab
# the others. If your network blocks huggingface.co, fetch them elsewhere and
# drop the .pt files into weights/ -- bench.py detects them.
HF=https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/efficientsam3_ft
for f in "${@:-efficientsam3_tinyvit}"; do
  [ -f "weights/$f.pt" ] && continue
  curl -fL --retry 3 -o "weights/$f.pt" "$HF/$f.pt" \
    || { rm -f "weights/$f.pt"; echo "WARN: could not fetch $f.pt from $HF"; }
done

uv sync
echo "EfficientSAM3 environment ready."
