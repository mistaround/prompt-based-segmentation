#!/usr/bin/env bash
# Fetch the pinned upstream EdgeSAM source and apply the inference-only patches.
set -euo pipefail
cd "$(dirname "$0")"
REPO_URL=https://github.com/chongzhou96/EdgeSAM.git
REPO_SHA=d24d99671f41a9c0003061248bded64a481e9059

if [ ! -d repo/.git ]; then
  git clone "$REPO_URL" repo
fi
git -C repo fetch --depth 1 origin "$REPO_SHA" 2>/dev/null || true
git -C repo checkout -q "$REPO_SHA"

# Makes the mmdet/mmengine imports lazy so inference needs no mmcv build.
python3 patches/apply_patches.py

mkdir -p weights
# Checkpoints are published only on HuggingFace (no GitHub release, no PyPI
# package, no mirror). If your network blocks huggingface.co, fetch them
# elsewhere and drop the .pth files into weights/ -- bench.py detects them.
HF=https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights
for f in edge_sam edge_sam_3x; do
  [ -f "weights/$f.pth" ] && continue
  curl -fL --retry 3 -o "weights/$f.pth" "$HF/$f.pth" \
    || { rm -f "weights/$f.pth"; echo "WARN: could not fetch $f.pth from $HF"; }
done

uv sync
echo "EdgeSAM environment ready."
