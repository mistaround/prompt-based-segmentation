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
# Checkpoints are published only on HuggingFace. This session's egress policy
# blocks huggingface.co, so fetch them wherever you run this and drop the .pth
# files into weights/. See ../docs/EXPERIENCE.md.
for f in edge_sam edge_sam_3x; do
  [ -f "weights/$f.pth" ] || echo "MISSING weights/$f.pth -> https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/$f.pth"
done

uv sync
echo "EdgeSAM environment ready."
