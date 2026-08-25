#!/usr/bin/env bash
# Fetch the pinned upstream FastSAM source and the model checkpoints.
set -euo pipefail
cd "$(dirname "$0")"
REPO_URL=https://github.com/CASIA-IVA-Lab/FastSAM.git
REPO_SHA=b4ed20c2fed75eadc5aa7d8b09fedd137b873b52

if [ ! -d repo/.git ]; then
  git clone "$REPO_URL" repo
fi
git -C repo fetch --depth 1 origin "$REPO_SHA" 2>/dev/null || true
git -C repo checkout -q "$REPO_SHA"

mkdir -p weights
# Upstream README links Google Drive; the ultralytics asset mirror is the same
# YOLOv8x/YOLOv8s-seg FastSAM checkpoints and is scriptable.
[ -f weights/FastSAM-x.pt ] || curl -fL --retry 3 -o weights/FastSAM-x.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/FastSAM-x.pt
[ -f weights/FastSAM-s.pt ] || curl -fL --retry 3 -o weights/FastSAM-s.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/FastSAM-s.pt

uv sync
echo "FastSAM ready."
