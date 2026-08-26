#!/usr/bin/env bash
# Fetch the pinned upstream RepViT-SAM source and its checkpoints.
set -euo pipefail
cd "$(dirname "$0")"
REPO_URL=https://github.com/THU-MIG/RepViT.git
REPO_SHA=298f42075eda5d2e6102559fad260c970769d34e

if [ ! -d repo/.git ]; then
  git clone "$REPO_URL" repo
fi
git -C repo fetch --depth 1 origin "$REPO_SHA" 2>/dev/null || true
git -C repo checkout -q "$REPO_SHA"

mkdir -p weights
[ -f weights/repvit_sam.pt ] || curl -fL --retry 3 -o weights/repvit_sam.pt \
  https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_sam.pt

uv sync
echo "RepViT-SAM ready."
