#!/usr/bin/env bash
# Fetch the pinned upstream TinySAM source and its checkpoints.
set -euo pipefail
cd "$(dirname "$0")"
REPO_URL=https://github.com/xinghaochen/TinySAM.git
REPO_SHA=11589bc1d98c16cff046c31d5ad4cd90a30f0897

if [ ! -d repo/.git ]; then
  git clone "$REPO_URL" repo
fi
git -C repo fetch --depth 1 origin "$REPO_SHA" 2>/dev/null || true
git -C repo checkout -q "$REPO_SHA"

# Source-level fixes (idempotent).
python3 patches/apply_patches.py

mkdir -p weights
[ -f weights/tinysam_42.3.pth ] || curl -fL --retry 3 -o weights/tinysam_42.3.pth \
  https://github.com/xinghaochen/TinySAM/releases/download/3.0/tinysam_42.3.pth

uv sync
echo "TinySAM ready."
