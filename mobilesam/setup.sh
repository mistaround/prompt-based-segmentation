#!/usr/bin/env bash
# Fetch pinned upstream MobileSAM. Checkpoint ships inside the repo (weights/mobile_sam.pt, 39 MB).
set -euo pipefail
cd "$(dirname "$0")"
REPO_URL=https://github.com/ChaoningZhang/MobileSAM.git
REPO_SHA=f706ad9c4eb7f219c00d9050e46328518ffb65d2

if [ ! -d repo/.git ]; then
  git clone "$REPO_URL" repo
fi
git -C repo fetch --depth 1 origin "$REPO_SHA" 2>/dev/null || true
git -C repo checkout -q "$REPO_SHA"

mkdir -p weights
cp -n repo/weights/mobile_sam.pt weights/mobile_sam.pt

uv sync
echo "MobileSAM ready."
