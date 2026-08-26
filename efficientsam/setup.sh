#!/usr/bin/env bash
# Fetch the pinned upstream EfficientSAM source and its checkpoints.
set -euo pipefail
cd "$(dirname "$0")"
REPO_URL=https://github.com/yformer/EfficientSAM.git
REPO_SHA=d525f622e6f640acf5a0fc37c7ca1f243da5bde0

if [ ! -d repo/.git ]; then
  git clone "$REPO_URL" repo
fi
git -C repo fetch --depth 1 origin "$REPO_SHA" 2>/dev/null || true
git -C repo checkout -q "$REPO_SHA"

mkdir -p weights
# Both checkpoints ship inside the upstream repo. The -S one is committed as a
# zip because it exceeds GitHub's 100 MB file limit.
cp -n repo/weights/efficient_sam_vitt.pt weights/ 2>/dev/null || true
[ -f weights/efficient_sam_vits.pt ] || python3 -c "import zipfile;zipfile.ZipFile('repo/weights/efficient_sam_vits.pt.zip').extractall('weights')"

uv sync
echo "EfficientSAM ready."
