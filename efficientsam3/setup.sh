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
# Checkpoints are published only on HuggingFace. This session's egress policy
# blocks huggingface.co, so fetch them wherever you run this and drop the .pt
# files into weights/. See ../docs/EXPERIENCE.md.
for f in efficientsam3_efficientvit efficientsam3_repvit efficientsam3_tinyvit; do
  [ -f "weights/$f.pt" ] || echo "MISSING weights/$f.pt -> https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/efficientsam3_ft/$f.pt"
done

uv sync
echo "EfficientSAM3 environment ready."
