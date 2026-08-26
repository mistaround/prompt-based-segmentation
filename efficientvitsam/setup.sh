#!/usr/bin/env bash
# Fetch the pinned upstream EfficientViT-SAM source and its checkpoints.
set -euo pipefail
cd "$(dirname "$0")"
REPO_URL=https://github.com/mit-han-lab/efficientvit.git
REPO_SHA=de7d7733cc0329f391b33f1f459271562ec27bd5

if [ ! -d repo/.git ]; then
  git clone "$REPO_URL" repo
fi
git -C repo fetch --depth 1 origin "$REPO_SHA" 2>/dev/null || true
git -C repo checkout -q "$REPO_SHA"

mkdir -p weights
# Checkpoints live on HuggingFace. L0 is the smallest (34.8M, 512px); pass another
# tag (l1/l2/xl0/xl1) to fetch a different one.
HF=https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main
for t in "${@:-l0}"; do
  [ -f "weights/efficientvit_sam_$t.pt" ] && continue
  curl -fL --retry 3 -o "weights/efficientvit_sam_$t.pt" "$HF/efficientvit_sam_$t.pt" \
    || { rm -f "weights/efficientvit_sam_$t.pt"; echo "WARN: could not fetch efficientvit_sam_$t.pt"; }
done

uv sync
echo "EfficientViT-SAM ready."
