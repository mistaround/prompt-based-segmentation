#!/usr/bin/env bash
# Run every model's benchmark against the shared coco128-seg protocol.
#
# Sequential on purpose: the box has 4 cores and each run pins torch to 4
# threads, so running them concurrently would contaminate every latency number.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)

run() {  # run <dir> <label> <args...>
  local dir=$1 label=$2; shift 2
  echo "=============================================================="
  echo "== $label"
  echo "=============================================================="
  ( cd "$ROOT/$dir" && uv run python -W ignore bench.py "$@" ) 2>&1 | grep -v "^No object detected.$"
  echo
}

run fastsam   "FastSAM-x"                  --model weights/FastSAM-x.pt --out outputs/results_x.json
run fastsam   "FastSAM-s"                  --model weights/FastSAM-s.pt --out outputs/results_s.json
run mobilesam "MobileSAM (TinyViT)"
run edgesam   "EdgeSAM (RepViT-M1)"
# EfficientSAM3 has no checkpoint here, so accuracy is not measurable; 32 images
# is plenty for a stable median latency and keeps the run to a few minutes.
run efficientsam3 "EfficientSAM3 TV-M"     --backbone tinyvit --max-images 32
echo "All runs complete."
