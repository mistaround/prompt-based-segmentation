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
# EfficientSAM3 runs the full set like the others. Its decoder is ~40x slower
# per prompt than the SAM-style models, so this stage dominates the wall clock.
run efficientsam3 "EfficientSAM3 TV-M"     --backbone tinyvit
run efficientsam    "EfficientSAM-Ti"       --variant vitt
run efficientsam    "EfficientSAM-S"        --variant vits
run efficientvitsam "EfficientViT-SAM-L0"   --model efficientvit-sam-l0
run repvitsam       "RepViT-SAM"
run tinysam         "TinySAM"
echo "All runs complete."
