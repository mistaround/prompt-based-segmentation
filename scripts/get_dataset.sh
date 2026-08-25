#!/usr/bin/env bash
# Fetch the shared evaluation set: coco128-seg (128 COCO train2017 images with
# instance polygon labels, YOLO-seg format). ~7.6 MB.
#
# Why this set: the session's egress policy only allows github.com + PyPI, so
# cocodataset.org / HuggingFace / davischallenge.org are unreachable. coco128-seg
# is a real COCO subset published as a GitHub release asset, which makes it the
# largest properly-labelled instance-segmentation set we can actually fetch here.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p datasets
if [ -d datasets/coco128-seg ]; then
  echo "datasets/coco128-seg already present."
  exit 0
fi
curl -fL --retry 3 -o datasets/coco128-seg.zip \
  https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128-seg.zip
python3 -c "import zipfile;zipfile.ZipFile('datasets/coco128-seg.zip').extractall('datasets')"
rm datasets/coco128-seg.zip
echo "Fetched $(ls datasets/coco128-seg/images/train2017 | wc -l) images."
