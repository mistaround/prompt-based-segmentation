#!/usr/bin/env python3
"""Benchmark RepViT-SAM on the shared coco128-seg protocol.

Usage:  uv run python bench.py [--checkpoint PATH] [--max-images N]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "repo/sam"))
sys.path.insert(0, str(HERE.parent / "common"))

from pbs_bench import load_coco128seg  # noqa: E402
from pbs_bench.runner import benchmark, measure_latency, save  # noqa: E402
from pbs_bench.sam_adapter import SamStyleAdapter, checkpoint_mb  # noqa: E402


def build(checkpoint: str) -> SamStyleAdapter:
    from repvit_sam import SamPredictor, sam_model_registry

    model = sam_model_registry["repvit"](checkpoint=checkpoint)
    return SamStyleAdapter(
        name="RepViT-SAM",
        variant=f"RepViT-M2.3 ({pathlib.Path(checkpoint).name})",
        model=model,
        predictor=SamPredictor(model),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(HERE / "weights" / "repvit_sam.pt"))
    ap.add_argument("--dataset", default=str(HERE.parent / "datasets" / "coco128-seg"))
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(HERE / "outputs" / "results.json"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)

    adapter = build(args.checkpoint)
    samples = load_coco128seg(args.dataset, max_images=args.max_images)
    print(f"{adapter.name} [{adapter.variant}]: {len(samples)} images, "
          f"{sum(len(s.instances) for s in samples)} instances")

    result = benchmark(adapter, samples)
    result["weights_available"] = True
    result["encoder_gflops_1024"] = adapter.encoder_gflops()
    result["checkpoint_mb"] = checkpoint_mb(args.checkpoint)
    dummy = np.zeros((1024, 1024, 3), dtype=np.uint8)
    result["latency"]["encoder_only_1024"] = measure_latency(
        lambda: adapter.set_image(dummy), warmup=2, repeat=10
    )
    save(result, args.out)
    acc, lat = result["accuracy"], result["latency"]
    print(f"box mIoU={acc['box_mIoU']:.4f}  boundary={acc['box_boundary_mIoU']:.4f}  "
          f"point mIoU={acc.get('point_mIoU', float('nan')):.4f}")
    print(f"encode={lat['encode_median_ms']:.1f} ms  decode={lat['decode_median_ms']:.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
