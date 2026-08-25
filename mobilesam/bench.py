#!/usr/bin/env python3
"""Benchmark MobileSAM on the shared coco128-seg protocol.

Usage:  uv run python bench.py [--max-images N] [--checkpoint PATH]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "repo"))          # upstream mobile_sam package
sys.path.insert(0, str(HERE.parent / "common"))  # shared harness

from mobile_sam import SamPredictor, sam_model_registry  # noqa: E402
from pbs_bench import load_coco128seg  # noqa: E402
from pbs_bench.runner import benchmark, count_flops, measure_latency, save  # noqa: E402


class MobileSAMAdapter:
    name = "MobileSAM"

    def __init__(self, checkpoint: str, device: str = "cpu"):
        self.variant = f"vit_t ({pathlib.Path(checkpoint).name})"
        self.device = device
        self.model = sam_model_registry["vit_t"](checkpoint=checkpoint)
        self.model.to(device).eval()
        self.predictor = SamPredictor(self.model)

    def set_image(self, bgr: np.ndarray) -> None:
        self.predictor.set_image(bgr[:, :, ::-1])  # SamPredictor expects RGB

    def predict_box(self, box: np.ndarray) -> np.ndarray:
        masks, _, _ = self.predictor.predict(
            box=np.asarray(box, dtype=np.float32), multimask_output=False
        )
        return masks[0]

    def predict_point(self, xy: np.ndarray) -> np.ndarray:
        # A single point is genuinely ambiguous (part / subpart / whole), so SAM
        # is designed to emit three candidates here. Taking the model's own
        # highest-scoring one is the standard single-point protocol.
        masks, scores, _ = self.predictor.predict(
            point_coords=np.asarray([xy], dtype=np.float32),
            point_labels=np.array([1]),
            multimask_output=True,
        )
        return masks[int(np.argmax(scores))]

    def param_stats(self):
        enc = sum(p.numel() for p in self.model.image_encoder.parameters()) / 1e6
        dec = sum(p.numel() for p in self.model.mask_decoder.parameters()) / 1e6
        pe = sum(p.numel() for p in self.model.prompt_encoder.parameters()) / 1e6
        return {
            "total": round(enc + dec + pe, 3),
            "image_encoder": round(enc, 3),
            "prompt_encoder": round(pe, 3),
            "mask_decoder": round(dec, 3),
        }

    def encoder_gflops(self):
        x = torch.zeros(1, 3, 1024, 1024)
        return count_flops(self.model.image_encoder, (x,))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(HERE / "weights" / "mobile_sam.pt"))
    ap.add_argument("--dataset", default=str(HERE.parent / "datasets" / "coco128-seg"))
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(HERE / "outputs" / "results.json"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)

    adapter = MobileSAMAdapter(args.checkpoint)
    samples = load_coco128seg(args.dataset, max_images=args.max_images)
    print(f"{adapter.name}: {len(samples)} images, "
          f"{sum(len(s.instances) for s in samples)} instances")

    result = benchmark(adapter, samples)
    result["encoder_gflops_1024"] = adapter.encoder_gflops()
    result["checkpoint_mb"] = round(pathlib.Path(args.checkpoint).stat().st_size / 1e6, 1)

    # Isolated encoder latency on a fixed 1024x1024 input, free of the JPEG
    # decode and resize that the accuracy loop also pays for.
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
