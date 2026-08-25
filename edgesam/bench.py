#!/usr/bin/env python3
"""Benchmark EdgeSAM on the shared coco128-seg protocol.

The released checkpoints (edge_sam.pth / edge_sam_3x.pth) are published only on
HuggingFace. If they are absent this still reports every weight-independent
metric -- parameter counts, encoder GFLOPs, encoder/decoder latency, peak RSS --
which depend on the architecture and not on the values in it, and marks the run
`weights_available: false` so the accuracy columns are never mistaken for real
measurements.

Usage:  uv run python bench.py [--checkpoint weights/edge_sam_3x.pth]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "repo"))
sys.path.insert(0, str(HERE.parent / "common"))

from edge_sam import SamPredictor, sam_model_registry  # noqa: E402
from pbs_bench import load_coco128seg  # noqa: E402
from pbs_bench.runner import benchmark, count_flops, measure_latency, save  # noqa: E402


class EdgeSAMAdapter:
    name = "EdgeSAM"

    def __init__(self, checkpoint: str | None, device: str = "cpu"):
        self.has_weights = checkpoint is not None
        self.variant = (
            f"RepViT-M1 ({pathlib.Path(checkpoint).name})"
            if checkpoint
            else "RepViT-M1 (RANDOM INIT -- no checkpoint)"
        )
        self.device = device
        self.model = sam_model_registry["edge_sam"](checkpoint=checkpoint)
        self.model.to(device).eval()
        self.predictor = SamPredictor(self.model)

    def set_image(self, bgr: np.ndarray) -> None:
        self.predictor.set_image(bgr[:, :, ::-1])

    def predict_box(self, box: np.ndarray) -> np.ndarray:
        # EdgeSAM can emit 1, 3 or 4 candidates; 1 is the box-prompt setting,
        # matching MobileSAM's multimask_output=False.
        masks, _, _ = self.predictor.predict(
            box=np.asarray(box, dtype=np.float32), num_multimask_outputs=1
        )
        return masks[0]

    def predict_point(self, xy: np.ndarray) -> np.ndarray:
        masks, scores, _ = self.predictor.predict(
            point_coords=np.asarray([xy], dtype=np.float32),
            point_labels=np.array([1]),
            num_multimask_outputs=3,
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
        return count_flops(self.model.image_encoder, (torch.zeros(1, 3, 1024, 1024),))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="path to edge_sam.pth / edge_sam_3x.pth")
    ap.add_argument("--dataset", default=str(HERE.parent / "datasets" / "coco128-seg"))
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(HERE / "outputs" / "results.json"))
    args = ap.parse_args()

    ckpt = args.checkpoint
    if ckpt is None:
        for cand in ("edge_sam_3x.pth", "edge_sam.pth"):
            p = HERE / "weights" / cand
            if p.is_file():
                ckpt = str(p)
                break

    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)

    adapter = EdgeSAMAdapter(ckpt)
    if not adapter.has_weights:
        print("WARNING: no checkpoint found in weights/ -- reporting "
              "weight-independent metrics only. Accuracy will NOT be measured.")

    samples = load_coco128seg(args.dataset, max_images=args.max_images)

    if adapter.has_weights:
        result = benchmark(adapter, samples)
    else:
        # Still walk the real images so encoder/decoder latency reflects the same
        # workload the other models were timed on.
        result = benchmark(adapter, samples, do_point=False, do_boundary=False)
        result["accuracy"] = {
            "_note": "not measured: random-initialized weights produce meaningless masks"
        }

    result["weights_available"] = adapter.has_weights
    result["encoder_gflops_1024"] = adapter.encoder_gflops()
    result["checkpoint_mb"] = (
        round(pathlib.Path(ckpt).stat().st_size / 1e6, 1) if ckpt else None
    )
    dummy = np.zeros((1024, 1024, 3), dtype=np.uint8)
    result["latency"]["encoder_only_1024"] = measure_latency(
        lambda: adapter.set_image(dummy), warmup=2, repeat=10
    )
    save(result, args.out)
    lat = result["latency"]
    print(f"encode={lat['encode_median_ms']:.1f} ms  decode={lat['decode_median_ms']:.2f} ms  "
          f"params={result['params_M']['total']} M")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
