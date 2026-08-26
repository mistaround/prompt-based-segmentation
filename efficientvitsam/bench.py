#!/usr/bin/env python3
"""Benchmark EfficientViT-SAM on the shared coco128-seg protocol.

Unlike the distilled students here, EfficientViT-SAM is trained on the full
SA-1B rather than distilled from SAM on a slice of it, so it sits at a different
point on the size/accuracy curve (L0 is 34.8M against MobileSAM's 10M).

Usage:  uv run python bench.py [--model efficientvit-sam-l0] [--max-images N]
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

from pbs_bench import load_coco128seg  # noqa: E402
from pbs_bench.runner import benchmark, count_flops, measure_latency, save  # noqa: E402
from pbs_bench.sam_adapter import SamStyleAdapter, checkpoint_mb  # noqa: E402


class EfficientViTSamAdapter(SamStyleAdapter):
    """EfficientViT-SAM keeps SAM's predictor API but names its encoder
    `image_encoder` inside a differently-shaped module tree, and L0/L1/L2 run at
    512x512 rather than 1024x1024."""

    def param_stats(self):
        def n(mod):
            return round(sum(p.numel() for p in mod.parameters()) / 1e6, 3)

        return {
            "total": round(sum(p.numel() for p in self.model.parameters()) / 1e6, 3),
            "image_encoder": n(self.model.image_encoder),
            "prompt_encoder": n(self.model.prompt_encoder),
            "mask_decoder": n(self.model.mask_decoder),
        }

    def encoder_gflops(self):
        s = self.encoder_input_size
        return count_flops(self.model.image_encoder, (torch.zeros(1, 3, s, s),))


def build(name: str, checkpoint: str) -> EfficientViTSamAdapter:
    from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
    from efficientvit.sam_model_zoo import create_efficientvit_sam_model

    model = create_efficientvit_sam_model(name=name, pretrained=True, weight_url=checkpoint)
    model.eval()
    # model.image_size is (train_res, eval_res); the second is what inference uses.
    res = int(model.image_size[-1])
    return EfficientViTSamAdapter(
        name="EfficientViT-SAM",
        variant=f"{name.replace('efficientvit-sam-', '').upper()} ({pathlib.Path(checkpoint).name}, {res}px)",
        model=model,
        predictor=EfficientViTSamPredictor(model),
        encoder_input_size=res,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientvit-sam-l0")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--dataset", default=str(HERE.parent / "datasets" / "coco128-seg"))
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tag = args.model.replace("efficientvit-sam-", "")
    ckpt = args.checkpoint or str(HERE / "weights" / f"efficientvit_sam_{tag}.pt")
    out = args.out or str(HERE / "outputs" / f"results_{tag}.json")

    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)

    adapter = build(args.model, ckpt)
    samples = load_coco128seg(args.dataset, max_images=args.max_images)
    print(f"{adapter.name} [{adapter.variant}]: {len(samples)} images, "
          f"{sum(len(s.instances) for s in samples)} instances")

    result = benchmark(adapter, samples)
    result["weights_available"] = True
    res = adapter.encoder_input_size
    result[f"encoder_gflops_{res}"] = adapter.encoder_gflops()
    result["checkpoint_mb"] = checkpoint_mb(ckpt)
    dummy = np.zeros((res, res, 3), dtype=np.uint8)
    result["latency"][f"encoder_only_{res}"] = measure_latency(
        lambda: adapter.set_image(dummy), warmup=2, repeat=10
    )
    save(result, out)
    acc, lat = result["accuracy"], result["latency"]
    print(f"box mIoU={acc['box_mIoU']:.4f}  boundary={acc['box_boundary_mIoU']:.4f}  "
          f"point mIoU={acc.get('point_mIoU', float('nan')):.4f}")
    print(f"encode={lat['encode_median_ms']:.1f} ms  decode={lat['decode_median_ms']:.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
