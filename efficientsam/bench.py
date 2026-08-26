#!/usr/bin/env python3
"""Benchmark EfficientSAM on the shared coco128-seg protocol.

EfficientSAM does not expose a SamPredictor. It is a single functional module:
`model(image, points, labels) -> (logits, iou)`, with no separate "set the image
once" step in its public API. To keep the encoder/decoder split that the rest of
this benchmark reports, we call the encoder ourselves in `set_image` and the
prompt decoder in `predict_*`, which is exactly how the module is structured
internally.

Its prompt encoding is also unusual: a *box* is two corner points carrying
labels 2 (top-left) and 3 (bottom-right), rather than a dedicated box input.

Usage:  uv run python bench.py [--variant vitt|vits] [--max-images N]
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

from efficient_sam.efficient_sam import build_efficient_sam  # noqa: E402
from pbs_bench import load_coco128seg  # noqa: E402
from pbs_bench.runner import benchmark, count_flops, measure_latency, save  # noqa: E402
from pbs_bench.sam_adapter import checkpoint_mb  # noqa: E402

# The two released variants; the upstream build_efficient_sam_vitt/vits wrappers
# hardcode a relative "weights/..." path with no override, so we call the
# underlying builder with an explicit checkpoint instead.
VARIANTS = {
    "vitt": dict(encoder_patch_embed_dim=192, encoder_num_heads=3, tag="Ti"),
    "vits": dict(encoder_patch_embed_dim=384, encoder_num_heads=6, tag="S"),
}


class EfficientSAMAdapter:
    name = "EfficientSAM"

    def __init__(self, variant: str, checkpoint: str, device: str = "cpu"):
        cfg = dict(VARIANTS[variant])
        tag = cfg.pop("tag")
        self.variant = f"{tag} ({pathlib.Path(checkpoint).name})"
        self.device = device
        self.model = build_efficient_sam(checkpoint=checkpoint, **cfg).eval().to(device)
        self._feats = None
        self._shape = None

    def set_image(self, bgr: np.ndarray) -> None:
        self._shape = bgr.shape[:2]
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)[None].to(self.device)
        self._img = x
        self._feats = self.model.get_image_embeddings(x)

    def _run(self, pts: np.ndarray, labels: np.ndarray) -> np.ndarray:
        p = torch.as_tensor(pts, dtype=torch.float32, device=self.device)[None, None]
        l = torch.as_tensor(labels, dtype=torch.float32, device=self.device)[None, None]
        logits, iou = self.model.predict_masks(
            self._feats, p, l,
            multimask_output=True,
            input_h=self._img.shape[2], input_w=self._img.shape[3],
            output_h=self._shape[0], output_w=self._shape[1],
        )
        # Candidates are not returned in score order; take the model's best.
        best = int(torch.argmax(iou[0, 0]))
        return (logits[0, 0, best] >= 0).cpu().numpy().astype(bool)

    def predict_box(self, box: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = [float(v) for v in box]
        # Labels 2 and 3 mark the box's top-left and bottom-right corners.
        return self._run(np.array([[x0, y0], [x1, y1]]), np.array([2, 3]))

    def predict_point(self, xy: np.ndarray) -> np.ndarray:
        return self._run(np.asarray([xy], dtype=np.float32), np.array([1]))

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
        return count_flops(self.model.image_encoder, (torch.zeros(1, 3, 1024, 1024),))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="vitt", choices=sorted(VARIANTS))
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--dataset", default=str(HERE.parent / "datasets" / "coco128-seg"))
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ckpt = args.checkpoint or str(HERE / "weights" / f"efficient_sam_{args.variant}.pt")
    out = args.out or str(HERE / "outputs" / f"results_{args.variant}.json")

    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)

    adapter = EfficientSAMAdapter(args.variant, ckpt)
    samples = load_coco128seg(args.dataset, max_images=args.max_images)
    print(f"{adapter.name} [{adapter.variant}]: {len(samples)} images, "
          f"{sum(len(s.instances) for s in samples)} instances")

    result = benchmark(adapter, samples)
    result["weights_available"] = True
    result["encoder_gflops_1024"] = adapter.encoder_gflops()
    result["checkpoint_mb"] = checkpoint_mb(ckpt)
    dummy = np.zeros((1024, 1024, 3), dtype=np.uint8)
    result["latency"]["encoder_only_1024"] = measure_latency(
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
