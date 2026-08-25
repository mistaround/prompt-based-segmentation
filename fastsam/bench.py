#!/usr/bin/env python3
"""Benchmark FastSAM on the shared coco128-seg protocol.

FastSAM is not a SAM-style encoder/decoder pair. It is a YOLOv8-seg instance
segmenter run in "everything" mode: one forward pass proposes every mask in the
image, and a "prompt" is then just a cheap selection over those proposals. We
map that onto the shared harness by treating the YOLO forward as `set_image` and
the proposal selection as the decoder, which is the honest comparison -- it is
exactly where FastSAM spends its time and where its accuracy ceiling comes from,
since a prompt can only ever retrieve a mask the detector already proposed.

Usage:  uv run python bench.py [--model weights/FastSAM-x.pt]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import cv2
import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "repo"))
sys.path.insert(0, str(HERE.parent / "common"))

from fastsam import FastSAM, FastSAMPrompt  # noqa: E402
from pbs_bench import load_coco128seg  # noqa: E402
from pbs_bench.runner import benchmark, count_flops, measure_latency, save  # noqa: E402


def _to_mask(raw, shape) -> np.ndarray:
    """Normalize FastSAM's assorted return types to a bool HxW at `shape`."""
    if raw is None or (hasattr(raw, "__len__") and len(raw) == 0):
        return np.zeros(shape, dtype=bool)
    arr = raw.cpu().numpy() if isinstance(raw, torch.Tensor) else np.asarray(raw)
    if arr.ndim == 3:
        arr = arr[0]
    arr = arr.astype(np.uint8)
    if arr.shape != shape:
        arr = cv2.resize(arr, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return arr.astype(bool)


class FastSAMAdapter:
    name = "FastSAM"

    def __init__(self, weights: str, imgsz: int = 1024, conf: float = 0.4,
                 iou: float = 0.9, device: str = "cpu"):
        self.variant = pathlib.Path(weights).stem
        self.device = device
        self.imgsz, self.conf, self.iou = imgsz, conf, iou
        self.model = FastSAM(weights)
        self._prompt = None
        self._shape = None

    def set_image(self, bgr: np.ndarray) -> None:
        self._shape = bgr.shape[:2]
        results = self.model(
            bgr, device=self.device, retina_masks=True,
            imgsz=self.imgsz, conf=self.conf, iou=self.iou, verbose=False,
        )
        self._prompt = FastSAMPrompt(bgr, results, device=self.device)

    def predict_box(self, box: np.ndarray) -> np.ndarray:
        # box_prompt() mutates the list it is handed, so pass a throwaway copy.
        bbox = [float(v) for v in box]
        try:
            return _to_mask(self._prompt.box_prompt(bbox=bbox), self._shape)
        except (AssertionError, IndexError, ValueError):
            # No proposal survived the confidence threshold for this region.
            return np.zeros(self._shape, dtype=bool)

    def predict_point(self, xy: np.ndarray) -> np.ndarray:
        try:
            raw = self._prompt.point_prompt(
                points=[[int(xy[0]), int(xy[1])]], pointlabel=[1]
            )
            return _to_mask(raw, self._shape)
        except (AssertionError, IndexError, ValueError):
            return np.zeros(self._shape, dtype=bool)

    def param_stats(self):
        total = sum(p.numel() for p in self.model.model.parameters()) / 1e6
        # Single-stage detector: there is no separable prompt decoder to report.
        return {"total": round(total, 3), "image_encoder": None,
                "prompt_encoder": None, "mask_decoder": None}

    def encoder_gflops(self):
        net = self.model.model
        x = torch.zeros(1, 3, self.imgsz, self.imgsz)
        return count_flops(net, (x,))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(HERE / "weights" / "FastSAM-x.pt"))
    ap.add_argument("--dataset", default=str(HERE.parent / "datasets" / "coco128-seg"))
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--iou", type=float, default=0.9)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(HERE / "outputs" / "results.json"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)

    adapter = FastSAMAdapter(args.model, imgsz=args.imgsz, conf=args.conf, iou=args.iou)
    samples = load_coco128seg(args.dataset, max_images=args.max_images)
    print(f"{adapter.name} [{adapter.variant}]: {len(samples)} images, "
          f"{sum(len(s.instances) for s in samples)} instances")

    result = benchmark(adapter, samples)
    result["encoder_gflops_1024"] = adapter.encoder_gflops()
    result["checkpoint_mb"] = round(pathlib.Path(args.model).stat().st_size / 1e6, 1)
    result["config"] = {"imgsz": args.imgsz, "conf": args.conf, "iou": args.iou}
    dummy = np.zeros((1024, 1024, 3), dtype=np.uint8)
    result["latency"]["encoder_only_1024"] = measure_latency(
        lambda: adapter.set_image(dummy), warmup=2, repeat=10
    )
    save(result, args.out)
    acc, lat = result["accuracy"], result["latency"]
    print(f"box mIoU={acc['box_mIoU']:.4f}  boundary={acc['box_boundary_mIoU']:.4f}  "
          f"point mIoU={acc.get('point_mIoU', float('nan')):.4f}")
    print(f"forward={lat['encode_median_ms']:.1f} ms  select={lat['decode_median_ms']:.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
