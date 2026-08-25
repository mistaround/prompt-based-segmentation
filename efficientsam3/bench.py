#!/usr/bin/env python3
"""Benchmark EfficientSAM3 on the shared coco128-seg protocol.

EfficientSAM3 is a promptable *concept* segmenter distilled from SAM3: it takes
text prompts as well as geometric ones, and a single prompt can return many
instances. The Stage-3 checkpoints are published only on HuggingFace; without
them this reports the weight-independent metrics (parameter counts, encoder
GFLOPs, encoder/decoder latency, peak RSS) and marks the run
`weights_available: false`.

Usage:  uv run python bench.py [--checkpoint weights/efficientsam3_tinyvit.pt]
                              [--backbone tinyvit|repvit|efficientvit]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "repo"))
sys.path.insert(0, str(HERE.parent / "common"))

from pbs_bench import load_coco128seg  # noqa: E402
from pbs_bench.runner import benchmark, count_flops, measure_latency, save  # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402
from sam3.model_builder import build_efficientsam3_image_model  # noqa: E402

# The three released Stage-3 variants and the (backbone, model_name) each needs.
VARIANTS = {
    "efficientvit": ("efficientvit", "b1", "EV-M"),
    "repvit": ("repvit", "m1.1", "RV-M"),
    "tinyvit": ("tinyvit", "11m", "TV-M"),
}


class EfficientSAM3Adapter:
    name = "EfficientSAM3"

    def __init__(self, backbone: str, checkpoint: str | None, device: str = "cpu",
                 resolution: int = 1008):
        backbone_type, model_name, tag = VARIANTS[backbone]
        self.has_weights = checkpoint is not None
        self.variant = (
            f"{tag} ({backbone_type}/{model_name}, {pathlib.Path(checkpoint).name})"
            if checkpoint
            else f"{tag} ({backbone_type}/{model_name}, RANDOM INIT -- no checkpoint)"
        )
        self.resolution = resolution
        self.model = build_efficientsam3_image_model(
            checkpoint_path=checkpoint,
            backbone_type=backbone_type,
            model_name=model_name,
            text_encoder_type="MobileCLIP-S0",
            text_encoder_context_length=16,
            load_from_HF=False,
            device=device,
        )
        self.model.eval()
        self.processor = Sam3Processor(self.model, resolution=resolution, device=device)
        self._state = None
        self._shape = None

    def set_image(self, bgr: np.ndarray) -> None:
        self._shape = bgr.shape[:2]
        pil = Image.fromarray(bgr[:, :, ::-1])
        self._state = self.processor.set_image(pil)

    def _best_mask(self, state) -> np.ndarray:
        masks = state.get("masks")
        if masks is None or len(masks) == 0:
            return np.zeros(self._shape, dtype=bool)
        scores = state.get("scores")
        idx = int(np.argmax(np.asarray(
            scores.cpu() if torch.is_tensor(scores) else scores))) if scores is not None else 0
        m = masks[idx]
        m = m.cpu().numpy() if torch.is_tensor(m) else np.asarray(m)
        return np.squeeze(m).astype(bool)

    def predict_box(self, box: np.ndarray) -> np.ndarray:
        # add_geometric_prompt wants normalized [cx, cy, w, h], not pixel XYXY.
        h, w = self._shape
        x0, y0, x1, y1 = [float(v) for v in box]
        cxcywh = [((x0 + x1) / 2) / w, ((y0 + y1) / 2) / h, (x1 - x0) / w, (y1 - y0) / h]
        self.processor.reset_all_prompts(self._state)
        state = self.processor.add_geometric_prompt(cxcywh, True, self._state)
        return self._best_mask(state)

    def predict_point(self, xy: np.ndarray) -> np.ndarray:
        self.processor.reset_all_prompts(self._state)
        state = self.processor.add_point_prompt([float(xy[0]), float(xy[1])], 1, self._state)
        return self._best_mask(state)

    def param_stats(self):
        stats = {"total": round(sum(p.numel() for p in self.model.parameters()) / 1e6, 3)}
        for child_name, child in self.model.named_children():
            n = sum(p.numel() for p in child.parameters()) / 1e6
            if n > 0:
                stats[child_name] = round(n, 3)
        return stats

    def encoder_gflops(self):
        x = torch.zeros(1, 3, self.resolution, self.resolution)
        return count_flops(self.model.backbone.forward_image, (x,))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="tinyvit", choices=sorted(VARIANTS))
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--dataset", default=str(HERE.parent / "datasets" / "coco128-seg"))
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ckpt = args.checkpoint
    if ckpt is None:
        p = HERE / "weights" / f"efficientsam3_{args.backbone}.pt"
        if p.is_file():
            ckpt = str(p)
    out = args.out or str(HERE / "outputs" / f"results_{args.backbone}.json")

    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)

    adapter = EfficientSAM3Adapter(args.backbone, ckpt)
    if not adapter.has_weights:
        print("WARNING: no checkpoint found in weights/ -- reporting "
              "weight-independent metrics only. Accuracy will NOT be measured.")

    samples = load_coco128seg(args.dataset, max_images=args.max_images)

    if adapter.has_weights:
        result = benchmark(adapter, samples)
    else:
        result = benchmark(adapter, samples, do_point=False, do_boundary=False,
                           report_accuracy=False)

    result["weights_available"] = adapter.has_weights
    result["encoder_gflops_1008"] = adapter.encoder_gflops()
    result["checkpoint_mb"] = (
        round(pathlib.Path(ckpt).stat().st_size / 1e6, 1) if ckpt else None
    )
    dummy = np.zeros((1008, 1008, 3), dtype=np.uint8)
    result["latency"]["encoder_only_1008"] = measure_latency(
        lambda: adapter.set_image(dummy), warmup=2, repeat=5
    )
    save(result, out)
    lat = result["latency"]
    print(f"encode={lat['encode_median_ms']:.1f} ms  decode={lat['decode_median_ms']:.2f} ms  "
          f"params={result['params_M']['total']} M")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
