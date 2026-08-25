"""Benchmark driver shared by all four models.

A model is plugged in through `ModelAdapter`, which splits inference the way
these architectures actually split it: `set_image` runs the image encoder once
per image (the dominant cost), and `predict_box` / `predict_point` run the
lightweight prompt decoder once per prompt. Reporting those two separately is
the whole point -- amortized over many prompts, decoder latency is what an
interactive annotation tool feels.
"""
from __future__ import annotations

import json
import platform
import resource
import time
from typing import Callable, Dict, List, Optional, Protocol, Sequence

import numpy as np

from .data import Sample
from .metrics import boundary_iou, by_size_bucket, mask_iou, summarize


class ModelAdapter(Protocol):
    name: str
    variant: str

    def set_image(self, bgr: np.ndarray) -> None:
        """Run the image encoder on a BGR uint8 image."""

    def predict_box(self, box: np.ndarray) -> np.ndarray:
        """Return a bool HxW mask for an XYXY box prompt."""

    def predict_point(self, xy: np.ndarray) -> np.ndarray:
        """Return a bool HxW mask for a single positive point prompt."""

    def param_stats(self) -> Dict[str, float]:
        """Parameter counts in millions, at least {'total'}."""


def peak_rss_mb() -> float:
    # ru_maxrss is kilobytes on Linux. It is a high-water mark for the whole
    # process, so it is only meaningful when read after the work is done.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def count_flops(model, example_inputs: tuple) -> Optional[float]:
    """GFLOPs for one forward pass, or None if the counter cannot handle the model.

    Uses torch's built-in FlopCounterMode so no extra dependency is needed.
    """
    try:
        import torch
        from torch.utils.flop_counter import FlopCounterMode

        counter = FlopCounterMode(display=False)
        with torch.no_grad(), counter:
            model(*example_inputs)
        return counter.get_total_flops() / 1e9
    except Exception:
        return None


def measure_latency(
    fn: Callable[[], object],
    warmup: int = 2,
    repeat: int = 10,
) -> Dict[str, float]:
    """Median-based timing. Median, not mean: a single scheduler hiccup on a
    shared 4-core box skews a mean badly, and we care about the typical call."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times)
    return {
        "median_ms": float(np.median(arr)),
        "mean_ms": float(arr.mean()),
        "p90_ms": float(np.percentile(arr, 90)),
        "min_ms": float(arr.min()),
    }


def benchmark(
    adapter: ModelAdapter,
    samples: Sequence[Sample],
    do_point: bool = True,
    do_boundary: bool = True,
    progress: bool = True,
    report_accuracy: bool = True,
) -> Dict[str, object]:
    """Run the protocol over `samples`.

    `report_accuracy=False` is for a model whose checkpoint is unavailable: the
    latency/params/memory columns are properties of the architecture and stay
    valid under random initialization, but the IoUs are noise, so they are
    neither printed nor written out.
    """
    import cv2

    box_ious: List[float] = []
    box_biou: List[float] = []
    point_ious: List[float] = []
    areas: List[int] = []
    encode_ms: List[float] = []
    decode_ms: List[float] = []
    n_instances = 0

    for idx, sample in enumerate(samples):
        bgr = cv2.imread(str(sample.path))
        if bgr is None:
            continue

        t0 = time.perf_counter()
        adapter.set_image(bgr)
        encode_ms.append((time.perf_counter() - t0) * 1000.0)

        for inst in sample.instances:
            n_instances += 1
            areas.append(inst.area)

            t0 = time.perf_counter()
            pred = adapter.predict_box(inst.box)
            decode_ms.append((time.perf_counter() - t0) * 1000.0)

            box_ious.append(mask_iou(pred, inst.mask))
            if do_boundary:
                box_biou.append(boundary_iou(pred, inst.mask))

            if do_point:
                pred_p = adapter.predict_point(inst.centroid)
                point_ious.append(mask_iou(pred_p, inst.mask))

        if progress and (idx + 1) % 20 == 0:
            running = (
                f", running box mIoU={np.mean(box_ious):.4f}" if report_accuracy else ""
            )
            print(
                f"  [{adapter.name}] {idx + 1}/{len(samples)} images, "
                f"{n_instances} instances{running}",
                flush=True,
            )

    result: Dict[str, object] = {
        "model": adapter.name,
        "variant": adapter.variant,
        "n_images": len(encode_ms),
        "n_instances": n_instances,
        "params_M": adapter.param_stats(),
        "accuracy": (
            {
                **summarize(box_ious, prefix="box_"),
                **({"box_boundary_mIoU": float(np.mean(box_biou))} if box_biou else {}),
                **(summarize(point_ious, prefix="point_") if point_ious else {}),
                **by_size_bucket(box_ious, areas),
            }
            if report_accuracy
            else {
                "_note": "not measured: random-initialized weights produce "
                         "meaningless masks"
            }
        ),
        "latency": {
            "encode_median_ms": float(np.median(encode_ms)) if encode_ms else None,
            "encode_mean_ms": float(np.mean(encode_ms)) if encode_ms else None,
            "decode_median_ms": float(np.median(decode_ms)) if decode_ms else None,
            "decode_mean_ms": float(np.mean(decode_ms)) if decode_ms else None,
            "images_per_s": float(1000.0 / np.median(encode_ms)) if encode_ms else None,
        },
        "peak_rss_mb": peak_rss_mb(),
        "env": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
        },
    }
    return result


def save(result: Dict[str, object], path: str) -> None:
    import pathlib

    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2))
    print(f"wrote {p}")
