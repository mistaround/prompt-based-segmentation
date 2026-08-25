"""Shared benchmark harness for the four promptable-segmentation models.

Each model lives in its own uv environment, so this package deliberately depends
only on what all four environments already have: numpy, opencv and the stdlib.
Model code is reached through the small `ModelAdapter` protocol in `runner.py`.
"""
from .data import Instance, Sample, load_coco128seg
from .metrics import boundary_iou, mask_iou, summarize
from .runner import ModelAdapter, benchmark, measure_latency

__all__ = [
    "Instance",
    "Sample",
    "load_coco128seg",
    "mask_iou",
    "boundary_iou",
    "summarize",
    "ModelAdapter",
    "benchmark",
    "measure_latency",
]
