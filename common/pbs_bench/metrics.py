"""Segmentation quality metrics."""
from __future__ import annotations

from typing import Dict, List, Sequence

import cv2
import numpy as np


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    union = np.count_nonzero(pred | gt)
    if union == 0:
        return 1.0
    return float(np.count_nonzero(pred & gt)) / union


def _boundary_region(mask: np.ndarray, dilation_ratio: float = 0.02) -> np.ndarray:
    """Pixels of `mask` within `d` of its boundary, per the Boundary IoU paper."""
    mask_u8 = mask.astype(np.uint8)
    h, w = mask_u8.shape
    d = max(1, int(round(dilation_ratio * np.sqrt(h * h + w * w))))
    # Erode, then subtract: what remains is the boundary shell of the mask.
    padded = cv2.copyMakeBorder(mask_u8, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(padded, kernel, iterations=d)[1:h + 1, 1:w + 1]
    return (mask_u8 - eroded).astype(bool)


def boundary_iou(pred: np.ndarray, gt: np.ndarray, dilation_ratio: float = 0.02) -> float:
    """IoU restricted to the boundary shells.

    Mask IoU is dominated by an object's interior, so a model that gets the blob
    roughly right scores well even with a mushy outline. Boundary IoU is the
    metric that separates the distilled students from their SAM teacher.
    """
    pb = _boundary_region(pred.astype(bool), dilation_ratio)
    gb = _boundary_region(gt.astype(bool), dilation_ratio)
    union = np.count_nonzero(pb | gb)
    if union == 0:
        return 1.0
    return float(np.count_nonzero(pb & gb)) / union


def summarize(ious: Sequence[float], prefix: str = "") -> Dict[str, float]:
    """mIoU plus the fraction of instances clearing each quality threshold."""
    arr = np.asarray(list(ious), dtype=np.float64)
    if arr.size == 0:
        return {}
    out = {
        f"{prefix}mIoU": float(arr.mean()),
        f"{prefix}median_IoU": float(np.median(arr)),
    }
    for t in (0.5, 0.75, 0.9):
        out[f"{prefix}IoU@{t}"] = float((arr >= t).mean())
    return out


def by_size_bucket(ious: Sequence[float], areas: Sequence[int]) -> Dict[str, float]:
    """COCO size buckets: small <32^2, medium <96^2, large otherwise."""
    arr = np.asarray(list(ious), dtype=np.float64)
    ar = np.asarray(list(areas), dtype=np.float64)
    buckets = {
        "small": ar < 32 ** 2,
        "medium": (ar >= 32 ** 2) & (ar < 96 ** 2),
        "large": ar >= 96 ** 2,
    }
    out: Dict[str, float] = {}
    for name, sel in buckets.items():
        if sel.any():
            out[f"mIoU_{name}"] = float(arr[sel].mean())
            out[f"n_{name}"] = int(sel.sum())
    return out
