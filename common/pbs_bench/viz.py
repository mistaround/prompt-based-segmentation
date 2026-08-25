"""Minimal mask visualization shared by the demo scripts."""
from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

# Distinguishable at a glance and colour-blind-safe enough for a handful of masks.
PALETTE = np.array([
    [ 31, 119, 180], [255, 127,  14], [ 44, 160,  44], [214,  39,  40],
    [148, 103, 189], [140,  86,  75], [227, 119, 194], [127, 127, 127],
    [188, 189,  34], [ 23, 190, 207],
], dtype=np.uint8)


def overlay_masks(
    bgr: np.ndarray,
    masks: Sequence[np.ndarray],
    alpha: float = 0.55,
    draw_contour: bool = True,
) -> np.ndarray:
    out = bgr.copy()
    for i, m in enumerate(masks):
        m = np.asarray(m).astype(bool)
        if m.shape != out.shape[:2] or not m.any():
            continue
        colour = PALETTE[i % len(PALETTE)].astype(np.float32)
        region = out[m].astype(np.float32)
        out[m] = (region * (1 - alpha) + colour * alpha).astype(np.uint8)
        if draw_contour:
            contours, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(out, contours, -1, tuple(int(c) for c in colour), 2)
    return out


def draw_box(bgr: np.ndarray, box, colour=(0, 255, 255), thickness: int = 2) -> np.ndarray:
    x0, y0, x1, y1 = [int(round(float(v))) for v in box]
    cv2.rectangle(bgr, (x0, y0), (x1, y1), colour, thickness)
    return bgr


def draw_point(bgr: np.ndarray, xy, colour=(0, 255, 255), radius: int = 7) -> np.ndarray:
    x, y = int(round(float(xy[0]))), int(round(float(xy[1])))
    cv2.circle(bgr, (x, y), radius + 2, (0, 0, 0), -1)
    cv2.circle(bgr, (x, y), radius, colour, -1)
    return bgr


def label(bgr: np.ndarray, text: str, org=(12, 32)) -> np.ndarray:
    cv2.putText(bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return bgr


def hstack_pad(images: Sequence[np.ndarray], pad: int = 8) -> np.ndarray:
    h = max(im.shape[0] for im in images)
    parts = []
    for im in images:
        if im.shape[0] != h:
            scale = h / im.shape[0]
            im = cv2.resize(im, (int(im.shape[1] * scale), h))
        parts.append(im)
        parts.append(np.full((h, pad, 3), 255, dtype=np.uint8))
    return np.hstack(parts[:-1])
