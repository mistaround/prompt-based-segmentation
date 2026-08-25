"""Shared body for the per-model demo scripts.

Every model exposes the same `ModelAdapter` protocol the benchmark uses, so one
demo routine covers all four: encode the image once, then show what a box prompt
and a point prompt each return.
"""
from __future__ import annotations

import pathlib
import time
from typing import Optional, Sequence

import cv2
import numpy as np

from .viz import draw_box, draw_point, hstack_pad, label, overlay_masks


def run_prompt_demo(
    adapter,
    image_path: str | pathlib.Path,
    box: Sequence[float],
    point: Sequence[float],
    out_path: str | pathlib.Path,
    max_width: int = 900,
) -> pathlib.Path:
    """Render input / box-prompt / point-prompt side by side."""
    image_path = pathlib.Path(image_path)
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(image_path)

    t0 = time.perf_counter()
    adapter.set_image(bgr)
    encode_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    box_mask = adapter.predict_box(np.asarray(box, dtype=np.float32))
    box_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    pt_mask = adapter.predict_point(np.asarray(point, dtype=np.float32))
    pt_ms = (time.perf_counter() - t0) * 1000

    src = label(draw_box(bgr.copy(), box), "input + prompts")
    draw_point(src, point)
    panel_box = label(overlay_masks(bgr, [box_mask]), f"box  {box_ms:.0f} ms")
    draw_box(panel_box, box)
    panel_pt = label(overlay_masks(bgr, [pt_mask]), f"point  {pt_ms:.0f} ms")
    draw_point(panel_pt, point)

    grid = hstack_pad([src, panel_box, panel_pt])
    if grid.shape[1] > max_width * 3:
        scale = (max_width * 3) / grid.shape[1]
        grid = cv2.resize(grid, (int(grid.shape[1] * scale), int(grid.shape[0] * scale)))

    banner = np.full((44, grid.shape[1], 3), 30, dtype=np.uint8)
    label(banner, f"{adapter.name} [{adapter.variant}] - encode {encode_ms:.0f} ms",
          org=(12, 31))
    grid = np.vstack([banner, grid])

    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    print(f"wrote {out_path}   encode={encode_ms:.0f} ms  "
          f"box={box_ms:.0f} ms  point={pt_ms:.0f} ms")
    return out_path


def default_prompt(image_path: str | pathlib.Path):
    """A box and a point over the main subject of each bundled demo image."""
    presets = {
        "dogs.jpg":     ((440, 180, 760, 580), (620, 360)),
        "cat.jpg":      ((180, 140, 1180, 1200), (660, 640)),
        "picture2.jpg": ((160, 110, 700, 640), (420, 380)),
        "picture3.jpg": ((300, 200, 1000, 800), (650, 500)),
    }
    return presets.get(pathlib.Path(image_path).name)
