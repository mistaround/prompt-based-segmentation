"""coco128-seg loader.

coco128-seg ships 128 COCO train2017 images with YOLO-seg labels: one line per
instance, `class x1 y1 x2 y2 ... xn yn` with polygon vertices normalized to
[0, 1]. We rasterize each polygon to a binary mask and derive a tight box from
that mask, so the box prompt and the ground-truth mask always agree -- which is
what makes box-prompted IoU a fair comparison across models.
"""
from __future__ import annotations

import dataclasses
import pathlib
from typing import Iterator, List, Optional

import cv2
import numpy as np


@dataclasses.dataclass
class Instance:
    cls: int
    mask: np.ndarray          # bool, HxW
    box: np.ndarray           # float32 [x0, y0, x1, y1], XYXY
    area: int                 # pixels
    centroid: np.ndarray      # float32 [x, y], guaranteed to lie inside `mask`


@dataclasses.dataclass
class Sample:
    path: pathlib.Path
    height: int
    width: int
    instances: List[Instance]


def _inside_point(mask: np.ndarray) -> np.ndarray:
    """A point that is inside the mask.

    The centre of mass of a banana- or donut-shaped mask can fall outside it, and
    a point prompt that lands on the background inverts the model's intent. The
    pixel furthest from the boundary (max of the distance transform) is always
    inside and is also the most unambiguous point to prompt with.
    """
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return np.array([x, y], dtype=np.float32)


def load_coco128seg(
    root: pathlib.Path | str,
    max_images: Optional[int] = None,
    min_area_px: int = 32 * 32,
    max_instances_per_image: Optional[int] = None,
) -> List[Sample]:
    """Load coco128-seg into rasterized instances.

    `min_area_px` drops very small instances. They are dominated by
    polygon-rasterization error, so including them measures label noise rather
    than model quality; 32x32 px is the COCO "small object" boundary.
    """
    root = pathlib.Path(root)
    img_dir = root / "images" / "train2017"
    lbl_dir = root / "labels" / "train2017"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"{img_dir} not found -- run scripts/get_dataset.sh")

    samples: List[Sample] = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.is_file():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        instances: List[Instance] = []
        for line in lbl_path.read_text().splitlines():
            parts = line.split()
            # Need a class plus at least a triangle (3 vertices = 6 numbers).
            if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
                continue
            cls = int(float(parts[0]))
            pts = np.asarray(parts[1:], dtype=np.float64).reshape(-1, 2)
            pts[:, 0] *= w
            pts[:, 1] *= h
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
            area = int(mask.sum())
            if area < min_area_px:
                continue
            ys, xs = np.nonzero(mask)
            box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)
            instances.append(
                Instance(
                    cls=cls,
                    mask=mask.astype(bool),
                    box=box,
                    area=area,
                    centroid=_inside_point(mask.astype(bool)),
                )
            )

        if not instances:
            continue
        if max_instances_per_image is not None:
            instances.sort(key=lambda i: -i.area)
            instances = instances[:max_instances_per_image]
        samples.append(Sample(path=img_path, height=h, width=w, instances=instances))
        if max_images is not None and len(samples) >= max_images:
            break
    return samples


def iter_instances(samples: List[Sample]) -> Iterator[Instance]:
    for s in samples:
        yield from s.instances
