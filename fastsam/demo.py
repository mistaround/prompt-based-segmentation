#!/usr/bin/env python3
"""FastSAM prompt demo -> outputs/demo_<image>.jpg (plus an everything-mode render)."""
import argparse, pathlib, sys
import cv2, torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "repo"))
sys.path.insert(0, str(HERE.parent / "common"))

from bench import FastSAMAdapter                          # noqa: E402
from pbs_bench.demo_common import default_prompt, run_prompt_demo  # noqa: E402
from pbs_bench.viz import label, overlay_masks            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--image", default=str(HERE.parent / "assets" / "dogs.jpg"))
ap.add_argument("--model", default=str(HERE / "weights" / "FastSAM-x.pt"))
a = ap.parse_args()

torch.set_num_threads(4)
torch.set_grad_enabled(False)
stem = pathlib.Path(a.image).stem
adapter = FastSAMAdapter(a.model)
box, point = default_prompt(a.image)
run_prompt_demo(adapter, a.image, box, point, HERE / "outputs" / f"demo_{stem}.jpg")

# FastSAM's native mode: every mask from a single forward pass.
masks = adapter._prompt.everything_prompt()
bgr = cv2.imread(a.image)
out = label(overlay_masks(bgr, [m.cpu().numpy() for m in masks]),
            f"FastSAM everything mode - {len(masks)} masks")
cv2.imwrite(str(HERE / "outputs" / f"everything_{stem}.jpg"), out)
print(f"wrote outputs/everything_{stem}.jpg ({len(masks)} masks)")
