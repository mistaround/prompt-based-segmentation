#!/usr/bin/env python3
"""EfficientSAM3 demo -> outputs/demo_<image>.jpg and outputs/text_<image>.jpg

Requires a checkpoint in weights/ (HuggingFace-only; see NOTES.md).
Also shows the capability the other three models do not have: a text prompt
returning every instance of a concept.
"""
import argparse, pathlib, sys
import cv2, numpy as np, torch
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "repo"))
sys.path.insert(0, str(HERE.parent / "common"))

from bench import EfficientSAM3Adapter                    # noqa: E402
from pbs_bench.demo_common import default_prompt, run_prompt_demo  # noqa: E402
from pbs_bench.viz import label, overlay_masks            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--image", default=str(HERE.parent / "assets" / "dogs.jpg"))
ap.add_argument("--backbone", default="tinyvit", choices=["tinyvit", "repvit", "efficientvit"])
ap.add_argument("--text", default="dog")
ap.add_argument("--checkpoint", default=None)
a = ap.parse_args()

ckpt = a.checkpoint or (
    str(HERE / "weights" / f"efficientsam3_{a.backbone}.pt")
    if (HERE / "weights" / f"efficientsam3_{a.backbone}.pt").is_file() else None
)
if ckpt is None:
    sys.exit(f"No checkpoint weights/efficientsam3_{a.backbone}.pt. Fetch it from "
             "HuggingFace (see NOTES.md) -- refusing to render random-weight masks.")

torch.set_num_threads(4)
torch.set_grad_enabled(False)
stem = pathlib.Path(a.image).stem
adapter = EfficientSAM3Adapter(a.backbone, ckpt)
box, point = default_prompt(a.image)
run_prompt_demo(adapter, a.image, box, point, HERE / "outputs" / f"demo_{stem}.jpg")

# Text prompt: one concept -> every instance of it.
bgr = cv2.imread(a.image)
adapter.set_image(bgr)
state = adapter.processor.set_text_prompt(a.text, adapter._state)
masks = [np.squeeze(m.cpu().numpy() if torch.is_tensor(m) else np.asarray(m)).astype(bool)
         for m in state.get("masks", [])]
out = label(overlay_masks(bgr, masks), f'text prompt "{a.text}" - {len(masks)} instances')
cv2.imwrite(str(HERE / "outputs" / f"text_{stem}.jpg"), out)
print(f"wrote outputs/text_{stem}.jpg ({len(masks)} instances)")
