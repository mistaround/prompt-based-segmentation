#!/usr/bin/env python3
"""EdgeSAM prompt demo -> outputs/demo_<image>.jpg

Requires a checkpoint in weights/ (HuggingFace-only; see NOTES.md). Refuses to
render with random weights rather than producing a meaningless picture.
"""
import argparse, pathlib, sys
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "repo"))
sys.path.insert(0, str(HERE.parent / "common"))

from bench import EdgeSAMAdapter                          # noqa: E402
from pbs_bench.demo_common import default_prompt, run_prompt_demo  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--image", default=str(HERE.parent / "assets" / "dogs.jpg"))
ap.add_argument("--checkpoint", default=None)
a = ap.parse_args()

ckpt = a.checkpoint
if ckpt is None:
    for cand in ("edge_sam_3x.pth", "edge_sam.pth"):
        if (HERE / "weights" / cand).is_file():
            ckpt = str(HERE / "weights" / cand)
            break
if ckpt is None:
    sys.exit("No checkpoint in weights/. Fetch edge_sam_3x.pth from HuggingFace "
             "(see NOTES.md) -- refusing to render random-weight masks.")

torch.set_num_threads(4)
torch.set_grad_enabled(False)
box, point = default_prompt(a.image)
run_prompt_demo(EdgeSAMAdapter(ckpt), a.image, box, point,
                HERE / "outputs" / f"demo_{pathlib.Path(a.image).stem}.jpg")
