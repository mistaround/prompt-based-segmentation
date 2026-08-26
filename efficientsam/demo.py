#!/usr/bin/env python3
"""EfficientSAM prompt demo -> outputs/demo_<image>.jpg"""
import argparse, pathlib, sys
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "common"))

import bench  # noqa: E402
from pbs_bench.demo_common import default_prompt, run_prompt_demo  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--image", default=str(HERE.parent / "assets" / "dogs.jpg"))
ap.add_argument("--variant", default="vitt", choices=["vitt", "vits"])
a = ap.parse_args()

torch.set_num_threads(4)
torch.set_grad_enabled(False)
box, point = default_prompt(a.image)
adapter = bench.EfficientSAMAdapter(a.variant, str(HERE / "weights" / f"efficient_sam_{a.variant}.pt"))
run_prompt_demo(adapter, a.image, box, point,
                HERE / "outputs" / f"demo_{pathlib.Path(a.image).stem}.jpg")
