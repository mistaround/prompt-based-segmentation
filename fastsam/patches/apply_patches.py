#!/usr/bin/env python3
"""Idempotent source patches for the pinned FastSAM checkout (repo/).

Each patch is a (path, old, new, why) tuple applied as an exact string
replacement. Re-running is a no-op. Run after `git checkout <SHA>`.
"""
from __future__ import annotations
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent / "repo"

PATCHES = [
    (
        "ultralytics/nn/tasks.py",
        "        return torch.load(file, map_location='cpu'), file  # load",
        "        return torch.load(file, map_location='cpu', weights_only=False), file  # load",
        "PyTorch >=2.6 flipped torch.load's `weights_only` default to True. The FastSAM "
        ".pt files are full pickled ultralytics SegmentationModel objects, not bare "
        "state_dicts, so they need weights_only=False. The checkpoints come from the "
        "official ultralytics release assets.",
    ),
    (
        "ultralytics/yolo/utils/torch_utils.py",
        "    x = torch.load(f, map_location=torch.device('cpu'))",
        "    x = torch.load(f, map_location=torch.device('cpu'), weights_only=False)",
        "Same weights_only issue in strip_optimizer().",
    ),
    (
        "fastsam/prompt.py",
        """        try:
            buf = fig.canvas.tostring_rgb()
        except AttributeError:
            fig.canvas.draw()
            buf = fig.canvas.tostring_rgb()
        cols, rows = fig.canvas.get_width_height()
        img_array = np.frombuffer(buf, dtype=np.uint8).reshape(rows, cols, 3)""",
        """        fig.canvas.draw()
        try:
            buf = fig.canvas.tostring_rgb()
            channels = 3
        except AttributeError:
            # matplotlib >=3.10 removed Canvas.tostring_rgb(); buffer_rgba() is the
            # supported replacement. Drop the alpha channel to keep the RGB layout
            # the rest of this function assumes.
            buf = fig.canvas.buffer_rgba()
            channels = 4
        cols, rows = fig.canvas.get_width_height()
        img_array = np.frombuffer(buf, dtype=np.uint8).reshape(rows, cols, channels)[:, :, :3]""",
        "matplotlib >=3.10 removed FigureCanvasAgg.tostring_rgb(); use buffer_rgba().",
    ),
]


def main() -> int:
    if not ROOT.is_dir():
        print(f"error: {ROOT} not found; run setup.sh first", file=sys.stderr)
        return 1
    for rel, old, new, why in PATCHES:
        path = ROOT / rel
        text = path.read_text()
        if new in text:
            print(f"[skip] {rel}: already patched")
            continue
        count = text.count(old)
        if count == 0:
            print(f"error: {rel}: anchor not found -- upstream changed?", file=sys.stderr)
            return 1
        path.write_text(text.replace(old, new))
        print(f"[ok]   {rel}: {count} site(s) -- {why.splitlines()[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
