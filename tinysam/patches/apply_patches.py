#!/usr/bin/env python3
"""Idempotent source patches for the pinned TinySAM checkout (repo/).

Exact string replacements; re-running is a no-op. Run after `git checkout <SHA>`.
"""
from __future__ import annotations
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent / "repo"

PATCHES = [
    (
        "tinysam/build_sam.py",
        """        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)""",
        """        with open(checkpoint, "rb") as f:
            # The released tinysam_42.3.pth was saved from CUDA tensors, so it
            # fails to load on a CPU-only machine without an explicit
            # map_location. Harmless on a GPU box -- the caller moves the model
            # to its device afterwards.
            state_dict = torch.load(f, map_location="cpu")""",
        "Load the CUDA-saved checkpoint on a CPU-only machine.",
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
        n = text.count(old)
        if n == 0:
            print(f"error: {rel}: anchor not found -- upstream changed?", file=sys.stderr)
            return 1
        path.write_text(text.replace(old, new))
        print(f"[ok]   {rel}: {n} site(s) -- {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
