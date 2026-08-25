#!/usr/bin/env python3
"""Idempotent source patches for the pinned EdgeSAM checkout (repo/).

Each patch is an exact string replacement; re-running is a no-op.
Run after `git checkout <SHA>`.
"""
from __future__ import annotations
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent / "repo"

PATCHES = [
    (
        "edge_sam/modeling/sam.py",
        """from mmdet.models.dense_heads import RPNHead, CenterNetUpdateHead
from mmdet.models.necks import FPN
from projects.EfficientDet import efficientdet
from mmengine import ConfigDict""",
        """# NOTE(pbs): mmdet/mmengine are imported lazily below instead of at module
# scope. Upstream's top-level import makes `import edge_sam` hard-fail unless
# mmdet + a torch-matched mmcv build are installed, but FPN / RPNHead /
# CenterNetUpdateHead / ConfigDict are only reachable when the model is built
# with an RPN head. The released edge_sam / edge_sam_3x inference checkpoints
# never use that branch, so inference works with no mmcv build at all.
def _import_mmdet():
    from mmdet.models.dense_heads import RPNHead, CenterNetUpdateHead
    from mmdet.models.necks import FPN
    from projects.EfficientDet import efficientdet
    from mmengine import ConfigDict
    return RPNHead, CenterNetUpdateHead, FPN, ConfigDict, efficientdet""",
        "Make mmdet/mmengine imports lazy so inference does not need an mmcv build.",
    ),
    (
        "edge_sam/build_sam.py",
        """            with open(checkpoint, "rb") as f:
                state_dict = torch.load(f)""",
        """            with open(checkpoint, "rb") as f:
                # The released edge_sam / edge_sam_3x checkpoints were saved from
                # CUDA tensors, so loading them on a CPU-only box fails without an
                # explicit map_location. Harmless on a GPU machine -- the model is
                # moved to its device by the caller afterwards.
                state_dict = torch.load(f, map_location="cpu")""",
        "Load the CUDA-saved checkpoints on a CPU-only machine.",
    ),
]

# Every site that actually dereferences one of the lazily-imported names has to
# bind it first. These are all inside `if rpn:` / detector branches.
LAZY_BINDINGS = [
    ("edge_sam/modeling/sam.py", "            self.fpn = FPN("),
    ("edge_sam/modeling/sam.py", "        cfg = ConfigDict("),
]

BIND_LINE = "            RPNHead, CenterNetUpdateHead, FPN, ConfigDict, efficientdet = _import_mmdet()\n"


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
        if old not in text:
            print(f"error: {rel}: anchor not found -- upstream changed?", file=sys.stderr)
            return 1
        path.write_text(text.replace(old, new))
        print(f"[ok]   {rel}: {why}")

    # Insert the lazy binding at the top of each block that uses the names.
    path = ROOT / "edge_sam/modeling/sam.py"
    lines = path.read_text().splitlines(keepends=True)
    if "_import_mmdet()\n" not in "".join(lines[10:]) or sum(
        1 for ln in lines if "= _import_mmdet()" in ln
    ) < 2:
        out, inserted = [], 0
        for line in lines:
            for _, anchor in LAZY_BINDINGS:
                if line.startswith(anchor) and "_import_mmdet" not in "".join(out[-2:]):
                    indent = " " * (len(line) - len(line.lstrip()))
                    out.append(indent + BIND_LINE.strip() + "\n")
                    inserted += 1
                    break
            out.append(line)
        path.write_text("".join(out))
        print(f"[ok]   edge_sam/modeling/sam.py: {inserted} lazy binding(s) inserted")
    else:
        print("[skip] edge_sam/modeling/sam.py: lazy bindings already present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
