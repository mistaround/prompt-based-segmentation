#!/usr/bin/env python3
"""Merge the per-model result JSONs into docs/RESULTS.md.

Runs on the system Python -- it only needs the stdlib, so it does not care which
of the four uv environments produced the inputs.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCES = [
    ("fastsam/outputs/results_x.json", "FastSAM-x"),
    ("fastsam/outputs/results_s.json", "FastSAM-s"),
    ("mobilesam/outputs/results.json", "MobileSAM"),
    ("edgesam/outputs/results.json", "EdgeSAM"),
    ("efficientsam3/outputs/results_tinyvit.json", "EfficientSAM3 TV-M"),
]


def fmt(v: Any, spec: str = ".4f", dash: str = "—") -> str:
    if v is None:
        return dash
    if isinstance(v, (int, float)):
        return format(v, spec)
    return str(v)


def load() -> List[Dict[str, Any]]:
    out = []
    for rel, label in SOURCES:
        p = ROOT / rel
        if not p.is_file():
            print(f"  (missing {rel} -- skipped)")
            continue
        d = json.loads(p.read_text())
        d["_label"] = label
        out.append(d)
    return out


def accuracy_table(rows: List[Dict[str, Any]]) -> str:
    head = (
        "| Model | Variant | Weights | box mIoU | boundary mIoU | IoU@0.5 | IoU@0.75 "
        "| IoU@0.9 | point mIoU | mIoU (M) | mIoU (L) |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for r in rows:
        a = r.get("accuracy", {})
        has = r.get("weights_available", True)
        if not has or "_note" in a:
            lines.append(
                f"| {r['_label']} | {r['variant']} | ❌ not available | "
                + " | ".join(["n/a"] * 8) + " |"
            )
            continue
        lines.append(
            f"| {r['_label']} | {r['variant']} | ✅ | "
            f"{fmt(a.get('box_mIoU'))} | {fmt(a.get('box_boundary_mIoU'))} | "
            f"{fmt(a.get('box_IoU@0.5'), '.3f')} | {fmt(a.get('box_IoU@0.75'), '.3f')} | "
            f"{fmt(a.get('box_IoU@0.9'), '.3f')} | {fmt(a.get('point_mIoU'))} | "
            f"{fmt(a.get('mIoU_medium'))} | {fmt(a.get('mIoU_large'))} |"
        )
    return head + "\n".join(lines) + "\n"


def performance_table(rows: List[Dict[str, Any]]) -> str:
    head = (
        "| Model | Params (M) | Encoder (M) | Decoder (M) | Ckpt (MB) | Encoder GFLOPs "
        "| Encode ms | Decode ms/prompt | img/s | Peak RSS (MB) |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for r in rows:
        p = r.get("params_M", {})
        lat = r.get("latency", {})
        gf = r.get("encoder_gflops_1024", r.get("encoder_gflops_1008"))
        lines.append(
            f"| {r['_label']} | {fmt(p.get('total'), '.2f')} | "
            f"{fmt(p.get('image_encoder') or p.get('backbone'), '.2f')} | "
            f"{fmt(p.get('mask_decoder') or p.get('segmentation_head'), '.2f')} | "
            f"{fmt(r.get('checkpoint_mb'), '.1f')} | {fmt(gf, '.1f')} | "
            f"{fmt(lat.get('encode_median_ms'), '.0f')} | "
            f"{fmt(lat.get('decode_median_ms'), '.1f')} | "
            f"{fmt(lat.get('images_per_s'), '.2f')} | "
            f"{fmt(r.get('peak_rss_mb'), '.0f')} |"
        )
    return head + "\n".join(lines) + "\n"


def main() -> int:
    rows = load()
    if not rows:
        print("no result files found -- run scripts/run_all.sh first")
        return 1
    ref = rows[0]
    env = ref.get("env", {})
    body = f"""# Measured results

All numbers below were produced in this repository by `scripts/run_all.sh`.
Nothing here is copied from a paper; published figures are kept separately in
[COMPARISON.md](COMPARISON.md) and clearly marked as such.

**Protocol.** {ref.get('n_instances', '?')} instances across
{ref.get('n_images', '?')} coco128-seg images. Each instance's polygon is
rasterized to a ground-truth mask; the model is prompted with the tight box
derived from that same mask, so the prompt and the target always agree.
Instances smaller than 32x32 px are dropped, since at that size
polygon-rasterization error dominates the measurement.

**Hardware.** CPU only — {env.get('processor', '?')}, 4 torch threads,
{env.get('platform', '?')}, Python {env.get('python', '?')}.
There is no GPU in this environment, so treat the latency columns as a
*relative* ranking between models, not as numbers comparable to the
GPU/iPhone figures the papers report.

## Accuracy (box- and point-prompted, coco128-seg)

{accuracy_table(rows)}
`mIoU (M)` / `mIoU (L)` are the COCO medium and large size buckets. There were
too few small instances after the 32x32 filter to report that bucket.

## Cost

{performance_table(rows)}
`Encode ms` is one image-encoder pass — paid once per image. `Decode ms/prompt`
is one prompt afterwards, which is what an interactive tool pays per click.

## Raw data

Per-model JSON lives in each model's `outputs/` directory.
"""
    out = ROOT / "docs" / "RESULTS.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(body)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
