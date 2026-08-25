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
`Encoder GFLOPs` is the image encoder alone at 1024x1024 (1008 for
EfficientSAM3, its native resolution).

### 关于 GFLOPs 口径

torch 的 `FlopCounterMode` 把一次乘加算作 **2 FLOPs**，而论文通常报的是 **MACs**
（乘加算 1）。所以本表的数字要 **除以 2** 才能和论文对齐：

| | 本表实测 | ÷2 | 论文值 |
|---|---|---|---|
| MobileSAM | 77.5 | **38.8** | 38.2 |
| EdgeSAM | 40.1 | **20.1** | 22.1（论文含解码器） |

MobileSAM 一项吻合到 1.5% 以内，可以确认口径判断。
但要注意 EdgeSAM 论文那张表里 **FastSAM 那一行（887.6）反而接近本表未折半的
845.1**，即同一张表里两种口径混用了。**跨论文比 FLOPs 时务必先确认口径。**

## 读数

- **MobileSAM 精度最高**（box mIoU 0.766），且 IoU@0.5 达到 0.920 —— 意味着它几乎不会
  完全答错，失分主要在轮廓精细度上（boundary mIoU 0.659）。
- **EdgeSAM 是最省的**：参数最少（9.58M）、GFLOPs 最低（约为 MobileSAM 的一半）、
  编码器最快（371ms vs 905ms，**2.4×**），峰值内存也最低。
  本环境拿不到权重所以没有精度数字，但论文报告它在参数量与 MobileSAM 持平的情况下
  COCO AP 高 2.8 —— 如果成立，它在这张表上是全面占优的。
- **FastSAM 的两个变体展示了它真正的取舍**：`-x` 精度尚可（0.645）但要 1877ms，
  `-s` 快 5× 却掉到 0.515。更关键的是 **它的"解码"只要 2.7ms（一次 IoU 匹配），
  而 SAM 系要 42ms** —— 密集提示场景下这个差距会放大成数量级。
  代价是 **点提示明显更弱**（0.547 vs 框提示 0.645），因为提示只能召回检测器已提议的 mask。
- **所有模型的点提示都明显弱于框提示**，这是提示信息量决定的，不是实现问题。
- **EfficientSAM3 的 decode 是 1730ms**，比 SAM 系高 **40×**。这不是缺陷而是架构差异：
  它每个提示都要走完整 grounding 解码。它换来的是另外三个都没有的能力
  （文本/概念提示、一次返回全部实例、跨帧跟踪），对等的比较对象是 SAM3 教师而非 MobileSAM。

> **注意**：EfficientSAM3 一行只跑了 32 张图（无权重时精度不可测，32 张足够稳定中位数时延）；
> 其余各行为全部 121 张图 / 524 个实例。

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
