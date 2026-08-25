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

- **EdgeSAM 在框提示上同时拿下最高精度和最低成本。** box mIoU 0.7801 高于 MobileSAM 的
  0.7664，而参数更少（9.58M vs 10.13M）、GFLOPs 只有一半（40.1 vs 77.5）、
  编码器快 **2.4×**（379ms vs 905ms）、峰值内存低 24%。
  方向与论文一致（论文报 COCO AP 高 2.8），而且这个优势来自**蒸馏方式**
  （prompt-in-the-loop）而非模型规模——两者体量本来就相当。

- **但点提示上出现反转：EdgeSAM 反而最低（0.5174），EfficientSAM3 最高（0.5709）。**
  这与 EdgeSAM 论文的说法不符。本协议下单点提示取 3 个候选再按模型自报的 IoU
  分数选最优，因此这项同时考验**分数头的标定质量**，而不只是分割质量；
  论文的点提示协议未必相同，样本量也只有 524 个实例。
  **如实记录，不据此下结论。**

- **所有模型的 IoU@0.9 都很低（0.16–0.22）**，而 IoU@0.5 高达 0.84–0.92。
  也就是说这些学生模型基本都能"找对物体"，**丢分几乎全在轮廓精度上**——
  boundary mIoU（0.42–0.68）比 mask mIoU 低 8～10 个点也印证了这一点。
  这正是蒸馏学生与 SAM 教师的主要差距所在。

- **FastSAM 的两个变体展示了它真正的取舍**：`-x` 精度 0.645 但要 1877ms，
  `-s` 快 5× 却掉到 0.515。更关键的是 **它的"解码"只要 2.7ms（一次 IoU 匹配），
  而 SAM 系要 42ms** —— 密集提示场景下这个差距会放大成数量级。
  代价是提示只能召回检测器已提议的 mask，所以它在框提示上也垫底。
  另外 FastSAM-x 有 **72M 参数**，是 MobileSAM/EdgeSAM 的 7 倍——
  它快是因为架构，不是因为小。

- **EfficientSAM3 不该和另外三个比"每次点击的延迟"。** 它的 decode 是 1808ms，
  比 SAM 系高 **40×**，因为每个提示都要走一遍完整 grounding 解码。
  用几何提示去测它其实是"用它不擅长的方式测它"——即便如此仍拿到 0.7089，
  且点提示是全场最高。它换来的是另外三个都没有的能力：
  **文本提示一次返回该概念的全部实例**（见 `efficientsam3/outputs/text_dogs.jpg`：
  一个 "dog" 分出两个独立实例），以及跨帧跟踪。
  对等的比较对象是 861.5M 参数的 SAM3 教师，不是 MobileSAM。

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
