# Measured results

All numbers below were produced in this repository by `scripts/run_all.sh`.
Nothing here is copied from a paper; published figures are kept separately in
[COMPARISON.md](COMPARISON.md) and clearly marked as such.

**Protocol.** 524 instances across
121 coco128-seg images. Each instance's polygon is
rasterized to a ground-truth mask; the model is prompted with the tight box
derived from that same mask, so the prompt and the target always agree.
Instances smaller than 32x32 px are dropped, since at that size
polygon-rasterization error dominates the measurement.

**Hardware.** CPU only — x86_64, 4 torch threads,
Linux-6.18.44-fc-v21-x86_64-with-glibc2.39, Python 3.10.20.
There is no GPU in this environment, so treat the latency columns as a
*relative* ranking between models, not as numbers comparable to the
GPU/iPhone figures the papers report.

## Accuracy (box- and point-prompted, coco128-seg)

| Model | Variant | Weights | box mIoU | boundary mIoU | IoU@0.5 | IoU@0.75 | IoU@0.9 | point mIoU | mIoU (M) | mIoU (L) |
|---|---|---|---|---|---|---|---|---|---|---|
| FastSAM-x | FastSAM-x | ✅ | 0.6453 | 0.5520 | 0.729 | 0.588 | 0.195 | 0.5474 | 0.6242 | 0.6726 |
| FastSAM-s | FastSAM-s | ✅ | 0.5152 | 0.4248 | 0.601 | 0.374 | 0.078 | 0.4464 | 0.5205 | 0.5083 |
| MobileSAM | vit_t (mobile_sam.pt) | ✅ | 0.7664 | 0.6589 | 0.920 | 0.672 | 0.198 | 0.5676 | 0.7556 | 0.7805 |
| EdgeSAM | RepViT-M1 (RANDOM INIT -- no checkpoint) | ❌ not available | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| EfficientSAM3 TV-M | TV-M (tinyvit/11m, RANDOM INIT -- no checkpoint) | ❌ not available | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

`mIoU (M)` / `mIoU (L)` are the COCO medium and large size buckets. There were
too few small instances after the 32x32 filter to report that bucket.

## Cost

| Model | Params (M) | Encoder (M) | Decoder (M) | Ckpt (MB) | Encoder GFLOPs | Encode ms | Decode ms/prompt | img/s | Peak RSS (MB) |
|---|---|---|---|---|---|---|---|---|---|
| FastSAM-x | 72.20 | — | — | 145.0 | 845.1 | 1877 | 2.7 | 0.53 | 2787 |
| FastSAM-s | 11.78 | — | — | 23.9 | 102.2 | 369 | 2.5 | 2.71 | 2560 |
| MobileSAM | 10.13 | 6.07 | 4.06 | 40.7 | 77.5 | 905 | 42.1 | 1.10 | 1461 |
| EdgeSAM | 9.58 | 5.52 | 4.06 | — | 40.1 | 371 | 43.3 | 2.70 | 1299 |
| EfficientSAM3 TV-M | 103.53 | 70.79 | 2.30 | — | 321.8 | 1489 | 1729.9 | 0.67 | 2433 |

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
