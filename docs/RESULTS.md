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
| EdgeSAM | RepViT-M1 (edge_sam_3x.pth) | ✅ | 0.7801 | 0.6770 | 0.924 | 0.700 | 0.223 | 0.5174 | 0.7766 | 0.7847 |
| EfficientSAM3 TV-M | TV-M (tinyvit/11m, efficientsam3_tinyvit.pt) | ✅ | 0.7089 | 0.6077 | 0.836 | 0.618 | 0.162 | 0.5709 | 0.7040 | 0.7152 |

`mIoU (M)` / `mIoU (L)` are the COCO medium and large size buckets. There were
too few small instances after the 32x32 filter to report that bucket.

## Cost

| Model | Params (M) | Encoder (M) | Decoder (M) | Ckpt (MB) | Encoder GFLOPs | Encode ms | Decode ms/prompt | img/s | Peak RSS (MB) |
|---|---|---|---|---|---|---|---|---|---|
| FastSAM-x | 72.20 | — | — | 145.0 | 845.1 | 1877 | 2.7 | 0.53 | 2787 |
| FastSAM-s | 11.78 | — | — | 23.9 | 102.2 | 369 | 2.5 | 2.71 | 2560 |
| MobileSAM | 10.13 | 6.07 | 4.06 | 40.7 | 77.5 | 905 | 42.1 | 1.10 | 1461 |
| EdgeSAM | 9.58 | 5.52 | 4.06 | 38.8 | 40.1 | 379 | 44.1 | 2.64 | 1105 |
| EfficientSAM3 TV-M | 103.53 | 70.79 | 2.30 | 492.8 | 321.8 | 1681 | 1808.1 | 0.59 | 2373 |

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
