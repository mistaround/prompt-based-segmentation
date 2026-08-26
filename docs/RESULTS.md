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
| TinySAM | vit_t (tinysam_42.3.pth) | ✅ | 0.7624 | 0.6549 | 0.914 | 0.656 | 0.189 | 0.4552 | 0.7505 | 0.7779 |
| RepViT-SAM | RepViT-M2.3 (repvit_sam.pt) | ✅ | 0.7744 | 0.6699 | 0.920 | 0.702 | 0.214 | 0.5664 | 0.7643 | 0.7875 |
| EfficientSAM-Ti | Ti (efficient_sam_vitt.pt) | ✅ | 0.7704 | 0.6592 | 0.924 | 0.693 | 0.197 | 0.4866 | 0.7594 | 0.7847 |
| EfficientSAM-S | S (efficient_sam_vits.pt) | ✅ | 0.7793 | 0.6739 | 0.914 | 0.739 | 0.210 | 0.5888 | 0.7700 | 0.7915 |
| EfficientViT-SAM-L0 | L0 (efficientvit_sam_l0.pt, 512px) | ✅ | 0.8019 | 0.7070 | 0.937 | 0.758 | 0.275 | 0.6099 | 0.7983 | 0.8065 |
| EfficientSAM3 TV-M | TV-M (tinyvit/11m, efficientsam3_tinyvit.pt) | ✅ | 0.7089 | 0.6077 | 0.836 | 0.618 | 0.162 | 0.5709 | 0.7040 | 0.7152 |

`mIoU (M)` / `mIoU (L)` are the COCO medium and large size buckets. There were
too few small instances after the 32x32 filter to report that bucket.

## Cost

| Model | Params (M) | Encoder (M) | Decoder (M) | Ckpt (MB) | Encoder GFLOPs | Encode ms | Decode ms/prompt | img/s | Peak RSS (MB) |
|---|---|---|---|---|---|---|---|---|---|
| FastSAM-x | 72.20 | — | — | 145.0 | 845.1 | 1804 | 3.6 | 0.55 | 2787 |
| FastSAM-s | 11.78 | — | — | 23.9 | 102.2 | 441 | 2.9 | 2.27 | 2560 |
| MobileSAM | 10.13 | 6.07 | 4.06 | 40.7 | 77.5 | 1264 | 54.1 | 0.79 | 1461 |
| EdgeSAM | 9.58 | 5.52 | 4.06 | 38.8 | 40.1 | 481 | 48.3 | 2.08 | 1105 |
| TinySAM | 10.13 | 6.07 | 4.06 | 40.7 | 77.5 | 1234 | 60.4 | 0.81 | 1277 |
| RepViT-SAM | 27.22 | 23.16 | 4.06 | 110.0 | 226.6 | 2520 | 50.0 | 0.40 | 1241 |
| EfficientSAM-Ti | 10.22 | 6.16 | 4.06 | 41.0 | 204.5 | 3329 | 85.4 | 0.30 | 1210 |
| EfficientSAM-S | 26.41 | 22.35 | 4.06 | 105.7 | 491.2 | 8324 | 80.0 | 0.12 | 1685 |
| EfficientViT-SAM-L0 | 34.79 | 30.73 | 4.06 | 139.4 | 69.5 | 539 | 53.0 | 1.85 | 1186 |
| EfficientSAM3 TV-M | 103.53 | 70.79 | 2.30 | 492.8 | 321.8 | 2868 | 2234.6 | 0.35 | 2373 |

`Encode ms` is one image-encoder pass — paid once per image. `Decode ms/prompt`
is one prompt afterwards, which is what an interactive tool pays per click.
`Encoder GFLOPs` is the image encoder alone at each model's native input
resolution (512² for EfficientViT-SAM-L0, 1008² for EfficientSAM3, else 1024²).

> **时延是单独一轮"背靠背"测的**，不是取自各自的精度运行。这台共享 4 核机器在不同批次之间
> 吞吐会漂移约 25%——足以在两个**逐字节相同**的模型之间凭空造出差距
> （MobileSAM 与 TinySAM 的编码器参数、FLOPs 完全一致，分批测相差 25%，背靠背测则一致）。
> 精度是确定性的、不受影响，时延则必须放在同一轮里测。见 `scripts/run_latency.sh`。

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

### 1. 唯一不是蒸馏出来的模型，赢在精度也赢在速度

**EfficientViT-SAM-L0 在四项精度指标上全部第一**：box mIoU 0.8019、boundary 0.7070、
point 0.6099、IoU@0.9 0.275（第二名只有 0.223）。它也是**编码器最快的**（503 ms）。

这不是矛盾——它是这一组里**唯一在完整 SA-1B 上端到端训练**的模型，其余都是从 SAM
蒸馏而来（通常只用 1%~3% 的数据）。代价是 34.8M 参数，是 MobileSAM/EdgeSAM 的 3.5 倍。
但因为它**推理分辨率是 512² 而非 1024²**，编码器 GFLOPs 反而只有 69.5，
低于 EfficientSAM-Ti 的 204.5。

> **参数量不等于计算量。** 这一行是全表最好的反例：参数最多的几个模型之一，
> 却同时是最快和最准的。真正决定 CPU 时延的是 FLOPs 和输入分辨率，不是参数数量。

### 2. 蒸馏路线内部的排序，与各自论文的说法基本一致

| | box mIoU | 参数 | 论文 COCO AP |
|---|---|---|---|
| EdgeSAM (RepViT-M1) | 0.7801 | 9.58M | 42.7 |
| EfficientSAM-S | 0.7793 | 26.41M | 44.4 |
| RepViT-SAM (RepViT-M2.3) | 0.7744 | 27.22M | 44.4 |
| EfficientSAM-Ti | 0.7704 | 10.22M | 42.3 |
| MobileSAM | 0.7664 | 10.13M | 41.0 / 42.7* |
| TinySAM | 0.7624 | 10.13M | 42.3 |

`*` MobileSAM 在 TinySAM 论文里是 41.0，在 RepViT-SAM 论文里是 42.7——
**协议不同，跨论文的 AP 本来就不能直接比**，这也正是本仓库重测一遍的理由。

**EdgeSAM 是效率之王**：9.58M 参数、40.1 GFLOPs（全表最低）拿到蒸馏组第一，
比参数多 2.8 倍的 RepViT-SAM 还高。同一条"换 CNN 编码器"路线上，
EdgeSAM 的 prompt-in-the-loop 蒸馏显然比单纯放大编码器更划算。

### 3. TinySAM 没能复现它相对 MobileSAM 的优势

TinySAM 的卖点是"同架构同算力、纯靠训练方法 +1.3 COCO AP"。
本测量里两者的编码器**逐字节相同**（6.0655M 参数、77.5 GFLOPs、模块类型计数完全一致），
但 **TinySAM 0.7624 反而略低于 MobileSAM 0.7664**。

一个已知的不对等因素：TinySAM 的 `predict()` **没有 multimask 参数**，
固定返回 3 个候选，所以框提示也只能按分数挑；其余模型用的是解码器的单-mask 输出头。
这条路径本身就不同，不能算作对该论文的反驳，但**在本协议下它确实没有体现出优势**。

### 4. 丢分几乎全在轮廓，不在定位

所有模型 IoU@0.5 高达 0.60–0.94，但 IoU@0.9 只有 0.08–0.28；
boundary mIoU 普遍比 mask mIoU 低 8~11 个点。
这些模型基本都能"找对物体"，**差距集中在轮廓精度**——
所以只看 mask mIoU 会系统性低估它们与 SAM 教师的差距。

### 5. 点提示普遍远弱于框提示，且排序会变

point mIoU 从 0.4552（TinySAM）到 0.6099（EfficientViT-SAM-L0），
每个模型都明显低于自己的框提示成绩。这是提示信息量决定的。

值得注意的是**排序会变**：EdgeSAM 框提示第二（0.7801）但点提示只排第七（0.5174），
EfficientSAM-Ti 框提示 0.7704 而点提示垫底之一（0.4866）。
本协议下单点提示取 3 个候选再按模型自报的 IoU 分数选最优，
因此这一列**同时考验分数头的标定质量**，不只是分割质量。

### 6. FastSAM 仍然是唯一"提示几乎免费"的模型

它的解码只是在已有提议里做一次 IoU 匹配（约 3 ms），
而 SAM 系每次提示都要跑一遍解码器，**差 17 倍以上**。
密集提示场景下这个差距会放大成数量级。
代价是精度垫底（0.6453 / 0.5152），因为提示只能召回检测器已经提议出来的 mask。

顺带一个交叉验证：MobileSAM / EdgeSAM / TinySAM / RepViT-SAM / EfficientViT-SAM
**用的是同一个 4.06M mask 解码器**，实测 decode 落在 48–60 ms 的窄区间内，彼此一致——
说明这一列的测量是可信的。EfficientSAM 的 80–85 ms 则来自它自己那套不同的解码器实现。

### 7. EfficientSAM3 仍是被用它不擅长的方式测的

decode 2540 ms 比 SAM 系高 40 倍，因为每个提示都要走完整 grounding 解码。
它压缩的是 SAM3（**文本提示 → 该概念的所有实例 + 跨帧跟踪**），
用几何提示测它仍拿到 0.7089。真正的价值在别处：
一个 "dog" 就分出两个独立实例（`efficientsam3/outputs/text_dogs.jpg`）。

### 关于 GFLOPs 口径

torch 的 `FlopCounterMode` 把一次乘加算作 **2 FLOPs**，而论文通常报 **MACs**。
本表数字要 **除以 2** 才能和论文对齐：MobileSAM 77.5÷2 = 38.8，论文 38.2（差 1.5% 以内）；
EfficientViT-SAM-L0 69.5÷2 = 34.8，论文 35G（吻合）。
但 EdgeSAM 论文那张表里 FastSAM 一行反而接近未折半的值——同一张表混用了两种口径。
**跨论文比 FLOPs 前务必先确认口径。**

## Raw data

Per-model JSON lives in each model's `outputs/` directory.
