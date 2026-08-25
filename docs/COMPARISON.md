# 四个模型的对比（论文公开数字）

> **这一页全部是论文/官方仓库里报告的数字，不是本仓库测的。**
> 本仓库实测的结果在 [RESULTS.md](RESULTS.md)，两者不要混用——
> 硬件、数据集、提示来源、评测协议都不一样。

## 1. 血统：它们压缩的不是同一个东西

```
SAM (2023)  几何提示 → 单个 mask
  ├── FastSAM      换思路：YOLOv8-seg 一次提议所有 mask，提示退化成筛选
  ├── MobileSAM    换编码器：ViT-H(632M) → TinyViT(5M)，解耦蒸馏
  └── EdgeSAM      换编码器 + 换蒸馏方式：RepViT(CNN) + prompt-in-the-loop

SAM3 (2025) 文本/概念提示 → 该概念的所有实例 + 跨帧跟踪
  └── EfficientSAM3  渐进分层蒸馏(PHD)，视觉和文本编码器一起压
```

**这一点决定了它们不能直接放在一张表里排名。** 前三个和 EfficientSAM3 解决的是不同任务：
前者回答"这个框/点里的东西是什么形状"，后者回答"图里所有的『狗』在哪"。

## 2. COCO 实例分割（前三个 + SAM）

这是唯一一张四方（SAM / FastSAM / MobileSAM / EdgeSAM）在**同一协议下**的表，
出自 [EdgeSAM 官方仓库](https://github.com/chongzhou96/EdgeSAM)。
用 ViTDet-H（box mAP 58.7）提供框提示，报 mask mAP，FLOPs 按 1024×1024 输入算。

| 方法 | 训练数据 | COCO AP | AP_s | AP_m | AP_l | GFLOPs | 参数(M) | FPS iPhone14 | FPS 2080Ti |
|---|---|---|---|---|---|---|---|---|---|
| SAM (ViT-H) | SA-1B | **46.1** | 33.6 | 51.9 | 57.7 | 2734.8 | 641.1 | — | 4.3 |
| FastSAM | 2% SA-1B | 37.9 | 23.9 | 43.4 | 50.0 | 887.6 | 68.2 | — | 25.0* |
| MobileSAM | 1% SA-1B | 39.4 | 26.9 | 44.4 | 52.2 | 38.2 | 9.8 | 4.9 | 103.5 |
| EdgeSAM | 1% SA-1B | 42.2 | 29.6 | 47.6 | 53.9 | **22.1** | **9.6** | **38.7** | **164.3** |
| EdgeSAM-3x | 3% SA-1B | 42.7 | 30.0 | 48.6 | 54.5 | 22.1 | 9.6 | 38.7 | 164.3 |
| EdgeSAM-10x | 10% SA-1B | **43.0** | 30.3 | 48.9 | 55.1 | 22.1 | 9.6 | 38.7 | 164.3 |

`*` 标注的数字由 MobileSAM 作者测得。

**读这张表的几个要点：**

- **EdgeSAM 在参数量和 MobileSAM 基本相同（9.6 vs 9.8M）的前提下，AP 高 2.8，GFLOPs 只有 42%。**
  差距来自蒸馏方式（prompt-in-the-loop）而不是模型大小——这是 EdgeSAM 最有说服力的一点。
- **iPhone 14 上的 FPS 差了近 8 倍（38.7 vs 4.9）**，而 2080Ti 上只差 1.6 倍。
  说明 RepViT 这种纯 CNN 结构对移动端 NPU 的友好程度，在服务器 GPU 上是体现不出来的。
  选型时如果只看服务器 benchmark，会严重低估 EdgeSAM 的优势。
- **FastSAM 参数量最大（68.2M）、GFLOPs 第二高（887.6），AP 却最低。**
  它的优势不在这张表里——见下一节。

## 3. FastSAM 的优势在别处

出自 [FastSAM 仓库](https://github.com/CASIA-IVA-Lab/FastSAM)（RTX 3090）：

**提示数量不影响耗时**——这是它的架构特性：

| 方法 | 参数 | 1点 | 10点 | 100点 | E(16×16) | E(32×32) | E(64×64) |
|---|---|---|---|---|---|---|---|
| SAM-H | 0.6G | 446 | 464 | 627 | 852 | 2099 | 6972 |
| SAM-B | 136M | 110 | 125 | 230 | 432 | 1383 | 5417 |
| FastSAM | 68M | **40** | **40** | **40** | **40** | **40** | **40** |

单位 ms。因为所有 mask 在一次前向里就生成完了，提示只是筛选。
**要密集提示（比如全图 everything 模式）时，FastSAM 是这几个里唯一不会爆炸的。**

物体提议召回率上它甚至超过 SAM：

| COCO | AR10 | AR100 | AR1000 |
|---|---|---|---|
| SAM-H E64 | 15.5 | 45.6 | **67.7** |
| FastSAM | 15.7 | 47.3 | 63.7 |

| LVIS bbox AR@1000 | all | small | med | large |
|---|---|---|---|---|
| SAM-H E32 | 50.3 | 33.1 | 76.2 | **89.8** |
| FastSAM | **57.1** | **44.3** | **77.1** | 85.3 |

显存：COCO2017 上 FastSAM 2608 MB vs SAM-H 7060 MB。

## 4. MobileSAM 官方数字

出自 [MobileSAM 仓库](https://github.com/ChaoningZhang/MobileSAM)（单卡 GPU）：

| 部件 | 原版 SAM | MobileSAM |
|---|---|---|
| 图像编码器 参数 / 速度 | 611M / 452ms | 5M / **8ms** |
| mask 解码器 参数 / 速度 | 3.876M / 4ms | 3.876M / 4ms（**完全相同**） |
| 整条流水线 | 615M / 456ms | 9.66M / **12ms** |

解码器和 SAM 逐字节相同——这就是"SAM 的代码几乎零改动就能换过来"的原因。

MobileSAM 作者对 FastSAM 的对比（两点提示下与原版 SAM 的 mIoU 对齐度）：

| 两点像素距离 | FastSAM | MobileSAM |
|---|---|---|
| 100 | 0.27 | **0.73** |
| 200 | 0.33 | **0.71** |

**点提示下 FastSAM 和 SAM 的对齐度很差**，这和本仓库实测的现象一致
（见 [RESULTS.md](RESULTS.md)，FastSAM 点提示 mIoU 显著低于框提示）。
原因就是第 3 节说的：点提示只能命中检测器已经提出来的 mask。

## 5. EfficientSAM3

出自 [论文 arXiv:2511.15833](https://arxiv.org/abs/2511.15833) 和
[官方仓库](https://github.com/SimonZeng7108/efficientsam3)。

Stage-3 完整模型的参数构成：

| 变体 | Vision | Text | Decoder | Other | 合计 | vs ImageSAM3 |
|---|---|---|---|---|---|---|
| EV-M (EfficientViT) | 22.2M | 42.5M | 21.0M | 3.5M | **89.2M** | 小 **90%** |
| RV-M (RepViT) | 25.6M | 42.5M | 21.0M | 3.5M | **92.7M** | 小 89% |
| TV-M (TinyViT) | 28.3M | 42.5M | 21.0M | 3.5M | **95.3M** | 小 89% |
| *ImageSAM3（教师）* | *463M* | *354M* | *30.3M* | *14.2M* | *861.5M* | — |

**文本编码器 42.5M 比视觉编码器还大**，这是概念分割路线独有的开销，前三个模型完全没有。

三阶段蒸馏：编码器蒸馏（SA-1B，prompt-in-the-loop）→ 时序记忆蒸馏（SA-V，Perceiver 压缩记忆）
→ 端到端微调（官方 SAM3 PCS 数据）。共发布 9 个学生变体（RepViT/TinyViT/EfficientViT × S/M/L）。

配套的 SAM3-LiteText 只换文本编码器、保留 SAM3 的 ViT-H 视觉编码器，
文本部分缩小 88%，整体 550M（小 36%），已被 ICMR2026 接收。

## 6. 选型建议

| 场景 | 选谁 | 理由 |
|---|---|---|
| 移动端/边缘设备交互式标注 | **EdgeSAM** | iPhone 30+ FPS，参数与 MobileSAM 持平但 AP 高 2.8，GFLOPs 只有 42% |
| 已有 SAM 代码，想直接提速 | **MobileSAM** | 解码器与 SAM 完全相同，接口零改动 |
| 全图密集分割 / everything 模式 | **FastSAM** | 耗时与提示数无关，密集提示下唯一不爆炸的 |
| 文本提示、概念级分割、视频跟踪 | **EfficientSAM3** | 唯一具备该能力；但每次提示的开销高一到两个数量级 |
| 精度优先、算力不限 | 原版 SAM / SAM3 | 学生模型都还有 3~4 个 AP 的差距 |

一个容易被忽略的点：**FastSAM 参数量（68M）是 MobileSAM/EdgeSAM（~10M）的 7 倍**，
"轻量"这个词对它其实不成立——它快是因为架构（一次前向出全部 mask），不是因为小。
