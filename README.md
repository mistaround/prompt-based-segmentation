# 可提示分割模型调研

把八个可提示分割（promptable segmentation）模型在同一台机器、同一套评测协议下从零跑通并横向对比。

| 模型 | 出处 | 年份 | 路线 |
|---|---|---|---|
| [FastSAM](fastsam/NOTES.md) | CASIA-IVA-Lab | 2023 | 换架构：YOLOv8-seg 一次提议全部 mask |
| [MobileSAM](mobilesam/NOTES.md) | Kyung Hee University | 2023 | 换编码器：TinyViT，解耦蒸馏 |
| [EdgeSAM](edgesam/NOTES.md) | NTU S-Lab | 2023 / IJCV 2025 | 换编码器 + prompt-in-the-loop 蒸馏（RepViT-M1） |
| [RepViT-SAM](repvitsam/NOTES.md) | 清华 THU-MIG | 2023 | 同路线更大一档（RepViT-M2.3） |
| [TinySAM](tinysam/NOTES.md) | 华为诺亚方舟 | AAAI 2025 | 同 TinyViT 同算力，全阶段蒸馏 + 难提示采样 |
| [EfficientSAM](efficientsam/NOTES.md) | Meta AI | CVPR 2024 | 换训练法：SAMI 掩码图像预训练 |
| [EfficientViT-SAM](efficientvitsam/NOTES.md) | MIT Han Lab | CVPRW 2024 | **非蒸馏**：EfficientViT + 完整 SA-1B 训练 |
| [EfficientSAM3](efficientsam3/NOTES.md) | Bristol / UvA / Edinburgh / SMU | 2025 | 压缩的是 **SAM3**：文本/概念提示 + 跟踪 |

八个模型都已完整评测，各自 `setup.sh` 会自动拉取权重
（GitHub release 资产 / 上游仓库内置 / HuggingFace，视模型而定）。

`bench.py` 是 weights-optional 的：检测不到 `weights/` 下的 checkpoint 时，
只报**与权重数值无关**的指标（参数量 / GFLOPs / 时延 / 峰值内存），
并在结果 JSON 里标 `weights_available: false`，**不会**用随机权重的 IoU 充数。
所以在拿不到某个权重的环境里，这套评测依然能跑出有意义的一半结果。

## 文档

| 文档 | 内容 |
|---|---|
| **[docs/EXPERIENCE.md](docs/EXPERIENCE.md)** | **调试经验记录**——每个模型踩到的坑、根因、修法 |
| [docs/RESULTS.md](docs/RESULTS.md) | 本仓库**实测**的精度与性能数字 |
| [docs/COMPARISON.md](docs/COMPARISON.md) | 论文**公开**数字的横向对比与选型建议 |
| `<模型>/NOTES.md` | 单个模型的说明、配置、坑与用法 |

## 目录结构

```
.
├── assets/                  演示图片
├── common/pbs_bench/        四个模型共用的评测框架（只依赖 numpy + opencv）
│   ├── data.py              coco128-seg 加载与多边形栅格化
│   ├── metrics.py           mask IoU / boundary IoU / 尺寸分桶
│   ├── runner.py            ModelAdapter 协议、时延测量、FLOPs、峰值内存
│   ├── viz.py               mask 可视化
│   ├── demo_common.py       demo 脚本共用逻辑
│   ├── sam_adapter.py       SAM 系预测器的共用适配器
├── datasets/                coco128-seg（scripts/get_dataset.sh 拉取，未入库）
├── docs/
├── scripts/
│   ├── get_dataset.sh       拉取统一测试集
│   ├── run_all.sh           串行跑完所有模型的评测
│   └── aggregate.py         汇总成 docs/RESULTS.md
└── <模型>/
    ├── pyproject.toml       独立 uv 环境（Python 3.10）
    ├── setup.sh             clone 固定 SHA → 打补丁 → 下权重 → uv sync
    ├── patches/             幂等的源码补丁（附原因说明）
    ├── repo/                上游源码（未入库，setup.sh 拉取）
    ├── weights/             权重（未入库）
    ├── bench.py             评测入口
    ├── demo.py              可视化 demo
    └── NOTES.md
```

**上游代码不入库**，而是由各自 `setup.sh` 按固定 SHA 拉取，
源码级不兼容用 `patches/apply_patches.py` 打上去（幂等，每条补丁都带原因说明）。
这样仓库干净、上游版本可追溯、改动意图有据可查。

## 快速开始

```bash
# 1. 统一测试集（coco128-seg，7.6 MB）
./scripts/get_dataset.sh

# 2. 逐个安装（clone + 打补丁 + 下权重 + uv sync）
for m in fastsam mobilesam edgesam repvitsam tinysam efficientsam efficientvitsam efficientsam3; do
  (cd $m && ./setup.sh)
done

# 3. 跑全部评测（串行，CPU 上约 2 小时）
./scripts/run_all.sh

# 4. 汇总
python3 scripts/aggregate.py     # → docs/RESULTS.md

# 5. 单个模型的可视化 demo
cd mobilesam && uv run python demo.py
```

## 环境

- **Python 3.10**（八个环境统一），uv 管理
- **torch 2.13.0**（八个环境统一钉同一版本）
  ——uv 从同一 cache 硬链接落盘，所以八个 venv 只占**一份**物理磁盘，
  而不是 8 × 5 GB。详见 [EXPERIENCE.md 第 1 节](docs/EXPERIENCE.md)。
- 纯 CPU（4 核 / 15 GB），无 GPU

> 因此本仓库的时延数字只能用作**模型间的相对排序**，
> 不能和论文里的 GPU / iPhone 数字直接比较。

## 评测协议

统一测试集：**coco128-seg**（128 张 COCO train2017 图 + 实例多边形标注）。
选它的原因是出网策略下 `cocodataset.org` 不可达，而它作为 GitHub release 资产可以拉到。

- 多边形栅格化成 GT mask，**再从这个 mask 反推紧致 box 作为提示**——
  保证提示和标注天然一致，不会因 COCO 的 bbox/mask 标注偏差污染 IoU
- 过滤 < 32×32 px 的实例（该尺度下栅格化误差已超过模型误差）
- 点提示用**距离变换最大值点**而非质心——环形/香蕉形 mask 的质心可能落在 mask 外，
  那样点提示的语义就反了
- 同时报 mask mIoU 和 **boundary mIoU**——mask IoU 被物体内部主导，
  轮廓糊一点看不出来；boundary IoU 才能区分蒸馏学生和 SAM 教师
- **编码器与解码器时延分开报**——图像编码器每图一次（大头），
  提示解码器每次点击一次（交互延迟）。合成一个端到端数字会抹掉最有意义的对比

局限：128 张图、几百个实例，**只够做模型间相对排序，不是论文级绝对精度**。

## 许可

各模型代码遵循各自上游许可（FastSAM / MobileSAM / EfficientSAM / EfficientSAM3: Apache-2.0；
EdgeSAM: S-Lab License 1.0；RepViT-SAM: Apache-2.0；TinySAM: Apache-2.0；
EfficientViT-SAM: Apache-2.0）。
本仓库自有的评测与封装代码可自由使用。
