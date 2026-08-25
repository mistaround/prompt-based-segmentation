# FastSAM

**Fast Segment Anything** — CASIA-IVA-Lab, 2023.
[论文 arXiv:2306.12156](https://arxiv.org/abs/2306.12156) ·
[代码](https://github.com/CASIA-IVA-Lab/FastSAM)

## 是什么

不是 SAM 的蒸馏版，而是**换了个思路**：用 YOLOv8-seg 这个纯 CNN 实例分割器，
在 SA-1B 的 2% 数据上训练，一次前向就把图里所有 mask 都提议出来（"everything 模式"）。
所谓"提示"只是在这些已有提议里做一次筛选。

这个设计决定了它的性能画像：

- **提示本身几乎免费**（一次 IoU 匹配，实测 ~1.6 ms/提示），跟提示数量无关。
  论文里 1 个点和 64×64 个点的耗时都是 40 ms。
- **但精度有天花板**：提示只能召回检测器已经提出来的 mask。
  置信度阈值 (`conf`) 以下的物体根本没有提议，再怎么提示也拿不到。这也是它点提示远弱于框提示的原因。

## 本目录的配置

- 环境：uv + Python 3.10 + torch 2.13.0
- 上游固定在 SHA `b4ed20c2fed75eadc5aa7d8b09fedd137b873b52`
- 权重从 ultralytics release 资产拉（不走 README 里的 Google Drive）：
  `FastSAM-x.pt` (145 MB, YOLOv8x) 和 `FastSAM-s.pt` (24 MB, YOLOv8s)

## 装 / 跑

```bash
./setup.sh                                   # clone + 打补丁 + 下权重 + uv sync
uv run python bench.py --model weights/FastSAM-x.pt
uv run python demo.py                        # 生成可视化到 outputs/
```

## 需要打的补丁（`patches/apply_patches.py`，幂等）

| 问题 | 原因 | 处理 |
|---|---|---|
| `No module named 'pkg_resources'` | setuptools ≥81 移除了它，内置的 ultralytics 8.0.120 却在模块级 import | 依赖里钉 `setuptools<81`（无需改码） |
| `UnpicklingError: Weights only load failed` | torch ≥2.6 把 `torch.load` 的 `weights_only` 默认翻成 `True`；FastSAM 的 `.pt` 是整个 pickle 的模型对象 | 两处 `torch.load(..., weights_only=False)` |
| `'FigureCanvasAgg' has no attribute 'tostring_rgb'` | matplotlib ≥3.10 删了该方法 | 改用 `buffer_rgba()`，按 4 通道 reshape 后切掉 alpha |

## API 上两个静默出错点

```python
# 1. box_prompt 收的是 XYXY，尽管 CLI 文档写 xywh（转换在 CLI 层）。
#    传成 xywh 不会报错，只会 IoU 很低。
ann = prompt_process.box_prompt(bbox=[x0, y0, x1, y1])

# 2. box_prompt 会原地改传进去的 list（内部做坐标裁剪）。
#    复用同一个 list 多次调用，第二次起结果就是错的 —— 每次传新 copy。
```

## 未验证

`text_prompt()` 需要 OpenAI CLIP 的权重，来源 `openaipublic.azureedge.net` 在本环境被出网策略挡住，
所以文本提示这条路径没跑通。`clip` 包本身可以从 GitHub 安装。

## 参数选择

`bench.py` 用的是 README 的默认值 `imgsz=1024, conf=0.4, iou=0.9`。
`conf` 直接影响召回：调低能提高小物体的框提示 IoU，代价是更多误检 mask。
