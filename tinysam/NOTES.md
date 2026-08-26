# TinySAM

**TinySAM: Pushing the Envelope for Efficient Segment Anything Model**
— 华为诺亚方舟实验室, AAAI 2025.
[论文 arXiv:2312.13789](https://arxiv.org/abs/2312.13789) ·
[AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/34255/36410) ·
[代码](https://github.com/xinghaochen/TinySAM)

## 是什么

和 MobileSAM 用**同一个 TinyViT 编码器、同样的 42.0 GFLOPs**，靠三件事把精度往上顶：

1. **全阶段知识蒸馏 + 在线难提示采样**（online hard prompt sampling）——
   不只蒸馏编码器，而是全流程蒸馏，并且在训练中**主动挑当前学生答不好的提示**来学
2. **训练后量化（PTQ）** 适配到可提示分割任务 → Q-TinySAM，20.3 GFLOPs
3. **分层 everything 策略**——加速全图分割，几乎不掉点

关键在于第 1 点：**它和 MobileSAM 计算量完全相同，纯靠训练方法拿到 +1.3 COCO AP**。

论文报告：

| 模型 | GFLOPs | COCO AP | LVIS AP |
|---|---|---|---|
| SAM-H | 2976 | 46.6 | 44.7 |
| SAM-B | 487 | 43.4 | 40.8 |
| FastSAM | 344 | 37.9 | 34.5 |
| MobileSAM | 42.0 | 41.0 | 37.0 |
| **TinySAM** | **42.0** | **42.3** | **38.6** |
| Q-TinySAM | 20.3 | 41.4 | 37.2 |

## 权重

GitHub release，可直接 curl：
`https://github.com/xinghaochen/TinySAM/releases/download/3.0/tinysam_42.3.pth`（39 MB）。
`setup.sh` 自动下载。

## 装 / 跑

```bash
./setup.sh
uv run python bench.py
```

## 坑 1：权重是用 CUDA 存的

和 EdgeSAM 一模一样的问题——`build_sam.py` 里 `torch.load(f)` 没给 `map_location`：

```
RuntimeError: Attempting to deserialize object on a CUDA device
but torch.cuda.is_available() is False.
```

补丁改成 `torch.load(f, map_location="cpu")`（两处）。
八个模型里这是**第二个**踩到同一个坑的，见 `../docs/EXPERIENCE.md`。

## 坑 2：`predict()` 没有 multimask 参数，永远返回 3 个候选

```python
def predict(self, point_coords=None, point_labels=None,
            box=None, mask_input=None, return_logits=False)
```

对比其他模型：

| 模型 | 控制方式 |
|---|---|
| MobileSAM / RepViT-SAM / EfficientViT-SAM | `multimask_output: bool` |
| EdgeSAM | `num_multimask_outputs: int`（1/3/4） |
| **TinySAM** | **没有——固定返回 3 个** |

它内部调 `mask_decoder(...)` 时压根不传 `multimask_output`，走的是解码器默认值。
所以**框提示也拿不到"单个 mask"**，只能从 3 个里按模型自报的 IoU 分数挑最优。

本目录的共用适配器为此加了 `multimask_kw=None` 模式。
这是本协议下最接近"向其他模型要单个 mask"的做法，但**并不完全等价**——
它多了一层分数头选择，评测时值得记住这一点差异。
