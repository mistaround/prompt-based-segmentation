# EfficientSAM3

**EfficientSAM3: Progressive Hierarchical Distillation for Video Concept Segmentation from SAM1, 2, and 3**
— University of Bristol / UvA / Edinburgh / SMU, 2025.
[论文 arXiv:2511.15833](https://arxiv.org/abs/2511.15833) ·
[代码](https://github.com/SimonZeng7108/efficientsam3) ·
[项目页](https://simonzeng7108.github.io/efficientsam3/)

## 是什么

和前三个不是一类东西。FastSAM / MobileSAM / EdgeSAM 压缩的是 **SAM1**（几何提示 → 单个 mask），
EfficientSAM3 压缩的是 **SAM3**（可提示概念分割 PCS：**文本提示 → 该概念的所有实例**，并可跨帧跟踪）。

用**渐进分层蒸馏（PHD）** 分三阶段：

1. **编码器蒸馏** — 在 SA-1B 上用 prompt-in-the-loop 对齐图像特征
2. **时序记忆蒸馏** — 用基于 Perceiver 的紧凑模块替换 SAM3 的稠密记忆，在 SA-V 上训练
3. **端到端微调** — 在官方 SAM3 PCS 数据上微调，保住概念级性能

发布了 9 个学生变体（RepViT / TinyViT / EfficientViT × S/M/L）。Stage-3 完整模型：

| 变体 | Vision | Text | Decoder | Other | 合计 | vs ImageSAM3 |
|---|---|---|---|---|---|---|
| EV-M | 22.2M | 42.5M | 21.0M | 3.5M | **89.2M** | 小 90% |
| RV-M | 25.6M | 42.5M | 21.0M | 3.5M | **92.7M** | 小 89% |
| TV-M | 28.3M | 42.5M | 21.0M | 3.5M | **95.3M** | 小 89% |
| *ImageSAM3（教师）* | 463M | 354M | 30.3M | 14.2M | *861.5M* | — |

注意 **文本编码器 42.5M 比视觉编码器还大**——这是"概念分割"路线的代价，
前三个模型完全没有这部分。

## 权重

Stage-3 checkpoint 只发布在 HuggingFace：

```
https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/efficientsam3_ft/efficientsam3_tinyvit.pt
https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/efficientsam3_ft/efficientsam3_repvit.pt
https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/efficientsam3_ft/efficientsam3_efficientvit.pt
```

每个约 470 MB。`setup.sh` 默认只下 tinyvit（TV-M），要别的变体就传名字：
`./setup.sh efficientsam3_repvit`。

若网络不通 HF，手动下好放进 `weights/` 即可——`bench.py` 会自动检测，
检测不到时只报与权重数值无关的指标，并写 `weights_available: false`。

另外，**建结构完全不需要联网**（BPE 词表随仓库分发），所以即使没有权重，
结构级指标也是实测的，不是抄的。

## 装 / 跑

```bash
./setup.sh
uv run python bench.py --backbone tinyvit          # 也可 repvit / efficientvit
```

## 坑 1：`requires-python = ">=3.12"` 是多写的

上游 `pyproject.toml` 声明 `>=3.12`，但它自己的 README 写的是 "Python 3.10+"。
实测 `sam3/` 下 **203 个 .py 文件在 3.10 上全部编译通过**（`py_compile`，0 失败）。

所以本目录**不用 `pip install -e .`**（会被 requires-python 卡住），
而是在自己的 `pyproject.toml` 里直接列依赖，再用 `sys.path` 指向 `repo/`。
既不改上游文件，也能在 3.10 上跑。

## 坑 2：包结构是套娃的

目录是 `repo/sam3/sam3/...`，外层 `repo/sam3/__init__.py` 是个 shim，靠改 `__path__` 把内层也挂上：

```python
__path__ = [str(_HERE), str(_HERE / "sam3")]
```

所以 **`sys.path` 要加 `repo/`（不是 `repo/sam3/`）**，然后 `from sam3.model_builder import ...` 才对。

## 坑 3：两个几何提示 API 的坐标约定不一致

这是最容易**静默算错**的地方——传错不报错，只是 mask 完全无关：

```python
# add_geometric_prompt：归一化的 [中心x, 中心y, 宽, 高]
cxcywh = [(x0+x1)/2/W, (y0+y1)/2/H, (x1-x0)/W, (y1-y0)/H]
state = processor.add_geometric_prompt(cxcywh, True, state)

# add_point_prompt：像素坐标 [x, y]（内部自己归一化）
state = processor.add_point_prompt([x, y], 1, state)
```

同一个类里两套约定，务必看清。

## 坑 4：每个提示都很贵

不像 SAM 系"编码器跑一次、解码器每提示很便宜"，EfficientSAM3 的**每个几何提示都要走一遍完整 grounding 解码**，
而且在没有文本提示时还会先跑一次文本编码器去编码 `"visual"` 这个占位词。

所以它的 `decode ms/prompt` 比 MobileSAM / EdgeSAM 高一到两个数量级。
**这是架构决定的，不是实现问题**——它换来的是前三个模型都没有的能力：
文本/概念提示，一次返回该概念的所有实例，并可跨帧跟踪。

拿它和 MobileSAM 比"每次点击的延迟"是不公平的比较；
真正对等的比较对象是 SAM3 教师本身（861.5M 参数）。

## 正常输出

构建时会打印 `Resizing positional embeddings from 77 to 16`，
是 MobileCLIP 文本编码器按 `context_length=16` 截断位置编码，属正常。

## 文本提示用法

```python
state = processor.set_image(pil_image)
state = processor.set_text_prompt("dog", state)
masks, scores = state["masks"], state["scores"]     # 该概念的所有实例
```
