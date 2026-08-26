# EfficientViT-SAM

**EfficientViT-SAM: Accelerated Segment Anything Model Without Performance Loss**
— MIT Han Lab, CVPRW 2024.
[论文](https://openaccess.thecvf.com/content/CVPR2024W/ELVM/papers/Zhang_EfficientViT-SAM_Accelerated_Segment_Anything_Model_Without_Performance_Loss_CVPRW_2024_paper.pdf) ·
[代码](https://github.com/mit-han-lab/efficientvit)

## 是什么

**这一组里唯一不是"蒸馏学生"的模型。** 其他几个都是从 SAM 蒸馏而来（通常只用 1%~3% 的 SA-1B），
EfficientViT-SAM 把编码器换成 EfficientViT 之后，**在完整 SA-1B 上端到端训练**。

代价是它明显更大——L0 就有 34.8M 参数，是 MobileSAM/EdgeSAM（约 10M）的 3.5 倍。
换来的是论文中 **COCO AP 45.7**，高于所有蒸馏学生，接近 SAM-ViT-H 的 46.x。

所以它不该被当成"另一个轻量 SAM"，而是**另一个尺度档位**：
在"接受 3 倍参数换 3 个 AP"这个取舍成立时才选它。

官方给出的变体（A100 上的时延）：

| 变体 | 分辨率 | COCO AP | LVIS AP | 参数 | GMac | A100 时延 |
|---|---|---|---|---|---|---|
| L0 | 512² | 45.7 | 41.8 | 34.8M | 35G | 8.2 ms |
| L1 | 512² | 46.2 | 42.1 | 47.7M | 49G | 10.2 ms |
| L2 | 512² | 46.6 | 42.7 | 61.3M | 69G | 12.9 ms |
| XL0 | 1024² | 47.5 | 43.9 | 117.0M | 185G | 22.5 ms |
| XL1 | 1024² | **47.8** | **44.4** | 203.3M | 322G | 37.2 ms |

本目录默认用 **L0**（最小、512² 输入），因为它是与其他模型最有可比性的那一档。

## 权重

HuggingFace：`https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_l0.pt`（133 MB）。
`setup.sh` 默认拉 l0，要别的档位传 tag：`./setup.sh l1`、`./setup.sh xl1`。

## 装 / 跑

```bash
./setup.sh
uv run python bench.py --model efficientvit-sam-l0
```

## 坑 1：仓库是个多应用 monorepo，依赖被拖累

`efficientvit` 仓库同时装着分类、分割、扩散模型（DC-AE）、GazeSAM 等应用，
而 `sam_model_zoo` 会经由 `apps.trainer` 把**训练/导出相关的模块**一并 import 进来。
于是只想跑 SAM 推理，却依次被这些模块级 import 挡住：

```
ModuleNotFoundError: No module named 'onnx'
ModuleNotFoundError: No module named 'onnxsim'
ModuleNotFoundError: No module named 'segment_anything'
```

处理方式是把这三个装上（都很轻），而不是改上游代码——
比给一个活跃的 monorepo 打补丁更稳。注意 `segment_anything` **不在 PyPI 上叫这个名字**，
要从 GitHub 装：

```toml
"segment-anything @ git+https://github.com/facebookresearch/segment-anything.git"
```

另外它的 `requirements.txt` 列了 `diffusers`、`torchdiffeq`、`wandb`、`gradio-*` 等一大堆，
**跑 SAM 推理完全用不到**，照抄会白装几个 GB。本目录只列实际需要的。

## 坑 2：`image_size` 是个二元组，不是标量

```python
model.image_size   # (1024, 512) —— (训练分辨率, 推理分辨率)
```

L0/L1/L2 **推理时是 512²**，不是其他 SAM 变体的 1024²。
取时延和 FLOPs 时要用第二个值，否则会按 1024² 去算，数字直接翻 4 倍。
本目录的适配器取 `model.image_size[-1]`。

## 坑 3：预处理用 torchvision，不吃负 stride

它的 `set_image` 走 torchvision 的 `ToTensor`，而 `bgr[:, :, ::-1]` 这种 BGR→RGB 翻转
产生的是**负 stride 视图**：

```
ValueError: At least one stride in the given numpy array is negative,
and tensors with negative strides are not currently supported.
```

其他几个模型内部用 numpy/cv2，不受影响。共用适配器现在统一 `np.ascontiguousarray(...)`。
