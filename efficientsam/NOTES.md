# EfficientSAM

**EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything**
— Meta AI (Yunyang Xiong et al.), CVPR 2024.
[论文 arXiv:2312.00863](https://arxiv.org/abs/2312.00863) ·
[CVPR 开放获取](https://openaccess.thecvf.com/content/CVPR2024/html/Xiong_EfficientSAM_Leveraged_Masked_Image_Pretraining_for_Efficient_Segment_Anything_CVPR_2024_paper.html) ·
[代码](https://github.com/yformer/EfficientSAM)

## 是什么

和 MobileSAM / EdgeSAM 的"直接蒸馏编码器输出"不同，EfficientSAM 走的是**预训练**路线：
提出 **SAMI**（SAM-leveraged Masked Image pretraining）——
用 MAE 式的掩码图像建模，让轻量 ViT 学着**重建 SAM 图像编码器的特征**，
预训练完成后再在 SA-1B 上微调整个 SAM 流水线。

区别在于：蒸馏是"对齐同一张图上的输出"，SAMI 是"从被遮挡的图里恢复教师特征"，
后者被认为能学到更强的通用视觉表征，而不只是模仿教师在特定输入上的响应。

论文报告（ViTDet 框作为提示的 COCO/LVIS 零样本实例分割）：

| 模型 | COCO AP | LVIS AP |
|---|---|---|
| EfficientSAM-Ti | 42.3 | 39.9 |
| EfficientSAM-S | **44.4** | **42.3** |

论文称相比其他快速 SAM 变体有约 **+4 AP** 的增益。

## 权重

**两个 checkpoint 都在上游仓库里**，不用额外下载：

- `weights/efficient_sam_vitt.pt`（40 MB，Ti）——直接可用
- `weights/efficient_sam_vits.pt.zip`（S）——**是 zip**，因为解压后 101 MB 超过 GitHub 单文件 100 MB 限制

`setup.sh` 会自动解压。

## 装 / 跑

```bash
./setup.sh
uv run python bench.py --variant vitt     # 或 vits
```

## 坑 1：build 包装函数写死了相对路径

```python
def build_efficient_sam_vitt():
    return build_efficient_sam(..., checkpoint="weights/efficient_sam_vitt.pt").eval()
```

**没有 checkpoint 参数**，路径是硬编码的相对路径。所以只有当前工作目录恰好含 `weights/` 时才能用。
官方 example 因此得靠 `cd` 到仓库根目录。

本目录不 `chdir`（那会污染其他相对路径），而是**直接调用底层的 `build_efficient_sam()`**
并显式传 checkpoint：

```python
from efficient_sam.efficient_sam import build_efficient_sam
model = build_efficient_sam(encoder_patch_embed_dim=192, encoder_num_heads=3,
                            checkpoint="<任意路径>")   # Ti
model = build_efficient_sam(encoder_patch_embed_dim=384, encoder_num_heads=6,
                            checkpoint="<任意路径>")   # S
```

## 坑 2：没有 SamPredictor，框是用"两个角点"表示的

这是它和其他所有 SAM 变体最大的接口差异。EfficientSAM **不提供 `SamPredictor`**，
只有一个函数式模块：

```python
logits, iou = model(image, points, labels)
```

而且**框提示不是独立输入，而是编码成两个点**：

| label | 含义 |
|---|---|
| 1 | 前景点 |
| 0 | 背景点 |
| **2** | **框的左上角** |
| **3** | **框的右下角** |

```python
# 框 [x0,y0,x1,y1] 的正确表达
points = [[x0, y0], [x1, y1]]
labels = [2, 3]
```

这个约定只在 `notebooks/EfficientSAM_example.ipynb` 里出现过一次（`input_label = np.array([2,3])`），
README 完全没提。**用错了不会报错，只会得到一个奇怪的 mask。**

## 坑 3：候选 mask 不是按分数排序的

`model()` 的输出需要自己按 `iou` 排序才能拿到最优的那个——官方 example 里就是这么做的：

```python
sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
```

本目录的适配器直接 `argmax(iou)` 取最优，效果等价。

## 编码器/解码器拆分

它公开 API 里没有"设置图像"这一步，但内部是分开的，所以本目录直接调用
`get_image_embeddings()`（编码器）和 `predict_masks()`（提示解码器），
以便和其他模型报同一套 encoder / decoder 时延。这不是 hack——
模块本身就是这么组织的，官方 ONNX 导出脚本也是按这两段拆的。
