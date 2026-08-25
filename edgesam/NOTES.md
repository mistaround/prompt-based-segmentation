# EdgeSAM

**EdgeSAM: Prompt-In-the-Loop Distillation for On-Device Deployment of SAM** — NTU S-Lab, 2023 / IJCV 2025.
[论文 arXiv:2312.06660](https://arxiv.org/abs/2312.06660) ·
[IJCV](https://link.springer.com/article/10.1007/s11263-025-02562-9) ·
[代码](https://github.com/chongzhou96/EdgeSAM)

## 是什么

比 MobileSAM 更进一步：把 SAM 的 ViT 编码器换成**纯 CNN 的 RepViT-M1**（对移动端 NPU/CoreML 更友好），
并且**蒸馏时把 prompt encoder 和 mask decoder 一起放进循环**——
用框和点提示参与训练（"prompt-in-the-loop"），让学生学到"用户输入 → mask"这个动态关系，
而不只是对齐一层中间特征。

这是它和 MobileSAM 的核心区别：MobileSAM 只对齐编码器输出，EdgeSAM 对齐的是最终的提示-响应行为。
论文报告 COCO mAP 42.7（EdgeSAM-3x）对 SAM 的 46.1，是第一个在 iPhone 14 上跑到 30+ FPS 的 SAM 变体。

**本环境实测**（edge_sam_3x，121 图 / 524 实例，纯 CPU）：

| | EdgeSAM | MobileSAM |
|---|---|---|
| box mIoU | **0.7801** | 0.7664 |
| boundary mIoU | **0.6770** | 0.6589 |
| point mIoU | 0.5174 | **0.5676** |
| 编码器时延 | **379 ms** | 905 ms |
| 参数 | **9.58M** | 10.13M |
| 编码器 GFLOPs | **40.1** | 77.5 |

**框提示上 EdgeSAM 以更小、更快的模型胜过 MobileSAM**，方向与论文一致（论文报 COCO AP 高 2.8）。

但**点提示这一项它反而更低**（0.5174 vs 0.5676），和论文的说法不一致。
本协议下单点提示取 `num_multimask_outputs=3` 再按模型自报的 IoU 分数选最优，
所以这项同时考验分数头的标定质量；论文的点提示协议未必相同。
样本量也只有 524 个实例。**这一条按实测如实记录，不宜据此下结论。**

## 权重

官方 checkpoint 只发布在 HuggingFace Space（无 GitHub release、无 PyPI 包、无镜像）：

```
https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/edge_sam.pth      # 42.1 COCO mAP
https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/edge_sam_3x.pth   # 42.7 COCO mAP
```

`setup.sh` 会自动下载（各 38 MB）。`bench.py` 默认优先用 `edge_sam_3x.pth`。

若网络不通 HF，手动下好放进 `weights/` 即可——`bench.py` 会自动检测，
检测不到时只报与权重数值无关的指标（参数量 / GFLOPs / 时延 / 峰值内存），
并在结果 JSON 里写 `weights_available: false`，**不会用随机权重的 IoU 充数**。

## 装 / 跑

```bash
./setup.sh
uv run python bench.py                                  # 自动探测 weights/ 下的 checkpoint
uv run python bench.py --checkpoint weights/edge_sam_3x.pth
```

## 坑 1（最大的）：模块级 import mmdet

`edge_sam/modeling/sam.py` 在模块顶层无条件 import 了 mmdet / mmengine / `projects.EfficientDet`，
所以 **`import edge_sam` 直接崩**，哪怕只想跑推理。

而装 mmdet 代价极高：它依赖 mmcv，mmcv 的预编译 wheel 在 `download.openmmlab.com`（本环境不可达），
源码编译要针对当前 torch 编 C++/CUDA 扩展，4 核机器上几十分钟且大概率失败。

**关键观察**：那几个符号（`FPN` / `RPNHead` / `CenterNetUpdateHead` / `ConfigDict`）
只在 `if rpn:` 分支里用到，而官方发布的推理 checkpoint 根本不走这个分支。
所以 `patches/apply_patches.py` 把它改成惰性 import + 在用到的分支里现绑定。

改完 `import edge_sam` 干净通过，**完全不需要 mmcv**。

## 坑 2：权重是用 CUDA 存的

`build_sam.py` 里是 `torch.load(f)`，没给 `map_location`，
而官方发布的 `.pth` 是从 CUDA 张量存下来的，所以在 CPU 机器上直接崩：

```
RuntimeError: Attempting to deserialize object on a CUDA device
but torch.cuda.is_available() is False.
```

补丁改成 `torch.load(f, map_location="cpu")`，GPU 机器上也无副作用
（模型随后由调用方 `.to(device)`）。

对照：MobileSAM 和 EfficientSAM3 都没有这个问题。**这类坑只有真正拿到权重才会暴露。**

## 坑 3：`predict()` 签名和 SAM 不一样 —— 别用位置参数

```python
def predict(self, features=None, point_coords=None, point_labels=None,
            box=None, mask_input=None, num_multimask_outputs=3, ...)
```

两个陷阱：

1. **第一个位置参数是 `features` 而不是 `point_coords`**。
   照搬 SAM 的 `predictor.predict(coords, labels)` 会把坐标当特征图传进去。全部用关键字参数。
2. **是 `num_multimask_outputs: int`（可选 1/3/4），不是 SAM 的 `multimask_output: bool`**。
   传 `multimask_output=False` 直接 `TypeError`。框提示对应 `num_multimask_outputs=1`。

EdgeSAM 能出 1/3/4 个候选（SAM 只有 1 或 3），4 是它特有的。

## ONNX

上游也提供 ONNX 和 CoreML 导出（同样在 HF 上）。`onnxruntime` 已经装进环境，
拿到 `edge_sam_3x_encoder.onnx` / `_decoder.onnx` 后可以直接用 `edge_sam/onnx/predictor_onnx.py`。
