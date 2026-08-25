# MobileSAM

**Faster Segment Anything: Towards Lightweight SAM for Mobile Applications** — Kyung Hee University, 2023.
[论文 arXiv:2306.14289](https://arxiv.org/abs/2306.14289) ·
[代码](https://github.com/ChaoningZhang/MobileSAM)

## 是什么

保持 SAM 的整体架构不动，只把那个 632M 参数的 ViT-H 图像编码器换成 5M 的 **TinyViT**，
用**解耦蒸馏**训练：不端到端蒸馏，而是单独让学生编码器去对齐教师编码器的输出特征，
再直接接上 SAM 原版的 prompt encoder 和 mask decoder（两者冻结、直接复用）。

所以：

- **prompt encoder / mask decoder 与 SAM 完全一致**，SAM 的调用代码可以原样搬过来。
- 整体 10.1M 参数，其中编码器 6.1M、解码器 4.1M。注意**解码器占了 40%**，
  这是"只换编码器"路线的固有结构——编码器已经很小了，解码器就成了不可忽视的部分。

## 本目录的配置

- 环境：uv + Python 3.10 + torch 2.13.0 + timm 1.0.28
- 上游固定在 SHA `f706ad9c4eb7f219c00d9050e46328518ffb65d2`
- 权重 `mobile_sam.pt` (39 MB) **直接在上游仓库里**，不用额外下载

## 装 / 跑

```bash
./setup.sh
uv run python bench.py
uv run python demo.py
```

## 四个模型里唯一零代码改动跑通的

只有两条无害的 `FutureWarning`：`timm.models.layers` 和 `timm.models.registry` 在 timm 1.x 里已弃用，
但兼容 shim 还在，功能正常，**不需要降级到上游提到的 `timm==0.4.12`**。

另外会有几条 `Overwriting tiny_vit_5m_224 in registry` 的 UserWarning，
因为 timm 自己也注册了同名模型。无害。

顺带一提，它**不受 torch ≥2.6 的 `weights_only` 变更影响**——`mobile_sam.pt` 存的是纯 state_dict
而不是 pickle 下来的模型对象，所以 `weights_only=True` 也能读。
（FastSAM 就是反例，见 `../fastsam/NOTES.md`。）

## 用法

标准 SAM 接口：

```python
from mobile_sam import sam_model_registry, SamPredictor

sam = sam_model_registry["vit_t"](checkpoint="weights/mobile_sam.pt")
predictor = SamPredictor(sam)
predictor.set_image(rgb)                                    # 编码器，每张图一次
masks, scores, _ = predictor.predict(box=xyxy, multimask_output=False)
```

单点提示时用 `multimask_output=True` 再按 score 取最优：单个点在语义上确实是有歧义的
（部件/子部件/整体），SAM 的设计就是让模型给三个候选，取它自己打分最高的那个。
`bench.py` 的点提示走的就是这个标准协议。
