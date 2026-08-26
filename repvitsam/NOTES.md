# RepViT-SAM

**RepViT-SAM: Towards Real-Time Segmenting Anything** — 清华大学 THU-MIG, 2023.
[论文 arXiv:2312.05760](https://arxiv.org/abs/2312.05760) ·
[代码](https://github.com/THU-MIG/RepViT)

## 是什么

思路和 EdgeSAM 高度相似——**把 SAM 的 ViT 编码器换成纯 CNN 的 RepViT**，
理由也一样：移动端上自注意力的显存和计算开销让 MobileSAM 的 TinyViT 难以落地。

区别在规模：它用的是 **RepViT-M2.3**（较大的一档），整体 27.2M 参数，
而 EdgeSAM 用 RepViT-M1（9.58M）。所以两者是同一条技术路线上的不同尺寸点。

论文报告的零样本实例分割（与 SegInW）：

| 模型 | COCO AP | AP_S | AP_M | AP_L | SegInW Mean AP |
|---|---|---|---|---|---|
| ViT-H-SAM | **46.8** | 31.8 | 51.0 | **63.6** | **48.7** |
| ViT-B-SAM | 42.5 | 29.8 | 47.0 | 56.8 | 44.8 |
| MobileSAM | 42.7 | 27.0 | 46.5 | 61.1 | 43.9 |
| **RepViT-SAM** | **44.4** | 29.1 | 48.6 | 61.4 | 46.1 |

时延（Core ML，1024² 标准分辨率）：

| 平台 | RepViT-SAM | MobileSAM | ViT-B-SAM | 解码器 |
|---|---|---|---|---|
| iPhone 12 | **48.9 ms** | OOM | OOM | 11.6 ms |
| Macbook M1 Pro | **44.8 ms** | 482.2 ms | 6249.5 ms | 11.8 ms |

**MobileSAM 在 iPhone 上直接 OOM**，而 RepViT-SAM 能跑——
这正是"CNN 对移动端更友好"这个论点最有说服力的证据。
Macbook 上编码器快约 **10×**。

> ⚠️ 注意这张表里 MobileSAM 是 42.7 AP，而 EdgeSAM 论文的表里 MobileSAM 是 39.4。
> **两篇论文的评测协议不同，跨论文的 AP 不能直接比。**
> 这也是本仓库自己重测一遍的原因。

## 权重

GitHub release，可直接 curl：
`https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_sam.pt`（105 MB）。
`setup.sh` 自动下载。

## 装 / 跑

```bash
./setup.sh
uv run python bench.py
```

## 几乎零改动跑通

八个模型里第二个不需要打补丁的（另一个是 MobileSAM）。三点注意：

1. **源码在子目录里**：SAM 部分是 `repo/sam/repvit_sam/`，
   所以 `sys.path` 要加 `repo/sam` 而不是 `repo`。
   仓库根目录是 RepViT 主干（分类/检测/分割），SAM 只是其中一个应用。
2. **registry 名字是 `repvit`**：`sam_model_registry["repvit"]`，
   注册表里同时有 `vit_h/vit_l/vit_b/vit_t`（继承自 SAM），别选错。
3. 权重存的是 CPU state_dict，**不受** torch≥2.6 的 `weights_only` 变更影响，
   也没有 EdgeSAM/TinySAM 那个 CUDA `map_location` 问题。
