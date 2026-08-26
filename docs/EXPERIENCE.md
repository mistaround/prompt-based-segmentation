# 调试经验记录

把八个可提示分割模型在同一台机器上从零跑通的过程记录：
FastSAM / MobileSAM / EdgeSAM / RepViT-SAM / TinySAM / EfficientSAM / EfficientViT-SAM / EfficientSAM3。
每一条都是实际踩到并解决掉的，不是照抄 README。

环境：Linux x86_64、**纯 CPU（4 核 / 15 GB 内存）**、Python 3.10、uv。

---

## 0. 最先要确认的：出网策略

这一条排第一，因为它决定了后面所有方案的可行性。

> **更新**：本次调研中途 `huggingface.co` 及其 CDN（`us.aws.cdn.hf.co`）已被放行，
> EdgeSAM 与 EfficientSAM3 的权重已成功下载，**四个模型现在都有完整精度数字**。
> 下面这张表保留的是最初的状态，因为由它推导出的几个设计决定（统一 torch 版本、
> 改用 coco128-seg、weights-optional 的评测脚本）仍然是当前代码的形态。

最初的 egress proxy 只放行：

| 可达 | 被封（CONNECT 返回 403） |
|---|---|
| `github.com`（含 releases 资产、git clone） | `huggingface.co` |
| `pypi.org` / `files.pythonhosted.org` | `download.pytorch.org` |
| | `cocodataset.org` / `images.cocodataset.org` |
| | `dl.fbaipublicfiles.com`（SAM 原版权重） |
| | `arxiv.org`、`hf-mirror.com`、`modelscope.cn` |

**排查方法**（不要靠猜，也不要试图绕过）：

```bash
curl -sS "$HTTPS_PROXY/__agentproxy/status"     # 会列出最近的 connect_rejected 记录
```

直接后果，按影响排序：

1. **EdgeSAM 和 EfficientSAM3 的权重只发布在 HuggingFace 上**（已解决，见上方更新）。
   当时确认过没有替代镜像：PyPI 上没有 `edge-sam` / `efficientsam3` 包，
   `ultralytics/assets` release 里也只有 `mobile_sam.pt` / `sam_b.pt` / `FastSAM-*.pt`，没有 `edge_sam*.pth`。
   **教训**：放行 HF 时只加 `huggingface.co` 不够——`resolve/main/...` 会 302 跳到
   CDN 域名（本次是 `us.aws.cdn.hf.co`），CDN 不放行的话握手能过但文件仍然拉不下来。
2. **拿不到 `+cpu` 版 torch**（那只在 `download.pytorch.org` 上）。只能装 PyPI 上的默认 Linux wheel，
   它会连带拉进 `nvidia-*` 一堆 CUDA 运行时，单个环境约 5 GB —— 在没有 GPU 的机器上这些完全用不到。
3. **COCO val2017 下不了**，所以统一测试集改用 `coco128-seg`（见第 5 节）。

> `bench.py` 全程设计成 weights-optional：检测到 `weights/` 下有 checkpoint 就跑完整评测，
> 没有就只报与权重无关的指标。所以权重到位后**一行代码都没改**，直接重跑即可。

---

## 1. uv 环境：八个环境，但 torch 只占一份磁盘

各模型对 torch 的要求本来是冲突的（EdgeSAM 锁 `torch==2.0.0`，EfficientSAM3 要 `>=2.7`），
所以必须环境隔离。但八份 torch = 40 GB，磁盘只有 30 GB。

**做法：八个环境全部锁同一个 `torch==2.13.0`。**
uv 默认从同一个 cache 用 **硬链接** 落盘，版本一致时所有 venv 只占一份物理空间。

实测：

```
第一个环境 (fastsam)   .venv 5.0 GB，磁盘可用 30G → 25G
第二个环境 (mobilesam) .venv 5.2 GB，磁盘可用 25G → 25G   ← 几乎不额外占用
后续环境同理：扩到八个环境后，四个新环境总共只多占约 1 GB（各自的独有依赖）。
```

所以 **"venv 目录 5 GB × 8" 是假象**，`du` 会把硬链接重复计数，看 `df` 才准。

几个副作用：
- EdgeSAM 官方 `requirements.txt` 钉 `torch==2.0.0`，实测在 2.13.0 上推理完全正常，那个钉子可以不管。
- 老代码 + 新 torch 会踩到 `weights_only` 的坑，见第 2 节。

关于 Python 版本：**torch 2.7 ~ 2.13 都有 cp310 的 Linux wheel**，所以用户要求的 Python 3.10 完全没问题。
（查证方式：读 `https://pypi.org/pypi/torch/json`，注意文件名是 `manylinux_2_28_x86_64` 而不是 `linux_x86_64`，
用错子串会误判成"没有 cp310 wheel"。）

## 1.1 不 vendor 上游代码

四个上游仓库没有提交进本仓库，而是各自 `setup.sh` 按 **固定 SHA** 拉取，
源码级的不兼容用幂等的 `patches/apply_patches.py` 打上去。好处是仓库干净、上游版本可追溯、补丁意图有文字说明。

| 模型 | 固定 SHA |
|---|---|
| FastSAM | `b4ed20c2fed75eadc5aa7d8b09fedd137b873b52` |
| MobileSAM | `f706ad9c4eb7f219c00d9050e46328518ffb65d2` |
| EdgeSAM | `d24d99671f41a9c0003061248bded64a481e9059` |
| EfficientSAM3 | `bd0936c788fed8d51fa799437f05abd97b401b06` |
| EfficientSAM | `d525f622e6f640acf5a0fc37c7ca1f243da5bde0` |
| EfficientViT-SAM | `de7d7733cc0329f391b33f1f459271562ec27bd5` |
| RepViT-SAM | `298f42075eda5d2e6102559fad260c970769d34e` |
| TinySAM | `11589bc1d98c16cff046c31d5ad4cd90a30f0897` |

---

## 2. FastSAM

### 2.1 权重不用走 Google Drive

README 给的是 Google Drive 链接（脚本化很麻烦）。同样的 checkpoint 在 ultralytics 的 release 资产上有，可以直接 curl：

```
https://github.com/ultralytics/assets/releases/download/v8.3.0/FastSAM-x.pt   # 145 MB
https://github.com/ultralytics/assets/releases/download/v8.3.0/FastSAM-s.pt   # 24 MB
```

### 2.2 `ModuleNotFoundError: No module named 'pkg_resources'`

FastSAM 仓库里内置了一份 ultralytics 8.0.120 的 fork，它在 `checks.py` 模块级 `import pkg_resources`。
而 **setuptools >= 81 已经把 `pkg_resources` 移除了**。

修法：在依赖里钉 `setuptools<81`。不需要改代码。

### 2.3 `_pickle.UnpicklingError: Weights only load failed`

**torch >= 2.6 把 `torch.load` 的 `weights_only` 默认值从 `False` 翻成了 `True`。**
FastSAM 的 `.pt` 是整个 pickle 下来的 `SegmentationModel` 对象（不是纯 state_dict），所以直接失败。

修法（`patches/apply_patches.py` 里两处）：

```python
torch.load(file, map_location='cpu', weights_only=False)
```

对照：**MobileSAM 不受影响**，因为它的 `mobile_sam.pt` 存的是纯 state_dict，`weights_only=True` 照样能读。
判断依据就是 checkpoint 里存的是对象还是 dict。

### 2.4 `'FigureCanvasAgg' object has no attribute 'tostring_rgb'`

只在可视化时触发。**matplotlib >= 3.10 删掉了 `Canvas.tostring_rgb()`**，替代是 `buffer_rgba()`。
注意 `buffer_rgba()` 出来是 4 通道，reshape 时要按 4 通道来再切掉 alpha，否则会得到一张错位的花屏图：

```python
buf = fig.canvas.buffer_rgba()
img = np.frombuffer(buf, np.uint8).reshape(rows, cols, 4)[:, :, :3]
```

### 2.5 API 上两个容易静默出错的地方

- **`box_prompt(bbox=...)` 收的是 XYXY**，尽管 CLI 文档写的是 xywh（转换在 CLI 层做的）。传错不会报错，只会 IoU 很低。
- **`box_prompt` 会原地修改传进去的 list**（内部做坐标裁剪）。如果把同一个 list 复用于多次调用，第二次开始结果就是错的。
  适配器里每次传一份新的 copy。

### 2.6 文本提示没验证

`text_prompt()` 需要 OpenAI CLIP，它的权重从 `openaipublic.azureedge.net` 下载 —— 本环境不可达，所以这条路径没跑通。
`clip` 包本身能从 github 装上，只是拿不到权重。

---

## 3. MobileSAM

**四个里唯一一个零改动跑通的。** 权重 (`weights/mobile_sam.pt`, 39 MB) 直接在仓库里，不用额外下载。

唯一噪音是 timm 的两条 `FutureWarning`：`timm.models.layers` 和 `timm.models.registry` 在 timm 1.x 里已弃用。
但兼容 shim 还在，timm 1.0.28 下功能正常，不用降级到 `timm==0.4.12`。

一个小提醒：`register_model` 会打印几条 "Overwriting tiny_vit_5m_224 in registry" 的 UserWarning，
原因是 timm 自己也注册了同名模型。无害。

---

## 4. EdgeSAM

### 4.1 `ModuleNotFoundError: No module named 'mmdet'`（最大的坑）

`edge_sam/modeling/sam.py` 在 **模块级** 无条件 import 了 mmdet / mmengine / `projects.EfficientDet`：

```python
from mmdet.models.dense_heads import RPNHead, CenterNetUpdateHead
from mmdet.models.necks import FPN
from projects.EfficientDet import efficientdet
from mmengine import ConfigDict
```

于是 `import edge_sam` 直接崩，哪怕只是想跑推理。

装 mmdet 又极其不划算：它依赖 mmcv，而 **mmcv 的预编译 wheel 在 `download.openmmlab.com`（不可达）**，
从源码编译要针对当前 torch 版本编 CUDA/C++ 扩展，4 核机器上几十分钟起步且很可能失败。

**关键观察**：这几个符号只在 `if rpn:` 分支里用到，而官方发布的 `edge_sam` / `edge_sam_3x` 推理 checkpoint 根本不走这个分支。
所以把 import 改成惰性的就行：

```python
def _import_mmdet():
    from mmdet.models.dense_heads import RPNHead, CenterNetUpdateHead
    from mmdet.models.necks import FPN
    from projects.EfficientDet import efficientdet
    from mmengine import ConfigDict
    return RPNHead, CenterNetUpdateHead, FPN, ConfigDict, efficientdet
```

再在每个用到这些名字的分支开头加一行绑定。改完 `import edge_sam` 干净通过，**完全不需要 mmcv**。

（写 patch 时被坑了一次：最初的 anchor 漏了中间那行 `from projects.EfficientDet import efficientdet`，
导致字符串匹配不上。改 patch 一定要 `cat -A` 看原文的确切行。）

### 4.2 `predict()` 的签名和 SAM 不一样 —— 千万别用位置参数

```python
def predict(self, features=None, point_coords=None, point_labels=None,
            box=None, mask_input=None, num_multimask_outputs=3, ...)
```

两个陷阱：

1. **第一个位置参数是 `features`，不是 `point_coords`**。照搬 SAM 的 `predictor.predict(coords, labels)` 会把坐标当特征图传进去。
   全部用关键字参数。
2. **是 `num_multimask_outputs: int`（可选 1/3/4），不是 SAM 的 `multimask_output: bool`**。
   传 `multimask_output=False` 会直接 `TypeError`。box 提示对应 `num_multimask_outputs=1`。

### 4.3 权重是用 CUDA 存的，CPU 机器直接加载会崩

拿到权重后立刻踩到的第二个坑：

```
RuntimeError: Attempting to deserialize object on a CUDA device
but torch.cuda.is_available() is False.
```

`build_sam.py` 里是 `torch.load(f)`，没给 `map_location`，
而官方发布的 `edge_sam.pth` / `edge_sam_3x.pth` 是从 CUDA 张量存下来的。
补丁改成 `torch.load(f, map_location="cpu")` 即可，在 GPU 机器上也无副作用
（模型随后由调用方 `.to(device)`）。

对照：**MobileSAM 和 EfficientSAM3 的权重都没有这个问题**——
前者存的是 CPU state_dict，后者的 builder 自己传了 `map_location`。
这类问题只有真正拿到权重才会暴露，光看代码看不出来。

### 4.4 权重缺失下仍然可测的部分

参数量、GFLOPs、encoder/decoder 时延、峰值内存 **只取决于网络结构，不取决于权重数值**，
所以随机初始化下这些指标全部有效。只有精度不行。`bench.py` 因此设计成 weights-optional，
并在结果里写 `weights_available: false`，避免把随机权重的 IoU 误当成真实精度。

---

## 5. EfficientSAM3

### 5.1 `requires-python = ">=3.12"` 是多写的

上游 `pyproject.toml` 声明 `>=3.12`，但 README 自己写的是 "Python 3.10+"。
实测：`sam3/` 下 **203 个 .py 文件在 3.10 上全部编译通过**（`py_compile`，0 失败）。

```bash
python3.10 -c "
import pathlib,py_compile,tempfile
bad=[p for p in pathlib.Path('repo/sam3').rglob('*.py')
     if not py_compile.compile(str(p),cfile=tempfile.mktemp(),doraise=False)]
print(len(bad))"
```

所以不要 `pip install -e .`（会被 requires-python 卡住），而是**在自己的 pyproject 里直接列依赖**，
再用 `sys.path` 指向 `repo/`。这样既不改上游文件，也能在 3.10 上跑。

### 5.2 包结构是套娃的

目录是 `repo/sam3/sam3/...`，外层 `repo/sam3/__init__.py` 是个 shim，靠改 `__path__` 把内层目录也挂上：

```python
__path__ = [str(_HERE), str(_HERE / "sam3")]
```

所以 **`sys.path` 要加的是 `repo/`（不是 `repo/sam3/`）**，然后 `from sam3.model_builder import ...` 才对。

### 5.3 `add_geometric_prompt` 收的是归一化 cxcywh

这是最容易静默算错的一处 —— 传 XYXY 像素坐标不会报错，只会得到完全无关的 mask：

```python
# 正确：归一化的 [中心x, 中心y, 宽, 高]
cxcywh = [(x0+x1)/2/W, (y0+y1)/2/H, (x1-x0)/W, (y1-y0)/H]
processor.add_geometric_prompt(cxcywh, True, state)
```

而同一个类里 **`add_point_prompt` 收的却是像素坐标 `[x, y]`**（内部自己做归一化）。两个 API 不一致，容易搞混。

### 5.4 构建模型不需要联网

`build_efficientsam3_image_model(..., load_from_HF=False, checkpoint_path=None)` 能在完全离线的情况下建出结构
（BPE 词表是仓库自带的）。这让"权重拿不到但结构指标照测"成为可能。

会打印一行 `Resizing positional embeddings from 77 to 16`，是 MobileCLIP 文本编码器按 `context_length=16` 截断位置编码，属正常。

### 5.5 每个提示都很贵

不像 SAM 系"编码器跑一次、解码器每个提示很便宜"，EfficientSAM3 的每个几何提示都要走一遍完整的 grounding 解码，
而且在没有文本提示时还会先跑一次文本编码器编码 `"visual"` 这个占位词。
所以它的 `decode ms/prompt` 比 MobileSAM / EdgeSAM 高一到两个数量级 —— 这是架构决定的，不是实现问题。
它换来的是 SAM 系没有的能力：**文本/概念提示，一次返回该概念的所有实例**。

---

## 6. 统一测试集的选择

原计划用 COCO val2017 box-prompted mIoU（各家论文的标准做法），但 `cocodataset.org` 不可达。

最后选了 **`coco128-seg`**：128 张 COCO train2017 图 + 实例多边形标注，以 GitHub release 资产发布，因此可达：

```
https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128-seg.zip   # 7.6 MB
```

评测协议的几个决定：

- **多边形栅格化成 GT mask，再从这个 mask 反推紧致 box 作为提示**。
  这样"提示"和"标注"天然一致，不会因为 COCO 的 bbox 标注和 mask 标注之间的偏差污染 IoU。
- **过滤掉小于 32×32 px 的实例**。这个尺度下多边形栅格化误差已经超过模型误差，测的是标注噪声而不是模型质量。
- **点提示用距离变换的最大值点，不用质心**。香蕉形/环形 mask 的质心可能落在 mask 外面，
  那样点提示的语义就反了（提示到背景）。距离变换最大点保证在 mask 内部，而且是最"不含糊"的一个点。
- **同时报 mask mIoU 和 boundary mIoU**。mask IoU 被物体内部主导，轮廓糊一点也看不出来；
  boundary IoU 才是能把蒸馏出来的学生模型和 SAM 教师区分开的指标。

局限性要说清楚：128 张图、几百个实例，**这个规模只够做模型间的相对排序，不能当成论文级的绝对精度**。
而且这些图来自 COCO train2017，对在 SA-1B 上训练的模型算 zero-shot，但严格说不是标准 benchmark split。

---

## 7. 时延测量的注意事项

- **必须串行跑**。机器只有 4 核，每个进程又 `torch.set_num_threads(4)`，并行跑会让所有时延数字互相污染。
  `scripts/run_all.sh` 因此是顺序执行的。
- **encoder 和 decoder 要分开报**。这几个模型的设计差异恰恰体现在这里：
  图像编码器每张图跑一次（大头），提示解码器每次点击跑一次（交互延迟）。
  合成一个"端到端时延"会把最有意义的对比抹掉。
- **取中位数不取均值**。共享机器上一次调度抖动就能把均值拉歪。
- **GFLOPs 不用额外装包**。torch 自带 `torch.utils.flop_counter.FlopCounterMode`，比 thop/fvcore 省一个依赖：

  ```python
  from torch.utils.flop_counter import FlopCounterMode
  counter = FlopCounterMode(display=False)
  with torch.no_grad(), counter:
      model(x)
  gflops = counter.get_total_flops() / 1e9
  ```

- **峰值内存用 `resource.getrusage(RUSAGE_SELF).ru_maxrss`**（Linux 上单位是 KB）。
  它是整个进程的高水位线，所以只有在跑完之后读才有意义。

---

## 7.5 时延必须"背靠背"测——这是本次最大的方法论教训

扩到八个模型后，第一版结果里出现了一个说不通的数字：
**MobileSAM 编码器 905 ms，TinySAM 1196 ms**——可这两个模型的编码器是**逐字节相同**的
（都是 TinyViT，6.0655M 参数、77.5 GFLOPs、模块类型计数完全一致，实测确认过）。
同样的架构不可能差 25%。

排查结论：**不是模型差异，是这台共享 4 核机器在不同批次之间的吞吐漂移。**
两者原本在不同批次里测（MobileSAM 在第一轮，TinySAM 在后来补测的一轮）。
背靠背连测两遍，结果立刻一致：

```
MobileSAM  median=1155.5   TinySAM  median=1207.2
MobileSAM  median=1172.7   TinySAM  median=1191.1
```

注意 MobileSAM 此时是 ~1160 ms，而它最初那轮测出来是 905 ms——**机器本身慢了约 25%**。

**处理方式**：把时延从精度运行里剥离出来，用 `scripts/run_latency.sh` 在**同一轮不间断的序列**
里重测全部十个变体，`docs/RESULTS.md` 的时延列只用这一轮的数字。
精度是确定性的，分批跑没问题；**时延不是**。

两个附带的坑：

1. **不要拿空白图测时延**。最初用 `np.zeros((1024,1024,3))`，结果 FastSAM 的 decode 测出 0.04 ms——
   因为空白图上它什么都检测不到，`box_prompt` 直接走空结果分支返回了。
   换成真实图片（`assets/dogs.jpg`）后是 ~3 ms，才是有意义的数字。
2. **短操作要多测几次**。decode 只有几十毫秒，8 次重复下调度抖动占主导，
   五个共用同一个 4.06M 解码器的模型测出 46~102 ms 的离散区间。
   提到 25 次重复后收敛到 48~60 ms，彼此一致——这反过来验证了测量的可信度。

---

## 7.6 权重的 CUDA/CPU 问题是个反复出现的类别

八个模型里**两个**（EdgeSAM、TinySAM）的官方 checkpoint 是从 CUDA 张量存下来的，
而它们的 `build_sam.py` 都写的是 `torch.load(f)`，没给 `map_location`：

```
RuntimeError: Attempting to deserialize object on a CUDA device
but torch.cuda.is_available() is False.
```

两个都用同样的补丁修（`map_location="cpu"`）。

**这类问题只有真正拿到权重才会暴露**——在只能做"结构级验证"的阶段，
代码 import 得通、模型建得出来、前向跑得动，看上去完全正常。
所以"跑通了"这个说法要区分两个层次：**结构跑通 ≠ 权重跑通**。

对照：MobileSAM / RepViT-SAM / EfficientSAM / EfficientViT-SAM / EfficientSAM3 都没有这个问题。

---

## 7.7 新增四个模型各自的坑（详见各目录 NOTES.md）

| 模型 | 坑 | 处理 |
|---|---|---|
| **EfficientSAM** | build 包装函数把 checkpoint 路径**硬编码成相对路径**，无参数可传 | 直接调底层 `build_efficient_sam()` 并显式传路径 |
| **EfficientSAM** | 无 `SamPredictor`；**框= 两个角点 + label `[2,3]`**，README 完全没写 | 见 NOTES；传错不报错，只是 mask 不对 |
| **EfficientViT-SAM** | monorepo，`sam_model_zoo` 会连带 import 训练/导出模块 | 装 `onnx` / `onnxsim` / `segment-anything`（后者只在 GitHub 上） |
| **EfficientViT-SAM** | `model.image_size` 是**二元组** `(1024, 512)`，L0/L1/L2 推理是 512² | 取 `[-1]`，否则 FLOPs 和时延都按 1024² 算，直接错 4 倍 |
| **EfficientViT-SAM** | 预处理走 torchvision，**不接受负 stride** | 共用适配器统一 `np.ascontiguousarray` |
| **TinySAM** | `predict()` **没有 multimask 参数**，固定返回 3 个候选 | 适配器加 `multimask_kw=None` 模式，按分数挑 |
| **RepViT-SAM** | SAM 部分在 `repo/sam/` 子目录，registry 名是 `repvit` | `sys.path` 加 `repo/sam` |

一个观察：**八个模型里只有 MobileSAM 和 RepViT-SAM 完全零改动跑通。**
其余六个都需要源码补丁或非平凡的依赖处理。

---

## 8. 复现步骤

```bash
# 1. 测试集
./scripts/get_dataset.sh

# 2. 每个模型各自装（会 clone 固定 SHA、打补丁、下权重、uv sync）
for m in fastsam mobilesam edgesam repvitsam tinysam efficientsam efficientvitsam efficientsam3; do
  (cd $m && ./setup.sh)
done

# 3. 精度评测（串行，CPU 上约 2 小时）
./scripts/run_all.sh

# 4. 时延单独一轮背靠背测（约 15 分钟）——见 7.5 节，这一步不能省
./scripts/run_latency.sh

# 5. 汇总成 docs/RESULTS.md
python3 scripts/aggregate.py
```

各模型的 `setup.sh` 会自动下载权重。若网络不通 HuggingFace，
手动下好放进对应的 `weights/` 目录即可，`bench.py` 会自动检测：

```bash
# EdgeSAM (38 MB each)
https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/edge_sam_3x.pth
# EfficientSAM3 (470 MB)
https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/efficientsam3_ft/efficientsam3_tinyvit.pt
```
