---
title: CV 模型 ONNX 导出与验证
created: 2026-01-29
updated: 2026-10-08
status: draft
blog: false
tags: [CV, YOLO, DEIMv2, EdgeCrafter, InsightFace, ONNX, 导出]
aliases: [/assets/tutorial/CV_ONNX导出与验证.html]
---

# CV 模型 ONNX 导出与验证

ONNX 是训练与平台编译之间的接口。它需要固定的不只是网络权重，还包括输入名称与 dtype、图像前处理、输出语义，以及哪些后处理已经包含在图中。不同模型都叫目标检测，也不代表它们共享输入输出。

本文承接 [CV 模型训练与微调](CV_训练与微调.md)，以静态 batch=1 建立可检查的导出基线。通过数值验证后，再进入 [多平台编译与部署](CV_多平台编译与部署.md)。

# 一、选择权重与导出边界

先确定使用哪份 checkpoint、普通模型还是 EMA、类别顺序以及模型规格。完整训练 checkpoint、纯 state_dict 和可直接加载的框架模型对象不是同一种文件，不能凭 .pt/.pth 后缀选择加载方式。

导出器至少需要知道：

| 信息 | 需要确定的内容 |
| --- | --- |
| 模型结构 | 家族、骨干、尺寸、任务类型、类别数 |
| 权重 | 参数所在键、EMA 选择、加载是否严格匹配 |
| 输入 | 图像尺寸、batch、额外元数据与 dtype |
| 输出 | 原始预测、解码结果、分数、掩码或关键点 |
| 后处理 | 是否包含 top-k、坐标缩放、NMS |

先调用 eval 并关闭训练期随机行为。仅将输入名称写进 ONNX，不会补齐前处理；仅修改 shape 声明，也不会让依赖固定 batch 的计算自动变成动态图。

# 二、按模型导出

## 2.1 YOLO 的静态基线

在可加载该权重的 Ultralytics 环境执行：

```bash
yolo export model=best.pt format=onnx imgsz=640 \
  batch=1 dynamic=False opset=17 simplify=False nms=False
```

这是 YOLO11 常规浮点检测权重的示例配置，opset 与导出选项需要结合模型版本和后端确认。先保留未简化的图，通过基线验证后再评估简化；不要把导出器自动选择的参数当作固定契约。

普通 Ultralytics 检测图常输出 `[N,4+nc,A]`，pose 还包含关键点，segment 通常增加 prototype。YOLO 的不同代际也可能改变输出和后处理方式，不能只根据“YOLO”名称套用同一个 decode。

## 2.2 DEIMv2 的部署图

现有导出工具使用配套网络定义，根据 checkpoint 参数结构识别模型规格和类别数，再构造 deploy model。图中包含模型专属 PostProcessor，对外返回 labels、boxes、scores。

编译工具包里的导出器可以单独执行。下面在转换工程根目录、已准备好的导出 Python 环境运行，仅生成 ONNX：

```bash
python TRT_Compile/toolkit/deimv2/scripts/export_onnx.py \
  /srv/models/deimv2-best.pth \
  /srv/export/deimv2.onnx \
  ""
```

最后的空字符串表示按识别出的模型规格使用默认尺寸。当前接口固定 batch=1；需要指定尺寸时用 `1,3,H,W`，并核对它与训练和位置编码配置是否兼容。

部署图有 `images` 和 `orig_target_sizes` 两个输入。后者为原图尺寸，本文对应导出器采用 `[W,H]` 顺序；其他导出器可能使用不同约定，必须从图内缩放逻辑确认。

## 2.3 EdgeCrafter 的任务输出

同一模型家族中的 detect、segment、pose 需要不同任务配置和输出名称，导出时明确指定任务：

```bash
python OM_Compile/toolkit/edgecrafter/scripts/export_onnx.py \
  --checkpoint /srv/models/ecdet-best.pth \
  --output /srv/export/ecdet.onnx \
  --task detect --input_shape 1,3,640,640
```

该示例导出检测模型；分割和姿态需要同时更换 checkpoint 与 task。导出器会选择相应网络定义，处理部署态转换和静态图所需的注意力路径。

| 任务 | 对外输出 |
| --- | --- |
| detect | labels、boxes、scores |
| segment | labels、boxes、scores、masks |
| pose | scores、labels、keypoints |

分割 masks 的空间尺寸与原图通常不同，姿态也不能从检测框输出中猜出关键点布局。以实际 ONNX 输出为准，并在验证时检查任务特有的还原过程。

## 2.4 已有 ONNX 的 InsightFace

已有 SCRFD 与 ArcFace ONNX 可以直接进入接口检查，无需为了统一流程再转回 PyTorch。SCRFD 输出多尺度检测结果与关键点，ArcFace 接收对齐人脸并输出 embedding。

两张图之间还存在检测解码、NMS 与五点对齐，不能把检测模型的整图输入直接送给识别模型。后续比对前对 embedding 做 L2 Normalize。几何缩放与对齐步骤同样属于模型流水线的一部分。

# 三、固定前处理和输出语义

| 模型路径 | 图像前处理 | 额外输入与后处理 |
| --- | --- | --- |
| YOLO11 常规检测 | RGB、Letterbox、/255、NCHW | 原始预测按该版本解码，再做筛选与 NMS |
| DEIMv2 DINOv3 系列 | RGB、直接 Resize、/255、ImageNet Normalize、NCHW | 原图尺寸；导出 PostProcessor 已输出检测结果 |
| DEIMv2 HGNetv2 系列 | RGB、直接 Resize、/255、NCHW | 同上，不追加 DINOv3 的 Normalize |
| EdgeCrafter | RGB、直接 Resize、/255、ImageNet Normalize、NCHW | 原图尺寸；按 detect/segment/pose 解析 |
| InsightFace | 检测与识别分别缩放、归一化；识别前先对齐 | SCRFD decode/NMS 与 ArcFace 特征归一化 |

ImageNet Normalize 使用 mean=[0.485,0.456,0.406]、std=[0.229,0.224,0.225]。这些数值不能作为所有 CV 模型的默认前处理。

Resize 和 Letterbox 都能产生相同 shape，但会改变几何关系。使用 Letterbox 时保存缩放比例与 padding；使用原图尺寸输入时选择非方形图片验证，防止 `[W,H]` 与 `[H,W]` 在方图上隐藏错误。DEIMv2 导出后的结果也不应再次执行 YOLO decode/NMS。

# 四、检查 ONNX 结构并保存真实输入

在安装了 ONNX 的环境检查模型路径及接口：

```python
import onnx

path = "model.onnx"
onnx.checker.check_model(path)
model = onnx.load(path, load_external_data=False)
initializers = {value.name for value in model.graph.initializer}
for kind, values in (("input", model.graph.input),
                     ("output", model.graph.output)):
    for value in values:
        if kind == "input" and value.name in initializers:
            continue
        tensor = value.type.tensor_type
        shape = [dim.dim_param or dim.dim_value for dim in tensor.shape.dim]
        print(kind, value.name, tensor.elem_type, shape)
```

checker 检查图格式与算子约束，不验证输入图像的颜色、数值范围或实际检测效果。外部权重文件要与 ONNX 一起保存，不能只复制 .onnx 主文件。

保存由正式前处理得到的输入 NPZ。双输入模型的例子：

```python
import numpy as np

# images 来自前处理；original_width/height 来自原图。
np.savez(
    "ref_inputs.npz",
    images=images.astype(np.float32),
    orig_target_sizes=np.array(
        [[original_width, original_height]], dtype=np.int64
    ),
)
```

固定输入用于比较 PyTorch、ONNX 与平台输出，避免每条验证路径各写一遍前处理后产生差异。用于平台端执行的输入可能已变成原始 RGB/YUV，后续还需按融合位置转换，不能不看接口就复用同一个数组。

# 五、建立数值与任务基线

用 CPU ONNX Runtime 保存参考输出：

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession(
    "model.onnx", providers=["CPUExecutionProvider"]
)
with np.load("ref_inputs.npz") as data:
    feed = {item.name: data[item.name] for item in session.get_inputs()}
names = [item.name for item in session.get_outputs()]
outputs = session.run(names, feed)
for name, value in zip(names, outputs):
    if np.issubdtype(value.dtype, np.floating):
        assert np.isfinite(value).all(), name
np.savez("ref_outputs.npz", **dict(zip(names, outputs)))
```

随后与同一输入下的 PyTorch 部署态输出比较。形状、dtype 与是否存在 NaN/Inf 是第一层；浮点误差与最终任务结果是下一层：

| 输出 | 比较内容 |
| --- | --- |
| 类别与检测分数 | 类别对应、top-k、阈值附近的差异 |
| 检测框 | 坐标系、原图映射、IoU |
| masks | logits/概率语义、插值与阈值后的区域 |
| keypoints | 点序、可见性与原图坐标 |
| embedding | L2 Normalize 后的余弦与排序 |

候选顺序改变时先按类别和几何关系匹配，再比较检测结果，不只逐下标比较框。整体余弦相近也不能替代对低幅值类别分数和阈值附近结果的检查。

如果做过图简化、注意力改写或采样算子替换，应将它作为独立版本重新跑同输入验证。通过后把 ONNX、外部权重、输入输出 NPZ、类别表和前处理说明交给 [多平台编译与部署](CV_多平台编译与部署.md)。

# 六、参考资料

- [Ultralytics ONNX 导出](https://docs.ultralytics.com/integrations/onnx/)
- [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2)
- [ONNX 外部权重与模型检查](https://onnx.ai/onnx/repo-docs/ExternalData.html)
- [ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html)
- [InsightFace](https://github.com/deepinsight/insightface)
