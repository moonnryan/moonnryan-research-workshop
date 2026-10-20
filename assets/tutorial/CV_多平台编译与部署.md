---
title: CV 模型多平台编译与部署
created: 2026-01-29
updated: 2026-10-08
status: draft
blog: false
aliases: [/assets/tutorial/YOLO_多平台训练部署教程.html, /topics/quantization-accuracy/NVIDIA_TensorRT量化边界.html]
tags: [CV, YOLO, DEIMv2, EdgeCrafter, InsightFace, NVIDIA, Ascend, Sophgo, Rockchip, Qualcomm, TensorRT, INT8, QDQ, 部署]
---

# CV 模型多平台编译与部署

平台编译从已经验证的 ONNX 接口出发，将模型转换成目标设备可加载的产物。统一的脚本入口可以减少重复操作，但 shape、量化、前处理融合和运行时依赖仍由后端与模型共同决定。

本文承接 [CV 模型 ONNX 导出与验证](CV_ONNX导出与验证.md)，按后端组织操作，集中说明模型差异。训练流程见 [CV 模型训练与微调](CV_训练与微调.md)，详细尺寸和接口见 [CV 多平台部署约束矩阵](../matrices/CV_多平台部署约束.md)。

# 一、准备模型、基线与编译环境

开始前准备 ONNX 及外部权重、类别表、固定输入输出 NPZ、前后处理说明和模型版本。示例以静态 batch=1 的浮点基线为主；量化和动态输入作为后续独立配置增加。

以下封装命令在模型转换工程根目录执行，调用其现有后端入口；它们不是厂商 SDK 自带命令。部署前先准备后端环境，入口脚本也可能自动创建环境或启动容器。

| 后端 | 编译位置与环境 | 产物 |
| --- | --- | --- |
| BModel | 匹配 TPU-MLIR 的 x86 Linux 容器 | .bmodel |
| OM | 匹配目标芯片的 CANN/ATC 环境 | .om |
| TensorRT | 目标 GPU 或匹配的 Jetson 构建环境 | .plan |
| RKNN | 匹配 RKNN-Toolkit2 的转换环境，通常在 x86 Linux | .rknn |
| QNN | 本文工具在目标 QCS9075、匹配 QAIRT 的环境生成与验证 | EPContext ONNX、部署清单及拆分模型 |

同一个 `--precision fp16` 在各工具中的作用不完全相同：可能允许混合精度、保留浮点路径，或指定平台模型类型，并不保证所有层与所有 I/O 都是 FP16。生成后重新读取模型属性。

# 二、按后端编译浮点基线

## 2.1 Sophgo BModel

```bash
bash BModel_Compile/bmodel_compile.sh --model_type yolo \
  --weight_path /srv/export/yolo11s.onnx \
  --output_name yolo11s-demo \
  --target bm1684x --precision fp16
```

环境、TPU-MLIR 转换与运行方法见 [Sophgo 模型编译部署教程](Sophgo_模型编译部署教程.md)。模型类型决定是否融合前处理，不能把 YOLO 的处理配置复制给所有模型。

DEIMv2 使用独立模型路径：

```bash
bash BModel_Compile/bmodel_compile.sh --model_type deimv2 \
  --weight_path /srv/export/deimv2.onnx \
  --target bm1684x --precision fp16
```

这里保留原始 bilinear GridSample；其他芯片与工具链版本需要分别验证。双输入中的原图尺寸可能在编译后变为 int32，运行程序按产物实际 dtype 送数。

## 2.2 Ascend OM

```bash
bash OM_Compile/om_compile.sh --model_type yolo \
  --weight_path /srv/export/yolo11s.onnx \
  --target Ascend310P1
```

目标字符串必须与实际芯片和 ATC 支持项一致。当前 YOLO 路径插入 AIPP，接收 YUV420SP 并完成颜色与数值转换；DEIMv2、EdgeCrafter 路径则不生成 YOLO AIPP，由调用方提供归一化图像与原图尺寸。

不能只把 `--model_type` 换成 deimv2，就继续沿用 YOLO 的输入缓冲区。尺寸、对齐与 dtype 应按 OM 描述读取。遇到 GridSample 导入失败，先保留失败日志和原始图，不把命令正常退出之外的推测写成“芯片已经支持”。

## 2.3 NVIDIA TensorRT

```bash
bash TRT_Compile/trt_compile.sh --model_type yolo \
  --weight_path /srv/export/yolo11s.onnx --precision fp16
```

当前模型转换不融合图像前处理，调用方按 ONNX 契约准备张量。AGX Orin 环境准备、原生 trtexec 与动态 profile 的操作见 [NVIDIA AGX 部署与 TensorRT](NVIDIA_AGX部署与TensorRT.md)。

DEIMv2 和 EdgeCrafter 的当前封装固定静态 batch=1 FP16，不能照搬 YOLO 的动态或校准参数。TensorRT 引擎的版本和硬件兼容有明确限制，目标 Orin 上应建立自己的可加载基线，不能把服务器生成的 plan 当作通用模型文件。

引擎可执行但数值明显偏离时，可参考 [TensorRT FP16 数值异常排查：QKV 布局的等价改写](../../topics/graphs-operators/NVIDIA_QKV布局与融合.md)，其中记录了从中间张量定位到导出图改写的过程。

## 2.4 Rockchip RKNN

Toolkit2 主机环境、板端 Runtime/Lite2 安装与最小推理流程见 [Rockchip RKNN 环境搭建与板端推理](Rockchip_RKNN环境与推理.md)。先确认这条基础链路，再使用模型转换入口。

```bash
bash RKNN_Compile/rknn_compile.sh --model_type yolo \
  --weight_path /srv/export/yolo11s.onnx \
  --target rk3576 --precision fp16
```

当前 YOLO 路径将 /255 融合到转换配置中，运行侧提供 RGB 像素，不能再除一次 255。Toolkit 的构建精度名称与 runtime 输入 dtype 需要分别检查。

DEIMv2、EdgeCrafter 的原始图含有 GridSample，本文对应的 RKNN 2.3.2 标准路径不能直接执行。不能因为有编译入口就把它标成可交付；CPU/GPU custom-op 实验也不等于 NPU 原生支持。具体边界见 [RKNN DEIM 部署边界](../../topics/graphs-operators/RKNN_DEIM部署边界.md)。

## 2.5 Qualcomm QNN

在目标 QCS9075 环境生成与验证 YOLO context：

```bash
bash QNN_Compile/qnn_compile.sh --model_type yolo \
  --weight_path /srv/export/yolo11s.onnx \
  --target qcs9075 --precision fp16 \
  --validation_input /srv/reference/yolo-inputs.npz \
  --reference_output /srv/reference/yolo-outputs.npz
```

这里的 NPZ 保存按模型名称组织的正式输入与参考输出，格式见前一篇。当前 YOLO HTP 路径覆盖经过接口检查的 detect/pose 单输出图，不能默认包含 segment；图中不负责外部 decode 与 NMS。

DEIMv2 的默认部署是 HTP encoder 与 CPU decoder/postprocessor 两段。输出不仅包含 context，还要同时交付 CPU 模型、部署清单和中间接口约定，加载程序保留两个 Session。相关取舍见 [Qualcomm DEIM 精度归因](../../topics/quantization-accuracy/Qualcomm_DEIM精度归因.md)。

EPContext 使用 ONNX 容器形式保存平台上下文，但不是可以随意交给 CPU 或其他设备执行的普通 ONNX。SoC、HTP 架构和 QAIRT/QNN 版本需要匹配。

# 三、把模型差异留在明确的位置

| 模型 | 当前公共基线 | 需要单独处理 |
| --- | --- | --- |
| YOLO | 静态浮点检测图 | pose/segment 输出、各代 decode、NMS、不同平台前处理融合 |
| DEIMv2 | 静态 batch=1，双输入 FP16 路径 | 骨干归一化、GridSample、尺寸输入 dtype、QNN 拆分 |
| EdgeCrafter | 静态 batch=1，按任务导出 | masks/keypoints 还原、GridSample 与任务覆盖范围 |
| InsightFace | 检测与识别两个静态模型 | 检测解码、五点对齐、特征归一化和两模型配套 |

InsightFace 的 ONNX 或双模型目录可以交给 `--model_type insightface`，但这只转换网络，检测解码、对齐与检索仍由运行程序完成。这里的流水线边界与单模型检测不同，不能套用一个 YOLO 后处理函数。

“存在入口”“生成模型”“板端执行”“任务数值通过”分别记录。某平台上的单模型成功不代表该平台、模型家族和所有精度组合都已通过。

# 四、明确前处理只执行一次

| 当前工具路径 | 运行侧负责 | 转换配置负责 |
| --- | --- | --- |
| YOLO → TensorRT/QNN | 按 ONNX 约定完成 RGB、Letterbox、/255、布局转换 | 不融合这套图像前处理 |
| YOLO → RKNN | 按接口准备尺寸与 RGB 像素 | mean/std 配置融合 /255 |
| YOLO → OM | 准备满足 AIPP 格式、尺寸与对齐的 YUV 输入 | 配置内的颜色转换、/255 等 |
| YOLO → BModel | 按产物接口准备几何与像素输入 | 按 model_transform/deploy 配置融合指定前处理 |
| DEIMv2/EdgeCrafter → OM/TRT | 归一化后的 images 与原图尺寸 | 不使用 YOLO AIPP |

前处理配置能否融合、哪些步骤真正成为模型的一部分，以最终部署选项和产物接口为准。keep_aspect_ratio 出现在配置中，不代表调用方就可以忽略 runtime 所需的 resize/padding。用同一张非方形图验证两端，确认几何、颜色与数值变换都等价。

# 五、在浮点基线上增加量化与动态输入

## 5.1 先固定校准输入

校准集使用接近部署分布的真实图片，按该后端正式输入规则生成。校准 cache 与模型及前处理绑定，换权重、shape、颜色顺序或归一化后，需要重新核对，不能仅凭 cache 文件名相同就继续使用。

## 5.2 TensorRT 的浮点校准与显式 QDQ

先检查 ONNX 如何表达量化，再选择编译入口。普通浮点图可以通过校准确定范围，含 QuantizeLinear / DequantizeLinear 的图则携带显式量化尺度。两条路径职责不同，不能在未检查图的情况下同时追加 INT8 开关和校准器。

```python
from collections import Counter
import onnx

model = onnx.load("model.onnx")
onnx.checker.check_model(model)
ops = Counter(node.op_type for node in model.graph.node)
print({k: ops[k] for k in ("QuantizeLinear", "DequantizeLinear")})
print([(x.name, x.type.tensor_type.elem_type) for x in model.graph.input])
```

| 路线 | 范围来源 | 应保存的材料 |
| --- | --- | --- |
| 浮点 ONNX＋校准 | 代表性样本与校准器 | 前处理、样本版本、cache、构建参数 |
| 显式 QDQ ONNX | 图中尺度与量化规则 | QDQ 图、尺度配置、后端支持范围 |

若存在 Q/DQ，检查 scale、zero point、axis、dtype 和连接关系，并确认目标 TensorRT 版本支持该组合。删除 Q/DQ 改变了量化表达，不能作为无条件的图清洗。`trtexec` 也不会直接把图片文件夹变成正确的校准集，需要按工具版本准备校准器或显式量化图。

一次 YOLO 编译中，含 Q/DQ 的 ONNX 在构建阶段触发 Myelin 错误，另一份不含这些节点的模型通过。这个对照提供了排查方向，但没有严格证明“重复量化”，也不能推断某个 YOLO 版本的所有导出图都自带 Q/DQ。可以保留失败图，按同图 FP16、同图 QDQ、重新导出的浮点图依次对照；先确定能否构建，再验证数值。

INT8 与 FP16 配合时，部分层可能保留更高精度。检查实际层精度、融合与 tactic，不能根据文件大小推断量化是否生效。静态输入、动态 profile 和硬件兼容同样需要单独验证，服务器生成的 plan 不默认适用于 AGX。

## 5.3 单独验证输出分支的量化误差

YOLO 单输出可能拼接像素级 boxes 与低幅值 scores。INT8 后要单独检查类别分数是否被量化范围吞没，不能只看整个输出的余弦。先确认后端实际采用的 scale；若两个分支共享的步长过大，降低阈值也无法恢复已经丢失的分数差异。必要时使用经过验证的混合精度或分支拆分方案，并同步检查输出顺序、各分支反量化与后处理。输出拼接本身不代表所有后端都会共享尺度，拆分也需要重新验证任务精度。校准与混合精度方法见 [Sophgo INT8 量化](../../topics/quantization-accuracy/Sophgo_INT8量化.md)。

当前工具中的 DEIMv2、EdgeCrafter 主要使用静态 FP16 路径，不能把统一参数表当成它们已经支持 INT8 的证据。

## 5.4 动态输入按组合声明

动态输入按“模型、后端、版本”组合声明：

- YOLO 的部分 OM、RKNN、TensorRT 路径提供动态能力，需检查实际实现和合法档位。
- BModel 与当前 QNN 路径使用静态输入。
- 动态 ONNX、编译 profile 与运行时 shape 三者必须一致，单加一个动态开关不构成完整适配。

# 六、板端验证与部署打包

先读取最终模型 I/O，再分配输入输出。用同一参考样本确认加载、数值、框或关键点还原，最后做数据集评估与真实链路测试。

分别记录解码、前处理、模型、数据搬运和后处理时间。QNN 的 HTP 子图耗时、TensorRT GPU 时间和完整视频帧时延口径不同，不能直接横向比较。多段模型还要计入中间张量传输和 Session 调度。

发布包至少保存：

```text
release/
├── models/          # 平台模型、外部权重及拆分子图
├── labels.json
├── manifest.json    # 版本、哈希、I/O、前后处理与运行依赖
└── reference/       # 固定样例及对应参考结果
```

manifest 记录源 ONNX 哈希、模型与任务类型、目标芯片、工具链版本、精度、动态范围和验证条件。模型文件、类别表和后处理作为同一版本更新；转换失败或数值未通过时保留旧版本，避免把半成品覆盖到正式路径。

# 七、参考资料

- [TPU-MLIR](https://github.com/sophgo/tpu-mlir)
- [Ascend CANN 开发文档](https://www.hiascend.com/document)
- [TensorRT 引擎兼容说明](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html)
- [TensorRT 量化类型与 QDQ](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html)
- [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2)
- [ONNX Runtime QNN Execution Provider](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)
- [ONNX Runtime EPContext](https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html)
