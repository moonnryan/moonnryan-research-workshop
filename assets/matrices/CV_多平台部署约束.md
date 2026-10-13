---
title: CV 多平台部署约束矩阵
created: 2026-08-10
updated: 2026-10-08
status: draft
blog: false
tags: [CV, ONNX, NVIDIA, TensorRT, Ascend, Sophgo, Rockchip, RKNN, Qualcomm, QNN, 约束矩阵]
---

# CV 多平台部署约束矩阵

本矩阵描述归档中固定 batch=1 的部署契约，不是各工具链能力的通用上限。动态 shape、不同模型修订或新的后端版本需单独建立验收。DEIM/EdgeCrafter 的 orig_target_sizes 顺序按这里的导出产物为 [W,H]；其他导出脚本可能不同，应从实际图的缩放逻辑核实，并用非方形原图测试。

# 一、整体约束

| 项目 | 约束 |
| --- | --- |
| batch | 导出、推理和平台编译统一固定为 1 |
| 输入 shape | 本链路采用静态，由具体模型定义；不在全局统一规定输入尺寸 |
| dynamic batch/shape | 本链路不使用；不代表所有平台均不支持 |
| 前处理 | 数学规则必须一致，并明确执行位置；平台可融合前处理，但不得与业务侧重复执行 |
| ONNX 与平台 | 输入名称、输入 dtype、输出顺序、输出含义和前处理必须一致 |
| INT8 校准 | 使用正式推理前处理和 batch=1 的真实业务图片 |
| 前处理次数 | 只能执行一次，禁止重复执行 /255 或 Normalize |

# 二、模型与 ONNX 约束

本节各表字段含义固定：ONNX images 前处理只描述图像张量的外部处理；ONNX 输入列列出所有输入的名称、dtype 和 shape；ONNX 输出列列出输出 tensor；最后一列描述静态导出与后处理约束。

## YOLO

YOLO 是本文归档链路常用的 Ultralytics 单阶段检测模型。detect 输出目标框与类别分数；pose 在此基础上额外输出人体关键点。两类模型均使用固定 640 x 640 的 ONNX 输入，但后处理规则不同。

| 模型/任务 | Backbone/说明 | ONNX images 前处理 | ONNX 输入 | ONNX 输出 | ONNX 导出/解析约束 |
| --- | --- | --- | --- | --- | --- |
| YOLO detect | Ultralytics detect | BGR -> RGB -> 保持比例 Letterbox 到 640 x 640，padding=114 -> float32 -> /255.0 -> NCHW | images：float32 [1,3,640,640] | 单输出，常见为 [1,4+nc,8400]，COCO 为 [1,84,8400] | batch=1；本链路固定静态；阈值、NMS、去 padding、坐标还原；ONNX 不内嵌 Letterbox、BGR -> RGB 或 /255。 |
| YOLO pose | Ultralytics pose | BGR -> RGB -> Letterbox 640 x 640，padding=114 -> float32 -> /255.0 -> NCHW | images：float32 [1,3,640,640] | 单输出，常见为 [1,4+nc+keypoint_dim,8400]，如 [1,56,8400] | batch=1；本链路固定静态；按 pose 通道规则解析关键点、去 padding 和坐标还原，不得按 detect 解析。 |

整体注意事项：本文归档链路的 Ultralytics YOLO 模型统一使用 Letterbox，不使用直接 Resize 拉伸。训练中的 Mosaic、透视等增强不属于部署前处理；验证、ONNX Runtime、量化校准和平台推理必须复用上述 Letterbox 规则。

## DEIMv2

DEIMv2 是端到端目标检测模型，按骨干分为 DINOv3/ViT 的 X/L/M/S 与 HGNetv2 的 N/Pico/Femto/Atto。每个 ONNX 都有两个输入：images 是完成下表前处理后的图像张量；orig_target_sizes 是原图 [W,H] 的 int64 元数据，不执行图像前处理，用于将输出框映射回原图。

| 模型/任务 | Backbone/说明 | ONNX images 前处理 | ONNX 输入 | ONNX 输出 | ONNX 导出/解析约束 |
| --- | --- | --- | --- | --- | --- |
| DEIMv2-X | DINOv3-S+ | RGB -> 直接 Resize 640 x 640 -> float32 -> /255.0 -> ImageNet Normalize；mean=[0.485,0.456,0.406]，std=[0.229,0.224,0.225] -> NCHW | images：float32 [1,3,640,640]；orig_target_sizes：int64 [1,2] | labels [1,300]、boxes [1,300,4]、scores [1,300] | batch=1；dynamic_axes=None；按 labels/boxes/scores 解析，不使用 YOLO decode/NMS。 |
| DEIMv2-L | DINOv3-S | 同上 | 同上 | 同上 | 同上 |
| DEIMv2-M | ViT-Tiny+，DINOv3-S 蒸馏权重 | 同上 | 同上 | 同上 | 同上 |
| DEIMv2-S | ViT-Tiny，DINOv3-S 蒸馏权重 | 同上 | 同上 | 同上 | 同上 |
| DEIMv2-N | HGNetv2 | RGB -> 直接 Resize 640 x 640 -> float32 -> /255.0 -> NCHW；不做 ImageNet Normalize | images：float32 [1,3,640,640]；orig_target_sizes：int64 [1,2] | 同上 | 同上 |
| DEIMv2-Pico | HGNetv2 | 同上 | 同上 | 同上 | 同上 |
| DEIMv2-Femto | HGNetv2 | RGB -> 直接 Resize 416 x 416 -> float32 -> /255.0 -> NCHW；不做 ImageNet Normalize | images：float32 [1,3,416,416]；orig_target_sizes：int64 [1,2] | 同上 | 同上 |
| DEIMv2-Atto | HGNetv2 | RGB -> 直接 Resize 320 x 320 -> float32 -> /255.0 -> NCHW；不做 ImageNet Normalize | images：float32 [1,3,320,320]；orig_target_sizes：int64 [1,2] | 同上 | 同上 |

整体注意事项：DEIMv2 的 ONNX 导出不应把外部图像前处理重复嵌入模型。官方 COCO 配置下，Atto/Femto/Pico/N 的 ONNX 对外输出维度与 X/L/M/S 相同，都是 PostProcessor 固定取出的 top-300 结果；这不等于它们的内部 query 数相同。导出自定义类别模型时，必须重新确认 num_top_queries 和实际 ONNX 输出，不能直接套用上述维度。

## EdgeCrafter

EdgeCrafter 是面向边缘端的任务专用模型族，包含 ECDet 检测、ECInsSeg 实例分割与 ECPose 人体姿态。三类 ONNX 都使用 images 和 orig_target_sizes 两个输入，图像前处理相同，输出由任务类型决定。

| 模型/任务 | Backbone/说明 | ONNX images 前处理 | ONNX 输入 | ONNX 输出 | ONNX 导出/解析约束 |
| --- | --- | --- | --- | --- | --- |
| EdgeCrafter detect | ECDet | RGB -> 直接 Resize 640 x 640 -> float32 -> /255.0 -> ImageNet Normalize；mean=[0.485,0.456,0.406]，std=[0.229,0.224,0.225] -> NCHW | images：float32 [1,3,640,640]；orig_target_sizes：int64 [1,2]，原图 [W,H] | labels：int64 [1,300]；boxes：float32 [1,300,4]；scores：float32 [1,300] | batch=1；dynamic_axes=None；按 labels/boxes/scores 解析。 |
| EdgeCrafter segment | ECInsSeg | 同上 | 同上 | labels：int64 [1,300]；boxes：float32 [1,300,4]；scores：float32 [1,300]；masks：float32 [1,300,160,160] | batch=1；dynamic_axes=None；masks 为 1/4 分辨率 logits，需按业务阈值和原图尺寸还原。 |
| EdgeCrafter pose | ECPose | 同上 | images：float32 [1,3,640,640]；orig_target_sizes：int64 [1,2]，原图 [W,H] | scores：float32 [1,60]；labels：int64 [1,60]；keypoints：float32 [1,60,17,2] | batch=1；dynamic_axes=None；按关键点输出解析。 |

整体注意事项：当前官方导出脚本声明了动态 batch；发布前必须改为 dynamic_axes=None，并以 batch=1 重新导出。图像前处理没有默认嵌入 ONNX，部署端必须按表执行。

## InsightFace

InsightFace 人脸流水线由 det_10g 完成人脸检测和 5 点关键点定位，再由 w600k_r50 对齐后的人脸图提取 512 维特征。两份 ONNX 不是 YOLO；检测模型在 stride 8、16、32 三个尺度上各输出人脸分数、框距离和 5 点关键点，因此共有 9 个输出 tensor。

| 模型/任务 | Backbone/说明 | ONNX images 前处理 | ONNX 输入 | ONNX 输出 | ONNX 导出/解析约束 |
| --- | --- | --- | --- | --- | --- |
| det_10g | SCRFD-10GF 人脸检测 | BGR -> Letterbox 640 x 640 -> RGB -> float32 -> (pixel - 127.5) / 128.0 -> NCHW | input.1：float32 [1,3,?,?]；部署 ONNX 固定为 [1,3,640,640] | 人脸分数：stride 8/16/32 分别为 [12800,1]/[3200,1]/[800,1]；框距离 l/t/r/b：分别为 [12800,4]/[3200,4]/[800,4]；5 点关键点：分别为 [12800,10]/[3200,10]/[800,10]。上述输出尺寸对应 640 输入。 | 每尺度有 2 个 anchors；先按 stride 解码框和关键点，再做分数过滤、NMS，最终输出原图 xyxy 框与 5 点。 |
| w600k_r50 | ArcFace ResNet50@WebFace600K | 先用检测模型的 5 点对齐，warp 到 112 x 112 -> RGB -> float32 -> (pixel - 127.5) / 127.5 -> NCHW | input.1：float32 [N,3,112,112]；部署 ONNX 固定为 [1,3,112,112] | embedding：float32 [1,512]，表示一张对齐人脸的特征向量 | 对外比对、入库和检索前必须执行 L2 Normalize；不是普通分类 ResNet50。 |

整体注意事项：det_10g 源 ONNX 的空间尺寸动态、w600k_r50 源 ONNX 的 batch 动态；部署与平台编译必须固定为 batch=1 的指定输入尺寸。检测先做 score 阈值、9 路 decode 和 NMS，随后才能用 5 点进行 ArcFace 对齐。

# 三、平台编译约束

| 平台 | 输出 | 输入约束 | 编译约束 | 前处理约束 |
| --- | --- | --- | --- | --- |
| TensorRT | plan | 使用模型定义的静态 shape，batch=1 | 本链路固定静态；plan 绑定 GPU、CUDA 和 TensorRT 版本 | 默认由业务侧按 ONNX 契约前处理；若 engine 融合前处理，必须关闭业务侧对应步骤。 |
| Ascend OM | om | 使用模型定义的静态 shape，batch=1 | 本链路固定静态；CANN、ATC 和芯片版本必须匹配 | AIPP 可融合 CSC、Resize、/255、mean/std；AIPP 全部参数必须与 ONNX 前处理逐项等价，业务侧不得重复执行。 |
| Rockchip RKNN | rknn | 使用模型定义的静态 shape，batch=1 | 本链路固定静态；明确 target 芯片；INT8 必须校准 | mean_values/std_values 可融合 /255 或 ImageNet/InsightFace Normalize。融合后通常传入原始 uint8 图像；颜色、几何和数值必须与 ONNX 契约等价。 |
| Sophon BModel | bmodel | 使用模型定义的静态 shape，batch=1 | 固定输入；Docker、processor 和目标芯片必须匹配 | model_transform 可融合颜色转换、Resize、scale、mean/std；keep_aspect_ratio 与 padding 规则必须和 ONNX 契约一致。 |
| Qualcomm QNN | context / EPContext | 使用模型定义的静态 shape，batch=1 | 本链路使用静态 shape；QAIRT 与设备 runtime 必须匹配 | 可在图或预处理配置中融合量化/前处理；必须核对最终 I/O，避免业务侧与模型内重复处理。 |

# 四、验收

1. 模型输入 shape、前处理和输出解析以对应模型表及实际 ONNX 文件为准。
2. 业务侧已经做过前处理时，平台侧不能再次做同样的处理；反之亦然。
3. 导出成功不代表转换正确，必须用同一张图片对比 PyTorch、ONNX Runtime 和平台输出。
4. INT8 校准图片必须使用正式推理前处理，不能使用另一种模型的校准脚本。


# 五、参考资料

- [Ultralytics ONNX 导出](https://docs.ultralytics.com/integrations/onnx/)
- [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2)
- [TensorRT 支持矩阵](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)
- [QNN EPContext](https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html)
