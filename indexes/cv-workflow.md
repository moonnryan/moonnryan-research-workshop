# CV 训练、导出与部署

按阶段维护公共流程，各篇内部说明模型差异。训练设备与部署目标分别选择，不为每一种模型和设备组合复制教程。

| 顺序 | 教程 | 结束时应具备 |
| --- | --- | --- |
| 1 | [CV 模型训练与微调](../assets/tutorial/CV_训练与微调.md) | 选定权重、配置、类别映射与评估基线 |
| 2 | [CV 模型 ONNX 导出与验证](../assets/tutorial/CV_ONNX导出与验证.md) | 固定 I/O、前后处理、ONNX 与参考输入输出 |
| 3 | [CV 模型多平台编译与部署](../assets/tutorial/CV_多平台编译与部署.md) | 通过目标设备验证的平台模型和部署清单 |

已有 ONNX 可以从第二篇开始。只有浮点基线通过后，再增加 INT8、动态输入或异构拆分。

## 平台环境与约束

- [NVIDIA AGX Orin 系统与 TensorRT 环境部署](../assets/tutorial/NVIDIA_AGX部署与TensorRT.md)
- [Sophgo 模型编译部署教程](../assets/tutorial/Sophgo_模型编译部署教程.md)
- [Ascend 服务器环境部署与信息查询](../assets/tutorial/Ascend_服务器环境部署.md)
- [Rockchip RKNN 环境搭建与板端推理](../assets/tutorial/Rockchip_RKNN环境与推理.md)
- [CV 多平台部署约束矩阵](../assets/matrices/CV_多平台部署约束.md)

矩阵描述固定版本下的已知接口和限制，模型或工具链更新时重新核对。操作入口存在与板端任务验证通过分别记录。

## 机制与取舍

- [DEIM 微调中的配置、前处理与权重产物](../topics/models-training/DEIM_微调与产物契约.md)
- [多平台模型任务的版本与产物管理](../topics/engineering-runtime/多平台_任务与产物管理.md)
- [RKNN DEIM 部署边界](../topics/graphs-operators/RKNN_DEIM部署边界.md)
- [Qualcomm DEIM 精度归因](../topics/quantization-accuracy/Qualcomm_DEIM精度归因.md)
