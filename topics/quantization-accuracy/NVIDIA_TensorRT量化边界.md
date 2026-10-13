---
title: NVIDIA TensorRT 的 QDQ 与 INT8 编译边界
created: 2026-04-24
updated: 2026-04-27
status: draft
blog: true
tags: [NVIDIA, TensorRT, INT8, QDQ, ONNX]
---

# NVIDIA TensorRT 的 QDQ 与 INT8 编译边界

TensorRT 的 INT8 编译需要先确认 ONNX 如何表达量化。普通浮点图依靠校准确定范围，含 QuantizeLinear / DequantizeLinear 的图则携带显式量化尺度。两条路径的职责不同，不能在没有检查图的情况下直接追加 INT8 开关和校准器。

归档 YOLO 编译中，某份含 Q/DQ 的 ONNX 在构建阶段触发 Myelin 错误，而另一份不含这些节点的模型通过。这个对照提供了排查方向；不能由此推断某个 YOLO 版本导出的所有 ONNX 都自带 Q/DQ，也不能把编译器报错简化为已严格证明的“重复量化”。

# 一、先检查模型

```python
from collections import Counter
import onnx

model = onnx.load("model.onnx")
onnx.checker.check_model(model)
ops = Counter(node.op_type for node in model.graph.node)
print({k: ops[k] for k in ("QuantizeLinear", "DequantizeLinear")})
print([(x.name, x.type.tensor_type.elem_type) for x in model.graph.input])
```

若存在 Q/DQ，检查 scale、zero point、axis、dtype 和算子连接，并确认目标 TensorRT 版本支持该组合。删除 Q/DQ 改变了量化表达，不能把删除节点当成无条件的图清洗。

# 二、两条编译路线

| 路线 | 范围来源 | 应保存的材料 |
| --- | --- | --- |
| 浮点 ONNX＋校准 | 代表性样本与校准器 | 前处理、样本版本、cache、构建参数 |
| 显式 QDQ ONNX | 图中尺度与量化规则 | QDQ 图、尺度配置、后端支持范围 |

`trtexec` 不会直接把一个图片文件夹变成正确校准集。需要将与线上一致的前处理送入校准器，或按当前工具版本支持的显式量化流程生成 QDQ 图。

校准 cache 与模型及前处理绑定。换权重、shape、颜色顺序或归一化后继续使用旧 cache，会让结果失去可解释性。文件名相同不代表 cache 仍然有效。

# 三、引擎精度与设备兼容

INT8 与 FP16 配合时，部分层可能保留更高精度，最终引擎大小和速度不只由权重位宽决定。需要查看实际层精度、融合与 tactic，不能根据文件大小判断量化是否生效。

静态输入、动态 profile 和硬件兼容也属于独立条件。桌面 GPU 生成的 plan 不能默认迁移到 AGX；即使两边都能构建，数值与性能也必须在目标设备验证。

# 四、一些反思

量化失败应先分成“图无法被表达或编译”和“图能运行但数值不可用”。前者检查格式、算子约束和版本；后者才围绕校准分布、尺度与敏感层展开。

遇到 Myelin 这类内部错误，适合保存失败图并做最小化对照：同图 FP16、同图 QDQ、移除量化表达后重新导出的浮点图。只有保持其他条件一致，才能把相关性收敛成可复查的原因。

# 五、参考资料

- [TensorRT 量化说明](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html)
- [TensorRT trtexec](https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/command-line-programs.html)

