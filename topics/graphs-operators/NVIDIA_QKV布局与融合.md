---
title: NVIDIA TensorRT 的 QKV 布局与融合边界
created: 2026-09-09
updated: 2026-09-09
status: draft
blog: true
tags: [NVIDIA, TensorRT, EdgeCrafter, ONNX, 图优化]
---

# NVIDIA TensorRT 的 QKV 布局与融合边界

计算图改写在数学上等价，编译器却可能对不同写法匹配不同融合路径。归档 AGX Orin / TensorRT 10.3 的 EdgeCrafter FP16 测试中，QKV 布局转换附近出现数值偏离，最终 scores 余弦约 0.6885；改为整体 reshape、permute 后再 Split，输出恢复到约 0.9999927。

这一结果说明图表达参与了后端执行选择。不能据此推断所有 QKV 融合都有错误，但可以用等价重写构造有信息的对照。

# 一、确认布局契约

Q、K、V 常从同一投影输出中拆分。通道分组、head 维与序列维的排列必须固定：

```python
# 假定线性输出按 [Q全部, K全部, V全部] 排列。
# 示例展示索引关系，实际图以模型定义为准。
qkv = projected.reshape(batch, tokens, 3, heads, head_dim)
qkv = qkv.permute(2, 0, 3, 1, 4)
q, k, v = qkv.unbind(0)
```

若原始权重按 head 交错排列，直接套用这段就不等价。重写前需要用 CPU 参考比较逐元素结果，并验证 RoPE 作用的维度与成对通道。

# 二、从最终输出向内定位

| 对照 | 作用 |
| --- | --- |
| PyTorch 与 ONNX | 检查导出 |
| ONNX 与 TensorRT | 检查后端路径 |
| RoPE 前 Q/K | 检查投影与布局 |
| RoPE 后 Q/K | 检查位置旋转与融合 |
| 等价图重写 | 检查融合模式影响 |

归档 RoPE 后 Q/K 余弦曾分别约 0.887 和 0.415。保存这些输出作为探针可以缩小范围，但新增输出也可能改变融合，应再用原正式输出验收修复，避免探针图本身绕开问题。

# 三、验证修复

先证明改写前后 ONNX 参考一致，再编译 TensorRT。比较 scores、类别、有效框数和匹配 IoU；最后移除探针输出，在实际部署图上重测。

图重写影响 tactic 与内存布局，因此还需记录性能。精度恢复并不保证速度不变，目标设备和原 TensorRT 版本是该结果的适用范围。

# 四、索引比 shape 更有信息

两个张量具有相同 shape，元素排列仍可能不同。构造带唯一编号的输入，沿 reshape、transpose、split 追踪某个 token、head 与 channel，可以在数值模型之前验证索引关系。

实际 attention 还包含 scale、mask、Softmax 和 RoPE。布局对齐后若仍有误差，应逐步固定上游输入，隔离各部分；若直接比较完整 attention，难以区分投影误差与后续累积。

验收加入非对称输入和不同 token 数，避免某些维度相等时错误置换恰好被隐藏。修复不依赖单张图片的偶然分布，才适合固化到通用导出工具。

# 五、一些反思

有时最小修复不是关闭所有优化，而是换一种更明确的等价表达，让编译器避开有问题的匹配。前提是先建立索引级等价与任务级验证。

探针也是干预。逐层输出帮助定位，但可能改变图融合；诊断图与交付图应分别验收，这一点在编译后端问题中尤其重要。

# 六、参考资料

- [TensorRT 精度考虑](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html)
- [ONNX Split](https://onnx.ai/onnx/operators/onnx__Split.html)

