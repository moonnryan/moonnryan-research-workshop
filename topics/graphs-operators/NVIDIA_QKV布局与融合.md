---
title: TensorRT FP16 数值异常排查：QKV 布局的等价改写
created: 2026-09-09
updated: 2026-09-09
status: draft
blog: true
tags: [NVIDIA, TensorRT, EdgeCrafter, ONNX, 图优化]
---

# TensorRT FP16 数值异常排查：QKV 布局的等价改写

同一个模型导出为 ONNX 后可以正常执行，编译成 TensorRT 引擎后却出现明显的输出偏差，并不一定要靠提高计算精度解决。AGX Orin / TensorRT 10.3 的 EdgeCrafter 部署记录中，异常集中在 QKV 布局转换及 RoPE 附近。保留 FP16 构建配置，仅调整导出图的拆分与转置顺序，最终 scores 的余弦相似度从约 0.6885 恢复到 0.9999927。

这个案例的重点是如何用等价图表达缩小问题范围。数值恢复支持将原因定位到这段图的后端处理路径，但尚不能据此确定某个 TensorRT 内部融合 kernel 的实现错误，也不代表所有 QKV 融合都会出现同类问题。

TensorRT 环境、FP16 基线构建与运行检查可先参考 [NVIDIA AGX Orin 系统与 TensorRT 环境部署](../tutorial/NVIDIA_AGX部署与TensorRT.md)。本文进一步讨论环境已经可用、引擎能执行后出现的数值偏差。

# 一、先区分导出问题和后端数值问题

引擎能生成、推理没有报错，只说明图可以被解析和执行。判断结果是否正确，需要给 PyTorch、ONNX 参考运行时与 TensorRT 提供同一份输入张量，先比较原始输出，再比较最终检测结果。

这里应固定已经完成前处理的输入，而不是让三个程序各自读取同一张 JPEG。颜色顺序、缩放、归一化和尺寸辅助输入的差异，都可能在进入模型之前改变结果。保存输入张量、模型哈希、ONNX opset、导出选项、目标设备和 TensorRT 构建配置，后面的对照才能复现。

| 比较对象 | 主要用于区分的问题 |
| --- | --- |
| PyTorch 与原 ONNX | 模型导出是否改变了计算语义 |
| 原 ONNX 与原 TensorRT 引擎 | 后端编译与执行是否引入明显偏差 |
| 原 ONNX 与改写 ONNX | 布局改写本身是否等价 |
| 改写 ONNX 与新 TensorRT 引擎 | 修改图表达后，后端结果是否恢复 |

各组都要记录参考端的实际精度。FP16 需要先检查输出范围、NaN/Inf，以及 Softmax 等敏感计算附近的误差；这些属于应排除的常见因素，不能用来直接解释本次异常。

# 二、两种 QKV 写法为什么等价

设输入为 $[B,T,C]$，head 数为 $H$，每个 head 的宽度为 $D$，且 $C=HD$。本模型的线性投影输出为 $[B,T,3HD]$，最后一维依次保存全部 Q、全部 K、全部 V。

原来的写法先拆出 Q/K/V，再分别把 token 与 head 维交换：

```python
qkv = projected.reshape(B, T, 3, H, D)
q, k, v = qkv.unbind(2)                    # 每个为 [B, T, H, D]
q, k, v = [x.transpose(1, 2) for x in (q, k, v)]
# 最终均为 [B, H, T, D]
```

改写后的顺序是先整体交换维度，再拆分：

```python
qkv = projected.reshape(B, T, 3, H, D)
qkv = qkv.permute(2, 0, 3, 1, 4)           # [3, B, H, T, D]
q, k, v = qkv.unbind(0)
# 最终仍为 [B, H, T, D]
```

令 $s=0,1,2$ 分别表示 Q、K、V，则两种表达都应满足：

$$
A_s[b,h,t,d]=\operatorname{projected}[b,t,sHD+hD+d]
$$

这里变的是取出元素的操作顺序，最终元素与坐标的对应关系没有变化。它不适用于所有 QKV 权重：若通道按 head 交错，或 Q 与 K/V 的 head 数不同，就需要重新推导，不能直接复制这段变换。

## 用编号张量验证索引

随机输入有助于比较数值，唯一编号则便于直接检查元素去向。下面使用非对称维度，避免 token 数与 head 数相同时掩盖错误：

```python
import numpy as np

B, T, H, D = 2, 5, 3, 4
projected = np.arange(B * T * 3 * H * D).reshape(B, T, 3 * H * D)
base = projected.reshape(B, T, 3, H, D)

before = tuple(base[:, :, s, :, :].transpose(0, 2, 1, 3)
               for s in range(3))
after_all = base.transpose(2, 0, 3, 1, 4)
after = tuple(after_all[s] for s in range(3))

for s in range(3):
    np.testing.assert_array_equal(before[s], after[s])
    assert after[s][1, 2, 4, 3] == projected[1, 4, s * H * D + 2 * D + 3]
```

这个检查证明示例布局的索引等价，不替代完整 ONNX 的验证。PyTorch 的 `unbind` 在导出后可能表现为 Split 与 Squeeze 等节点组合，具体以导出器、opset 和实际图为准。节点数量不同，并不直接说明计算语义不同。

# 三、从 scores 向内找到偏差位置

最终 scores 不一致时，直接修改整个 attention 很难判断哪个变化有效。更有用的顺序是逐步检查投影结果、拆分后的 Q/K/V、RoPE 后 Q/K、attention 输出，再回到检测头。

| 探针位置 | 要核对的内容 |
| --- | --- |
| QKV 投影输出 | 权重、偏置与输入是否一致 |
| 布局转换后 Q/K/V | head、token、channel 的元素对应关系 |
| RoPE 后 Q/K | 位置、通道配对、sin/cos 广播及计算结果 |
| attention logits 与输出 | scale、mask、Softmax 及误差放大 |
| 检测头输出 | 上游偏差是否传播到分类与框回归 |

归档中，RoPE 后 Q/K 的余弦相似度曾分别约为 0.887 和 0.415。这能把检查范围缩到相应计算附近，却不能仅凭 RoPE 后误差大，就认定 RoPE 公式有错：它也可能接收到已经错位的输入，或与前后节点共同进入了不同的后端实现路径。

如果完整网络中的某个中间输出已经异常，可以将固定的参考输入送入局部子图，再比较这段计算。这样能减少上游误差的干扰。但局部子图也会改变优化条件，因此它用于建立对照，仍需回到完整网络复核。

## RoPE 要核对的不只是输出 shape

本地实现将每个 head 的最后一维分为前后两半，构造 `[-后半, 前半]`，再与原值分别乘 sin/cos 后相加。它与按相邻奇偶通道配对的写法不是直接可互换的布局。

模型还保留首个特殊 token，仅对后续 patch token 的 Q/K 应用 RoPE。定位时应同时检查这一 token 是否被保留、sin/cos 的位置索引是否对应 patch 顺序，以及广播轴是否落在预期维度。shape 相同但 token 顺序或通道配对错误，结果仍会明显偏离。

只比较整个张量的余弦相似度也不够。可以按 head、token 分组查看最大绝对误差和均方误差；若误差集中在某些通道或位置，它们比单个全局指标更容易暴露排列问题。接近零的参考值则不适合只用相对误差衡量。

# 四、等价改写改变的是编译器看到的图

数学上等价的拆分与转置顺序，在 ONNX 中具有不同的连接方式。TensorRT 可能据此执行不同的融合、布局传播、格式转换或 tactic 选择。这里的 tactic 指某层或融合区域可采用的具体执行实现，不是模型数学定义的一部分。

本次修复将整体转置放在拆分之前，并只在 ONNX 导出路径启用这一表达。训练和普通 PyTorch 执行仍保留原写法，未修改权重、RoPE 公式或 attention 定义，也未以强制全 FP32 作为修复手段。

不过，保留 `--fp16` 等构建设置，不等于证明每个引擎层都采用 FP16。若需要比较改写前后的实际精度、布局和 tactic，应在构建时保留详细 profiling 信息，再通过 Engine Inspector 检查引擎。ONNX 中看不到的融合，不能只靠节点截图推断。

| 证据 | 可以说明 | 仍不能说明 |
| --- | --- | --- |
| 两种参考表达输出一致 | 改写保留了相应计算语义 | 两个编译结果一定等价 |
| 改写后 TensorRT 输出恢复 | 异常与该图表达的后端处理有关 | 已确定内部某个 kernel 的具体缺陷 |
| 更改精度后现象变化 | 执行路径或数值行为对精度敏感 | 原问题必然只是舍入误差 |
| 引擎层信息发生变化 | 编译器选择了不同实现或布局 | 某一项变化单独构成根因 |

现有案例适用于归档的 AGX Orin / TensorRT 10.3 组合。新版本、其他模型或不同构建选项需要重新验证，不应将这个 workaround 扩大为统一的 QKV 导出规则。

固定中间输入、区分上游漂移与局部计算误差的另一组对照，见 [Qualcomm HTP 上 DEIM 的数值误差与拆图](../quantization-accuracy/Qualcomm_DEIM精度归因.md)。那组实验最终选择显式拆图，本例则通过保持数学语义的布局改写恢复输出。

# 五、诊断图与正式引擎分别验证

把中间张量标记为输出，有助于定位，但也可能阻止融合、改变张量生命周期或内存规划。如果一加探针结果就恢复，说明问题对图结构敏感，不能把它当成原图已经正确的证据。

可以保留两类模型：诊断图只暴露少量待检查张量，正式图维持原有输入输出。每次缩小范围后，都要移除探针、重新构建正式引擎，再验证最终输出。仅在带探针的图上通过，不足以交付。

本次记录中，最终 scores 余弦相似度恢复到约 0.9999927，是该对齐样例的结果，不是检测准确率，也不意味着所有类别和阈值附近的结果都一致。进一步验证应覆盖：

- **原始数值**：scores 与框坐标的绝对误差、异常值、各输出分支的对应关系
- **检测结果**：类别、有效框数、置信度阈值附近的变化，以及匹配目标的框 IoU
- **输入覆盖**：不同图片内容和宽高比；仅测试部署配置实际支持的输入形状
- **执行稳定性**：重复推理、重新构建引擎，以及目标设备上的持续运行
- **性能**：同输入、同预热及功耗条件下的推理时延、内存与端到端耗时

局部索引示例可以使用不同 batch 和 token 数发现错误，但本案例部署封装仍是静态 batch=1，不能因此宣称正式引擎支持动态输入。现有记录也没有给出改写前后完整的性能对照，不能将数值修复进一步写成性能提升。

# 六、从这个案例保留的排查方法

首先固定输入与参考结果，再寻找最早出现明显偏差的位置。最终检测框变化可能是后续放大，直接在检测头补偿，反而会留下前面的错误。

等价改写提供了一种可控的实验手段：不同时改变权重、精度和模型结构，只改变编译器看到的图表达。它能帮助判断问题是否与某段后端处理有关，但修复有效与内部根因已经查明，是两个不同层次的结论。

最后，探针图、正式图和性能测试需要各自验收。图优化的目标是保持计算语义并改善执行，既不能因引擎能运行就忽略数值，也不能因精度恢复就默认性能没有变化。

# 七、参考资料

- [TensorRT 数值精度与敏感计算](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html)
- [TensorRT Engine Inspector 与调试](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-tools.html)
- [ONNX Transpose](https://onnx.ai/onnx/operators/onnx__Transpose.html)、[Split](https://onnx.ai/onnx/operators/onnx__Split.html)

