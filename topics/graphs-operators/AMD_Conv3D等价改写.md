---
title: 视觉 patch embedding 的等价改写：从 Conv3D 到 Linear
created: 2026-09-21
updated: 2026-09-24
status: ready
blog: true
tags: [AMD, NVIDIA, Qwen3-VL, Qwen3.5, LLaMA-Factory, Conv3D, 图优化, 训练加速]
---

# 视觉 patch embedding 的等价改写：从 Conv3D 到 Linear

模型使用 Conv3D，不一定意味着它在做一般意义上的时空滑动卷积。Qwen3-VL 和 Qwen3.5 的视觉 patch embedding 接收的是已经按独立 patch 整理的输入；在对应结构下，每个 patch 的输出只有一个空间位置，这一层可以写成展平后的 Linear。

这个优化最初用于处理 AMD R9700 上的慢卷积路径，后来也接入了 CUDA 训练镜像。更值得保留的是改造方法：先识别模型实际使用的结构约束，再选择等价且更适合后端的算子表达，而不是将它归为某一种硬件的专用技巧。

# 一、为什么 Conv3D 在这里可以变成矩阵乘

一般 Conv3D 会在多个位置提取局部窗口，窗口之间还可能重叠。本例的输入已经是独立 patch，每项形状为 $[C,T_p,H_p,W_p]$，卷积核覆盖整个 patch。原实现使用匹配的 kernel/stride、无 padding、dilation=1、groups=1，输出为 $[D,1,1,1]$。

设一批共有 $P$ 个 patch，展平后的长度为 $K=CT_pH_pW_p$，卷积权重为 $[D,C,T_p,H_p,W_p]$。将输入和权重按相同顺序展平，就得到：

$$
X_{\mathrm{flat}}\in\mathbb{R}^{P\times K},\qquad
W_{\mathrm{flat}}\in\mathbb{R}^{D\times K},\qquad
Y=X_{\mathrm{flat}}W_{\mathrm{flat}}^{\mathsf T}+b
$$

每个输出仍然是对同一个 patch 的全部元素做加权求和，没有改变 patch 的位置，也没有改变投影参数。一般卷积转 GEMM 往往需要 im2col；这里前处理已经完成了 patch 整理，不需要再展开所有滑动窗口。

| 检查项 | 本例满足的条件 | 条件变化后需要重新判断的内容 |
| --- | --- | --- |
| 输入单元 | 一项对应一个独立 patch | 是否仍含多个待滑动的位置 |
| kernel 与输入尺寸 | 核恰好覆盖整个 patch | 是否产生多个输出位置 |
| padding / dilation | 无填充、无膨胀 | 是否改变参与计算的元素 |
| groups | 1 | 是否存在分组连接关系 |
| 元素顺序 | 输入与权重展平顺序一致 | 通道、时间与空间维是否被交换 |
| 输出接口 | 一个 patch 对应一个 D 维向量 | 下游是否仍获得原有 patch 顺序 |

这些是本次替换采用的一组充分条件，不是任何 Conv3D 都能改成一个普通 Linear。对于完整视频上的重叠卷积或多位置输出，需要额外的窗口组织，不能直接套用这里的写法。

# 二、改计算表达，保留原参数

下面示例假设 `patches` 已经恢复为五维 patch 输入，并且结构检查已经通过：

```python
# patches: [P, C, Tp, Hp, Wp]
# conv.weight: [D, C, Tp, Hp, Wp]
patches = patches.to(dtype=conv.weight.dtype)
flat_input = patches.reshape(patches.shape[0], -1)
flat_weight = conv.weight.reshape(conv.out_channels, -1)
output = torch.nn.functional.linear(flat_input, flat_weight, conv.bias)
# output: [P, D]
```

这里直接使用原卷积层的 weight 和 bias，没有新建一套 Parameter。reshape 保留自动求导关系，优化器仍然更新原参数，checkpoint 中的名称与五维形状也不需要改变。

如果改为新建 `nn.Linear` 并复制数值，就要另外处理参数注册、优化器引用与 checkpoint 转换。尤其在优化器已经创建之后替换模块，可能出现前向使用新参数、优化器仍持有旧参数的问题。保留参数对象，通常更容易将改写限制在计算路径内。

输入顺序同样不能省略。相同的元素总数和输出 shape，不代表投影正确；把 `[C,T,H,W]` 改成 `[T,H,W,C]` 后再展平，若权重仍沿用旧顺序，得到的是另一个计算。

# 三、如何接入 LLaMA-Factory 训练栈

本地训练运行时以 LLaMA-Factory 为入口，模型由 Transformers 加载。实际等价补丁作用于 `Qwen3VLVisionPatchEmbed` 和 `Qwen3_5VisionPatchEmbed` 的 forward；训练启动、LoRA、优化器和数据接口沿用原链路。

构建镜像时，补丁脚本检查目标模型文件，匹配预期的原始实现后替换 forward，并保留环境开关：

```bash
# 使用等价矩阵乘路径
export TRAIN_LM_PATCH_EMBED_MATMUL=1

# 对照实验时恢复原 Conv3D 路径
export TRAIN_LM_PATCH_EMBED_MATMUL=0
```

同一份脚本已被 ROCm 和 CUDA 两套 Dockerfile 调用，因此这项改写不再只服务于 AMD。它依赖的是模型结构，不是某个品牌的设备判断；迁移到其他平台时，仍要验证 Linear 的支持情况、数值与性能，不能从已接入两套镜像推断所有平台都更快。

升级 Transformers 后，模型类或 forward 可能变化。补丁需要逐个确认预期模型是否命中，不能只检查脚本退出码为零；原实现未匹配时应保留明确结果，避免镜像构建成功而模型实际没有替换。

LLaMA-Factory 中针对某些 PyTorch 版本的 Conv3D 性能检查，是另外一层兼容逻辑。允许版本继续加载，不等于慢算子已经消失；应在完成替换、确认实际路径之后，再判断是否需要调整对应检查。

# 四、训练改写要同时验证前向与反向

对于 $Y=XW^{\mathsf T}+b$，给定上游梯度 $G$，有：

$$
\frac{\partial L}{\partial X}=GW,\qquad
\frac{\partial L}{\partial W}=G^{\mathsf T}X,\qquad
\frac{\partial L}{\partial b}=\sum_{p=1}^{P}G_p
$$

将输入梯度恢复为原 patch 形状、权重梯度恢复为原五维形状，就应与卷积路径对应。只验证前向结果接近，还不足以将替换用于训练。

局部对照应使用相同输入、参数与上游梯度，分别执行两条路径，再比较输出、输入梯度、weight/bias 梯度。先用 FP32 检查布局，再覆盖实际 BF16/FP16 与 autocast；浮点累加顺序不同，可以使用合理容差，但不应接受明显的排列错误。

还要检查冻结视觉塔、解冻视觉层和梯度检查点重计算等调用方式。不同设置下参与反向的张量不同，纯前向冒烟不能覆盖所有训练配置。

空媒体输入应在模型入口明确旁路。为了让 reshape 成功而临时拼一个空 patch，或为纯文本伪造视觉输入，会改变训练路径，也可能引入新的错误。非连续张量则需要核对 reshape 是否发生复制，不能为省复制而无条件换成 view。

AMD 上从故障隔离、库回退到恢复快速路径的完整过程，见 [AMD Qwen3.5 训练的故障定位与性能优化](../engineering-runtime/AMD_Qwen3.5训练故障.md)；这里集中讨论投影改写本身的结构条件和梯度对应。

# 五、为什么同一种改写能在不同后端复用

AMD 的发现过程是：环境因兼容问题关闭 MIOpen 后，这一 Conv3D 进入异常慢的回退路径，视觉投影占据了大部分训练时间。对结构做等价改写，可以只改变这一层的执行方式，而不必假设所有卷积都应关闭或全部替换。

CUDA 环境也出现过 Conv3D 慢路径。PyTorch 上游 issue #166122 记录了特定版本与 AMP 下的性能回归，本地 CUDA 训练镜像同时包含较新的 PyTorch、FLA 和同一视觉补丁。两种环境遇到慢路径的原因并不完全相同，但都能利用这里的结构约束，将投影交给矩阵乘实现。

这也说明“后端支持 Conv3D”与“当前形状运行高效”是两个问题。算子名相同，实际输入规模、dtype、库版本和执行实现仍可能改变耗时。等价改写的意义是增加一个可验证的执行选择，最终是否保留，要看目标环境的实测。

# 六、性能记录按实验口径分别保存

| 记录 | 观测结果 | 适合说明什么 |
| --- | --- | --- |
| AMD 视觉层 profiling | 某次记录中 Conv3D 约占 278/307 秒 | 原环境的主要耗时集中在视觉投影 |
| AMD 同一组 20 步测试 | 总耗时约 2329.8 → 89.8 秒 | 特定测试中绕开慢路径的端到端收益 |
| CUDA 训练栈整体调整 | 记录约 190 → 2.94 秒/步 | 新 PyTorch、FLA 与补丁等组合后的结果 |

第二组总时长比约为 25.9。早期摘要另写过 21.4 倍，但没有对应的完整计时口径，因此这里保留原始时长计算，不混用倍率。第三组同时更换了多个组件，不能将全部收益归到 Conv3D 改写。

从 profiling 比例还能判断优化上限。若某层占总耗时比例为 $f$，局部提速为 $s$，其余部分不变，则整体加速比约为：

$$
S=\frac{1}{(1-f)+f/s}
$$

这是一种估算方法，不是把不同测试的时长拼接计算。某次投影占比很高时，改写可能带来明显端到端收益；在另一个环境中该层本来就很快，整体收益可能很小，甚至被额外 reshape 或数据复制抵消。

正式计时应固定 patch 数、图像预算、batch、梯度累积、冻结设置和精度，预热后再同步设备测量。局部前反向、稳态训练 step、评估保存与整次任务总时长分别记录，避免将初始化或数据缓存变化当成算子收益。

# 七、从算子名回到实际计算

这项改造不是发明一种新的卷积算法，而是识别当前层已经退化为独立 patch 的稠密投影。通过换一种等价表达，可以保留模型参数与训练接口，同时利用不同后端已有的矩阵乘实现。

可复用的过程是：先定位真实瓶颈，再确认结构条件，证明索引与梯度对应，接入可回退路径，最后在目标后端比较正确性与端到端收益。模型或框架升级后，这些条件仍然需要重新核对，而不是将补丁永久当作固定模板。

# 八、参考资料

- [PyTorch Conv3d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv3d.html)
- [PyTorch Conv3D AMP 性能回归记录](https://github.com/pytorch/pytorch/issues/166122)
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
