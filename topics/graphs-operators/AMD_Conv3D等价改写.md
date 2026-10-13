---
title: AMD 视觉 patch embedding 的 Conv3D 等价改写
created: 2026-09-21
updated: 2026-09-24
status: draft
blog: true
tags: [AMD, Qwen3-VL, Conv3D, 图优化, 训练加速]
---

# AMD 视觉 patch embedding 的 Conv3D 等价改写

Qwen3-VL 的视觉 patch embedding 可以表现为 Conv3D，但实际输入已经组织成独立 patch。在 kernel 覆盖单个 patch、输出空间为一点的条件下，这个卷积等价于 flatten 后的 Linear。

AMD R9700 的归档训练中，关闭 MIOpen 后该 Conv3D 进入很慢的回退路径，占据大部分步骤时间。针对这层结构进行等价改写，能够绕开通用卷积路径，而无需修改模型参数。

# 一、等价条件

设每个 patch 为 $P_i\in\mathbb{R}^{C\times T\times H\times W}$，卷积权重为 $K\in\mathbb{R}^{D\times C\times T\times H\times W}$。当 kernel 恰好覆盖 patch，且 stride、padding、dilation 保证单点输出时：

$$
y_i=\operatorname{vec}(P_i)K_{\mathrm{flat}}^{\mathsf T}+b
$$

这不是任意 Conv3D 都能改成一个 Linear。输入布局、flatten 顺序和权重顺序需要完全一致。

```python
# 局部替换：一个输入 patch 对应一个输出向量
patches = x.reshape(x.shape[0], -1)
weight = conv.weight.reshape(conv.out_channels, -1)
y = torch.nn.functional.linear(patches, weight, conv.bias)
```

执行前检查实际输入尺寸、kernel 与卷积参数，防止把 padding 或多点卷积路径误用为本例。

# 二、前向与梯度验证

训练替换同时要求前向和反向等价。固定权重、输入与随机种子，对比输出、输入梯度和权重梯度；再测试真实多模态 batch、不同 patch 数以及空媒体分支。

浮点实现的累加顺序可能不同，应使用合理 dtype 容差，而不把 bitwise 相等作为所有设备的通用要求。checkpoint 的参数名称与形状不变，转换也更容易维护。

# 三、性能口径

归档记录同一 20 步测试的总耗时从约 2329.8 s 降到 89.8 s，按这两个总时长计算约为 25.9 倍。其他摘要曾写 21.4 倍，缺少对应计时边界，因此这里保留原始时长，避免混用倍率。

端到端时长包含数据准备、评估或保存时，还需单独测稳态每步与视觉层占比。收益来自原环境的异常慢路径，不能外推成所有 Conv3D 或所有 AMD GPU 都能获得同等提升。

# 四、为什么不能只比较前向

Conv3D 与 Linear 的输入梯度分别由其反向 kernel 计算；对于展平映射，需要把 Linear 的输入梯度还原到相同 patch 顺序。权重梯度也必须映射回原来的五维参数。只验证输出相等，可能漏掉训练阶段的错误。

同一测试还应覆盖非连续张量。reshape 可能产生复制，view 则要求符合 stride；避免为了省一次复制而破坏布局假设。性能计时需要在前后同步 GPU，否则只测到 CPU 提交时间。

替换层的参数仍由原模块持有，优化器与 state_dict 不变。若通过新建 Linear 复制参数实现，要额外验证优化器是否仍引用旧参数，以及 checkpoint 加载后是否重复初始化。

# 五、一些反思

模型结构给出了比算子名字更具体的优化机会。通用卷积 kernel 需要处理很多情况，独立 patch 的单点输出则可以降为矩阵乘；利用真实约束，比无条件关闭优化库更精确。

补丁也需要版本边界。给替换加入结构断言和可关闭开关，并在升级 transformers 后回归，避免模型实现变化后静默应用错误改写。

# 六、参考资料

- [PyTorch Conv3d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv3d.html)
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)

