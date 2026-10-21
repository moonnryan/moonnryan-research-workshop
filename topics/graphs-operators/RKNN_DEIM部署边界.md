---
title: RKNN 上 DEIM 的采样语义与部署边界
created: 2026-08-11
updated: 2026-08-12
status: draft
blog: true
tags: [Rockchip, RKNN, DEIMv2, GridSample, 异构推理]
---

# RKNN 上 DEIM 的采样语义与部署边界

DEIMv2 的可变形注意力依赖连续坐标采样。RKNN 能生成文件，不代表该采样已经在 NPU 上实现；注册 CPU 或 GPU 自定义算子可以补足执行，但会引入图切分、同步和布局转换。部署边界同时包含可执行性、数学语义和延迟。

本文对应 RKNN-Toolkit2/Runtime 2.3.2、RKNPU Driver 0.9.8 的 RK3576 复测及 RK3588 历史对照。导出基线为 DEIMv2-S、静态 batch=1、640×640，不将这些结果泛化到其他模型规格和版本。

Toolkit2、板端 Runtime 与最小模型的环境验证见 [Rockchip RKNN 环境搭建与板端推理](../tutorial/Rockchip_RKNN环境与推理.md)；DEIM 权重、前处理与导出参考见 [DEIM 微调中的配置、前处理与权重产物](../models-training/DEIM_微调与产物契约.md)。

# 一、GridSample 的作用

```python
torch.nn.functional.grid_sample(
    feature, grid,
    mode="bilinear",
    padding_mode="zeros",
    align_corners=False,
)
```

它根据 decoder 预测的连续坐标读取多尺度特征，再按注意力权重聚合，不是固定 Resize。采样位置随输入与 query 改变，不能预先折叠成一张固定缩放表。归档 DEIMv2-S 的 4 层 decoder、3 个特征层级形成 12 次 GridSample，无法直接删去而保持模型语义；这也不是所有 DEIM 变体都固定有 12 次调用。

对于 align_corners=False，归一化横坐标 $u$ 映射到像素坐标：

$$
x=\frac{(u+1)W-1}{2}
$$

纵坐标同理使用特征图高度 $H$。双线性插值读取相邻四点，按坐标距离加权；例如特征为 `[[0,10],[20,30]]`，像素坐标 $(x,y)=(0.25,0.5)$ 的结果为 12.5，不能用某一个整数位置的值替代。

`align_corners=False` 时，$u=-1$ 映射到 $x=-0.5$，并不是第一个像素中心。使用 zeros padding 时，应按四个邻点各自是否越界决定补零，不能先将整个采样坐标 clamp 到图内，那会变成不同的边界处理。坐标顺序 `(x,y)`、半像素偏移、插值和补零都需要一致，输出 shape 相同不足以证明实现正确。

# 二、回退成本

在该版本组合中，转换器可能生成带有待执行 GridSample 节点的 RKNN，而板端默认 Runtime 缺少对应实现，报 `Unsupport CPU op: GridSample`。因此要分别检查 ONNX 导出、RKNN 构建、Runtime 初始化和真实前向，文件生成成功不是最后一步。

注册 CPU/GPU 自定义算子补的是回退实现，不是给 NPU 动态安装一个新 kernel。GPU 回调也只接管相应节点，不会把整个 RKNN 网络改成 Mali GPU 图；若要比较整图 GPU，需要使用另一个支持该模型的 GPU 推理后端。

| 设备与方案 | 单次整图前向归档结果 | 边界 |
| --- | ---: | --- |
| RK3588，12 次 CPU 回退 | 约 961.3 ms | 双线性语义 |
| RK3588，融合 CPU | 约 856.5 ms | 减少边界，仍非实时 |
| RK3588，融合 GPU | 约 952.5 ms | GPU kernel 不等于整图加速 |
| RK3576，CPU 回退 | 约 1.05–1.21 s | 12 次调用 |
| RK3576，discrete | 约 0.93–1.04 s | 采样语义改变 |

RK3576 的 GPU 自定义算子在初始化后注册失败，不能当作零耗时或模型已能运行。RK3588 数字来自前期历史记录，RK3576 数字是少量重复调用的范围；表中混合了平均值与范围，不是统一条件下的跨芯片性能排名，也不包含完整视频链路。

融合 CPU 方案从约 961.3 ms 降至 856.5 ms，整图耗时减少约 10.9%。它说明减少回退调度和边界有收益，却没有让模型进入几十毫秒级。融合 GPU 仍约 952.5 ms，也说明“把慢算子改成 GPU”不是端到端提速的充分条件。

官方 v2.5.0 变更记录已列出 GridSample 的 GPU 支持。升级后应重新核对 SoC、采样模式、padding、布局与实际落点，不能沿用 2.3.2 的失败结论；GPU 支持也不等于 NPU 原生支持，本文没有新版的整图复测结果。

另一个减少异构边界的实际方案见 [Qualcomm HTP 上 DEIM 的数值误差与拆图](../quantization-accuracy/Qualcomm_DEIM精度归因.md)：将三层特征作为切点，整个 decoder 放在 CPU。它处理的是 HTP 数值偏差，不能直接当作本设备上已经验证的部署方案。

# 三、discrete 的采样差异

discrete 用整数索引和 Gather 替代双线性采样，可避开 GridSample，却改变了计算。历史实现使用未经变换的采样位置 $r\in[0,1]$，按 `r * [W,H] + 0.5` 转整数，而默认分支先将 $r$ 映射到 $[-1,1]$ 再执行 GridSample。两者不能只因输出 shape 一致，就认定坐标与采样等价。

边界也不同：整数索引路径通过 clamp 避免越界，默认路径则对越界邻点补零。历史 discrete 实现还对两个坐标统一 clamp 到 `H-1`，矩形特征图需要分别核对横轴 `W-1` 与纵轴 `H-1`。修正这个索引边界，只能修复该处问题，不会恢复双线性插值语义。

按双线性训练的模型直接切换 discrete，应作为另一份模型重新比较 boxes、scores、labels 与数据集 AP，特别检查小目标、边界和采样位置敏感样本。若进一步按离散路径训练或微调，也需要保存新的训练与导出配置，不能继续沿用原双线性模型的评测结论。

即使不再出现不支持算子，DETR 图中的 MatMul、Gather、Transpose 和小算子调度仍可能昂贵。因此离散路径“可转换”不能推导出“可实时”。

# 四、边界成本的测量

设一次推理产生 $k$ 次 CPU/NPU 边界，端到端时间可以拆为：

$$
T=T_{\mathrm{NPU}}+T_{\mathrm{fallback}}
+\sum_{i=1}^{k}(T_{\mathrm{sync},i}+T_{\mathrm{layout},i}+T_{\mathrm{transfer},i})
$$

这个分解用于梳理串行关键路径，实际存在重叠时以端到端计时为准；$k$ 是实际执行边界数，不应直接等同于 ONNX 中 GridSample 的节点数。只对 CPU kernel 计时会遗漏其余三项，DMA-BUF 共享减少显式拷贝，也不能消除同步、缓存一致性处理和布局转换。

部署验证可以按三层保留对照：

| 层次 | 固定内容 | 需要回答的问题 |
| --- | --- | --- |
| 单算子 | 同一份特征与 grid，对比 PyTorch/ONNX 参考 | 采样数值、边界与 dtype 是否一致 |
| 整图 | 同图分别测原始双线性回退、融合回退与 discrete | 结构改写和边界减少各带来什么变化 |
| 任务 | 正式图片与数据集，包含前后处理 | 框、分数、AP 与真实延迟是否仍可接受 |

预热后分别统计纯前向与端到端 P50/P95，并记录 Toolkit、Runtime、驱动、CPU 线程数和 NPU 核配置。未分项 profile 前，不应仅凭 discrete 仍慢，就断言某一种 MatMul 或 Gather 是唯一瓶颈。

对于矩形输入与边界坐标，建立专门的插值单元样本：中心、四角、越界和半像素位置。方形图像上的一次推理通过，无法验证坐标缩放和 clamp 在另一 shape 下正确。

整图还要给 `orig_target_sizes` 输入真实原图 `[W,H]`，并查询平台模型实际接受的 dtype。用 `[0,0]` 跑性能可能仍能完成前向，却会让输出框缩放到零，不能据此判断检测有效。`labels/boxes/scores` 已由导出 PostProcessor 组织，不应再套用 YOLO decode。

# 五、一些反思

算子支持表只是部署的第一层。真正的成本取决于支持子图如何连接，以及数据在哪些设备和布局之间流动。融合回退调用可以改善延迟，但也应明确改善上限。

模型选型应在大规模训练前做目标设备最小部署。若核心结构需要频繁异构回退，换一个已验证的 CNN 检测基线，可能比继续补算子更符合任务时延。

# 六、参考资料

- [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2)
- [RKNN 官方版本变更记录](https://github.com/airockchip/rknn-toolkit2/blob/master/README.md#changelog)
- [PyTorch grid_sample](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html)
- [ONNX GridSample 语义](https://onnx.ai/onnx/operators/onnx__GridSample.html)
- [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2)

