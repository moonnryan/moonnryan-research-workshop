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

本文对应 RKNN-Toolkit2/Runtime 2.3.2 的 RK3576 复测及 RK3588 历史对照，不将这些结果泛化到其他版本。

# 一、GridSample 的作用

```python
torch.nn.functional.grid_sample(
    feature, grid,
    mode="bilinear",
    padding_mode="zeros",
    align_corners=False,
)
```

它根据 decoder 预测的连续坐标读取特征，不是固定 Resize。归档 DEIMv2-S 的 4 层 decoder、3 个特征层级形成 12 次 GridSample，无法直接删去而保持模型语义。

对于 align_corners=False，归一化横坐标 $u$ 映射到像素坐标：

$$
x=\frac{(u+1)W-1}{2}
$$

插值、越界补零和坐标约定都必须一致。CPU 自定义实现仅仅“输出 shape 相同”，不足以证明正确。

# 二、回退成本

| 设备与方案 | 单帧归档结果 | 边界 |
| --- | ---: | --- |
| RK3588，12 次 CPU 回退 | 约 961.3 ms | 双线性语义 |
| RK3588，融合 CPU | 约 856.5 ms | 减少边界，仍非实时 |
| RK3588，融合 GPU | 约 952.5 ms | GPU kernel 不等于整图加速 |
| RK3576，CPU 回退 | 约 1.05–1.21 s | 12 次调用 |
| RK3576，discrete | 约 0.93–1.04 s | 采样语义改变 |

RK3576 的 GPU 自定义算子注册在该组合中失败。通用自定义算子目标为 CPU/GPU，注册回调不会给 NPU 新增一个硬件 GridSample kernel。

# 三、discrete 的边界

discrete 用整数索引替代双线性采样，可避开 GridSample，却改变了计算。按双线性训练的模型直接切换 discrete，应重新比较 AP 和边界、小目标样本。

即使不再出现不支持算子，DETR 图中的 MatMul、Gather、Transpose 和小算子调度仍可能昂贵。因此离散路径“可转换”不能推导出“可实时”。

# 四、边界成本的测量

设一次推理产生 $k$ 次 CPU/NPU 边界，端到端时间可以拆为：

$$
T=T_{\mathrm{NPU}}+T_{\mathrm{fallback}}
+\sum_{i=1}^{k}(T_{\mathrm{sync},i}+T_{\mathrm{layout},i}+T_{\mathrm{transfer},i})
$$

只对 GridSample 的 CPU kernel 计时，会遗漏其余三项；DMA-BUF 共享减少显式拷贝，也不能消除同步和布局转换。可分别测试固定张量的单算子、NPU 前后段和整链，对比纯 CPU 图，判断异构方案是否真正值得保留。

对于矩形输入与边界坐标，建立专门的插值单元样本：中心、四角、越界和半像素位置。方形图像上的一次推理通过，无法验证坐标缩放和 clamp 在另一 shape 下正确。

# 五、一些反思

算子支持表只是部署的第一层。真正的成本取决于支持子图如何连接，以及数据在哪些设备和布局之间流动。融合回退调用可以改善延迟，但也应明确改善上限。

模型选型应在大规模训练前做目标设备最小部署。若核心结构需要频繁异构回退，换一个已验证的 CNN 检测基线，可能比继续补算子更符合任务时延。

# 六、参考资料

- [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2)
- [PyTorch grid_sample](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html)
- [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2)

