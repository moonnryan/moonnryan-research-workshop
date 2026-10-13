---
title: Qualcomm llama.cpp 的后端放置与性能边界
created: 2026-07-27
updated: 2026-09-09
status: draft
blog: true
tags: [Qualcomm, llama.cpp, MNN, Hexagon, OpenCL, 性能]
---

# Qualcomm llama.cpp 的后端放置与性能边界

HTP prefill 更快而 decode 更慢并不矛盾。prefill 处理多个 token，容易形成较大矩阵计算；decode 每步生成一个 token，更易受权重读取、CPU 回退和同步影响。峰值 TOPS 不能直接换算成生成速度。

一组 QCS8550 多模态归档测试中，HTP prompt processing 约 194.57 token/s、decode 约 10.52 token/s；CPU 分别约 21.96 与 13.88 token/s。HTP 改善首段计算，但生成仍慢于 CPU。

# 一、卸载层数与算子放置

ngl all 表示尝试卸载模型层，不能保证所有节点由目标执行。视觉图中的布局与插值节点可能回退 CPU，输出投影也受形状和量化类型限制。

| 检查项 | 证据 |
| --- | --- |
| 语言层 | 实际 offload 日志 |
| 视觉编码 | CLIP/MTMD 设备日志 |
| 回退节点 | graph split 与 profiler |
| 输出投影 | supports_op 条件 |
| 同步成本 | 分阶段计时 |

归档 Hexagon 路径对量化输出投影的大词表维度有限制。词表约 248320、隐藏维度 2048 的权重形成很大投影；每 token 若这部分落 CPU，decode 收益可能被抵消。

# 二、buffer 与能力声明

OpenCL 单 buffer 最大分配量与总可用内存不同。模型总量能放入内存，不代表一个权重 tensor 能分配成功。应把设备最大分配量与具体 tensor 大小比较。

Hexagon 的维度保护也需要理解。删除 32768 等限制前，检查它保护的 VTCM、索引和 kernel 条件。F16 混合权重可以实验，但改变体积、带宽和计算成本，不保证更快。

# 三、MNN 对照的解释

同一 QCS9075 的归档 Qwen3.5 对照中，MNN W4A16 的 prefill 未优于 llama.cpp Q8。限制包括 CPU i8mm 路径未启用、OpenCL 解量化与 kernel 差异，以及 HTP 上部分线性注意力回退。

这些结果比较“模型表示＋框架实现＋硬件路径”。W4 权重更小不能说明 kernel 更快，其他 SoC 的优势也不能直接外推。

# 四、测量条件

$$
T_{\mathrm{request}}=T_{\mathrm{media}}+T_{\mathrm{vision}}
+T_{\mathrm{prefill}}+T_{\mathrm{decode}}+T_{\mathrm{queue}}
$$

固定图像、prompt、生成长度、CPU 线程、功耗和预热。跨模型比较优先记录 TTFT 与请求总延迟，因为 tokenizer 和视觉 token 数不同会改变 token/s 的含义。

# 五、单路 decode 与批处理

归档 Hexagon 路径中，单路 decode 的矩阵行数常为 $M=1$，部分 HMX 路径要求更大的 M，因此落到另一类实现。增加并行会话可能提升总体吞吐，但同时改变每请求延迟和 K/V 容量，不能解释为单路 decode 自动加速。

多 HTP session 可以缓解单个映射空间限制，也会引入切分与同步。归档双 session 能加载原先受限的 F16 模型，但生成速度没有相同幅度收益；“成功加载”与“更快运行”需要分别验收。

多模态再增加一个视觉后端，可能形成 HTP、OpenCL 和 CPU 混合链。应在同图请求中记录视觉阶段与语言阶段，避免纯文本基准掩盖实际视频复判的放置成本。

# 六、一些反思

先问哪部分没有实际加速。放置与阶段分开测量后，才能决定修 kernel 支持、调整量化表示还是减少同步边界。

框架选型依赖真实请求分布。短回答中 decode 慢的影响有限，长文本会持续放大 CPU 回退的代价；需要同时比较数值精度与整请求体验。

# 七、参考资料

- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [MNN](https://github.com/alibaba/MNN)
- [OpenCL 设备查询](https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clGetDeviceInfo.html)

