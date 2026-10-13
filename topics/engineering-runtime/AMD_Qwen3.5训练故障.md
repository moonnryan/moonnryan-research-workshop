---
title: AMD Qwen3.5 训练的 hipBLASLt fault 定位
created: 2026-09-22
updated: 2026-09-23
status: draft
blog: true
tags: [AMD, R9700, ROCm, Qwen3.5, hipBLASLt]
---

# AMD Qwen3.5 训练的 hipBLASLt fault 定位

Qwen3.5 在 R9700（gfx1201）上的训练曾在第 2–8 步触发 GPU memory access fault，而相同设备上的 Qwen3-VL 可以运行。故障随输入与内存布局变化，单次普通 GEMM 探针通过并不足以排除底层库问题。

固定数据、训练配置和宿主驱动后，旧栈关闭 hipBLASLt 可完成 100 步；换成包含相关 gfx1201 修复的新用户态栈后，重新启用 hipBLASLt 也完成 100 步。该对照支持把故障收敛到旧用户态计算路径。

# 一、排除模型与数据因素

纯文本也能复现，排除了视觉塔作为唯一原因。FLA delta-rule 单独探针在多种长度上通过，不能再因为模型使用线性注意力就直接归因于 FLA。

同一 tokenized 数据在修复栈完成训练、评估和保存，支持排除数据损坏；同宿主驱动不变，说明修复不依赖一次宿主升级。每项结论都来自固定条件的对照，而非错误出现的 Python 行号。

# 二、关闭路径要核实实际调用

仅设置 PyTorch 的 BLAS 偏好仍可能由 rocBLAS 转交 hipBLASLt。归档旧栈的稳定回退需要三项变量共同生效：

```bash
export TORCH_BLAS_PREFER_HIPBLASLT=0
export ROCBLAS_USE_HIPBLASLT=0
export DISABLE_ADDMM_HIP_LT=1
```

这些是版本相关的诊断配置。需要用库日志确认实际调用，而不是只检查环境变量存在。新栈复测则要求确实重新进入 hipBLASLt，才能形成“关闭稳定、修复后开启稳定”的闭环。

# 三、结果与因果边界

| 路线 | 完成情况 | 平均步耗时 | train loss |
| --- | --- | ---: | ---: |
| 旧栈正常路径 | 早期 fault | 未完成 | 未完成 |
| 旧栈三变量回退 | 100 步 | 23.77 s | 1.0896 |
| 固定新栈，hipBLASLt 开启 | 100 步 | 6.20 s | 1.0899 |

固定数据、LoRA、BF16、梯度累积和 checkpoint 设置下，新栈较稳定回退快约 3.83 倍，数值轨迹接近。该测试验证所测配置，并非无限时长稳定性保证。

现象与 gfx1201 GLTr 转置 kernel 越界修复高度一致，但没有逐 commit 构建二分，因此不能严格归因于一个提交。对共享负载上的 Qwen3-VL 耗时波动，也不应直接断言为新版性能回归。

# 四、一些反思

GPU fault 容易被误当成显存碎片。allocator 改变布局可能改变触发时间，却不能修复越界 kernel；应以执行路径对照和长测结果判断。

稳定版本需要保存完整用户态组合和 wheel 来源。浮动 nightly 无法重复复现。升级后除了原故障模型，还应回归已运行的另一架构，避免修复一个模型时引入新的兼容问题。

# 五、参考资料

- [ROCm rocm-libraries PR #8909](https://github.com/ROCm/rocm-libraries/pull/8909)
- [rocBLAS 环境变量](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/reference/env-variables.html)
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)

