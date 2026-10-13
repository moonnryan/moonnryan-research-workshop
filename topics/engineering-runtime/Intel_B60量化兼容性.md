---
title: Intel B60 的量化模型与优化路径兼容性
created: 2026-08-12
updated: 2026-08-12
status: draft
blog: true
tags: [Intel, B60, Qwen3.5, GPTQ, vLLM, KV Cache]
---

# Intel B60 的量化模型与优化路径兼容性

量化模型在 XPU 上能够加载，不代表生成路径与所有优化组合都正确。归档 Qwen3.5-9B GPTQ 测试中，旧镜像曾产生无关回答和非法 JSON；升级后又出现量化层字段与 GDN 输出投影优化不兼容的问题。

因此版本升级需要同时验证加载、首请求、图文生成、结构化输出和重复请求，不能仅以服务返回 health 200 判定完成。

# 一、量化表示与实际执行

GPTQ 权重常以 qweight、qzeros、scales 保存，未必存在普通浮点 Linear 的 weight。优化代码若直接访问 weight，就可能在首次执行时失败：

```text
AttributeError: 'RowParallelLinear' object has no attribute 'weight'
```

归档 `intel/llm-scaler-vllm:0.21.0-b3` 提供 `DISABLE_ESIMD_GDN_OUTPROJ=1` 开关，禁用冲突的 GDN 输出投影优化后，单 B60 的 GPTQ＋FP8 KV＋eager 组合通过已记录回归。

```bash
export DISABLE_ESIMD_GDN_OUTPROJ=1
```

这个开关只对应所测版本与故障路径，不能写成所有 B60 部署都需要的永久配置。

# 二、分开评估优化项

| 项目 | 主要改变 | 验证内容 |
| --- | --- | --- |
| 权重量化 | Linear 表示与 kernel | 生成内容、视觉与 JSON |
| FP8 KV | 缓存容量与读写 | 长上下文数值与容量 |
| XPU graph | 捕获与执行路径 | 首请求、重复请求、shape |
| GDN 优化 | 线性注意力实现 | dtype、权重接口与数值 |

FP8 KV 可以提升可用 token 容量，但容量提升倍率受权重、缓存预留和实现影响，不能直接解释为等倍率吞吐改善。eager 基线通过后再启用 graph，便于定位新故障。

# 三、版本与因果强度

旧镜像后续有相关精度修复，支持把版本列为重要因素；但旧服务重启或调整参数后也曾短暂通过，尚不能严格证明旧版本是唯一原因。

发布说明应写成“固定新组合在回归中未复现”，同时保存模型哈希、镜像 digest 和参数。不要把不同量化包、TP 数和运行状态混在同一升级对照中。

# 四、建立最小回归集

先用自然语言描述确认视觉输入有效，再用同一张图片要求结构化结果；如果前者正确而后者失败，应分别检查语言生成、量化执行与格式约束，不把问题一概归到图片下载。

回归集可以固定短数学题、普通问答、单图描述、图文 JSON 和重复请求，另外加入首次请求、容器重启后的请求与长上下文。保存原始输出、退出状态和 OOM 标记，而非只保存解析后的 hit。

升级时从 eager 与已知模型开始，再逐项恢复 FP8 KV、graph 和其他优化。若同时换模型包、镜像与全部参数，成功只能说明新组合可用，无法判断旧故障究竟由哪一项消除。

# 五、一些反思

优化路径假设的权重结构，是量化兼容性的隐藏边界。新优化不只需要计算等价，还需要正确处理不同 Linear 表示。

接口回归也应包含输出语义。重复换行、无关内容和 JSON 合法性都可以在进程不退出时出现；服务可达只是最外层检查。

# 六、参考资料

- [Intel llm-scaler](https://github.com/intel/llm-scaler)
- [Qwen3.5](https://github.com/QwenLM/Qwen3.5)

