---
title: Sophgo W4 通道对齐问题
created: 2026-04-21
updated: 2026-04-21
status: draft
blog: true
tags: [BM1684X, TPU-MLIR, Qwen3.5, W4, 权重重排]
---

# Sophgo W4 通道对齐问题


在 BM1684X 上将 Qwen3.5-2B 转为 W4BF16 时，编译出现 `invalid scale channel`。W4BF16 表示四位权重配合 BF16 激活，是权重量化路径；量化后的 scale、zero point 与权重本身都需要转换为目标后端能够读取的布局。

复现组合为 tpu_mlir 1.28.1、BM1684X、Qwen3.5-2B。`q_group_size=128` 和 64 都失败，BF16 转换通过。这首先是后端模型生成问题，还没有进入“量化模型可运行，但任务精度下降”的阶段。

因此，“工具支持 W4”只是入口条件。实际模型的层形状、分组方式、元数据布局和 kernel 约定也必须匹配，才会生成可用产物。

# 一、分组与通道对齐

按输出通道组织 Linear 权重为 $[N,K]$，其中 $K$ 是输入特征数、$N$ 是输出特征数。对于此处分组量化的逻辑布局，设 group size 为 $g$，scale 形状为：

$$
\operatorname{shape}(S)=[N,K/g]
$$

这一表达式假设所选路径满足分组条件；实际模型还要核对转置和已发生的重排。

```text
scale: [ N, K/g ]
         ↑    ↑
      通道数  输入方向的组数
```

BM1684X 对应的重排路径需要把通道分配到 $P$ 个 NPU lane，概念变换为：

```text
[N, K/g]
    → reshape [N/P, P, K/g]
    → permute [P, N/P, K/g]
```

在该 BM1684X 路径中 $P=64$。必须先满足 $N\bmod P=0$，才能按这个布局整分。改变 $g$ 影响的是 $K/g$，不会让 $N$ 自动变成 64 的倍数。

| 调整项 | 主要影响 | 是否自动修复通道对齐 |
| --- | --- | --- |
| `q_group_size` | K 方向的分组数量和元数据量 | 否 |
| 输出通道 N | scale 的通道维 | 还需保持模型语义 |
| 浮点路径 | 不使用这套 W4 重排 | 可用于隔离问题，仍需完整验收 |

这里的 64 指芯片内部 lane 数，不是 64 张卡，与 `num_device` 也不是同一个概念。

# 二、重排的约束

关键检查及后续维度计算可以压缩为：

```cpp
int64_t npu_num = backend::Arch::NPU_NUM;
if (scale_shape[0] % npu_num) {
    llvm_unreachable("invalid scale channel");
}
auto groups = scale_shape[1];
auto channels_per_lane = scale_shape[0] / npu_num;
```

这个检查并非多余限制。后续重排以整数 $N/P$ 为基础计算偏移和目标形状，尾部通道没有被这条路径处理。直接删除断言，可能把显式编译失败变成错误布局、输出损坏或越界。

一种值得研究的修复方向是 padding，但它要同时覆盖权重、scale、zero point、kernel 和输出裁剪。只把一个 shape 改大，并不等于完成等价转换。

例如对单个 Linear，按行补零可写为：

$$
W' = \begin{bmatrix}W\\0\end{bmatrix},\qquad
Y'=X(W')^{\mathsf T},\qquad Y=Y'[:,0:N]
$$

这是局部数学等价的示意。它不能直接证明量化后的布局等价，也不能让额外通道不经裁剪就进入后续 attention、reshape 或残差。修复必须在正确边界上补齐，再恢复原接口。

# 三、排查与验证

先保存失败 block 的 MLIR，核对 A16MatMul 的权重、scale 和 zero point 形状；然后确认到底进入哪条重排分支。这样才能把报错落到具体层及布局条件，而不是继续盲调 group size。

| 检查层次 | 要回答的问题 |
| --- | --- |
| 导出图 | 哪个投影层产生了不满足条件的 N？ |
| 量化数据 | scale 的转置、分组和 padding 是否一致？ |
| 编译路径 | 浮点通过而 W4 失败，差异集中在哪里？ |
| 修复验收 | 生成模型后，局部输出和整模型任务是否都正确？ |

已有证据解释了为何修改 group size 无效，但尚未记录具体投影层的 N，也没有新版工具链复测。修复方案应当继续验证，不能把这篇诊断当成已完成补丁。

# 四、一些反思

量化故障至少分成两类：一种是模型已经可运行，需要研究误差如何影响输出；另一种是后端根本无法表示当前量化布局。校准集和敏感层搜索主要帮助前者，对后者应先检查 shape、对齐和数据重排。

判断参数是否值得调整，要先知道它控制哪个维度。group size、NPU lane 数和多设备数量虽然都表现为整数配置，作用对象却不同；定位到实际 shape 后，许多无效尝试可以直接排除。

修复断言类问题时还应追问：这个条件保护了什么后续不变量？满足条件的等价变换通常比删除检查更有价值，因为它同时解释正确性和可维护性。

# 五、参考资料

- [TPU-MLIR issue #269](https://github.com/sophgo/tpu-mlir/issues/269)
- [BM1684X A16MatMul 权重重排](https://github.com/sophgo/tpu-mlir/blob/master/lib/Dialect/Tpu/Transforms/WeightReorder/BM1684X/A16MatMul.cpp)
- [LLM Linear 与 scale 形状构造](https://github.com/sophgo/tpu-mlir/blob/master/python/llm/LlmConverter.py)
