---
title: NVIDIA KV Cache 量化中的 NVFP4 与 TurboQuant
created: 2026-10-20
updated: 2026-10-20
status: draft
blog: true
tags: [NVIDIA, vLLM, KV Cache, NVFP4, TurboQuant]
---

# NVIDIA KV Cache 量化中的 NVFP4 与 TurboQuant

长上下文和多请求推理中，权重能放进显存，不代表服务还留有足够的 KV Cache 空间。历史 K/V 随序列增长，也会在解码时被反复读取。降低缓存位宽既能增加容量，也可能减少访存，但收益取决于量化误差、解包开销和实际 attention 内核。

NVFP4 与 TurboQuant 都能把缓存压到接近 4 bit，却采用不同的数值表示。前者使用分块缩放的浮点格式，后者在常见实现中结合旋转、码本与标量量化。比较时需要同时看存储布局、计算路径和服务负载，单凭“硬件原生”或“压缩倍数”还不能判断哪一种更合适。

# 一、KV Cache 到底占多少空间

对各层结构相同、保留完整历史 K/V 的模型，缓存数据量可近似写成：

$$
M_{\mathrm{KV}} \approx
2 L B T H_{\mathrm{kv}} d_h \frac{b_{\mathrm{eff}}}{8}
$$

其中 $L$ 为 attention 层数，$B$ 为同时驻留的序列数，$T$ 为每条序列缓存长度，$H_{\mathrm{kv}}$ 为 KV 头数，$d_h$ 为头维度；$b_{\mathrm{eff}}$ 要包含 scale、范数等元数据。长度不等时，用各序列长度之和替代 $BT$。

GQA 减少的是 KV 头数。滑动窗口、MLA，以及混合线性 attention 模型则需要按各自状态布局计算，不能统一套用上式。Mamba、Gated DeltaNet 等层的状态也不会因为完整 attention 层换成 4-bit KV，就按同样比例缩小。

权重量化与缓存量化是两个设置。模型名中的 NVFP4 通常描述权重格式；服务是否使用 NVFP4 KV，仍要检查缓存配置和实际选中的后端。

# 二、NVFP4：每 16 个数共享一个尺度

NVFP4 使用 E2M1 表示数值，并为每 16 个元素保存一个 FP8 E4M3 block scale；格式还可配合更外层的全局尺度。解码可概括为：

$$
\hat x_i = s_{\mathrm{global}}\,s_{\mathrm{block}}\,operatorname{FP4}(q_i)
$$

每个小块有自己的尺度，可以适应局部数值范围，避免整条向量共用一个尺度时被少量大值拉宽。但块内仍只有有限的浮点表示，缩放不能消除量化误差。

16 个 FP4 数占 8 字节，再加 1 字节 block scale，合计 9 字节。忽略全局尺度、分页对齐和工作区后：

$$
b_{\mathrm{eff}}=4+\frac{8}{16}=4.5\ \mathrm{bit},\qquad
\frac{M_{\mathrm{NVFP4}}}{M_{\mathrm{FP8}}}=\frac{9}{16}=56.25\%
$$

因此相同缓存预算下，理论容量约为 FP8 的 $16/9\approx1.78$ 倍、BF16 的 $32/9\approx3.56$ 倍。这里不能直接按 4 bit 推成 BF16 的四倍。

一份 Qwen3-VL-8B 容量记录中，FP8 为 272,576 tokens，NVFP4 为 484,544 tokens，比例约 1.78，与上述存储比例接近。这份记录只说明可分配 token 池的变化，没有同时测得延迟或任务精度；它也不等于单请求能使用这么长的上下文，或并发数必然增加相同倍数。

# 三、TurboQuant：先改变分布，再减少表示位数

直接对向量做低位宽均匀量化时，少数离群坐标容易占据大部分数值范围。TurboQuant 的核心思路是先归一化并进行随机正交旋转，让信息分散到各坐标，再使用适合该分布的标量码本。

设 $R$ 为正交矩阵，向量 $x$ 的范数为 $n$，则一种简化表达是：

$$
z=R\frac{x}{n},\qquad
q_i=\arg\min_j |z_i-c_j|,\qquad
\hat x=nR^T c_q
$$

$c_j$ 是码本中的重建值。4-bit 索引可以选择 16 个值，但这些值不必等间距。Lloyd–Max 量化会把更多表示能力放在概率较高的区域，以减小均方误差。

论文讨论的随机旋转与工程中的快速旋转还要区分。Hadamard 变换能降低旋转成本，但特定结构化旋转不能直接等同于理论分析中的 Haar 随机矩阵。论文还包含用于内积误差校正的 QJL 残差方案，框架中的同名实现未必完整采用这条路线。

以当前 vLLM 源码中的预设为例，K 使用 Hadamard 旋转和 Lloyd–Max 码本，V 使用带 scale、zero point 的均匀量化。`turboquant_4bit_nc` 开启的是 **norm correction，范数校正**：修正码本重建造成的范数偏差。它与 QJL 残差校正不是一回事。

按该布局，头维度为 128 时，4-bit K 占 $64+2$ 字节，4-bit V 占 $64+4$ 字节；一对 K/V 共 134 字节，相比 BF16 的 512 字节，压缩约 3.82 倍。这个结果包含向量元数据，但尚未计入整个服务的额外开销。

MNN 的实现采用不同分块与元数据布局，不能照搬 vLLM 的压缩率。它在旋转域内组织 attention 的做法则很有参考价值：正交旋转保留内积关系，查询可以旋转后与压缩 K 配合计算，V 的加权结果也可以在最后统一变换回来，从而减少整份缓存还原和搬运。

# 四、缓存是 FP4，不代表矩阵乘法也是 FP4

KV dtype 决定缓存如何保存，attention 后端决定读取之后如何计算。二者之间可能有解包、缩放、类型转换和临时缓冲，不能把 NVFP4 KV 简化成“直接做原生 FP4 attention”，也不能把 TurboQuant 简化成“每次先还原整份 BF16 缓存”。

LMSYS 的 SGLang 实现就是一个例子：SM120 decode 内核读取 NVFP4 数据和 block scale，在内核中重建成 BF16，再进行矩阵乘法。它仍然节省缓存读取流量，因为反量化在内核内部完成，没有把整份高精度缓存写回显存。

vLLM 的 TurboQuant 代码也同时包含压缩缓存 decode 内核和完整反量化路径。初次 prefill 可以使用未压缩的当前 K/V；续写 prefill 需要访问历史缓存，处理方式又不同。解包是否融合、临时张量有多大，必须沿实际分支检查。

| 比较项 | NVFP4 KV | vLLM TurboQuant 4-bit 预设 |
| --- | --- | --- |
| 数值表示 | FP4 数值与分块尺度 | K 为旋转后的码本索引，V 为均匀量化值 |
| 额外信息 | block scale、全局尺度等 | K 范数、V scale 与 zero point 等 |
| 主要处理 | 解包、尺度恢复与类型转换 | 旋转、查码本、范数校正和 V 重建 |
| 计算精度 | 由具体内核决定 | 由具体内核决定 |
| 性能来源 | 少读缓存、扩大驻留容量，以及后端优化 | 少读缓存、扩大驻留容量，以及融合解码实现 |

FP8 同样不能理解为所有 attention 运算都在 FP8 上完成。即使缓存 dtype 相同，GPU 架构、prefill 与 decode 后端也可能不同。

# 五、旧架构如何读写 NVFP4

vLLM v0.32.0 的发布说明已包含 NVFP4 KV 在 SM8x、SM90 和 SM12x 上的支持。沿源码看，这套适配把缓存表示与计算类型分开：显存中的 K/V 保持压缩，GPU 读取后恢复数值，再交给已有的 FP16/BF16 attention 路径。软件恢复也在 GPU 上完成，没有把 attention 转到 CPU。

| 架构与改动 | 对应路径 | 需要关注的区别 |
| --- | --- | --- |
| SM8x，PR #46963 | FlashInfer 写入，FA2 prefill / decode | 通过软件恢复 NVFP4 数值 |
| SM90，PR #60620 | FA2 prefill / decode | 不能直接套用 FP8 的 FA3 / XQA 路径性能 |
| SM12x，PR #46963、#60452 | FA2 prefill，满足条件时 XQA decode | 后端选择还会影响图捕获和多 token 验证 |
| SM100，PR #59300 | 扩展到 MiniMax-M3 稀疏 attention | 属于特定模型和内核支持，不能外推到所有模型 |

**架构选择与写入。** `flashinfer.py` 的 `_nvfp4_kv_on_fa2()` 将 SM8x、SM90、SM12x 分到兼容路径；`do_kv_cache_update()` 再调用 FlashInfer 的 slot-mapping writer。它按调度器给出的 token 槽位追加新 K/V，不是每生成一个 token 就重新量化整个历史缓存。

写入接口接收 FP16/BF16 K/V，输出打包的 `uint8` 数据和独立的 FP8 block scale。`_nvfp4_kv_views()` 只是把同一块缓存拆成数据与尺度视图，适配 NHD/HND 排列；这里不会额外复制一份完整的高精度 KV。

**旧 GPU 怎样还原 FP4。** FlashInfer 的 `vec_dtypes.cuh` 对 SM100 以下保留软件转换：取出每个 4-bit 编码的符号和幅值索引，将幅值映射到 $\{0,0.5,1,1.5,2,3,4,6\}$，再组成 FP16/BF16 表示。向量化分支通过寄存器中的位操作与 `prmt` 完成查表，另有标量 LUT 回退。旧 GPU 不必认识 FP4 算术指令，也能读取这组编码。

block scale 同样需要转成计算类型。SM8x 缺少该实现使用的硬件 FP8 转换时，尺度也走软件转换；SM90 可使用硬件 FP8 转换，但 FP4 数据本身仍需恢复。这是两代旧架构处理成本的一个区别，不能仅按“都不支持 FP4 Tensor Core”视为完全相同。

**恢复后在哪里计算。** 在 FA2 的 `compute_qk()` 中，可以看到先通过 `vec_cast` 展开 K，再乘对应 block scale，最后调用 16-bit 输入的 MMA；V 的加权计算采用相应处理。部分 tile 还会先由 `repack_fp4_tile_to_16b()` 恢复到共享内存，减少重复转换。这里使用的是寄存器或共享内存中的局部数据，没有先把全部历史 KV 展开并写回显存。

Q 和输出保持模型类型，常见为 FP16 或 BF16；累加类型另由内核配置决定。vLLM 对这条路径传入 `q_scale=1.0`，也不会启用仅供 SM100 trtllm-gen NVFP4 路径使用的 FP8 输出中间缓冲。因此，旧架构获得的是缓存容量和访存上的压缩收益，矩阵乘法精度没有随缓存一起降到 FP4。

**v0.32.0 仍然拒绝的组合。** 后端的能力检查与 metadata builder 都有显式限制，不能只看 dtype 出现在支持列表中。

| 条件 | 该版本的处理 | 原因或影响 |
| --- | --- | --- |
| head size 为 64 | 拒绝 NVFP4 FA2 路径 | prefill 存在已知读取错误；不是泛指所有 64 维模型都不支持量化 |
| attention sink | 拒绝该组合 | 对应 prefill wrapper 尚不能读取 NVFP4 block scale |
| DCP 大于 1 | metadata 构建时报错 | DCP prefill wrapper 不能读取这些尺度；不等于禁止普通 TP 多卡 |
| `nvfp4_4over6` | 旧架构路径不接受 | slot-mapping writer 没有对应的 scale search |
| KV page size | 此后端在旧架构提供 16、32、64 | 更大的页属于另一条内核路径 |
| T4 等 SM75 设备 | 不在这次支持范围 | 该版本 FlashInfer 后端的最低能力检查为 SM80 |

SM8x 包括 A100 的 SM80、RTX 3090 / A10 的 SM86，以及 RTX 4090 / L40S 的 SM89。它们通过同一架构分支，但 PR 作者给出的旧架构实测主要在 SM80；SM86、SM89 不能仅凭分支匹配就视为已有完整验证。SM90 的评审中有 H200 的数值和任务测试，但没有给出吞吐结论。Jetson Orin 的 SM87 也不能只靠架构号判断整套 vLLM 环境可用，还要解决设备软件栈和依赖构建问题。

**对性能的实际含义。** 对 SM8x，较小的 KV 读取量要与软件解包、尺度恢复的代价比较；源码本身无法证明哪一项占主导。对 SM90，还多了一次后端变化：该版本的 FP8 可以使用 FA3 prefill / XQA decode，而 NVFP4 使用 FA2。NVFP4 的图支持级别也收敛到 `UNIFORM_SINGLE_TOKEN_DECODE`，不能直接复用 XQA 多 token 验证的图执行收益；这不代表普通单 token decode 必须关闭 CUDA Graph。

因此，在 A100、3090、4090 等设备上，NVFP4 适合加入长上下文或容量不足时的候选测试；在 H100/H200 上，则尤其需要和 FP8 的较成熟内核对照。这里的选择依据来自执行路径分析，尚不是这些设备上的性能排名。

同一版本还加入 ROCm 的 `ultraquant_4bit`，这是另一种量化方案，不能与 TurboQuant 混用名称。GLM 的 FP8、MLA 优化也属于各自模型和内核的结果，不能作为普通 GQA 模型的通用性能承诺。

依赖版本也会改变结论。vLLM v0.32.0 标签中的 `requirements/cuda.txt` 固定的是 `flashinfer-python==0.7.0.post1`；升级 0.7.1 的 PR #60948 虽然已经合并到主线，却不能替代发布标签的依赖记录。上述路径按 v0.32.0 与其固定版本核对，主线、发布 wheel 与自行升级依赖的镜像应分别记录。

# 六、容量、精度与速度要分开比较

NVFP4 的公开评测可以说明它在特定模型上有效，但不足以建立对 TurboQuant 的统一精度排名。LMSYS 使用相同 FP8 权重比较 SGLang 中的 FP8 与 NVFP4 KV，Qwen3.8-27B 的 GSM8K 相差 0.31 个百分点，SWE-bench Verified 相差 1.60 个百分点。换模型、后端和任务后，这些差值未必成立。

旧架构支持 PR 中也报告过小模型量化后的明显退化，因此“NVFP4 几乎无损”仍需具体到模型与测试。类似地，vLLM TurboQuant 配置注释中的 PPL 增幅是参考结果，不应当作当前模型的实测值，更不能拿它与另一模型的任务准确率直接比较。

误差的影响也不只看 K/V 的重建均方误差。K 的变化会进入 $QK^T$，经过 softmax 改变关注位置；V 的变化影响加权后的内容。`turboquant_k8v4` 保留 FP8 K、压缩 V，提供了一种非对称选择，但是否比同时压缩 K/V 更合适，仍取决于任务和容量需求。

性能测试则应区分两种条件：

- **固定并发与上下文**：观察量化、访存和内核路径本身的代价，比较 TTFT、每 token 延迟与吞吐
- **固定显存预算**：逐步增加驻留请求，观察何时开始等待、抢占或重算，以及缓存容量能否推迟吞吐下降

低并发时，原有缓存已经足够，压缩后的计算开销可能更明显。高并发或长上下文下，缓存更容易成为限制，减少访存和重算可能抵消甚至超过这些开销。前缀缓存命中率的提高还会影响 TTFT，这又不同于单个内核加速。

所以不能把“NVFP4 比 FP8 慢固定百分比”或“TurboQuant 只适合极限省显存”作为通用结论。它们都可能在某些负载下增加延迟，也可能通过容量与访存改善服务吞吐。

# 七、在目标服务中如何选择

对 SM120 上的长上下文服务，可以先保留 FP8 作为对照，再测试 NVFP4 与 `turboquant_4bit_nc`；需要保留更高精度 K 时补充 `turboquant_k8v4`。已有容量记录支持继续评估 NVFP4，但尚不足以确定最终配置。

| 检查内容 | 应记录的条件与指标 |
| --- | --- |
| 实现 | vLLM、FlashInfer、CUDA 版本，实际 attention 后端，图捕获与推测解码设置 |
| 内存 | KV 池实际字节数、token 容量、工作区、图内存，以及是否有未压缩层 |
| 性能 | 多档输入长度与并发，TTFT、ITL / TPOT 的 P50 与 P95，全局输出吞吐、抢占与重算 |
| 质量 | 短文本任务、长上下文召回、结构化输出，以及真实图文输入 |
| 复用 | 冷缓存与前缀命中分开测试，包含带历史缓存的继续生成 |

比较时固定权重、请求集、生成长度、调度参数和显存预算。换 KV dtype 后，如果后端也发生变化，应把整条执行路径一起记录。`gpu-memory-utilization` 相同仍可能因工作区差异产生不同的 KV 池，应同时核对实际分配。

混合模型还需检查完整 attention 的分页与线性状态如何共用资源。历史 issue #41560 中的 page-size 对齐失败来自特定版本，不能据此排除所有混合模型；换成 NVFP4 也不能自动证明所有组合都兼容。启动成功以后，仍要验证续写、前缀复用和多请求切换的结果。

若服务同时承担视频解码、CV 与 VLM 请求，整卡内存、计算和排队预算见 [NVIDIA 多路视频 AI 的资源预算与容量规划](../inference-performance/NVIDIA_视频容量规划.md)。

# 八、一些反思

低位宽 KV 的价值要放回服务负载中看。单次 decode 更快、相同显存容纳更多请求、前缀缓存保留更久，是几种不同的收益，也可能对应不同的最佳配置。

NVFP4 在旧架构上的支持进一步说明，数据格式与硬件计算能力并不是一一对应的。选型时更有用的问题是：缓存减少了多少读写，恢复数据花了多少代价，实际任务允许多少误差，以及这些变化能否改善目标并发下的延迟和吞吐。

# 九、参考资料

- [TurboQuant 论文](https://arxiv.org/abs/2504.19874)
- [NVIDIA：NVFP4 KV 格式与推理优化](https://developer.nvidia.cn/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/)
- [LMSYS：NVFP4 KV 的内核实现与精度评测](https://www.lmsys.org/blog/2026-09-16-nvfp4-kv-cache)
- [MNN 中的 TurboQuant 实现](https://zhaode.wang/blog/2026/turboquant/)
- [vLLM：架构选择、能力限制与缓存读写](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/attention/backends/flashinfer.py)
- [FlashInfer：FP4 软件转换实现](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.0.post1/include/flashinfer/vec_dtypes.cuh)
- [vLLM：TurboQuant 预设与存储布局](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/turboquant/config.py)
- 旧架构适配与测试：[SM8x / SM12x](https://github.com/vllm-project/vllm/pull/46963)、[SM90](https://github.com/vllm-project/vllm/pull/60620)
