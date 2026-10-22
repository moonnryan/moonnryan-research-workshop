---
title: Ascend SAM3 的图文配对与动态编译
created: 2026-05-19
updated: 2026-07-23
status: ready
blog: true
tags: [Ascend, 910B, SAM3, ONNX, 动态形状, 子图, 批处理]
---

# Ascend SAM3 的图文配对与动态编译

SAM3 的图像 batch 与文本 batch 并不天然相同。一张图可以查询多个概念，同一个概念也可以用于多张图；进入 grounding decoder 时，真正的处理单位是图像与提示词的配对。导出时如果只给所有输入标上同一个动态 batch，容易得到形状看似灵活、配对关系却被固定的计算图。

在昇腾 910B 上，这个问题还涉及编译器如何表达动态形状。最终采用三段式部署：图像编码器复用视觉特征，文本编码器独立处理提示词，decoder 以规则的图文网格接收输入。三段都获得了动态输入 OM，但分别使用 shape range、动态 batch 和动态维度档位，运行时需要按各自接口设置输入。

CANN、芯片算子包和设备运行环境的准备见 [Ascend 服务器环境部署与信息查询](../tutorial/Ascend_服务器环境部署.md)。下面在已确认环境可执行的前提下，讨论图文配对与动态形状。

# 一、batch 到底表示什么

设图像数量为 $I$，第 $i$ 张图有 $K_i$ 个概念，图文配对总数为：

$$
P=\sum_{i=0}^{I-1}K_i
$$

只有每张图的概念槽位数量相同、均为 $K$ 时，才有 $P=IK$。概念是一个完整文本提示，文本长度 $L$ 是另一个维度；把多个概念拼成一句话，会改变提示语义，无法替代多个独立配对。

| 导出接口 | 图像侧 batch | 文本侧 batch | 配对方式 |
| --- | --- | --- | --- |
| 单体统一模型 | $P$ | $P$ | 每个位置是一对图文，同图多概念需要复制图片 |
| 索引式拆分 decoder | $I$ | $P$ | `image_indices[p]` 指定第 $p$ 条文本对应哪张图 |
| text-grid decoder | $I$ | 输入组织为 $[I,K,L,D]$ | 按图像优先顺序展开成 $P=IK$ 个配对 |

例如两张图分别查询两个和一个概念，索引式接口可以表达为：

```python
prompts = ["person", "bicycle", "dog"]
image_indices = [0, 0, 1]
# 输出第 0、1 项属于图 0，第 2 项属于图 1。
```

输出 shape 相同也可能配错图。`[0,0,1,1]` 与 `[0,1,0,1]` 都能匹配四条文本，却采用不同的遍历顺序。文本、图像索引、框提示以及输出还原必须使用同一份配对顺序。

# 二、按复用边界拆图

单体模型适合建立输出基线；同图多概念时，复制图片到 pair batch 则会重复执行视觉编码。拆图的主要收益来自消除这部分重复计算。

这里的单体导出封装了 HF `Sam3Model.forward` 的五个输入：图像、token、attention mask、框和框标签。它没有点提示输入，不能仅因为原模型支持多种提示，就把导出产物也描述为支持全部接口。同图多概念通过复制图像组成配对 batch 实现；三段拆分则把这份重复从视觉编码前移到了编码之后。

| 子图 | 输入与输出 | 可复用的内容 |
| --- | --- | --- |
| image encoder | 图像 $[I,3,1008,1008]$ → 多尺度 FPN 与位置编码 | 同一前处理图像的视觉特征 |
| text encoder | token 与 mask $[M,32]$ → 文本特征 $[M,32,256]$ | 相同提示词及其 attention mask |
| grounding decoder | 按配对组织的视觉、文本特征 → mask、box、score、presence | 每对图文的融合与预测仍需计算 |

$M$ 是本次实际编码的提示词数量。可以直接编码 $P$ 条文本，也可以先对重复文本去重，编码后再按配对顺序组装。文本特征必须连同 mask 一起复用；模型版本、tokenizer、截断长度变化时，缓存也要失效。

输出同样需要按实际导出图定义。当前接口保留候选目标维度，尚未做原图尺寸还原和最终筛选：

| 输出 | 当前导出图的形状 | 含义 |
| --- | --- | --- |
| `pred_masks` | $[P,200,288,288]$ | 每个配对的 200 路候选掩码 logits |
| `pred_boxes` | $[P,200,4]$ | 归一化 `xyxy` 候选框 |
| `pred_logits` | $[P,200]$ | 各候选的检测 logits |
| `presence_logits` | $[P,1]$ | 当前图文配对的目标存在性 logits |

输入图像为 1008×1008，并不意味着原始 mask 输出也是这个分辨率。将输出写成每对只有一个 1008×1008 mask，会同时低估候选输出的内存和后处理工作量。后处理还需根据实际缩放、padding 与原图尺寸恢复掩码和坐标。

视觉特征在图文融合前与提示词无关，适合作为缓存边界。融合编码器的输出已经依赖文本，不能只按图片缓存后用于另一个概念。图像缓存则应绑定媒体内容、缩放与归一化方式、模型版本，不能只使用文件名。

拆图也有传输成本。若子图之间经由 NumPy 数组交换特征，会引入设备到宿主、宿主到设备的搬运。需要在编码复用收益之外，测量中间张量的传输时间与峰值内存，再决定是否使用设备侧缓冲区衔接。

# 三、动态导出需要改写张量组织

几何编码器原来的 RoI 路径按 batch 拆开框张量，再传给 `roi_align`。在这里使用的 `dynamo=False` 导出方式下，沿 batch 做 `unbind(0)` 会形成与示例 batch 数量相关的拆分结构。给输入补一个 `dynamic_axes` 标记，并不会把已展开的 Python 容器重新变成动态张量计算。

改写后，将框展平为 $P\times N$ 个 RoI，并把所属 batch 索引作为第一列传入。核心张量组织如下，`boxes_xyxy` 已转换为特征图上的坐标：

```python
# boxes_xyxy: [P, N, 4]，P 为配对数，N 为每对的框槽位数。
pair_count, box_count = boxes_xyxy.shape[:2]
batch_id = torch.arange(pair_count, device=boxes_xyxy.device)
batch_id = batch_id.to(boxes_xyxy.dtype)
batch_id = batch_id[:, None, None].expand(-1, box_count, 1)
rois = torch.cat([
    batch_id.reshape(-1, 1),
    boxes_xyxy.reshape(-1, 4),
], dim=1)                                  # [P*N, 5]
pooled = roi_align(features, rois, roi_size)
# 后续投影完成后恢复为 [P, N, hidden_size]。
```

这里保留的是张量中的 batch 索引，避免依赖 Python 列表长度。几何编码器内部接收归一化 `cxcywh`，进入 RoIAlign 前转成特征图上的 `xyxy`。同为四个数，坐标含义和尺度不同。

该改写解决了导出中的一个动态 batch 障碍，但索引式 decoder 还保留 `index_select`、几何提示和 RoIAlign。它适合任意配对与框提示，编译时却需要处理更多动态索引和形状关系。

导出产物的组织也是独立问题。单体导出使用 external data，并将权重合并为一个外置数据文件，部署时需与 `.onnx` 一起保留，维持图中记录的相对路径。外置权重用于处理大模型的序列化与分发，不会自动降低执行时的权重或激活内存；拆分子图是否外置，应按各自大小决定。

本次导出使用 opset 15，但 opset 只规定算子的版本语义。是否能在 NPU 上编译，还取决于具体算子的 dtype、输入形状、属性和动态形状传播。算子分析 CSV 适合筛选可疑节点，最终仍需结合 ATC 的首个编译错误和实际执行结果定位。

# 四、用 grid 接口换取更规则的计算图

纯文本概念查询可以单独导出 text-grid decoder：不传框提示，去掉几何编码器与 `image_indices`，让每张图拥有相同数量的文本槽位。视觉特征先沿概念维展开，再与文本一起展平成 pair batch：

```python
# feature: [I, C, H, W]；text: [I, K, L, D]。
pair_features = feature[:, None].expand(-1, K, -1, -1, -1)
pair_features = pair_features.reshape(I * K, C, H, W)
pair_text = text.reshape(I * K, L, D)
pair_mask = attention_mask.reshape(I * K, L)
# 输出第 p 项对应 i = p // K，k = p % K。
```

这个布局用规则展开替代任意索引，$I$、$K$ 与 $P$ 的关系直接体现在图中，便于按组合编译档位。代价是每张图的槽位数必须相同；概念内容可以不同，但数量不齐时要分组、分批，或填充槽位并丢弃对应输出。token 的 padding mask 只描述文本内部的有效位置，不能代替整对图文是否有效的标记。

text-grid decoder 只接收 `fpn0/fpn1/fpn2/pos2` 四路视觉输入，省去几何路径使用的额外位置编码接口。需要框提示时继续使用独立路径，不能把删除 geometry 分支当作所有提示类型都等价的优化。用 `-10` 填充 dummy box 表示框槽位无效，也不宜直接假设它与完全不传几何提示的整张图计算相同。

验证时，将 text-grid 与原生不传框的纯文本路径比较，检查四路输出：`pred_masks`、`pred_boxes`、`pred_logits`、`presence_logits`。置信度还要组合检测分数与 presence 分数：

$$
s_{p,q}=\sigma(\ell_{p,q})\,\sigma(a_p)
$$

其中 $p$ 表示图文配对，$q$ 表示候选目标。只检查输出 batch 等于 $IK$，能确认张量组织，却无法发现图文错配或分数处理不一致。测试中还应使用不同图片、不同概念，按配对分别后处理，并保持 NMS 分组方式一致。

# 五、三段采用不同的昇腾动态输入方式

在 910B2、CANN 8.5 的转换测试中，三段得到的可用方案如下。图像分辨率固定为 1008×1008，文本长度固定为 32，这里的动态主要针对图像和概念数量。

| 子图 | 可用编译方式 | 已记录的运行组合 |
| --- | --- | --- |
| image encoder | `input_shape_range`，图像数范围 1～16 | batch 1、2、4、8、16 |
| text encoder | `dynamic_batch_size` | prompt batch 1、2、4、8、16 |
| text-grid decoder | `dynamic_dims` | $(I,K)$ 为 (1,1)、(1,2)、(1,4)、(2,2)、(2,4)、(4,4) |

decoder 同时改变图像数 $I$ 和概念槽位数 $K$，无法只靠一个共享首维 batch 表达。`dynamic_dims` 的每个档位需要按输入顺序列出所有 `-1` 的值，重复出现的同一逻辑维度也要重复填写。

例如四路视觉输入的首维均为 $I$，文本特征和 mask 的前两维均为 $I,K$，共八个动态位置。$(I,K)=(2,4)$ 的编译档位写成：

```text
输入顺序：fpn0, fpn1, fpn2, pos2, text_features, attention_mask
动态位置： I     I     I     I       I,K              I,K
档位值：2,2,2,2,2,4,2,4
```

运行工具要求的却可能是带名称、包含固定维度的完整 shape 串，不能直接把这八个值当作运行参数。档位是预先编译的组合，(2,3) 这样的未列组合并不会自动获得支持。

image encoder 的关键取舍不同：`dynamic_batch_size`、`dynamic_dims` 曾生成 OM，但运行时无法按对应档位接口执行。改用 shape range 后，同一个 OM 跑通了多个 batch。采用的转换参数为：

```bash
atc --framework=5 \
  --model=./image_encoder_grid_outputs.onnx \
  --output=./image_encoder_grid_shape_range \
  --input_format=NCHW \
  --input_shape="pixel_values:-1,3,1008,1008" \
  --input_shape_range="pixel_values:[1~16,3,1008,1008]" \
  --soc_version=Ascend910B2
```

这个四输出图像编码器使用 shape range 运行接口，benchmark 时需要设置实际输入 shape，并显式给出各输出缓冲区大小，否则工具报 `out i:0 size is zero`。缓冲区按实际 shape 与 dtype 计算字节数，执行后再读取真实输出描述。

因此，image 使用动态 shape，text 使用动态 batch，decoder 使用动态 dims，服务层负责把三者接起来。完整请求还必须同时满足三段约束：image 能接受 16 张图，不代表 decoder 已有 $(16,1)$ 档位；文本去重后出现三个提示词，也需要选择可用的文本档位或分批处理。

`input_shape_range` 是这套 CANN 8.5 环境的可用经验。新版 ATC 文档已将该参数列为弃用选项，升级时应核对新版 `input_shape` 的范围表达方式和运行接口，不把旧命令当作跨版本保证。

静态 ATC、ATC multi-batch 和 ORT CANN EP 的结果需要分开记录。静态输入提供了更明确的形状条件，可能使部分 shape 计算、广播和 reshape 在编译期确定；multi-batch 则要求保留下来的中间动态形状也满足其编译规则。将几何框数量固定为 1，仍没有消除单体图的中间动态形状问题。

以下对照来自同一阶段的转换与运行记录，CANN EP 使用 ONNX Runtime 1.24.4：

| 路径 | 已记录结果 | 可以得出的结论 |
| --- | --- | --- |
| 单体、静态 ATC、batch=1 | 转换和 benchmark 通过，NPU compute mean 约 160.22 ms | 单体图在固定形状下可以离线部署 |
| 单体、ATC dynamic batch | `Multi-batch not support middle dynamic shape` | 此图不满足当时 multi-batch 的中间形状约束 |
| 单体、CANN EP 图模式 | `aclgrphParseONNXFromMem` 失败 | 在线图解析路径没有通过，尚不能据此指定某个算子为根因 |
| 单体、CANN EP 单算子模式 | geometry 的 `Mul_4` 路径报 `BroadcastToD` 错误 | 故障缩小到该广播执行路径，仍需核对具体形状与类型 |
| 拆分 image、CANN EP | 图模式失败；单算子模式在 LayerNorm 的 `Sub` 路径报广播错误 | 拆图没有自动解决视觉编码器的在线执行问题 |
| 拆分 text 与索引式 decoder、CANN EP | 分别跑通 1、2 条文本或配对的动态输入 | 这些子图可以执行，不代表三段完整链路已经通过 |

当时的 CANN EP 测试脚本关闭了 ORT 图优化，显式设置 `force_fp16`，并分别切换 CANN 图模式和单算子模式。上述结果针对这一组图、版本和参数，不能概括为 CANN EP 永远不支持 SAM3，也不能把 `BroadcastToD` 报错直接解释成 ONNX 的 `Mul` 或 `Sub` 普遍不受支持。

ORT CANN EP 是另一条执行路径，其图模式、单算子模式和允许子图切分的模式有不同的覆盖范围；允许子图切分可能使部分节点落到 CPU。选择它作为验证入口时，需要查看实际节点分配与拷贝开销，仅看到 provider 名称包含 CANN 不能证明全图都在 NPU 上执行。

text 和 decoder 的测试还观察到新 shape 首次执行需要较长的在线编译时间，同 shape 重复运行则明显缩短。性能记录应分开保存 Session 创建、新 shape 首请求和预热后的耗时。提前预热已知档位可以减少首请求等待，但不等于支持任意形状而没有额外开销。

如果部署目标只需要固定的小 batch，静态单体 OM 可以保留为更简单的基线。已有记录确认的是 batch=1；为 batch 2、4、8、16 分别编译静态模型仍属于待验证方案。拆分加 grid 的价值在于特征复用和明确的配对组织，不需要把所有场景都改成同一种接口。

精度也应按子图选择。文本编码器曾出现 FP16 数值风险，不能因为图像与 decoder 能用低精度，就把整条链路统一设成 `force_fp16`。先检查各段输出是否有限、与浮点基线的差异，再对掩码、框和最终分数进行比较。

# 六、复用收益与 batch 的真实代价

图像编码复用避免了重复编码，但 grid 展开后的 decoder 仍要处理 $IK$ 个配对。`expand` 在张量语义上是广播，后续 reshape 和编译执行可能将其物化，不能按“只是一个 view”估计设备内存。

以导出接口中的 shape、FP32 张量为例，四路视觉特征与掩码输出的大小为：

| 张量 | 单图或单配对的数据量 |
| --- | --- |
| FPN：$[256,288,288]$、$[256,144,144]$、$[256,72,72]$ | 每张图约 106.3 MiB |
| `pos2`：$[256,72,72]$ | 每张图约 5.1 MiB |
| mask：$[200,288,288]$ | 每个配对约 63.3 MiB |

仅 16 个配对的原始 mask 输出就接近 1 GiB，尚未计算 decoder 中间激活。减少无用位置编码输出、避免宿主往返拷贝，以及及时筛选不需要的候选，都可能比继续扩大 batch 更有效；若改变输出图以提前筛选，仍需保持排序、阈值与掩码处理语义。

以下是该环境中单段 decoder 的 benchmark 记录：

| 图像数 $I$ | 每图概念数 $K$ | 配对数 $P$ | 单次耗时 / ms | 配对吞吐 / pairs·s⁻¹ |
| --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 25.63 | 39.0 |
| 1 | 2 | 2 | 42.34 | 47.2 |
| 1 | 4 | 4 | 78.18 | 51.2 |
| 2 | 2 | 4 | 77.65 | 51.5 |
| 2 | 4 | 8 | 155.33 | 51.5 |
| 4 | 4 | 16 | 306.32 | 52.2 |

从 4 个配对继续增大到 16 个配对，吞吐增长已经很小，单次等待与内存占用却继续增加。图像编码器的另一组记录中，batch 1 为 145.22 ms，batch 16 为 2049.26 ms，吞吐约从 6.9 增至 7.8 图/s，也没有随 batch 线性增长。

decoder 的 pairs/s 与完整服务的图/s 需要分别统计。完整时延还包括预处理、编码、数据搬运和后处理；缓存命中的同图多概念查询，与每次输入新图的请求，也应分开报告。测试脚本中先计算编码特征、再仅计 decoder 时间的用例，应按单段性能理解。

# 七、一些反思

拆图边界应同时服务于计算复用和编译可控性。图像、文本编码可以独立缓存，图文融合则按配对重新执行；为了纯文本路径的部署可行性，框提示可以保留为独立模型。这样牺牲了一部分单入口的通用性，却减少了动态索引、几何计算和输出维度相互牵制的问题。

动态能力也需要拆开看：ONNX 声明可变维度，ATC 生成对应模型，运行时能够设置形状并正确分配输出，三者共同决定它是否可用。不能只凭转换成功或输出 shape 正确，就判断完整推理已经通过。

图文配对是整条链路最应保持明确的约定。图片特征可以共享，概念可以去重，batch 可以分档，但每一个输出属于哪张图、哪条提示词始终需要可追踪。接口清楚之后，才有条件比较缓存、动态编译和批处理各自带来的收益。

拆分也不自动带来流水线并行。顺序调用三个子图，只改变计算组织；要让不同请求在编码与解码阶段重叠，还需要设备资源、调度和缓冲区的配合。复用减少了多少重复计算，与完整服务加速多少倍，应分别测量。

# 八、参考资料

- [SAM3](https://github.com/facebookresearch/sam3)
- [Transformers SAM3 接口与特征复用](https://huggingface.co/docs/transformers/model_doc/sam3)
- [ONNX 外置权重格式](https://onnx.ai/onnx/repo-docs/ExternalData.html)
- [ONNX Runtime CANN Execution Provider](https://onnxruntime.ai/docs/execution-providers/community-maintained/CANN-ExecutionProvider.html)
- [ATC 动态维度档位](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/devaids/devtools/atc/atlasatcparam_16_0020.html)
- [ATC shape range 与版本变化](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/devaids/atctool/atlasatcparam_16_0017.html)
- [msIT benchmark 动态 shape 与输出内存](https://gitee.com/ascend/msit/blob/master/msit/docs/benchmark/API_GUIDE.md)

