---
title: Qualcomm GenieX 的 GGUF 与 QAIRT 部署
created: 2026-09-01
updated: 2026-09-02
status: ready
blog: true
tags: [Qualcomm, QCS9075, GenieX, GGUF, QAIRT, 多模态, 部署]
aliases: [/assets/tutorial/Qualcomm_GenieX部署.html]
---

# Qualcomm GenieX 的 GGUF 与 QAIRT 部署

GenieX 在同一套模型管理与服务接口下提供两条运行路径：本地 GGUF 使用 llama.cpp runtime，Qualcomm 预编译 bundle 使用 QAIRT NPU runtime。服务入口可以相同，模型资产、可调参数和实际执行方式却不同。

对于已经提供目标芯片适配资产的模型，QAIRT 路径能利用面向设备编译、量化和图执行的专门优化，适合作为性能优先的部署选择。GGUF 路径更方便导入自定义模型及切换计算设备，但仍受所用 llama.cpp 后端的能力限制。两条路径需要分别建立正确性和性能基线。

本文基于 QCS9075 Linux、GenieX CLI v0.5.0 的部署记录。模型目录、命令选项与芯片支持列表会随版本变化，使用前固定 CLI、runtime 和资产版本。

下面对插件和加载流程的说明补充参考官方当前源码，不代表这些实现细节已经在归档的 v0.5.0 安装包中逐项复测。升级后应重新核对模型 ID、参数与日志，尤其不要将旧版本的服务行为直接推断为当前版本的功能限制。

# 一、确认环境与运行路径

先完成板端 FastRPC 通信检查；GGUF 若需使用 OpenCL，还要确认 Adreno runtime 与 ICD 可用。安装匹配 Linux ARM64 的官方 GenieX 发布包后，检查：

```bash
geniex --version
geniex config list
geniex list
```

安装包中附带的 QAIRT runtime 与系统 BSP 依赖需要匹配。能识别芯片，只说明模型管理器获得了设备信息，实际推理还需要验证。

| 项目 | GGUF 路径 | QAIRT 路径 |
| --- | --- | --- |
| 资产 | GGUF，VLM 还需配套 mmproj | 针对设备生成的预量化、预编译 bundle |
| runtime | llama.cpp | QAIRT |
| 计算设备 | 按模型与后端支持选择 CPU/GPU/NPU/hybrid | 对应 bundle 的 NPU 路径 |
| 权重量化 | 选择实际提供的 GGUF 版本 | 由 bundle 固定 |
| 上下文 | 可在模型与内存允许范围内配置 | 编译时确定，运行参数不能扩展编译窗口 |

**模型管理与推理插件分别负责什么**

GenieX 的模型管理负责查找、下载或导入资产，CLI 和服务层将请求交给统一 SDK，再由对应插件执行。`pull` 本身不会把 GGUF 自动转换成 QAIRT context binary；下载来源与计算设备也不是同一个参数，指定 `--compute npu` 不会改变权重格式。

```text
本地 GGUF + mmproj → llama_cpp 插件 → GGML 后端 → CPU / GPU / HTP
AI Hub bundle     → qairt 插件     → QAIRT/QNN → HTP 执行图
```

两条路径都可能用到 Hexagon 和 FastRPC，但前者由 llama.cpp 的后端实现承担算子执行，后者加载事先准备的 QAIRT 图资产。判断实际路径时，既要看插件，也要看设备放置，不能仅凭日志中出现了 NPU 名称就认为两者采用同一种实现。

# 二、导入本地 GGUF

为缓存指定独立目录，再导入包含主模型与配套 mmproj 的目录：

```bash
export GENIEX_DATADIR=/srv/geniex

geniex pull local/demo-vlm \
  --model-hub localfs --model-type vlm \
  --local-path /srv/models/demo-vlm-gguf
geniex list
```

`pull` 会复制资产到 GenieX 缓存，多个量化版本和 mmproj 会增加磁盘占用。导入后按列表中的量化名称运行，例如已有 Q8_0 时：

```bash
geniex infer local/demo-vlm:Q8_0 \
  --compute cpu --think=false --max-tokens 64 \
  --prompt "请用一句话说明你的能力" --verbose
```

先验证 CPU 文本与单图结果，再分别测试 `gpu`、`npu` 或 `hybrid`。`hybrid` 是 runtime 的设备策略，不表示每个阶段都更快。归档同图测试中，它没有获得超过 CPU 的整体收益。

GGUF 选择 NPU 后仍需检查实际算子放置。量化 lm-head 回退、视觉分支回退和内存分配限制不会仅因换了服务入口而消失，具体分析见 [Qualcomm llama.cpp 部署与性能分析](Qualcomm_llama.cpp部署教程.md)。

**GGUF 是怎样被执行的**

官方 `llama_cpp` 插件先初始化 GGML 后端，再将模型参数映射为 llama.cpp 的层卸载、设备列表和上下文配置。运行时根据模型结构建立计算图，由后端支持情况决定具体节点的执行位置。GGUF 保存权重和模型元信息，并不是一份已经针对目标 HTP 编译好的执行图。

VLM 还需要单独初始化多模态处理上下文。源码中的 `mtmd_init_from_file()` 使用 mmproj 与语言模型建立配套关系，图像经过视觉编码后再参与语言模型计算。因此主模型能正常回答文本问题，只验证了语言路径，不能证明图片分支已经加载成功。

当前实现对视觉分支初始化失败有回退 CPU 的处理，也有为视觉与语言模型选择不同 HTP session 的逻辑，以减少内存资源争用。它说明视觉和语言路径的设备放置需要分别核对；不能仅根据 `--compute npu` 推断整次多模态请求都在 NPU 上执行。

导入自定义 VLM 时，主模型、mmproj、tokenizer 与 chat template 应来自同一套转换结果。维度匹配只是最低条件，视觉预处理、特殊 token 和模板不一致，也可能出现模型可以加载但图片理解错误的情况。先用固定图片确认语义，再评估不同后端的速度。

# 三、使用 QAIRT 预编译资产

先查看检测到的芯片有哪些可用资产，再选择模型。归档使用的是官方 Qwen3-VL-4B-Instruct bundle：

```bash
geniex model list
geniex pull qualcomm/Qwen3-VL-4B-Instruct \
  --model-hub aihub --model-type vlm
geniex list
geniex infer qualcomm/Qwen3-VL-4B-Instruct \
  --compute npu --think=false
```

模型名称以当前列表为准。QAIRT bundle 已包含匹配的模型分片与运行配置，不再选择 GGUF Q4/Q8，也不另挂 GGUF mmproj。多模态资产按其接口输入图片与文本。

QAIRT 的效率优势来自针对模型和设备准备好的执行图与 kernel 路径，代价是可调整范围更小。`--ngl` 不能重新分配已编译 bundle 的层，`--nctx` 也不能把其编译上下文从 4K 改成 8K。需要更长上下文时，应选择对应资产或重新编译；这与 llama.cpp 运行时调整 KV 容量是两回事。

## 3.1 bundle 不只是一组权重文件

官方当前 `qairt` 插件从模型目录解析 tokenizer、运行配置和模型分片，再创建对应模型的 pipeline。语言图分片按 `genie_config.json` 中 `dialog.engine.model.binary.ctx-bins` 指定的文件与顺序读取，而不是扫描目录下全部 `.bin`。有些 bundle 还带有 CPU 侧 embedding 数据，不能把这些文件当成 QNN 图一起加载。

VLM 加载器另外解析视觉编码图，并把它与语言图分片区分开，再创建视觉语言流水线。QAIRT 路径不使用 GGUF mmproj，但仍然需要匹配的视觉编码资产、预处理和文本配置，不能理解为只下载语言权重就自动获得多模态能力。

迁移模型时应保留完整目录及相对路径，核对配置引用的分片是否齐全，避免只复制体积最大的几个文件。新模型也需要有对应的运行时适配，不能仅修改配置里的模型名称，就让另一套结构复用现有执行图。

## 3.2 预编译之后，运行时还在做什么

QAIRT 预编译省去了部署时重新构建通用模型执行图的步骤，但生成过程仍然需要运行时组织。当前官方 QAIRT core 包含模型图、KV cache 与位置编码等管理代码，图执行封装最终调用 QNN 接口；tokenizer、输入准备、采样和输出解码仍是整条生成链路的一部分。

prefill 消化输入序列，decode 则复用缓存逐 token 生成。运行时需要维护有效位置、attention mask 和缓存状态，并按资产支持的形状组织调用。能处理不同长度的 prompt，不意味着底层图可以接受任意形状，也不能把服务端的 `max_tokens` 当成编译上下文大小。

在当前 core 的 `prefillChunks()` 中，输入按资产的 prefill 长度分块，每块依次调用模型分片、传递隐藏状态并更新 KV。进入 decode 前还会调整 KV 的布局，以匹配另一阶段的读取方式。假设资产每块处理 128 个输入位置，300 个 token 就需要分成三块，而不是临时生成一个长度为 300 的新图；最后一块还要按有效输入位置取得输出。这里的数值仅用于说明分块关系。

源码还对独立的 lm-head 分片做了条件优化：普通生成不需要每个中间输入块的词表 logits，可以跳过非末块的这类输出计算。因此影响 TTFT 的不仅是 NPU 算力，还有资产的 prefill 档位、分片结构及运行时如何组织它们。同为 QAIRT bundle，也不能默认具有相同的执行开销。

对于 VLM，上下文预算还需要计入图像相关输入，具体数量由模型的图像预处理和 token 组织方式决定。相同文字 prompt 配上不同尺寸、数量的图片，可能改变视觉耗时、prefill 工作量和剩余生成空间，应一起检查。

当前 QAIRT 插件还会明确拒绝非零的 `n_gpu_layers` 和 `n_ctx` 配置。因此不应向 QAIRT 请求套用 GGUF 的调参方式；即使某个 CLI 版本忽略了这些参数，也不表示图已按参数重新编译。部分新版本提供滑动窗口，通过移除旧上下文继续对话，这也不等于增加模型一次能够访问的上下文。

## 3.3 运行库与模型资产一起固定

QAIRT 的主机库、HTP stub/skel、板端固件及模型 context binary 需要配合使用。排查时确认实际加载的库目录，而不只记录系统安装了哪个 SDK；环境变量和动态库搜索路径可能让进程加载到另一套版本。

库能加载、模型能生成文本，都不足以单独证明组合正确。部署归档至少保留 GenieX 与 runtime 版本、板端 BSP、模型资产来源及哈希，并使用固定输入检查输出。自行替换 QAIRT 库后需要重新建立正确性基线，不能只看 token/s 是否正常。

# 四、启动本地服务

两条路径都可以通过 `serve` 提供接口，先绑定回环地址测试。QAIRT 示例：

```bash
export GENIEX_DATADIR=/srv/geniex
geniex serve --host 127.0.0.1:18181 --compute npu
```

另一个终端查询模型列表，并发送短文本请求：

```bash
curl http://127.0.0.1:18181/v1/models
curl http://127.0.0.1:18181/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qualcomm/Qwen3-VL-4B-Instruct",
    "messages": [{"role": "user", "content": "请用一句话介绍自己"}],
    "temperature": 0,
    "max_tokens": 64
  }'
```

GGUF 服务改用列表中的本地模型 ID，并选择相应 compute 配置。再增加图片输入、长提示与连续请求，分别保存生成内容和服务日志。

OpenAI 风格的接口不意味着所有格式约束都等价实现。归档 v0.5.0 的 GGUF 服务曾未按预期执行 `response_format: json_object`，返回自然语言；QAIRT 单独测试中曾返回有效 JSON。一次有效输出不能代替 JSON 约束功能的系统验证，应统计固定提示下的可解析率和语义正确率。

多模态请求还要确认文件路径由谁解析。本地服务读取的是服务进程可见的文件；容器部署时，客户端主机上的路径不一定存在于容器内。可以先使用服务端可读的测试图片，明确路径映射后再接入 URL 或 base64，避免把取图失败误判为视觉模型异常。

服务能接收多个 HTTP 连接，也不能直接推断同一个模型实例支持并行生成。会话历史、KV cache 的隔离与请求调度需要结合所用版本验证。连续发送两份内容不同的请求，检查第二份是否受到第一份历史影响，再继续测试多轮对话与并发。

# 五、性能结果与选择

QCS9075 的归档 QAIRT Qwen3-VL-4B 测试中，平均 TTFT 约 0.638 s，完整响应约 16.63 s，有效生成约 15.6 token/s。这组结果属于特定多模态请求，没有与自定义 2B GGUF 或 MNN 做完全同条件比较，不能直接换算成框架加速倍率。

部署选择可以先看资产条件：已有适配目标设备、任务质量也满足要求的 QAIRT bundle，优先测试其 NPU 路径；需要自定义权重或官方尚无资产时，再选择 GGUF 或自行导出 MNN。QAIRT 更偏向利用硬件专门优化，GGUF/MNN 则保留更多模型和构建选择。

记录性能时同时固定图片尺寸、prompt、生成长度、采样方式、预热与功耗配置。视觉耗时、语言 prefill、decode 和服务总时延分别比较；仅 decode 快，不代表首响应更快，仅首响应快，也不能说明长文本生成更快。

冷启动还包含模型读取、图加载和运行时初始化，应与预热后的推理分开统计。比较 GGUF 与 QAIRT 时，尽量使用相同模型、任务输入及输出长度，并记录实际 token 数与图片处理方式。不同量化格式可能改变输出和提前结束的位置，不能只拿一次完整响应时间相除。

QAIRT 的性能潜力来自硬件适配、图划分、量化和运行时配合，并非使用 `.bin` 就天然更快。对 GGUF，主要检查后端覆盖、回退及跨设备开销；对 QAIRT，主要检查资产形状、分片调用、缓存与实际库版本。定位到具体阶段后，再决定更换资产、调整输入或升级 runtime。

需要自行导出模型并编译端侧 runtime 时，可继续参考 [Qualcomm MNN 本地编译与模型部署](Qualcomm_MNN编译与部署.md)，按同一输入和计时口径与现有 bundle 比较。

# 六、参考资料

- [Qualcomm GenieX](https://github.com/qualcomm/GenieX)
- [GenieX CLI 与 runtime 参数](https://github.com/qualcomm/GenieX/blob/main/docs/en/run/cli/reference.mdx)
- [GenieX 本地服务](https://github.com/qualcomm/GenieX/blob/main/docs/cn/run/cli/local-server.mdx)
- [GGUF 多模态加载与设备配置](https://github.com/qualcomm/GenieX/tree/main/sdk/plugins/llama_cpp/src)
- [QAIRT bundle 加载实现](https://github.com/qualcomm/GenieX/tree/main/sdk/plugins/qairt/src)
- [官方 QAIRT core、运行库与图执行实现](https://github.com/qualcomm/geniex-qairt-plugin)
