---
title: Qualcomm GenieX 的 GGUF 与 QAIRT 部署
created: 2026-09-01
updated: 2026-09-02
status: draft
blog: true
tags: [Qualcomm, QCS9075, GenieX, GGUF, QAIRT, 多模态, 部署]
---

# Qualcomm GenieX 的 GGUF 与 QAIRT 部署

GenieX 在同一套模型管理与服务接口下提供两条运行路径：本地 GGUF 使用 llama.cpp runtime，Qualcomm 预编译 bundle 使用 QAIRT NPU runtime。服务入口可以相同，模型资产、可调参数和实际执行方式却不同。

对于已经提供目标芯片适配资产的模型，QAIRT 路径能利用面向设备编译、量化和图执行的专门优化，适合作为性能优先的部署选择。GGUF 路径更方便导入自定义模型及切换计算设备，但仍受所用 llama.cpp 后端的能力限制。两条路径需要分别建立正确性和性能基线。

本文基于 QCS9075 Linux、GenieX CLI v0.5.0 的部署记录。模型目录、命令选项与芯片支持列表会随版本变化，使用前固定 CLI、runtime 和资产版本。

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

# 五、性能结果与选择

QCS9075 的归档 QAIRT Qwen3-VL-4B 测试中，平均 TTFT 约 0.638 s，完整响应约 16.63 s，有效生成约 15.6 token/s。这组结果属于特定多模态请求，没有与自定义 2B GGUF 或 MNN 做完全同条件比较，不能直接换算成框架加速倍率。

部署选择可以先看资产条件：已有适配目标设备、任务质量也满足要求的 QAIRT bundle，优先测试其 NPU 路径；需要自定义权重或官方尚无资产时，再选择 GGUF 或自行导出 MNN。QAIRT 更偏向利用硬件专门优化，GGUF/MNN 则保留更多模型和构建选择。

记录性能时同时固定图片尺寸、prompt、生成长度、采样方式、预热与功耗配置。视觉耗时、语言 prefill、decode 和服务总时延分别比较；仅 decode 快，不代表首响应更快，仅首响应快，也不能说明长文本生成更快。

# 六、一些反思

GenieX 统一的是资产管理和调用方式，底层执行机制仍由 runtime 决定。先确认输入资产走哪条路线，再解释参数、错误和性能，可以避免把 GGUF 的问题归到 QAIRT，或期待服务封装自动解决 kernel 限制。

选择专门优化的预编译模型，可以减少后端适配工作；自行编译则能够接入更多自定义模型。两者的取舍最终落在任务质量、设备适配、时延和维护成本上。

# 七、参考资料

- [Qualcomm GenieX](https://github.com/qualcomm/GenieX)
- [GenieX CLI 与 runtime 参数](https://github.com/qualcomm/GenieX/blob/main/docs/en/run/cli/reference.mdx)
- [GenieX 本地服务](https://github.com/qualcomm/GenieX/blob/main/docs/cn/run/cli/local-server.mdx)
- [Qualcomm AI Hub](https://aihub.qualcomm.com/)
