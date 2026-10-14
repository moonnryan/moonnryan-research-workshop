---
title: Intel B60 上的 Qwen3-VL vLLM 部署
created: 2025-12-11
updated: 2026-01-22
status: draft
blog: false
tags: [Intel, B60, XPU, vLLM, Qwen3-VL]
---

# Intel B60 上的 Qwen3-VL vLLM 部署

Qwen3-VL-32B 的部署同时受权重、视觉编码和 KV cache 容量约束。本文整理双张 24 GiB Intel B60 上的 AWQ 部署流程，使用 Intel XPU 适配的 vLLM 镜像提供 OpenAI 兼容接口。目标是先建立可重复的启动与测试基线，再调整上下文和并发。

本流程对应归档验证的 `intel/llm-scaler-vllm:0.10.2-b6`。后续版本可能改变参数和量化执行路径，升级时应重新验证，不应直接套用旧启动脚本。

# 一、检查设备与模型

宿主机需要安装匹配 B60 的驱动，并能通过 `xpu-smi` 识别两张卡。先检查设备节点，实际编号以主机为准：

```bash
xpu-smi discovery
ls -l /dev/dri/
docker version
```

准备完整模型目录，包括配置、权重、tokenizer 和 processor。本文用 `/srv/models/Qwen3-VL-32B-Instruct-AWQ` 作为示例路径。模型来源、修订版本和校验值应一起记录；只复制权重文件不足以构成完整多模态模型。

# 二、启动隔离环境

下列节点编号是示例，需与检查结果对应。容器内的 GPU 数量应再次核验；可见卡数与 TP 数不同，会在通信初始化阶段产生额外问题。

```bash
docker run --rm -it --name xpu-vllm \
  --device=/dev/dri/renderD129 \
  --device=/dev/dri/renderD130 \
  --group-add "$(stat -c '%g' /dev/dri/renderD129)" \
  --group-add "$(stat -c '%g' /dev/dri/renderD130)" \
  --shm-size=40g \
  -p 8000:8000 \
  -v /srv/models:/models:ro \
  --entrypoint /bin/bash \
  intel/llm-scaler-vllm:0.10.2-b6
```

如果镜像的驱动接口另有设备或权限要求，应按对应版本的说明补齐，并在容器内执行设备探针。原始部署使用过特权容器；这里将权限收敛为设备节点直通，实际环境还需验证该方式是否满足驱动要求。

# 三、启动模型服务

在容器内设置服务密钥，避免把固定凭据写进脚本。下面以 8192 个 token 作为批处理调优起点，其余参数沿用旧镜像的兼容配置；8k 预算仍需在目标设备完成容量与性能验收。

```bash
read -rs -p "API key: " VLLM_API_KEY
export VLLM_API_KEY
export TORCH_LLM_ALLREDUCE=1
export VLLM_USE_V1=1
export CCL_ZE_IPC_EXCHANGE=pidfd
export VLLM_WORKER_MULTIPROC_METHOD=spawn

python3 -m vllm.entrypoints.openai.api_server \
  --model /models/Qwen3-VL-32B-Instruct-AWQ \
  --served-model-name qwen-vl \
  --host 0.0.0.0 --port 8000 \
  --api-key "$VLLM_API_KEY" \
  --dtype float16 --quantization awq \
  --tensor-parallel-size 2 \
  --enforce-eager \
  --enable-chunked-prefill \
  --gpu-memory-utilization 0.85 \
  --max-model-len 40960 \
  --max-num-batched-tokens 8192 \
  --block-size 64
```

`max-model-len` 是单请求长度上限，`max-num-batched-tokens` 控制一次调度的 token 预算；二者不能互相替代。40960 是归档环境在 64k 上限出现容量问题后采用的配置，并不代表任意图片数量或并发下都安全。先以短文本完成启动，再逐步增加图片、序列长度和同时在途请求数。

**批处理预算与延迟取舍**

启用 chunked prefill 后，调度器可以把 prefill 分块，并与 decode 请求共同调度。较小预算限制一轮 prefill 的工作量，通常更有利于控制正在生成请求的 token 间延迟；较大预算可以处理更多 prefill token，可能改善首 token 等待与总体吞吐，但也增加单轮计算和临时内存压力。

| 预算 | 调优方向 | 需要观察 |
| --- | --- | --- |
| 4096 | 较小 prefill 分块 | token 间延迟、prefill 调度次数 |
| 8192 | 更多 prefill 工作 | TTFT、总吞吐、内存峰值 |

因此 4096 不是错误配置，8192 也不是所有 B60 模型的最优值。该 32B 双卡组合应固定输入长度、输出长度和并发，对比两个预算。批处理 token 预算也不等于允许 8192 个并发请求；在途序列还受 `max-num-seqs`、KV 容量和请求分布限制。

**eager 模式的兼容性与性能**

`--enforce-eager` 保留在这套旧镜像配置中，用于绕开模型执行路径的 Torch Dynamo 兼容问题。它不是性能优化参数：eager 路径不能获得相应图编译或图重放的收益，decode 中反复调度小算子时，宿主调度与 kernel 启动开销可能更明显。eager 仍使用 XPU 算子和优化内核，不能把它理解成完全没有硬件加速。

Intel 在 2026 年 7 月的 `0.21.0-b1` 才公告实验性 XPU graph 支持；该能力不能反推到本文的 `0.10.2-b6`，也不能由“新版有 graph”推断所有量化模型都可用。升级后应分别验收 eager 与图模式的首次请求、不同输入形状、数值和连续运行，再比较性能，不直接从旧启动脚本删除这个开关。

# 四、验证接口与输出

客户端请求中的 `model` 必须与 `served-model-name` 相同。端口可达而返回“模型不存在”时，先检查服务实际模型名：

```bash
curl --fail http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer $VLLM_API_KEY"

curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen-vl","messages":[{"role":"user","content":"请用一句话说明什么是张量并行"}],"max_tokens":128}'
```

图文测试随后加入 `image_url` 内容，分别验证自然语言描述和固定 JSON 输出。图片文件要固定，不能依赖测试期间会变化的外部资源。每轮记录输入 token、输出 token、图片尺寸、并发、耗时和失败原因。

# 五、并发与容量验收

| 层次 | 测试内容 | 观察项 |
| --- | --- | --- |
| 单请求 | 短文本、长文本、单图 | 输出正确性、首 token 延迟 |
| 多模态 | 不同尺寸与图片数 | 视觉 token、内存增长 |
| 并发 | 逐级增加在途请求 | 成功率、尾延迟、吞吐 |
| 持续运行 | 重复请求与冷启动 | EngineCore、容器重启、OOM |

10 并发的纯文本用例报告请求成功率为 100%，可作为运行基线，不能代表单请求峰值或最优并发。旧测试采用字符数近似 token，并在收集完整流式响应后才计首 token 时间，因此不将其速度和 prefill 数字作为标准性能指标。

**统一测速口径**

使用服务端实际输出 token 数，或匹配模型的 tokenizer 计数；SSE chunk 数和字符数都不等于 token 数。首 token 时间在接收到第一个非空生成内容时即时记录，不能等完整响应结束后再解析。

| 指标 | 口径 | 主要反映 |
| --- | --- | --- |
| TTFT | 发出请求到收到首个生成 token | 排队、输入处理与 prefill |
| TPOT | 首 token 到最后 token 的时间 / 后续 token 数 | 单请求 decode 体验 |
| 总输出吞吐 | 全部完成请求的输出 token / 整轮墙钟时间 | 服务整体容量 |
| 成功率与 p95 延迟 | 同一固定测试集 | 稳定性与尾延迟 |

输入处理与网络延迟包含在端到端 TTFT 中，因此它也不能直接当成纯模型 prefill 时间。无首 token 或仅生成一个 token 的请求，单独标记，避免参与无定义的 TPOT 计算。

**找出并发拐点**

固定模型、输入与输出长度，分别测试 1、2、4、8、10、16 等并发档位。每档记录单请求 TPOT、TTFT p50/p95、总吞吐、设备内存及抢占情况，并区分纯文本与图文请求。

低并发时，单请求可以更充分地占用计算资源；提高并发后，批处理可能提升总吞吐，同时拉长单请求 token 间隔。若总吞吐趋于饱和、尾延迟开始急升，继续增加并发往往只是延长等待。明显陡降还应检查 KV 不足导致的抢占与重算、prefill/decode 竞争、双卡通信及宿主调度，不能只凭一个 10 并发结果判定设备偏慢。

服务上限应按允许的 TTFT 与 TPOT 选择，入口使用有界队列或限流。面向交互延迟的配置与离线吞吐配置可以不同，不必追求同一个并发数覆盖所有负载。

遇到 `EngineDead`，保存服务端首个异常及容器状态。客户端 500 是后果，不能仅根据它判断是显存不足。把共享内存、设备通信、量化内核和输入容量分开检查，每次只改变一个变量。

# 六、参考资料

- [Intel llm-scaler](https://github.com/intel/llm-scaler)
- [vLLM 项目](https://github.com/vllm-project/vllm)
- [vLLM 0.10.2 调度与性能调优](https://docs.vllm.ai/en/v0.10.2/configuration/optimization.html)

