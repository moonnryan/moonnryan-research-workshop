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

在容器内设置服务密钥，避免把固定凭据写进脚本。以下参数属于该版本基线：

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
  --max-num-batched-tokens 4096 \
  --block-size 64
```

`max-model-len` 是单请求长度上限，`max-num-batched-tokens` 控制一次调度的 token 预算；二者不能互相替代。40960 是归档环境在 64k 上限出现容量问题后采用的配置，并不代表任意图片数量或并发下都安全。先以短文本完成启动，再逐步增加图片、序列长度和同时在途请求数。

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

归档纯文本测试在 10 并发下成功率为 100%，单请求生成约 10–13 token/s，全局吞吐约 90–113 token/s。这些数字只描述当时的请求负载；图文、长上下文和更长生成需要单独测量。

遇到 `EngineDead`，保存服务端首个异常及容器状态。客户端 500 是后果，不能仅根据它判断是显存不足。把共享内存、设备通信、量化内核和输入容量分开检查，每次只改变一个变量。

# 六、参考资料

- [Intel llm-scaler](https://github.com/intel/llm-scaler)
- [vLLM 项目](https://github.com/vllm-project/vllm)

