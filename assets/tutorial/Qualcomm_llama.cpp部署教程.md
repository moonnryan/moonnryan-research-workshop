---
title: Qualcomm llama.cpp 部署与性能分析
created: 2026-07-23
updated: 2026-09-09
status: draft
blog: true
tags: [Qualcomm, llama.cpp, Hexagon, OpenCL, ABI, MNN, 性能]
aliases: [/topics/inference-performance/Qualcomm_llama.cpp性能边界.html]
---

# Qualcomm llama.cpp 部署与性能分析

Qualcomm 设备上的 llama.cpp 可以使用 ARM CPU、Adreno OpenCL 和 Hexagon HTP。Hexagon 后端由 ARM64 宿主库、FastRPC 与 HTP kernel 组成，与 QNN graph API 是不同路径。部署需分别确认构建选项、目标 ABI 和系统运行时。

本文整合 QCS8550 与 QCS9075 材料，先建立三后端的可运行基线，再对照实际算子放置与分阶段性能选择部署方案。命令对应归档源码的 Snapdragon preset，使用前检查目标 commit 中 preset 与参数是否存在。

# 一、记录设备条件

```bash
uname -m
getconf GNU_LIBC_VERSION
ls -l /dev/dri/
ls -l /dev/fastrpc*
clinfo
```

| 后端 | 发布包 | 系统依赖 |
| --- | --- | --- |
| CPU | llama 与 GGML ARM64 库 | libc、libstdc++ |
| OpenCL | ggml-opencl | Adreno runtime、ICD、驱动 |
| HTP | ggml-hexagon 与匹配架构的 HTP 库 | FastRPC、CDSP 固件与用户态 |

设备节点存在不代表用户态已安装。QCS9075 归档设备使用 HTP v73，架构应依据设备说明确认。

# 二、交叉编译

固定源码 commit，选择包含 Hexagon SDK、ARM64 工具链与 OpenCL headers 的环境。目标 glibc 2.35 时使用对应或更低 ABI 的 sysroot。

在已配置的工具链环境中：

```bash
cmake --list-presets
cmake --preset arm64-linux-snapdragon-release \
  -B build-snapdragon \
  -DGGML_OPENCL=ON \
  -DGGML_HEXAGON=ON \
  -DGGML_OPENCL_USE_ADRENO_KERNELS=ON
cmake --build build-snapdragon --parallel 8
```

Android 另用 NDK 与 Bionic，不能复用 Linux glibc 二进制。归档的 Jammy 工具链 Dockerfile 属于原项目修改，不是所有上游版本自带文件。

# 三、发布包与 ELF 检查

```text
llama-runtime/
├── bin/        # llama-cli、llama-server、llama-bench
└── lib/        # libllama、libggml、OpenCL、Hexagon 与 HTP 架构库
```

保留所选版本生成的库与软链接。发布前用 file、readelf 检查架构及 GLIBC/GLIBCXX 版本，上板后检查：

```bash
cd /opt/llama-runtime
export LD_LIBRARY_PATH="$PWD/lib:$LD_LIBRARY_PATH"
export ADSP_LIBRARY_PATH="$PWD/lib"
ldd ./bin/llama-cli
./bin/llama-cli --list-devices
```

若所用源码仍标记 Hexagon 为实验性，按该版本要求设置 `GGML_HEXAGON_EXPERIMENTAL=1`。FastRPC 用户态与 OpenCL 驱动通过厂商源安装，不由 llama.cpp 包替代。

# 四、建立三后端基线

```bash
# CPU
./bin/llama-cli -m /srv/models/model.gguf \
  --device none -ngl 0 --no-kv-offload -p "Hello"
# HTP
./bin/llama-cli -m /srv/models/model.gguf \
  --device HTP0 -ngl all --no-mmap -fa on -p "Hello"
# OpenCL
./bin/llama-cli -m /srv/models/model.gguf \
  --device GPUOpenCL -ngl all --no-mmap -fa on -p "Hello"
```

名称以设备枚举为准。先短文本，再增加上下文；记录实际卸载与 CPU fallback，不能仅用 ngl all 判断所有节点都在加速器上。

# 五、多模态与测速

```bash
MTMD_BACKEND_DEVICE=HTP0 ./bin/llama-server \
  -m /srv/models/model.gguf \
  --mmproj /srv/models/mmproj-f16.gguf \
  --device HTP0 -ngl all --mmproj-offload \
  --no-mmap -c 4096 --parallel 1 \
  --host 0.0.0.0 --port 8080
```

检查 CLIP/MTMD 日志与 fallback。mmproj、tokenizer 和视觉配置需属于同一组合。

```bash
./bin/llama-bench -m /srv/models/model.gguf \
  --device HTP0 -ngl all --no-mmap -p 128 -n 64
```

分别报告 prompt processing、token generation、TTFT、峰值内存与温度。多模态 TTFT 包含视觉处理，不能与纯文本 prefill token/s 混为一个指标。

# 六、后端放置与性能分析

HTP prefill 更快而 decode 更慢并不矛盾。prefill 处理多个 token，容易形成较大矩阵计算；decode 每步生成一个 token，更易受权重读取、CPU 回退和同步影响。峰值 TOPS 不能直接换算成生成速度。

一组 QCS8550 多模态归档测试中，HTP prompt processing 约 194.57 token/s、decode 约 10.52 token/s；CPU 分别约 21.96 与 13.88 token/s。HTP 改善首段计算，但生成仍慢于 CPU。

## 6.1 实际卸载与回退

ngl all 表示尝试卸载模型层，不能保证所有节点由目标执行。视觉图中的布局与插值节点可能回退 CPU，输出投影也受形状和量化类型限制。

| 检查项 | 证据 |
| --- | --- |
| 语言层 | 实际 offload 日志 |
| 视觉编码 | CLIP/MTMD 设备日志 |
| 回退节点 | graph split 与 profiler |
| 输出投影 | supports_op 条件 |
| 同步成本 | 分阶段计时 |

归档 Hexagon 路径对量化输出投影的大词表维度有限制。词表约 248320、隐藏维度 2048 的权重形成很大投影；每 token 若这部分落 CPU，decode 收益可能被抵消。

## 6.2 内存分配与 kernel 条件

OpenCL 单 buffer 最大分配量与总可用内存不同。模型总量能放入内存，不代表一个权重 tensor 能分配成功。应把设备最大分配量与具体 tensor 大小比较。

Hexagon 的维度保护也需要理解。删除 32768 等限制前，检查它保护的 VTCM、索引和 kernel 条件。F16 混合权重可以实验，但改变体积、带宽和计算成本，不保证更快。

归档 Hexagon 路径中，单路 decode 的矩阵行数常为 $M=1$，部分 HMX 路径要求更大的 M，因此落到另一类实现。增加并行会话可能提升总体吞吐，但同时改变每请求延迟和 K/V 容量，不能解释为单路 decode 自动加速。

多 HTP session 可以缓解单个映射空间限制，也会引入切分与同步。归档双 session 能加载原先受限的 F16 模型，但生成速度没有相同幅度收益；“成功加载”与“更快运行”需要分别验收。

多模态再增加一个视觉后端，可能形成 HTP、OpenCL 和 CPU 混合链。应在同图请求中记录视觉阶段与语言阶段，避免纯文本基准掩盖实际多模态请求的放置成本。

## 6.3 MNN 对照的解释

同一 QCS9075 的归档 Qwen3.5 对照中，MNN W4A16 的 prefill 未优于 llama.cpp Q8。限制包括 CPU i8mm 路径未启用、OpenCL 解量化与 kernel 差异，以及 HTP 上部分线性注意力回退。

这些结果比较“模型表示＋框架实现＋硬件路径”。W4 权重更小不能说明 kernel 更快，其他 SoC 的优势也不能直接外推。

## 6.4 测量完整请求

$$
T_{\mathrm{request}}=T_{\mathrm{media}}+T_{\mathrm{vision}}
+T_{\mathrm{prefill}}+T_{\mathrm{decode}}+T_{\mathrm{queue}}
$$

固定图像、prompt、生成长度、CPU 线程、功耗和预热。跨模型比较优先记录 TTFT 与请求总延迟，因为 tokenizer 和视觉 token 数不同会改变 token/s 的含义。

# 七、一些反思

先问哪部分没有实际加速。放置与阶段分开测量后，才能决定修 kernel 支持、调整量化表示还是减少同步边界。

框架选型依赖真实请求分布。短回答中 decode 慢的影响有限，长文本会持续放大 CPU 回退的代价；需要同时比较数值精度与整请求体验。

# 八、参考资料

- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Snapdragon 后端文档](https://github.com/ggml-org/llama.cpp/tree/master/docs/backend)
- [Radxa Q900 FastRPC 配置](https://docs.radxa.com/fogwise/airbox-q900/ai-dev/fastrpc-setup)
- [MNN](https://github.com/alibaba/MNN)
- [OpenCL 设备查询](https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clGetDeviceInfo.html)
