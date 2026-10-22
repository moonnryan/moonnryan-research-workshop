---
title: Qualcomm llama.cpp 部署与性能分析
created: 2026-07-23
updated: 2026-09-09
status: ready
blog: true
tags: [Qualcomm, llama.cpp, Hexagon, OpenCL, ABI, GGUF, 量化, 性能, 部署]
aliases: [/assets/tutorial/Qualcomm_llama.cpp部署教程.html, /topics/inference-performance/Qualcomm_llama.cpp性能边界.html]
---

# Qualcomm llama.cpp 部署与性能分析

Qualcomm 设备上的 llama.cpp 可以使用 ARM CPU、Adreno OpenCL 和 Hexagon HTP。Hexagon 后端由 ARM64 宿主库、FastRPC 与 HTP kernel 组成，与 QNN graph API 是不同路径。部署需分别确认构建选项、目标 ABI 和系统运行时。

宿主侧的 `libggml-hexagon.so` 负责 GGML 接口、能力判断、内存与任务提交；`libggml-htp-vNN.so` 才是运行在 DSP 上的算子库。调用经过 FastRPC/dspqueue，再进入 HVX 向量或 HMX 矩阵 kernel。加载了宿主库，不等于 DSP 通信和实际算子都已经通过。

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

设备节点存在不代表用户态已安装。先确认 FastRPC 通信与 OpenCL 设备发现，再加载模型。Radxa 官方系统可按其软件源安装 FastRPC 测试工具和 Adreno runtime；其他 BSP 不能直接照搬包名。

归档构建与运行记录使用 HTP v73 库，实际部署应核对具体设备、DSP 库和后端报告的架构。厂商 FastRPC 测试示例中的 `-a v75` 是该测试程序的参数，不能单凭这条命令选择 llama.cpp 的 DSP 库。后续模型验证应保存实际加载的 `libggml-htp-vNN.so` 信息。

# 二、交叉编译

固定源码 commit，选择包含 Hexagon SDK、ARM64 工具链与 OpenCL headers 的环境。目标 glibc 2.35 时使用对应或更低 ABI 的 sysroot。

在已配置的工具链环境中：

```bash
cp docs/backend/snapdragon/CMakeUserPresets.json .
cmake --list-presets
cmake --preset arm64-linux-snapdragon-release \
  -B build-snapdragon \
  -DGGML_OPENCL=ON \
  -DGGML_HEXAGON=ON \
  -DGGML_OPENCL_USE_ADRENO_KERNELS=ON
cmake --build build-snapdragon --parallel 8
cmake --install build-snapdragon --prefix "$PWD/release-snapdragon"
```

CMake 命令在源码根目录执行，安装前缀位于交叉编译环境中。日志应分别出现 CPU、OpenCL 和 Hexagon 后端；只看到编译成功，不能确认三种后端都已启用。

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

ABI 检查还要区分 `GLIBC_*` 与 `GLIBCXX_*`。现场曾在满足 glibc 要求后，继续遇到板载 libstdc++ 缺少 `GLIBCXX_3.4.30`。处理方式是在发布包私有目录中携带兼容目标 glibc 的 ARM64 libstdc++、libgcc_s，通过进程库路径加载，而非替换系统库。

```bash
file ./bin/llama-server ./lib/libggml-hexagon.so
readelf --version-info ./bin/llama-server
readelf --version-info ./lib/libggml-hexagon.so
readelf --version-info ./lib/libstdc++.so.6
```

最后一条仅在发布包携带该库时执行。程序和宿主 `.so` 应是 ARM64 ELF，HTP 库则是 DSP ELF；不能用宿主库的 glibc 检查逻辑解释 DSP 产物。

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

若需要 HTP 与 OpenCL 按层混合，显式列出设备和分层比例，例如：

```bash
./bin/llama-cli -m /srv/models/model.gguf \
  --device HTP0,GPUOpenCL --split-mode layer --tensor-split 3,1 \
  -ngl all --no-mmap -fa on -p "Hello"
```

这只是分配策略示例，不是已验证的最优比例。HTP 不支持某个节点，不意味着 scheduler 会自动按期望将它转给 OpenCL。统一内存也不代表两个后端能直接复用各自的量化布局与缓冲区。

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

初次验证多模态时，可先将主模型和视觉编码都放在 CPU：主模型使用 `--device none -ngl 0 --no-kv-offload`，视觉侧使用 `--no-mmproj-offload`，并移除上面的 `MTMD_BACKEND_DEVICE=HTP0`。先确认同一张图能得到正确回答，再分别改变语言和视觉后端，避免同时引入多个变量。

```bash
./bin/llama-bench -m /srv/models/model.gguf \
  --device HTP0 -ngl all --no-mmap -p 128 -n 64
```

分别报告 prompt processing、token generation、TTFT、峰值内存与温度。多模态 TTFT 包含视觉处理，不能与纯文本 prefill token/s 混为一个指标。

# 六、词表、量化与内存限制

HTP prefill 更快而 decode 更慢并不矛盾。prefill 处理多个 token，容易形成较大矩阵计算；decode 每步生成一个 token，更易受权重读取、CPU 回退和同步影响。峰值 TOPS 不能直接换算成生成速度。

QCS8550 的一组 2B Q8 模型对照显示，纯文本与多模态请求需要分开理解：

| 场景与后端 | 日志 Prompt 速度 / token·s⁻¹ | Decode / token·s⁻¹ | 请求总耗时 |
| --- | ---: | ---: | ---: |
| 纯文本、HTP | 86.70 | 12.09 | 未列 |
| 纯文本、CPU | 75.32 | 17.91 | 未列 |
| 同图请求、HTP | 194.57 | 10.52 | 约 9.84 s |
| 同图请求、CPU | 21.96 | 13.88 | 约 49.23 s |

多模态这组 Prompt 计时包含视觉处理影响，不能把约 9 倍差距直接解释为纯语言模型 prefill 加速。两种场景也不是相同 token 序列，不能横向用 Prompt 数字推断硬件效率。它们共同说明，HTP 可以改善请求前段，却没有让当时的单路 decode 快于 CPU。

## 6.1 实际卸载与回退

ngl all 表示尝试卸载模型层，不能保证所有节点由目标执行。视觉图中的布局与插值节点可能回退 CPU，输出投影也受形状和量化类型限制。

| 检查项 | 证据 |
| --- | --- |
| 语言层 | 实际 offload 日志 |
| 视觉编码 | CLIP/MTMD 设备日志 |
| 回退节点 | graph split 与 profiler |
| 输出投影 | supports_op 条件 |
| 同步成本 | 分阶段计时 |

归档模型使用 tied embeddings，没有独立的 `output.weight`。共享 `token_embd.weight` 既用于输入 token lookup，也作为输出 lm-head；隐藏维度 2048、词表 248320 的 Q8_0 张量约为：

$$
2048\times248320\times\frac{34}{32}\ \mathrm{bytes}
\approx515.31\ \mathrm{MiB}
$$

其中每 32 个 Q8_0 权重包含 32 个整数值和一个 FP16 scale。归档 Hexagon 后端对部分量化类型存在输出维度检查，超过 32768 时在宿主 `supports_op()` 阶段就拒绝该矩阵。即使日志显示主模型 25/25 层卸载，仍可同时看到约 515 MiB 的 CPU 模型缓冲区。若共享输出投影每生成一个 token 都在 CPU 执行，就会持续引入 CPU 计算和后端同步。

这个现象支持输出投影是重要瓶颈，但没有逐算子计时就不能把全部 decode 耗时都归到它。视觉侧也有独立回退：归档 CLIP 图中的两处 IM2COL 和一处 UPSCALE 未在 HTP 执行，形成多个 graph split。

## 6.2 OpenCL 分层与单 buffer 限制

OpenCL 的两个故障需要按发生阶段区分。首先是自动分层逻辑：现场 runtime 报告 global memory 为 1 GiB，后端再扣除 1 GiB 预留，得到零可用量。通用加载器按可用量归一化 split 时出现零除，最终表现为 `vector::_M_range_check`，尚未进入模型 kernel 执行。

单 GPU 显式指定 `--tensor-split 1` 后，这个索引错误消失，但随后仍然分配失败：该环境的单 buffer 上限为 256 MiB，前述共享 embedding 却需要 515.31 MiB。两者是一条排查链上的不同问题，修复分层不能增加真实分配容量。

| 现象 | 定位层次 | 已记录结果 |
| --- | --- | --- |
| 自动 split 导致 vector 越界 | 加载器与设备内存报告 | 显式单设备 split 可绕过 |
| 515.31 MiB 张量申请失败 | OpenCL 单 buffer 分配 | 普通模式失败 |
| 启用 Adreno large-buffer | runtime/驱动分配能力 | 仍报 `CL_INVALID_BUFFER_SIZE (-61)` |

模型总量能放进系统内存，不代表单个设备 buffer 能分配。不能将前一个错误修好，就宣布纯 GPU 模型已经可运行。

## 6.3 量化输出投影与大词表

Hexagon 的 32768 检查则属于另一个后端。上游量化 MatMul 重构改用了 32×32 tiled repack，并由宿主 solver 按 shape 和 VTCM 需求选择 HMX/HVX kernel；归档版本仍保留对大输出投影的保护。它不是公开的“HTP 词表硬件上限”，删除检查也不代表 repack、索引和写回都已支持该形状。

不改 kernel 的一个对照办法，是从原始 F16/BF16 GGUF 重新制作“主体 Q8、共享 embedding F16”的混合模型：

```bash
./bin/llama-quantize --token-embedding-type f16 \
  /srv/models/model-F16.gguf \
  /srv/models/model-Q8-embedding-F16.gguf Q8_0
```

参数以归档版本为准。tied embedding 模型只设置 `--output-tensor-type f16` 不足以改变共享张量；已有 Q8 模型转回 F16，也无法恢复量化前的信息。共享张量改成 F16 后约为 970 MiB，虽然进入另一条已声明支持的算子路径，却增加内存与带宽开销，仍需验证加载位置、logits 和生成结果。

源码改动应对应具体问题：split 零除属于加载器健壮性，单 buffer 失败属于分配能力，输出投影被拒绝属于 kernel 能力声明。放宽一个限制不会连带修好另外两类问题。先用显式分层、独立后端和混合权重建立对照，再决定是否需要修改 kernel。

归档 Hexagon 路径中，单路 decode 的矩阵行数常为 $M=1$，部分 HMX 路径要求更大的 M，因此落到另一类实现。增加并行会话可能提升总体吞吐，但同时改变每请求延迟和 K/V 容量，不能解释为单路 decode 自动加速。

## 6.4 单路 decode 与 HTP session

HTP session 是另一个内存约束。单个 Process Domain 的映射空间有限，较大模型可以按该版本接口创建多个 session，再复用 layer split：

```bash
GGML_HEXAGON_NDEV=2 ./bin/llama-cli \
  -m /srv/models/model.gguf --device HTP0,HTP1 \
  --split-mode layer -ngl all --no-mmap -p "Hello"
```

新增 session 不等于增加一颗独立 NPU，也不会让单路 token 自动并行。它可能缓解映射容量问题，同时增加分段调度与同步；“成功加载”与“更快运行”需要分别测试。模型容量还需计入 REPACK、KV cache、计算 buffer 和视觉模型，不能只按 GGUF 文件大小估算。

多模态再增加一个视觉后端，可能形成 HTP、OpenCL 和 CPU 混合链。应在同图请求中记录视觉阶段与语言阶段，避免纯文本基准掩盖实际多模态请求的放置成本。

## 6.5 测量完整请求

$$
T_{\mathrm{request}}=T_{\mathrm{media}}+T_{\mathrm{vision}}
+T_{\mathrm{prefill}}+T_{\mathrm{decode}}+T_{\mathrm{queue}}
$$

固定图像、prompt、生成长度、CPU 线程、功耗和预热。跨模型比较优先记录 TTFT 与请求总延迟，因为 tokenizer 和视觉 token 数不同会改变 token/s 的含义。

生成内容也属于基准的一部分。重复字符、异常 logits、格式截断需要与后端异常分别记录：CPU 与 HTP 同样输出不完整 JSON 时，应先检查模板、输出长度和格式约束；同一量化包在多个后端都异常时，则优先比较浮点与重新导出的模型。只有语言和视觉语义通过，吞吐数字才有比较价值。

GGUF 与 QAIRT 的统一服务入口见 [Qualcomm GenieX 的 GGUF 与 QAIRT 部署](Qualcomm_GenieX部署.md)，自行编译和导出其他模型的路径见 [Qualcomm MNN 本地编译与模型部署](Qualcomm_MNN编译与部署.md)。

# 七、参考资料

- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Snapdragon 后端文档](https://github.com/ggml-org/llama.cpp/tree/master/docs/backend)
- [Hexagon 量化 MatMul 与 tiled repack 重构](https://github.com/ggml-org/llama.cpp/pull/24954)
- [Radxa Q900 FastRPC 配置](https://docs.radxa.com/fogwise/airbox-q900/ai-dev/fastrpc-setup)
- [OpenCL 设备查询](https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clGetDeviceInfo.html)
