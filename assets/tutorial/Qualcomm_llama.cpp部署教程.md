---
title: Qualcomm llama.cpp 三后端编译与部署
created: 2026-07-23
updated: 2026-08-21
status: draft
blog: false
tags: [Qualcomm, llama.cpp, Hexagon, OpenCL, ABI]
---

# Qualcomm llama.cpp 三后端编译与部署

Qualcomm 设备上的 llama.cpp 可以使用 ARM CPU、Adreno OpenCL 和 Hexagon HTP。Hexagon 后端由 ARM64 宿主库、FastRPC 与 HTP kernel 组成，与 QNN graph API 是不同路径。部署需分别确认构建选项、目标 ABI 和系统运行时。

本文整合 QCS8550 与 QCS9075 材料。命令对应归档源码的 Snapdragon preset，使用前检查目标 commit 中 preset 与参数是否存在。

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

# 六、参考资料

- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Snapdragon 后端文档](https://github.com/ggml-org/llama.cpp/tree/master/docs/backend)
- [Radxa Q900 FastRPC 配置](https://docs.radxa.com/fogwise/airbox-q900/ai-dev/fastrpc-setup)

