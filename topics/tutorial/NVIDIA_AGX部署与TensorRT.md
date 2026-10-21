---
title: NVIDIA AGX Orin 系统与 TensorRT 环境部署
created: 2026-03-30
updated: 2026-06-09
status: draft
blog: false
tags: [NVIDIA, AGX Orin, Jetson, JetPack, CUDA, TensorRT, ABI]
aliases: [/assets/tutorial/NVIDIA_AGX部署与TensorRT.html]
---

# NVIDIA AGX Orin 系统与 TensorRT 环境部署

Jetson AGX Orin 的部署需要同时处理板级系统、计算库和应用运行环境。Jetson Linux 提供 bootloader、内核和驱动等 BSP 组件，JetPack 在此基础上提供 CUDA、TensorRT 等计算组件。只刷入系统，并不代表 Python binding、模型编译工具和容器运行时都已安装完整。

本文以归档中的 Jetson Linux R36.5、CUDA 12.6、TensorRT 10.3 环境为主，覆盖从刷机到引擎部署的操作顺序。命令中的路径和输入名称均为示例；安装、刷机、修改网络和重启 Docker 是环境搭建步骤，已有设备排查时先执行查询命令，按缺失项处理。

# 一、先确认系统与软件栈

## 1.1 区分 BSP、计算组件和 Python 环境

先记录系统信息，再决定安装哪些包：

```bash
cat /etc/nv_tegra_release
cat /etc/os-release
uname -r
uname -m
getconf GNU_LIBC_VERSION
dpkg-query -W 'nvidia-l4t-core' 'nvidia-jetpack'
dpkg-query -W 'cuda-cudart-*' 'libnvinfer*' 'python3-libnvinfer*'
readlink -f /usr/local/cuda
```

部分包未安装时，`dpkg-query` 会返回非零状态，应结合输出逐项判断。没有 `nvidia-jetpack` 元包，也不代表 CUDA 和 TensorRT 不存在：它们可能被单独安装。

归档中的版本基线如下：

| 层次 | 版本或状态 | 主要核对方式 |
| --- | --- | --- |
| Jetson Linux | R36.5.0 | `/etc/nv_tegra_release`、L4T 包 |
| 操作系统 | Ubuntu 22.04、aarch64 | `os-release`、`uname -m` |
| 内核 | 5.15.185-tegra | `uname -r` |
| 系统 Python | 3.10 | `/usr/bin/python3 --version` |
| CUDA | 12.6 | 安装包、CUDA 路径、`nvcc` |
| TensorRT | Python 导入版本 10.3.0 | Python binding 与系统库共同检查 |

JetPack 6.2.2 的官方组合包含 Jetson Linux 36.5、CUDA 12.6 和 TensorRT 10.3。版本表用于确定这篇教程的适用范围，不要求其他版本的设备照表降级。计算组件也可以按官方支持的方式升级，不能只凭一个 JetPack 名称推断所有已安装版本。

## 1.2 检查解释器与编译工具

```bash
command -v python3
python3 --version
/usr/bin/python3 --version
command -v nvcc
nvcc --version
command -v trtexec
ls -l /usr/src/tensorrt/bin/trtexec
```

`nvcc` 不存在可能只是没有安装 CUDA 开发工具，不能据此判定 GPU 运行库不可用。反过来，`nvcc --version` 只说明这个编译器的版本，不能证明当前 Python 进程加载的库来自同一套 CUDA。

# 二、刷机、登录与存储准备

## 2.1 确定刷机目标

刷机前确认模块、载板和启动介质。AGX Orin 开发套件的配置不能直接套到所有第三方载板；eMMC 与外置 NVMe 的刷机流程也不同。先备份目标盘数据，再选择对应版本的 SDK Manager 或 BSP 工具。

使用 SDK Manager 时，依次选择目标硬件和 JetPack 版本，连接 Recovery USB，完成系统刷写和首次启动，再按需要安装目标端 SDK 组件。主机 Ubuntu 版本以对应 SDK Manager 和 BSP 的要求为准。

命令行方式需要同一版本的 BSP 与 Sample Root Filesystem。准备过程为：

1. 解压 BSP，得到 `Linux_for_Tegra`。
2. 将配套 Sample Root Filesystem 解压到 `rootfs/`，保留文件权限。
3. 在 BSP 目录按该版本要求执行 `sudo tools/l4t_flash_prerequisites.sh`，再执行 `sudo ./apply_binaries.sh`。
4. 使设备进入 Force Recovery，在主机检查 USB 识别结果。
5. 使用匹配板型和介质的命令刷写。

```bash
# 在刷机主机执行，仅查询 USB 设备。
lsusb -d 0955:
```

归档中开发套件写入内部存储使用：

```bash
# 在已准备好的 BSP 目录执行；此命令会刷写目标设备。
sudo ./flash.sh jetson-agx-orin-devkit internal
```

外置 NVMe 应采用对应 BSP 的 `l4t_initrd_flash.sh` 流程，并核对目标盘与分区配置，不把上面的 `internal` 简单替换成一个盘符。Recovery 阶段的 USB 识别只说明刷机通道可用；刷写成功后，还要确认设备能从预期介质正常启动。

## 2.2 无头初始化与网络

没有显示器时，可以通过开发套件支持的 USB 串口完成首次用户和网络配置。正常启动后的 USB 网络与 Recovery 是两种状态：前者提供运行系统的通信入口，后者用于刷写，不能混为一谈。

先查询实际连接，不预设网卡一定叫 `eth0`：

```bash
ip -brief address
ip route
nmcli device status
nmcli connection show
```

需要固定地址时，修改已有连接配置，明确地址前缀、网关和 DNS。在远程连接下改当前网卡会中断会话，应保留串口或其他管理入口。教程不固定某个私网网段，部署时由实际网络分配。

## 2.3 确认根分区与数据盘

```bash
findmnt /
lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINTS
df -hT
docker info --format '{{.DockerRootDir}}'
```

模型、编译产物和镜像层很容易占满根分区。数据盘挂载完成后，还应检查挂载路径的可写权限和持久化配置，避免重启后服务把数据写入根分区中的空目录。

Docker 与使用 containerd image store 的环境可能分别占用不同目录。迁移前先确认实际存储后端，停服务、复制数据、修改配置和重启验证应作为独立维护操作，不能只修改一个路径就假定所有镜像数据都已迁走。

# 三、补齐 CUDA 与 TensorRT 组件

先检查 NVIDIA APT 源是否与 BSP 匹配：

```bash
grep -RhE '^[[:space:]]*deb .*repo.download.nvidia.com/jetson' \
  /etc/apt/sources.list /etc/apt/sources.list.d/
apt-cache policy nvidia-jetpack python3-libnvinfer libcudla-12-6
dpkg --audit
```

不要为安装一个新包而混入另一代 L4T 的软件源。在源和包依赖正确的全新系统上，可以安装完整 JetPack：

```bash
sudo apt-get update
sudo apt-get install nvidia-jetpack
```

存储有限时，可以按官方组件清单选择运行包和开发包。TensorRT 原生 runtime、Python binding、头文件和 `trtexec` 是不同组件：只安装 lean 或 dispatch binding，不等于具备完整的模型构建环境。

需要补包时先模拟依赖变化，例如：

```bash
apt-get --simulate install python3-libnvinfer python3-libnvinfer-dev
```

确认不会意外替换现有计算栈后再执行实际安装。包管理器显示“可自动删除”的 CUDA 包也需要逐项核对，不能直接把 `apt autoremove` 当作部署收尾步骤。

# 四、定位 Python 与动态库问题

## 4.1 先确认导入的到底是什么

```bash
/usr/bin/python3 -B - <<'PY'
import sys
import tensorrt as trt

print("python:", sys.executable)
print("TensorRT:", getattr(trt, "__version__", "<missing>"))
print("module:", getattr(trt, "__file__", None))
PY
```

正常环境应同时给出版本和明确的模块位置。导入成功却没有 `__version__`，应先排查当前目录中的同名文件、空目录或残留 namespace package；不能仅凭 `import` 没报错就认定 binding 完整。

系统 APT 安装的 binding 通常位于系统 Python 的 `dist-packages`，独立虚拟环境不一定可见。若希望沿用系统 TensorRT，可以用同一 Python 小版本建立环境：

```bash
/usr/bin/python3 -m venv --system-site-packages .venv
. .venv/bin/activate
python -B -c 'import tensorrt as trt; print(trt.__version__, trt.__file__)'
```

该方式也会暴露其他系统 Python 包。若需要更强隔离，应采用匹配平台的完整环境；不要把 Python 3.10 的二进制扩展直接复制到另一小版本，也不要用一次 `pip install --upgrade tensorrt` 覆盖整套 JetPack 依赖。

## 4.2 区分缺库与搜索路径错误

遇到：

```text
ImportError: libcudla.so.1: cannot open shared object file
```

先查看系统包、文件和链接缓存：

```bash
dpkg-query -W 'libcudla*'
ls -l /usr/local/cuda-12.6/targets/aarch64-linux/lib/libcudla.so*
ldconfig -p | grep -E 'libcudla|libnvinfer'
dpkg -L python3-libnvinfer | grep -E '\.so($|\.)'
```

再对上一步找到的实际 Python 扩展执行 `ldd`。归档中的路径为：

```bash
ldd /usr/lib/python3.10/dist-packages/tensorrt/tensorrt.so
```

| 检查结果 | 处理方向 |
| --- | --- |
| 主系统缺少文件，只有旧系统备份或容器层里存在 | 安装当前 CUDA 对应的 cuDLA 包 |
| 正确版本的文件存在，但依赖解析为 `not found` | 检查链接缓存、配置和符号链接 |
| 终端可导入，服务启动失败 | 检查服务使用的解释器和库搜索环境 |
| 导入的是空模块或其他目录中的同名模块 | 检查 Python 包位置和 binding 是否完整 |

在 CUDA 12.6 环境，运行库缺失时补充 `libcudla-12-6`；只有开发工作需要时再加开发包：

```bash
sudo apt-get install libcudla-12-6
```

文件已经存在、但系统搜索配置确实缺失时，才采用归档中用过的路径修复：

```bash
printf '%s\n' '/usr/local/cuda-12.6/targets/aarch64-linux/lib' \
  | sudo tee /etc/ld.so.conf.d/cuda-12-6-aarch64.conf
sudo ldconfig
```

修复后重新检查 `ldconfig -p`、`ldd` 和 Python 导入。旧 CUDA 目录或 Docker overlay 中找到一个同名库，并不能说明它适合当前宿主系统；也不能用软链接把不匹配的 SONAME 伪装成所需版本。

## 4.3 把媒体依赖与推理环境分开

原生媒体程序常同时依赖 FFmpeg、OpenCV、Python 和 C++ 运行库。排查时从实际二进制的 `ldd` 开始，按缺少的 SONAME 找对应构建版本，而不是笼统升级一个 FFmpeg 命令行程序。

交互终端、虚拟环境与 systemd 服务的环境变量不同。只在个人 shell 中设置 `LD_LIBRARY_PATH`，不会自动修复服务进程。服务需要的私有库应限定在该服务的启动环境中；不要把 conda 的整套库目录加入全系统搜索路径，否则可能替换媒体插件原本使用的 `libstdc++`、FFmpeg 或 OpenCV。

# 五、配置 GPU 容器环境

先查询 Docker 和 NVIDIA runtime：

```bash
docker version
docker info --format '{{json .Runtimes}}'
command -v nvidia-container-runtime
command -v nvidia-ctk
dpkg-query -W 'nvidia-container-toolkit*'
```

若缺少 toolkit，按 NVIDIA 安装指南配置对应 APT 源并安装。提供 `nvidia-ctk` 的版本可以这样配置 Docker：

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

这一步会更新 Docker 配置并重启服务，应安排在维护窗口。显式指定 runtime 即可，不必为了一个推理容器把所有容器的默认 runtime 都改成 NVIDIA。

镜像需要同时满足 aarch64、Jetson/iGPU 支持和计算库兼容性。aarch64 的服务器镜像也不必然支持 Jetson；镜像内 CUDA 或 TensorRT 与宿主版本不同，需要依据该镜像的支持说明判断。

对于已经包含 Python TensorRT binding 的兼容镜像，可以先做导入检查：

```bash
# jetson-trt:validated 代表已经核对兼容性的本地镜像。
docker run --rm --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  jetson-trt:validated \
  python3 -B -c 'import tensorrt as trt; print(trt.__version__, trt.__file__)'
```

导入通过后，再单独验证实际 GPU 计算。容器启动成功、库能导入、GPU 算子能执行是三个检查层次；不要通过挂载宿主的整套 `/usr/lib` 来掩盖镜像依赖不匹配。

如果应用还需要 PyTorch，应选择 NVIDIA 的 Jetson wheel 或明确支持当前 JetPack 的容器，核对 Python、CUDA 和框架扩展版本。`torch.cuda.is_available()` 返回真仍不足以覆盖所有扩展；`no kernel image is available` 需要进一步检查报错算子所属二进制的 GPU 架构支持。

# 六、在目标环境编译 TensorRT 引擎

## 6.1 固定输入契约

编译前保存 ONNX 及其外部权重文件，核对输入名称、dtype、维度和输出含义。检测模型还要固定 RGB/BGR、缩放或 Letterbox、归一化、类别顺序和 NMS 所在位置，先得到 PyTorch 与 ONNX 的数值基线。

找到实际 `trtexec` 后查看当前版本帮助：

```bash
/usr/src/tensorrt/bin/trtexec --help
```

工具也可能由本地软链接加入 PATH。核对链接目标和依赖，避免 shell 调用旧版工具、Python 却导入另一版 runtime。

## 6.2 建立静态 FP16 基线

下面以 TensorRT 10.3、静态输入 ONNX 为例：

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=model.onnx \
  --saveEngine=model-fp16.plan \
  --fp16 \
  --skipInference
```

`--skipInference` 让此步骤只构建和保存引擎。构建本身也会进行计算与 tactic 测量，应在资源可用时进行。

静态 ONNX 不需要额外设置动态 profile。固定 batch=1 的图，要增加 batch 应重新导出相应静态图，或导出真正的动态维度，不能只在命令后面追加一个 batch 数。

`--fp16` 允许编译器使用 FP16，并不要求每一层和输入输出都是 FP16。先验证浮点路径，再增加 INT8 及校准逻辑，便于区分模型导出、编译与量化误差。

## 6.3 动态 batch 与 profile

假设输入叫 `images`，只有首维动态，通道和分辨率固定为 3×640×640：

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=model-dynamic.onnx \
  --saveEngine=model-dynamic-fp16.plan \
  --fp16 \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:4x3x640x640 \
  --maxShapes=images:8x3x640x640 \
  --skipInference
```

这里的 1、4、8 只是示例。min/max 限定接受范围，opt 是构建优化的重要目标，应贴近常见请求；不能只按“能装下多大的 batch”设置范围。多输入模型需要覆盖各动态输入，并维持输入间的维度关系。

TensorRT 8.5 与 10.3 的脚本也应分开核对：

| 配置 | TensorRT 10.3 示例 | 迁移时要检查 |
| --- | --- | --- |
| 只构建引擎 | `--skipInference` | 旧脚本可能使用 `--buildOnly` |
| workspace 限制 | `--memPoolSize=workspace:1024` | 旧 `--workspace` 参数不能直接照搬 |
| batch | ONNX 维度与 profile | 不沿用 implicit batch 的 `--maxBatch` 思路 |
| 日志 | 按当前帮助选择 `--verbose` 等 | 不添加不存在的 `--quiet=false` |

workspace 限制也不是整个构建进程或运行时的总内存上限。Orin 使用共享物理内存，模型权重、激活、CPU 应用和媒体缓冲区会共同占用资源。

## 6.4 引擎的兼容边界

`.plan` 与 `.engine` 通常只是序列化引擎的不同扩展名。引擎包含面向目标硬件和 runtime 的优化结果，不能因为服务器生成成功，就认为能复制到 Orin 运行。

常规部署优先在目标 Orin 或匹配的 Jetson 构建环境生成引擎。版本兼容与硬件兼容是不同选项，并且有平台限制；JetPack 不支持的硬件兼容模式，不能用来实现桌面 GPU 到 Orin 的通用引擎迁移。

# 七、运行验证与部署打包

## 7.1 分开检查数值和性能

静态引擎加载与计时示例：

```bash
/usr/src/tensorrt/bin/trtexec \
  --loadEngine=model-fp16.plan \
  --warmUp=1000 \
  --duration=10
```

动态引擎运行时还要设置具体输入形状：

```bash
/usr/src/tensorrt/bin/trtexec \
  --loadEngine=model-dynamic-fp16.plan \
  --shapes=images:4x3x640x640 \
  --warmUp=1000 \
  --duration=10
```

这些命令会实际占用 GPU。默认生成的测试数据适合检查执行和计时，不代表真实图片上的精度已经通过。数值验证应给 PyTorch、ONNX 和 TensorRT 输入相同的前处理张量，逐步比较原始输出与最终结果；动态模型还要覆盖 profile 的边界和常用形状。

记录功耗模式与频率状态时可以查询：

```bash
sudo nvpmodel -q
sudo jetson_clocks --show
```

`jetson_clocks --show` 是查询，不等于执行不带参数的锁频操作。比较性能时同时记录温度、内存和其他负载，区分 GPU 计算时间、数据传输与端到端时延，不直接把 `trtexec` 的吞吐当成视频服务处理速度。

## 7.2 打包可复现的部署材料

| 材料 | 需要记录的内容 |
| --- | --- |
| 系统清单 | L4T、内核、架构、CUDA、TensorRT、Python |
| 原始模型 | ONNX、外部权重、模型哈希 |
| 构建信息 | 完整命令、profile、精度、构建日志与所需插件 |
| 推理约定 | 输入输出名称、dtype、前后处理、类别表 |
| 运行环境 | 镜像 digest 或包版本、动态库配置、启动命令 |
| 验证材料 | 固定输入、参考结果、误差标准和性能测试条件 |

```bash
sha256sum model.onnx model-fp16.plan
```

涉及外部权重和自定义插件时也要计算对应哈希。部署包中的路径使用可配置目录，账号、网络地址和服务凭据由运行环境注入。

遇到问题时按“系统包 → 动态库 → Python binding → 引擎 → 应用”的顺序缩小范围。系统 Python 与原生 TensorRT 正常后，再接入虚拟环境和容器，最后进入真实媒体链路，能减少多个环境问题叠在一起时的定位成本。

# 八、参考资料

- [JetPack 6.2.2 软件栈与安装方式](https://developer.nvidia.com/embedded/jetpack-sdk-622)
- [Jetson Linux R36.5 刷机指南](https://docs.nvidia.com/jetson/archives/r36.5/DeveloperGuide/SD/FlashingSupport.html)
- [Jetson Linux 软件包与更新机制](https://docs.nvidia.com/jetson/archives/r36.5/DeveloperGuide/SD/SoftwarePackagesAndTheUpdateMechanism.html)
- [NVIDIA Container Toolkit 安装与 Docker 配置](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [Jetson PyTorch 安装与兼容要求](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)
- [TensorRT trtexec 参数](https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/command-line-programs.html)
- [TensorRT 引擎兼容边界](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html)
- [TensorRT 10.x 支持矩阵](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/support-matrix.html)

