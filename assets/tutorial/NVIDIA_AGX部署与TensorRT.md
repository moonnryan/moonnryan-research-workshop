---
title: NVIDIA AGX Orin 系统与 TensorRT 环境部署
created: 2026-03-30
updated: 2026-06-09
status: draft
blog: false
tags: [NVIDIA, AGX Orin, Jetson, TensorRT, ABI]
---

# NVIDIA AGX Orin 系统与 TensorRT 环境部署

Jetson AGX Orin 的驱动、CUDA 和 TensorRT 随 Jetson Linux / JetPack 发布组合演进。安装桌面 GPU 的普通 CUDA 包，或只更新 Python 的 TensorRT 包，可能破坏这套组合。部署应先确认 BSP，再处理容器、动态库和模型引擎。

# 一、刷机与系统基线

按 NVIDIA 对应版本文档选择主机系统、目标板配置和存储介质，通过 SDK Manager 或 BSP 提供的 flash 工具刷机。Recovery 接入后核对 USB 设备，再执行该板型的刷机流程；不同 carrier board 的配置不能混用。

系统启动后归档：

```bash
cat /etc/nv_tegra_release
uname -a
uname -m
dpkg-query -W 'nvidia-l4t-*'
getconf GNU_LIBC_VERSION
```

完成用户、网络和磁盘配置，并确认重启后持久化。服务账号、网络地址和设备序列号不进入公开部署文档。

# 二、检查 GPU 容器运行时

安装与 JetPack 匹配的 NVIDIA Container Toolkit。若该版本提供 `nvidia-ctk`，配置方式为：

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

在维护窗口执行 Docker 重启。选择明确支持当前 Jetson Linux 的镜像，再验证容器能加载宿主驱动。容器与宿主共享 kernel 和设备驱动，容器内的库版本仍需单独匹配。

# 三、修复动态库查找

归档环境为 TensorRT 10.3、CUDA 12.6。曾出现 `import tensorrt` 找不到 `libcudla.so.1`，库实际位于 CUDA 的 ARM64 target 目录，问题来自动态链接器路径。

先检查文件与依赖：

```bash
ls -l /usr/local/cuda-12.6/targets/aarch64-linux/lib/libcudla.so.1
ldconfig -p | rg libcudla
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

确认路径正确后，加入搜索配置：

```bash
printf '%s\n' '/usr/local/cuda-12.6/targets/aarch64-linux/lib' \
  | sudo tee /etc/ld.so.conf.d/cuda-12-6-aarch64.conf
sudo ldconfig
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

使用不同 CUDA 版本时替换实际路径。若文件不存在，应恢复相应系统包，不能通过软链接把不匹配版本伪装成需要的库。

# 四、编译静态 FP16 引擎

先在目标 AGX 环境建立基线：

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=model.onnx \
  --saveEngine=model-fp16.plan \
  --fp16
```

静态 ONNX 不需要额外定义动态 batch profile。若模型确实使用动态维度，需依据对应 TensorRT 版本设置完整的 min/opt/max shape，并单独测试边界。

`.plan` 和 `.engine` 常作为同类序列化引擎的扩展名，名称不同不能带来兼容性。跨 GPU 编译应依据官方支持范围核验；不能因为服务器生成成功，就默认该引擎能在 Orin 上加载。

# 五、验收与打包

| 阶段 | 验证内容 |
| --- | --- |
| 原始模型 | PyTorch 与 ONNX 同输入对照 |
| 引擎加载 | 目标设备反序列化成功 |
| 数值 | 分数、框、类别或特征对齐 |
| 性能 | 预热、重复测量、功耗模式一致 |
| 服务 | 实际媒体链路、重启与错误恢复 |

打包时记录 JetPack、CUDA、TensorRT、模型哈希、输入输出和前处理。将 Python 环境与媒体服务的动态库环境分开配置，避免 conda 的 `LD_LIBRARY_PATH` 污染系统媒体插件。

# 六、参考资料

- [Jetson Linux 文档](https://docs.nvidia.com/jetson/archives/)
- [TensorRT 支持矩阵](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)

