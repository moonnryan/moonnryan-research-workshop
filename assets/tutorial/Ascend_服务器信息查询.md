---
title: Ascend 服务器硬件与环境信息查询
created: 2026-01-05
updated: 2026-01-28
status: draft
blog: false
tags: [Ascend, 910B, Linux, CANN, 环境配置]
---

# Ascend 服务器硬件与环境信息查询

接手一台昇腾服务器时，通常需要先弄清 CPU 架构、内存和存储配置、NPU 型号与数量，以及驱动、固件和 CANN 的安装情况。这些信息决定了后续选择什么软件包、如何配置运行环境，也能为排查部署问题提供依据。

下面以 Ascend 910B 服务器为例，按宿主硬件、NPU、软件环境和网络四部分整理常用查询命令。安装路径和设备编号以实际机器为准。

# 一、查询宿主硬件

```bash
cat /etc/os-release
uname -a
lscpu
free -h
lsblk
df -h
lspci -nn
lspci -tv
```

需要管理员权限的信息单独采集：

```bash
sudo dmidecode -t processor
sudo dmidecode -t memory
sudo nvme list
```

| 模块 | 查询内容 | 主要命令 |
| --- | --- | --- |
| CPU | 架构、型号、核数、NUMA | `lscpu` |
| 内存 | 总容量、当前占用 | `free -h` |
| 内存条 | 插槽、规格、配置频率 | `dmidecode -t memory` |
| 存储 | 磁盘容量、分区、挂载 | `lsblk`、`df -h` |
| PCIe | 设备型号、总线拓扑 | `lspci -nn`、`lspci -tv` |

`free` 显示操作系统可用的内存，`dmidecode` 显示内存条和插槽信息；两者关注的层次不同。存储也类似：`lsblk` 用于查看磁盘与分区关系，`df` 用于查看已挂载文件系统的剩余空间。

# 二、查询 NPU 与驱动信息

```bash
npu-smi info
npu-smi -v
ls -l /dev/davinci*
cat /usr/local/Ascend/driver/version.info
cat /usr/local/Ascend/firmware/version.info
cat /etc/ascend_install.info
```

`npu-smi info` 用于查看设备型号、数量、健康状态、温度、功耗及内存使用等信息，具体字段随设备和工具版本变化。`npu-smi -v` 查询管理工具版本，驱动与固件版本则查看对应的 `version.info` 文件。

需要观察任务运行期间的变化时，可以定时刷新：

```bash
watch -n 1 npu-smi info
```

`/dev/davinci*` 是后续容器设备映射的重要依据。宿主机能看到 NPU，而容器中看不到时，可以先对比两侧的设备节点和驱动库挂载。

# 三、查询 CANN 与框架环境

先查看实际安装目录中的版本文件，以及当前 Python 环境：

```bash
ls -l /usr/local/Ascend/ascend-toolkit/
find /usr/local/Ascend/ascend-toolkit -name version.info -type f
python3 --version
python3 -m pip show torch torch-npu
```

找到版本文件后，用 `cat` 读取对应内容。机器上可能同时保留多套 CANN，查询时需要区分安装了哪些版本、当前进程使用哪个版本；`latest` 链接的指向和环境变量中的路径可以帮助定位：

```bash
readlink -f /usr/local/Ascend/ascend-toolkit/latest
printenv ASCEND_HOME_PATH ASCEND_OPP_PATH LD_LIBRARY_PATH PYTHONPATH
```

驱动、固件、CANN、PyTorch 和 `torch_npu` 需要作为一组记录。准备安装或升级时，再根据这组版本查对应的兼容关系。

如果还需要确认框架能调用 NPU，可以加载实际安装的 CANN 环境，执行一次简单的矩阵乘法：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 - <<'PY'
import torch
import torch_npu

print(torch.__version__, torch_npu.__version__)
print("device count:", torch.npu.device_count())
x = torch.ones((1024, 1024), device="npu")
y = x @ x
torch.npu.synchronize()
print(y[0, 0].cpu().item())
PY
```

正常结果为 1024。版本查询只读取环境信息，这一步会实际执行 NPU 运算，用于检查从 PyTorch 到设备的基本调用链。

# 四、查询网络接口与链路

```bash
ip -br link
ip -br addr
ip route
lspci -nn | grep -Ei 'ethernet|network'
```

先确认接口名称，再查询某个物理接口的链路信息。下面的 `eth0` 需要替换为实际名称：

```bash
sudo ethtool eth0
```

重点看 `Speed`、`Duplex` 和 `Link detected`，分别表示当前协商速率、双工模式和链路状态。这里得到的是接口链路信息；端到端吞吐还受对端设备、交换机和传输路径影响，需要另做网络测试。

# 五、整理环境清单

查询结果可以整理成一份简短清单，后续部署或排障时直接对照：

| 类别 | 建议记录 |
| --- | --- |
| 宿主机 | 操作系统、内核、CPU 架构与 NUMA、内存、存储 |
| NPU | 型号、数量、内存、设备节点、当前健康状态 |
| 软件栈 | 驱动、固件、CANN、Python、PyTorch、`torch_npu` |
| 网络 | 物理接口、链路速率、路由、用途 |
| 调用检查 | 是否执行张量测试、输出或异常 |

保留命令输出比只写“910B 服务器”更有用，尤其是机器存在多套软件环境时。软件升级后同步更新清单；对外分享时去掉序列号、网络地址、账号和管理平台信息。

# 六、参考资料

- [Ascend PyTorch](https://github.com/Ascend/pytorch)
- [Ascend 官方文档](https://www.hiascend.com/document)

