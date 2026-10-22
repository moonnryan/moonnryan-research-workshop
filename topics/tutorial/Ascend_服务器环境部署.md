---
title: Ascend 服务器环境部署与信息查询
created: 2026-01-05
updated: 2026-08-18
status: ready
blog: true
tags: [Ascend, 910B, 310P, CANN, 驱动, 部署]
aliases: [/assets/tutorial/Ascend_服务器环境部署.html, /assets/tutorial/Ascend_服务器信息查询.html, /assets/tutorial/Ascend_设备生命周期.html]
---

# Ascend 服务器环境部署与信息查询

昇腾服务器的环境部署涉及宿主机驱动、设备固件、CANN Toolkit、芯片专属 OPS，以及应用使用的 PyTorch 和 torch_npu。安装完成后，还需要查询硬件与软件的实际状态，并通过设备运算确认调用链可用。

本文先整理软件栈的安装流程，再集中列出硬件、NPU、框架和网络信息的查询方法，最后说明运行验证与后续维护。以 910B 服务器为主，涉及 310P 时按对应产品替换安装包和设备配置；具体版本与升级顺序以选定产品的官方说明为准。

# 一、确认安装对象与版本组合

安装前先确认设备型号、CPU 架构、操作系统和运行内核，用于选择安装包和编译依赖：

```bash
uname -m
uname -r
cat /etc/os-release
lspci -nn
```

已有环境还应保存 `npu-smi info`、驱动与固件版本、CANN 版本和 Python 包清单。尚未安装驱动时，`npu-smi` 不存在或设备节点未生成属于预期状态，不能用这一点判断硬件损坏。

版本选择需要覆盖 Driver、Firmware、CANN、芯片 OPS、Python、PyTorch 和 torch_npu。按官方兼容矩阵选择完整组合，保留 wheel 的 `.postN` 后缀；同一个 PyTorch 版本下的不同 torch_npu 包，可能对应不同 CANN 版本。

驱动安装还需按对应系统说明准备当前内核的 headers、编译工具和 DKMS 等依赖。记录已有组件使用 apt、yum 还是 runfile 安装，后续由同一种机制管理同一组件。

# 二、安装驱动、固件与 CANN

## 2.1 宿主机驱动与固件

先从官方软件中心取得匹配设备型号和系统架构的 Driver、Firmware 安装包，核对完整文件名、版本与校验信息。910B、310P 等产品的包不能交叉使用。

首次安装与覆盖升级的顺序不同：首次安装通常为 Driver 后 Firmware，同时升级则按 Firmware、Driver、CANN 的顺序。实际执行时仍需对照对应产品和发行版本的说明，使用规定的安装参数及重启方式。

驱动与固件安装后，按安装器要求重启，再检查：

```bash
npu-smi info
ls -l /dev/davinci*
journalctl -k -b
```

确认设备型号、数量和健康状态正常，启动日志没有新的驱动符号或设备初始化错误，再进入用户态软件安装。内核驱动属于宿主机，容器中的 CANN 镜像不能替代它。

## 2.2 CANN Toolkit 与芯片 OPS

下面以已配置官方软件仓库的 910B、apt 环境为例，版本变量必须替换为实际存在且兼容的版本：

```bash
CANN_VERSION="<target-version>"
sudo apt-get install "ascend-cann-toolkit=$CANN_VERSION"
source /usr/local/Ascend/cann/set_env.sh
sudo apt-get install "ascend-cann-910b-ops=$CANN_VERSION"
```

310P 等产品替换为相应芯片的软件包；runfile 安装则沿用该发行版提供的安装入口。不要把一套安装的目录交给另一种包管理机制覆盖。

新旧 CANN 的目录布局可能不同，环境入口可能在 `cann/set_env.sh` 或 `ascend-toolkit/set_env.sh`。先确认文件及软链接指向，再加载对应环境。进程需要使用哪一套 CANN，就在其启动入口中明确加载，避免反复把不同版本写入全局 shell。

NNAL 只在应用需要 ATB 或 AscendSiPBoost 时安装。安装前按要求加载 Toolkit 环境，并在安装后确认真实库文件和版本信息存在。ATB、AscendSiPBoost 的环境按具体应用选择，不同时追加到全局配置。

## 2.3 Python 框架与容器环境

根据兼容矩阵，在独立 Python 环境中安装固定版本的 PyTorch、torch_npu 和所需 torchvision。安装后记录完整版本，避免后续添加依赖时自动改变框架组合。

```bash
python3 --version
python3 -m pip show torch torch_npu torchvision
```

使用容器时，宿主机提供内核驱动，容器提供用户态 CANN 与框架。设备节点和驱动库的映射按所用 Ascend 容器运行时说明配置；不要将宿主机空目录或旧 Toolkit 目录挂载到镜像的有效 CANN 目录上。

# 三、查询硬件与环境信息

## 3.1 CPU、内存、存储与 PCIe

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

需要管理员权限的信息单独采集，相关工具需已安装：

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

`free` 显示操作系统可用的内存，`dmidecode` 显示内存条和插槽信息。存储也类似：`lsblk` 用于查看磁盘与分区关系，`df` 用于查看已挂载文件系统的剩余空间。

## 3.2 NPU、驱动与固件

```bash
npu-smi info
npu-smi -v
ls -l /dev/davinci*
cat /usr/local/Ascend/driver/version.info
cat /usr/local/Ascend/firmware/version.info
cat /etc/ascend_install.info
```

安装记录与版本文件的位置随安装方式变化，不存在时应回到对应组件的实际安装目录查询。`npu-smi info` 可查看型号、数量、健康状态、温度、功耗和内存使用等信息，具体字段随设备和工具版本变化。`npu-smi -v` 查询管理工具版本，不能代替驱动与固件各自的版本文件。

观察任务运行期间的变化时，可以定时刷新：

```bash
watch -n 1 npu-smi info
```

宿主能看到 NPU，而容器中看不到时，先对比两侧的设备节点和驱动库挂载。设备健康、框架能发现设备与真实算子运行成功，需要分别检查。

## 3.3 CANN 与框架

机器可能同时保留多套 CANN，需要区分“安装了哪些版本”和“当前进程使用哪个版本”：

```bash
ls -l /usr/local/Ascend/
readlink -f /usr/local/Ascend/cann
readlink -f /usr/local/Ascend/ascend-toolkit/latest
printenv ASCEND_HOME_PATH ASCEND_TOOLKIT_HOME ASCEND_OPP_PATH
printenv LD_LIBRARY_PATH PYTHONPATH
python3 -m pip show torch torch_npu torchvision
```

再到实际版本目录读取 `version.info` 或对应组件的版本文件。`latest` 链接和环境变量可以帮助定位，但已启动的服务可能仍在使用旧库；运行异常时需要进一步核对进程的动态库映射。

## 3.4 网络接口与链路

```bash
ip -br link
ip -br addr
ip route
lspci -nn | grep -Ei 'ethernet|network'
```

先确认接口名称，再查询物理接口的链路信息，下面的 `eth0` 替换为实际名称：

```bash
sudo ethtool eth0
```

重点看 `Speed`、`Duplex` 和 `Link detected`，分别表示当前协商速率、双工模式和链路状态。端到端吞吐还受对端、交换机和传输路径影响，需要另做网络测试。

# 四、验证框架到设备的调用链

版本查询只读取环境信息，下面的探针会实际执行 NPU 运算。先在进程入口加载所选 CANN 环境，再确认设备可用：

```bash
python3 -c "import acl; print('PyACL OK')"
python3 -c "import torch, torch_npu; print(torch.npu.is_available())"
```

在空闲设备上测试一次矩阵乘法，设备编号按实际分配修改：

```python
import torch
import torch_npu

print(torch.__version__, torch_npu.__version__)
print("device count:", torch.npu.device_count())
x = torch.ones((1024, 1024), device="npu:0")
y = x @ x
torch.npu.synchronize()
print(y[0, 0].cpu().item())
```

正常结果为 1024。随后根据负载分别测试原生随机数、Conv2d 前反向和一个真实 batch，每项单独同步并在独立进程中设置超时，保存首个错误；最后才进入多卡通信和完整模型。

PyACL 导入成功、框架发现设备和完整训练通过，是不同层次的证据。若出现异步超时，报错位置可能只是后续同步点，具体排查见[Ascend 910B 驱动异常与异步错误定位](../../topics/engineering-runtime/Ascend_驱动与异步错误.md)。

# 五、升级与卸载维护

| 情况 | 维护范围 | 验证重点 |
| --- | --- | --- |
| CANN 升级 | Toolkit＋匹配 OPS，按需 NNAL | 驱动兼容范围 |
| Python 依赖冲突 | 应用环境 | torch/torch_npu/CANN |
| 模块与内核不匹配 | 驱动与内核 | 模块来源、符号错误 |
| 设备不枚举 | 驱动、固件或硬件 | PCIe、供电、设备日志 |

先根据故障层次选择维护范围。涉及驱动卸载、固件和重启时，安排维护窗口，停止所有受影响的 NPU 负载，保存版本、配置和日志。Python 环境冲突无需直接扩大成整机驱动重装。

移除 CANN 时，按反向依赖处理可选 NNAL、芯片 OPS 和 Toolkit。apt 环境先预览明确包名的移除计划：

```bash
apt-get -s remove ascend-cann-nnal \
  ascend-cann-910b-ops ascend-cann-toolkit
```

示例针对已安装这些包的 910B 环境。runfile 组件使用对应卸载器，Driver 和 Firmware 也分别按官方流程处理，不直接递归删除整个 `/usr/local/Ascend`。

卸载后检查手工添加的环境、软链接、旧动态库路径和 DKMS 模块来源，只清理已经确认属于旧安装的残留。遇到 bind mount，先确认挂载关系；卸载器无法删除挂载点，不等于应该删除底层数据。

按对应安装或升级流程恢复环境，必要时重启，再依次重跑设备查询、独立算子和实际负载。安装器成功、包状态正常和运行验证通过，都应留下记录。

基础运行链路通过后，再按模型路径检查能力：多模态权重与量化格式选择见 [Ascend 多模态模型的量化与工具链边界](../quantization-accuracy/Ascend_多模态量化边界.md)，多段 OM 的动态形状与运行接口见 [Ascend SAM3 的图文配对与动态编译](../graphs-operators/Ascend_SAM3子图与档位.md)。

# 六、保存环境清单

| 类别 | 建议记录 |
| --- | --- |
| 宿主机 | 操作系统、内核、CPU 架构与 NUMA、内存、存储 |
| NPU | 型号、数量、内存、设备节点、当前健康状态 |
| 软件栈 | Driver、Firmware、CANN/OPS、Python、PyTorch、torch_npu |
| 安装来源 | 安装方式、目录、软链接、容器镜像与挂载 |
| 网络 | 物理接口、链路速率、路由、用途 |
| 运行检查 | 已执行的探针、实际负载、输出或异常 |

保留命令输出比只写“910B 服务器”更有用。软件升级后同步更新清单，对外分享时去掉序列号、网络地址、账号和管理平台信息。

# 七、参考资料

- [Ascend 官方文档与软件中心](https://www.hiascend.com/document)
- [Ascend PyTorch 版本配套表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.md)
- [CANN 安装说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0008.html)
- [CANN 升级前必读](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/softwareinst/instg/instg_0028.html)
