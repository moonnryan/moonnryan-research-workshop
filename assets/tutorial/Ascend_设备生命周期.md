---
title: Ascend 驱动、固件与 CANN 的安装和维护
created: 2026-02-24
updated: 2026-08-18
status: draft
blog: false
tags: [Ascend, 910B, 310P, CANN, 驱动, 维护]
---

# Ascend 驱动、固件与 CANN 的安装和维护

Ascend 软件栈维护涉及驱动、固件、CANN Toolkit、芯片专属 OPS 和可选 NNAL。安装完成并不代表组合兼容；维护流程应从精确识别硬件与现有版本开始，在每层验证后再进入下一层。

本文覆盖 910B 和 310P 等产品的通用流程。具体安装文件、版本组合和升级顺序，以对应产品及 release 的官方说明为准，不能把不同芯片的包交叉使用。

# 一、盘点环境与负载

```bash
uname -m
uname -r
cat /etc/os-release
npu-smi info
lspci -nn
lsmod
docker ps
```

Debian 系采集 `dpkg -l`，RPM 系采集 `rpm -qa`，记录安装方式及安装前缀。检查内核模块的 `vermagic`、当前 kernel、安装日志和设备错误。

不要因为 Python 报错就直接重装驱动。先确认宿主能枚举设备，再运行最小同步探针；NPU 异步错误可能在后续拷贝或同步处才被报告。

# 二、选择维护范围

| 情况 | 维护范围 | 验证重点 |
| --- | --- | --- |
| CANN 升级 | Toolkit＋匹配 OPS，按需 NNAL | 驱动兼容范围 |
| Python 依赖冲突 | 应用环境 | torch/torch_npu/CANN |
| 模块与 kernel 不匹配 | 驱动与内核 | 模块加载、符号错误 |
| 设备不枚举 | 固件、驱动或硬件 | PCIe、供电、设备日志 |

维护前停止目标设备工作负载，保存配置和数据。移除、重启及固件操作在维护窗口执行。每个组件使用一种安装机制，避免 apt/yum 与 runfile 同时管理相同文件。

# 三、卸载与残留处理

按反向依赖移除可选 NNAL、芯片 OPS、Toolkit。apt 系先预览明确包名的计划：

```bash
apt-get -s remove ascend-cann-nnal \
  ascend-cann-910b-ops ascend-cann-toolkit
```

示例针对实际安装了这些包的 910B 环境；310P 替换为其确切包名。runfile 安装则使用相同 installer 提供的卸载入口。不要递归删除整个 `/usr/local/Ascend`。

检查 shell 中手工添加的环境、旧动态库路径和 DKMS 链接，只清理已经移除安装的残留。驱动维护后按产品说明重启，再检查 kernel 日志。

# 四、安装固定组合

先从官方兼容矩阵取得匹配的驱动和固件。归档 runfile 流程中，首次安装为 Driver 后 Firmware，原地升级为 Firmware 后 Driver；这两条路径不同，执行前仍需对照选定产品的安装文档。

设备重启并恢复健康后再安装 Toolkit 和专属 OPS。例如已配置官方仓库的 apt 环境：

```bash
CANN_VERSION="<target-version>"
sudo apt-get install "ascend-cann-toolkit=$CANN_VERSION"
source /usr/local/Ascend/cann/set_env.sh
sudo apt-get install "ascend-cann-910b-ops=$CANN_VERSION"
```

`<target-version>` 必须替换为仓库实际存在、且兼容当前设备的版本。旧安装布局的环境入口可能在 `ascend-toolkit` 下，不能把路径是否存在当成版本判断。

NNAL 只在应用需要 ATB 或 AscendSiPBoost 时安装。按应用加载需要的环境，避免把互相影响的可选库全部写进全局 shell。

# 五、逐层验收

```bash
npu-smi info
python3 -c "import acl; print('PyACL OK')"
python3 -c "import torch, torch_npu; print(torch.npu.is_available())"
```

之后依次测试设备张量、逐元素计算、矩阵乘、随机数、Conv2d 前反向和一个真实 batch。每个探针单独同步并设定超时，保存首个错误；最后才进入多卡通信和完整模型。

容器挂载的驱动库需与宿主相符，容器 CANN 与框架也需兼容。宿主健康、容器导入成功和模型运行成功是三个不同验收阶段。

# 六、参考资料

- [Ascend 官方文档与软件中心](https://www.hiascend.com/document)
- [Ascend PyTorch](https://github.com/Ascend/pytorch)

