---
title: Ascend 910B 服务器验收流程
created: 2026-01-05
updated: 2026-01-28
status: draft
blog: false
tags: [Ascend, 910B, 验收, CANN, 多卡]
---

# Ascend 910B 服务器验收流程

Ascend 910B 服务器验收需要同时回答三个问题：交付配置是否一致，设备及软件栈是否健康，真实训练和推理任务是否可用。单条 `npu-smi info` 只能证明设备当前可枚举，不能替代多卡通信、持续负载和模型功能验证。

本文把验收材料压缩为可执行流程。卡数、内存容量、网络速率和性能阈值以采购与交付说明为准，不将某种 910B 配置的标称值套到所有设备上。

# 一、记录基础配置

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

| 模块 | 核验项 | 归档内容 |
| --- | --- | --- |
| CPU | 架构、型号、核数、NUMA | 命令输出与交付对照 |
| 内存 | 容量、规格、频率 | 插槽与总容量 |
| 存储 | 型号、容量、挂载 | 可用空间与健康状态 |
| 网络 | 型号、速率、链路 | 接口名称与实测吞吐 |
| NPU | 型号、卡数、容量 | 每卡输出与拓扑 |

# 二、核对 Ascend 软件栈

```bash
npu-smi info
npu-smi -v
ls -l /dev/davinci*
cat /usr/local/Ascend/driver/version.info
cat /usr/local/Ascend/firmware/version.info
cat /etc/ascend_install.info
```

部分安装布局的版本文件位置不同，应按实际安装目录采集。记录驱动、固件、CANN、Python、PyTorch 和 `torch_npu` 的完整组合，确认其兼容矩阵，不能只比较单个组件是否“最新”。

加载实际安装的 CANN 环境后，执行最小张量探针：

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

该输出应为 1024。它验证基本调用链，尚未覆盖 HCCL、多进程或长时间负载。

# 三、测试健康与性能

先在空闲维护窗口运行与设备型号匹配的 Ascend DMI 测试；具体参数以所安装版本的 `--help` 为准。归档功耗、温度、内存带宽和链路结果，并检查是否持续出现错误或降频。

CPU、磁盘和网络独立测试。磁盘测试使用专门创建的文件，不对系统盘块设备直接写入：

```bash
fio --name=acceptance --filename=/srv/acceptance-fio.bin \
  --size=2G --rw=randread --bs=4k --iodepth=16 \
  --ioengine=libaio --direct=1 --runtime=60 --time_based \
  --group_reporting
```

网络 `iperf3` 需要对端配合，且吞吐受链路、网卡、NUMA 和 CPU 共同影响。把接口标称值与端到端测试分开记录。

# 四、验证实际模型

| 任务 | 基础条件 | 通过标准 |
| --- | --- | --- |
| YOLO 多卡训练 | 固定小数据集、固定步数 | 每卡参与、loss 有限、能保存 |
| 语言模型微调 | 固定配置与随机种子 | 前反向、评估、checkpoint 完成 |
| 多模态在线推理 | 固定文本和图片样本 | 输出合理、接口及连续请求正常 |
| 多卡通信 | 明确卡数与 HCCL 配置 | 无超时、无卡长期闲置 |

每个模型先单卡再多卡。多卡结果需要记录实际 batch、梯度累积和数据加载时间；总耗时下降并不必然等于线性扩展。

# 五、生成验收报告

报告包括环境清单、各项命令、原始输出、阈值来源、模型配置和失败项。无法完成的测试标记为未验证，不能用另一项成功代替。

交付时保留可复现的测试配置及依赖版本；对外版本去掉序列号、网络地址、账号和管理平台信息。验收完成后若驱动或 CANN 升级，应重跑张量探针、通信与关键模型冒烟。

# 六、参考资料

- [Ascend PyTorch](https://github.com/Ascend/pytorch)
- [Ascend 官方文档](https://www.hiascend.com/document)
- [fio](https://github.com/axboe/fio)

