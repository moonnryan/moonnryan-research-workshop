---
title: Ascend 910B 驱动异常与异步错误定位
created: 2026-08-07
updated: 2026-08-13
status: draft
blog: true
tags: [Ascend, 910B, 驱动, CANN, 异步执行]
---

# Ascend 910B 驱动异常与异步错误定位

NPU 任务超时在 Python 中可能表现为拷贝、loss 或同步失败。报错位置是错误被观察的地方，不一定是最初出错的 kernel。若最小随机数或矩阵运算也失败，应先退出模型层排查。

归档 910B 故障中，一度出现设备枚举失败、Unknown symbol 与驱动模块路径混乱。清理匹配组件并重装后，最小探针及原 YOLO 训练恢复；证据支持宿主驱动栈问题，但没有直接证明固件本身损坏。

# 一、区分可枚举与可执行

```bash
npu-smi info
lspci -nn
ls -l /dev/davinci*
journalctl -k -b
modinfo drv_pcie_host
```

PCIe 能识别设备、设备节点存在、npu-smi 正常和真实算子通过，是逐层增强的证据。vermagic 匹配当前 kernel，也不能排除模块之间的符号或依赖冲突。

历史更新目录、DKMS 链接和 modules.dep 可能同时残留不同版本。检查实际加载文件与依赖，不能只看安装器显示成功。

# 二、最小算子与同步

```python
import torch
import torch_npu

x = torch.ones((256, 256), device="npu")
y = x @ x
torch.npu.synchronize()
print(y[0, 0].cpu().item())
```

随机数、Conv 前反向和真实 batch 分别作为独立探针执行，避免前一个异步错误污染后面的结论。测试在空闲设备上运行，超时则保留首个 kernel 日志。

组合命令没有完整输出，不足以证明其中某个随机数算子必现故障。归档重装后的独立复测通过，因此需要撤销早期“RNG 持续异常”的判断。

# 三、修复与验收

安装方式、安装前缀和宿主 kernel 一起盘点。遇到 bind mount，卸载器无法删除挂载点属于存储布局问题，不能直接递归删除底层数据。

维护后先确认无 Unknown symbol、设备健康及探针通过，再恢复容器。归档原 YOLO 任务恢复训练和验证，说明修复支持原负载；包状态、import 成功和实际训练结果仍需分别记录。

# 四、一些反思

异步错误适合通过同步点和最小探针定位。把完整训练拆到标量、随机数、矩阵乘与真实 batch，可以迅速判断问题是否已超出数据和模型范围。

重装是恢复措施，也会一次改变多个组件。没有逐组件对照时，应说明“清理并重建组合后恢复”，而非给某个单包赋予未经证明的唯一根因。

# 五、参考资料

- [Ascend PyTorch](https://github.com/Ascend/pytorch)
- [Ascend 官方文档](https://www.hiascend.com/document)

