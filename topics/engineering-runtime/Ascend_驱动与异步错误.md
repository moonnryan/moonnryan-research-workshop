---
title: Ascend 910B 驱动异常与异步错误定位
created: 2026-08-07
updated: 2026-08-13
status: ready
blog: true
tags: [Ascend, 910B, 驱动, CANN, 异步执行]
---

# Ascend 910B 驱动异常与异步错误定位

910B 上的训练停在第一个 batch，最终报出流同步超时，Python 栈却落在 loss 中一次很普通的张量创建上。排查期间，基础前向和小网络反向曾经通过，替换 PyTorch、torch_npu 和 CANN 后，完整训练仍然挂起。直到同步测试把问题缩小到模型之外，再结合重启后的模块加载失败，才找到宿主驱动安装残留这条更明确的线索。

这类问题需要核对版本配套，但版本表只是起点。安装记录中的版本、进程实际加载的动态库、当前驻留的内核模块，以及重启后将要装载的模块，可能并不属于同一套安装。真正需要确认的是这些组件能否组成一条一致的执行链。

安装顺序、设备查询和基础运算探针见 [Ascend 服务器环境部署与信息查询](../tutorial/Ascend_服务器环境部署.md)。本篇关注这些检查之后仍然出现的挂起，以及如何确认实际加载的驱动来源。

# 一、版本匹配需要核对到哪一层

容器使用宿主机的内核与 NPU 驱动模块，镜像中的 CANN 和 Python wheel 无法替换宿主 `.ko`。因此，更新训练镜像只能改变用户态的一部分，不能自动修复驱动加载或设备状态。

| 层次 | 需要核对的内容 | 容易遗漏的地方 |
| --- | --- | --- |
| 内核与驱动模块 | 运行内核、模块来源、依赖符号 | 同一内核目录残留多套模块 |
| Driver 与 Firmware | 产品型号、完整版本、兼容声明 | 固件包版本与内部版本命名不同 |
| CANN 与芯片 OPS | Toolkit、Runtime、算子包及实际路径 | 只升级 Toolkit，或挂载覆盖镜像文件 |
| PyTorch 与 torch_npu | 完整 wheel 版本及配套 CANN | 忽略 `.postN` 对应的产品版本 |
| 应用进程 | 环境变量、动态库映射、启动参数 | shell 检查正常，服务仍用旧环境 |

官方配套表中，`torch_npu 2.8.0.post2` 对应 CANN 8.5.0，`2.8.0.post4` 对应 CANN 9.0.0。现场曾只将 post2 换成 post4，而进程仍加载 CANN 8.5.0；这组对照没有建立配套环境，因此不能据此判断框架升级是否有效。后续使用的 `torch 2.9.0 + torch_npu 2.9.0.post6 + CANN 9.1.0` 则属于配套表中的组合。

几组测试放在一起，更容易看出只比较 wheel 版本会遗漏什么：

| 组合与状态 | 观察结果 |
| --- | --- |
| torch_npu 2.8.0.post2＋CANN 8.5.0，宿主驱动未清理 | 训练首批挂起 |
| 只换成 torch_npu 2.8.0.post4，仍使用 CANN 8.5.0 | 同类挂起，且没有形成官方配套组合 |
| torch_npu 2.9.0.post6＋CANN 9.1.0，宿主驱动未清理 | 仍挂起，最小设备随机数也超时 |
| 保留上述框架组合，清理并重装驱动、固件及 CANN/OPS 后重启 | 原生随机数与原训练恢复 |

配套组合仍然失败时，需要继续核查宿主状态和实际加载来源，而不是无止境地更换 Python 包。

Driver、Firmware 和 CANN 也不能靠版本数字是否相近来判断兼容。现场 Driver 为 `26.0.rc1`，Firmware 内部版本为 `9.0.0.0.205`，其包版本仍为 `26.0.rc1`；原有 CANN 8.5.0 的内部版本系列也包含在驱动的兼容声明中。数字看起来不同，不等于已经发现冲突。

这些版本用于解释本次故障，不作为后续部署的固定推荐。选型时应按目标硬件及对应发行说明核对整套组合。

# 二、从环境变量查到实际加载文件

原容器曾把多个变量指向不存在的 CANN 8.5.1，实际安装的却是 8.5.0。修正路径后可以导入框架，但训练仍然挂起，说明路径错误确实存在，却没有解释全部现象。

后续镜像升级到 CANN 9.1 时，又将宿主机空的 Toolkit 目录挂载到容器同一路径，遮住了镜像中的入口。登录 shell 因找不到 `set_env.sh` 报错，但训练进程实际映射的库来自有效的 CANN 9.1 路径。此时应修正挂载，同时避免把 shell 警告直接当作设备超时的根因。

排查时至少保留三份信息：安装版本、启动环境和进程映射。以下命令用于读取现场，PID 替换为目标训练进程：

```bash
python3 -m pip show torch torch_npu torchvision
printenv | rg '^(ASCEND|PATH=|PYTHONPATH=|LD_LIBRARY_PATH=)'
readlink -f /usr/local/Ascend/ascend-toolkit/latest

ASCEND_DIAG_PID=12345
rg 'libascendcl|libge_|libhccl|libopapi|\(deleted\)' \
  "/proc/$ASCEND_DIAG_PID/maps"
```

`/proc/<pid>/maps` 能说明进程已经映射了哪些文件，但不会列出尚未加载的所有依赖。若出现 CANN 库的 `(deleted)` 映射，说明文件已被移除，进程仍持有旧映射；继续运行不能证明重启后还能恢复。重装用户态库后，需要重新启动相关进程才能验证新环境。

# 三、同步超时为什么会落在 loss 里

现场错误包含：

```text
rtStreamSynchronize execution failed, reason=task timeout
ACL error: 107020
```

Python 栈落在 loss 中根据特征图形状创建设备张量的位置。该路径触发了主机与设备间的拷贝或同步，发现之前提交的任务已经超时。它只能指出错误在这里被观察到，不能证明 loss、张量创建或拷贝就是最先失败的操作。

同时，设备日志反复报告相同的任务位置和未完成数量：

```text
SynchronizeExecutedTask: report three minutes timeout!
stream_id=47, task_id=1280, sqHead=500, pendingNum=781
```

这里有价值的是多次采样后队列位置仍不前进，而不是某个 stream 编号本身。结合 AICore 长期空闲，可以判断任务执行缺少进展；显存已经分配、总体利用率显示很高，都不足以证明正在正常计算。单次低利用率也不能排除首轮编译，需要结合持续日志判断。

定位时在新进程启动前开启同步模式：

```bash
export ASCEND_LAUNCH_BLOCKING=1
export TASK_QUEUE_ENABLE=0
```

`ASCEND_LAUNCH_BLOCKING=1` 用于让算子同步执行，使报错更接近实际调用位置；`TASK_QUEUE_ENABLE=0` 用于减少该框架版本的任务队列干扰。前者是调试配置，会降低性能，定位结束后不应沿用它测量训练速度。

训练侧同时固定 `workers=0`、`batch=1`、关闭 AMP、缓存与数据增强，并按所用适配代码关闭 pin memory。启动日志必须确认这些值真的生效。若参数使用 `argparse type=bool`，传入字符串 `False` 仍可能得到真值；这种情况下，命令看似关闭了功能，实际对照条件却没有改变。

# 四、最小探针把问题移出模型

同步模式下，积压任务从数百个缩小到少数几个。逐层前向测试还没进入模型第一层，创建 NPU 随机输入并同步就已经挂起。此时继续调整检测 loss 或数据增强，无法解释这个更小的复现。

| 探针 | 故障现场结果 | 能说明什么 |
| --- | --- | --- |
| NPU `ones`、填充及拷回 | 通过 | 部分内存与执行路径可用 |
| CPU 生成随机数后复制到 NPU | 通过 | 该输入下的 H2D 路径可用 |
| NPU 原生 `rand`、`randn`、`randint` | 同步超时 | 故障可在模型之外复现 |
| 关闭 worker、AMP、pin memory 后训练 | 仍挂起 | 这些功能不足以解释当前故障 |

每个探针应放在独立进程中，分别同步并由外层设置超时。不要在一次设备错误后连续执行其他测试，再把后续失败当作新的独立证据。矩阵乘与随机数可以分别测试，例如：

```python
# probe.py：每次启动只选择一个分支
import sys
import torch
import torch_npu

if sys.argv[1] == "matmul":
    x = torch.ones((256, 256), device="npu:0")
    y = x @ x
elif sys.argv[1] == "rng":
    y = torch.randn((1,), device="npu:0")
else:
    raise ValueError("expected matmul or rng")

torch.npu.synchronize()
print(y.flatten()[0].cpu().item())
```

```bash
timeout 60s python3 probe.py matmul
timeout 60s python3 probe.py rng
```

超时输出与退出状态应一起保存；进程未退出时，应先确认其状态，再发起下一项测试。探针只使用空闲且已授权的设备。

NPU 随机数失败并不意味着“随机数必须先在 CPU 生成”。CPU 生成再复制只绕过了设备随机数路径，训练内部仍可能隐式调用其他随机算子。重装后的独立复测中，原生随机数也恢复了，因此这里的 RNG 是暴露故障的探针，尚不足以认定为随机数算法本身的缺陷。

# 五、vermagic 一致仍可能装载错模块

整机重启后，PCI 总线仍能识别加速卡，主驱动却没有正常绑定，设备节点消失，内核日志出现大量 `Unknown symbol`。这时问题已经落到宿主驱动加载链，不需要再从训练框架入手。

进一步检查发现，同一运行内核下存在两套模块：新安装的实体文件位于 `updates/`，旧模块的符号链接位于 `updates/dkms/`，链接指向另一处历史安装目录。模块依赖索引与 `modprobe` 实际选择了后者。

```text
当前内核的模块目录
├── updates/          新安装的模块
└── updates/dkms/     指向历史模块的旧链接

模块依赖索引 → 旧链接 → 历史模块 → 符号依赖失败
```

两套文件都可能针对相同内核编译。因此 `vermagic` 与 `uname -r` 一致，只能排除一类内核构建不匹配，不能证明驱动内部组件来自同一次安装。符号还必须由兼容且正确装载的其他模块提供。

以下检查分别回答“当前内核是什么”“工具准备加载哪个文件”“依赖链指向哪里”：

```bash
uname -r
lsmod | rg 'ascend|drv_pcie|drv_devmm|davinci'
modinfo -n drv_pcie_host
modinfo -F vermagic drv_pcie_host
modprobe -n -v drv_pcie_host
modprobe --show-depends drv_pcie_host
journalctl -k -b | rg 'Unknown symbol|ascend|drv_pcie|davinci'
```

`modprobe -n -v` 是预览，不实际加载模块；`modinfo` 显示的是磁盘上解析到的文件，也不等于已经证明当前驻留模块的来源。应把这些结果与安装日志、软链接目标和当前启动日志一起核对。

这条模块来源冲突，是重启后驱动整体无法启动的最强证据。它也与重启前执行链异常相吻合，但不能反过来证明每一次训练超时都由同一个旧模块直接造成。

# 六、重建一致环境后如何验证

修复围绕同一目标展开：让驱动模块、依赖索引、固件与用户态组件恢复为可追踪的一套安装。维护中使用对应组件的卸载器处理旧安装，核对残留链接及模块来源，再按产品说明重装、重启并验证。手工混用两套路径 `insmod`，会让来源问题更难判断。

现场还有安装目录被 bind mount 的情况，卸载器因此无法删除挂载点；这与普通文件占用不同，不能靠递归删除目录解决。NNAL 则出现过包状态看似正常、实际子组件未完整安装的情况，需要加载 Toolkit 环境后重新安装并检查真实库文件。两者都说明安装命令的返回信息需要与文件和运行结果交叉核对。

清理后恢复的组合为 Driver `26.0.rc1`、Firmware `9.0.0.0.205`、CANN/910B OPS `9.1.0`，容器使用 `torch 2.9.0 + torch_npu 2.9.0.post6`。验证按以下顺序推进：

1. 重启后确认设备节点和健康状态，检查启动日志没有新的符号依赖错误
2. 确认模块解析不再指向旧 DKMS 链接，容器没有覆盖有效 CANN 目录
3. 分别执行设备填充、拷贝、矩阵乘和原生随机数，并完成同步
4. 恢复原训练命令，确认多个 epoch 的训练与验证能够持续完成

原训练随后持续通过多个 epoch，近期日志没有再次出现同类超时。训练脚本未修改，支持故障主要来自运行环境，而非通过改模型绕过。由于重建同时涉及多个组件，这仍不是驱动、固件和 CANN 各自影响的独立对照；尤其没有证据证明固件文件本身损坏。

# 七、一些反思

版本匹配有两层含义：选定的组合在兼容表中成立，机器实际使用的文件也确实属于这个组合。前者可以通过文档核对，后者需要看路径、模块索引、进程映射和重启后的状态。只记录 `pip list` 或安装器版本，容易漏掉真正生效的旧组件。

异步超时则需要另一条证据链。先把报错观察点与实际执行点分开，再用同步和独立探针缩小范围。最小随机数已经失败时，模型 loss 的栈位置就不应继续主导排查。

这次恢复最有价值的结果，是重建一致环境后，同一份训练代码重新工作。它支持清理安装残留的必要性，也提醒后续升级应保存模块来源和完整组件组合，让下一次故障有一个可比较的基线。

# 八、参考资料

- [Ascend PyTorch 版本配套表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.md)
- [Ascend PyTorch](https://github.com/Ascend/pytorch)
- [ASCEND_LAUNCH_BLOCKING 同步定位说明](https://www.hiascend.com/document/detail/zh/canncommercial/80RC22/apiref/envvar/envref_07_0050.html)
- [Ascend 官方文档与软件中心](https://www.hiascend.com/document)
