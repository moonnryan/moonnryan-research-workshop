---
title: Sophgo TPU hang 与进程生命周期
created: 2026-04-13
updated: 2026-04-23
status: draft
blog: true
tags: [Sophgo, TPU, GDB, 并发, 故障定位]
---

# Sophgo TPU hang 与进程生命周期

同一条媒体推理链路可能同时出现段错误、堆损坏、TPU hang 和设备内存不足。进程崩溃栈落在视频输入函数，不一定说明视频函数就是最初故障点；异步设备错误和 native 内存破坏，可能延迟到后续同步或析构时才暴露。

归档排查中，CV 模型压力测试与 Qwen3-VL 量化模型的进程中断表现不同。大模型加载、推理及异常终止的生命周期成为重要复现条件，同时还发现 Python 跨线程调用、重复释放和动态库污染等独立问题。

# 一、按故障层分开记录

| 层次 | 典型证据 | 下一步 |
| --- | --- | --- |
| Python 逻辑 | 非协程被 await、字段缺失 | 最小请求与异常回溯 |
| native 内存 | SIGSEGV、SIGABRT、malloc 错误 | core、线程栈、对象所有权 |
| 运行时 | 加载失败、地址越界、断言 | SDK 版本与最小模型 |
| 设备 | hang、恢复失败 | kernel 日志与设备状态 |
| 服务环境 | 不兼容动态库 | 清理环境后独立启动 |

这些故障可能叠加，应分别建立复现，不能用“TPU 环境异常”覆盖所有调用层问题。

# 二、缩小复现条件

从媒体系统退回 `bmrt_test` 或最小推理程序，保持同一 BModel 和 SDK。分别测试 CV 单模型、大模型、二者并发，以及大模型加载或推理期间的进程退出。

归档复现中，大模型配合中断与重启能够触发 hang，CV 路径未出现相同现象。这支持进一步检查资源释放与设备状态恢复，但不足以直接证明某个闭源内存管理函数越界。

```bash
gdb /path/to/executable /path/to/core
# 在 GDB 中：
thread apply all bt full
info sharedlibrary
```

core 分析需要同版本二进制和动态库。只保存 core 而随后升级 SDK，会使调用栈和符号难以匹配。复现包应包括精简入口、版本、模型哈希和首次 kernel 错误。

# 三、宿主侧资源管理

调用 Python C API 的 native 线程必须遵守 GIL 规则；持有 GIL 也不会保护 TPU handle。设备资源则应明确唯一所有者，析构只执行一次，并避免停止服务时还有线程正在访问已释放实例。

模型单例配合会话锁可以减少重复加载的内存压力。退出流程应先停止接收请求、取消或等待在途任务，再释放模型和 device handle。若 SDK 无法保证强制终止后的恢复，应把设备健康检测作为重启前置条件。

# 四、一些反思

故障定位需要同时保存“观察到什么”和“已经证明什么”。随机 kill 可以建立压力复现，但它本身不能给出闭源 runtime 的具体根因；core 显示析构处崩溃，也不能排除更早的写越界。

一个可独立运行的最小复现比完整服务日志更有价值。它能把媒体、Python、运行库和设备四层分开，也让修复后的回归条件明确。服务恢复成功与根因闭环应分别记录。

# 五、参考资料

- [Sophon demo](https://github.com/sophgo/sophon-demo)
- [Python C API 线程状态与 GIL](https://docs.python.org/3/c-api/init.html#thread-state-and-the-global-interpreter-lock)
- [GDB 手册](https://sourceware.org/gdb/current/onlinedocs/gdb.html/)

