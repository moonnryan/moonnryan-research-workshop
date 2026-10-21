---
title: Sophgo Qwen3-VL 异常退出后的 TPU hang
created: 2026-04-13
updated: 2026-04-23
status: ready
blog: true
tags: [Sophgo, Qwen3-VL, TPU, GDB, 异常退出, 故障定位]
---

# Sophgo Qwen3-VL 异常退出后的 TPU hang

在算能盒子开发媒体推理服务的过程中，出现过段错误、堆损坏、bmlib 地址越界和 TPU hang。进一步缩小测试后，Qwen3-VL 的 W4BF16、W8BF16 模型配合进程中断与重启能够复现 hang，单独运行 CV 模型的压力测试中没有记录到相同现象。

这使排查重点从单次推理转向加载、中断、退出和重新初始化。现有记录保留了复现现象和向厂商提交日志的过程，但还没有定位到具体驱动、固件或运行库缺陷。Python 跨线程调用、重复释放和动态库冲突也需要单独处理，避免把多个故障归为同一个原因。

# 一、区分进程崩溃与设备挂起

`SIGSEGV` 表示进程发生了非法内存访问；`malloc(): smallbin double linked list corrupted` 表示宿主堆管理检测到结构损坏。触发检查的函数可能只是最后访问到损坏内存的位置，需要沿对象分配、访问和释放继续查找。

TPU hang 则需要设备侧证据。一次 HTTP 超时、进程仍存在或内存占用没有下降，都不足以单独确认设备挂起。排查时同时检查进程线程栈、运行时返回值和内核日志，并观察新建的最小推理任务是否还能完成。宿主堆报错也不能仅凭现象归因为 `/dev/shm` 不足；应另查共享内存分配失败、OOM 和设备内存错误。

| 层次 | 典型证据 | 下一步 |
| --- | --- | --- |
| Python 逻辑 | 非协程被 await、字段缺失 | 最小请求与异常回溯 |
| native 内存 | SIGSEGV、SIGABRT、malloc 错误 | core、线程栈、对象所有权 |
| 运行时 | 加载失败、地址越界、断言 | SDK 版本与最小模型 |
| 设备 | hang、恢复失败 | kernel 日志与设备状态 |
| 服务环境 | 不兼容动态库 | 清理环境后独立启动 |

这些故障可能叠加。按时间顺序对齐服务日志与内核日志，重点找第一次异常：是进程先发生非法访问，随后设备调用失败，还是先出现设备错误，随后宿主程序在处理失败时崩溃。

SDK 与原生例程的准备见 [Sophgo 模型编译部署教程](../tutorial/Sophgo_模型编译部署教程.md)；视觉编码、prefill 和 decode/cache 的执行关系见 [Sophgo Qwen3-VL 子图导出与部署边界](../graphs-operators/Sophgo_Qwen3-VL导出与部署.md)，便于按阶段构造最小复现。

# 二、缩小复现条件

固定 BModel、SDK、运行库和测试输入，再逐步去掉媒体接入、插件和 HTTP 层。CV 模型可以先用 `bmrt_test` 检查加载和执行；Qwen3-VL 则保留最小原生例程，覆盖视觉编码、prefill、decode 和缓存状态。直接用随机输入运行其中一个子图，只能检查该子图的可运行性。

| 对照 | 记录中的现象 | 可继续缩小的范围 |
| --- | --- | --- |
| CV 模型压力测试 | 未记录到同类 hang | 保留为对照，不能据此认定所有 CV 负载都安全 |
| Qwen3-VL 量化模型配合中断、重启 | 可复现 hang | 加载、执行中断与设备恢复路径 |
| Python 插件与媒体服务 | 出现 GIL、重复释放、动态库问题 | 宿主程序的独立故障 |

后续测试应把加载期间退出、推理期间退出、正常释放后重启分开，并记录信号类型。`SIGTERM` 是否触发正常收尾取决于应用的处理逻辑；`SIGKILL` 无法被捕获，程序没有机会执行析构与清理。将两者混成随机 kill，会丢失最重要的退出条件。

```bash
gdb /path/to/executable /path/to/core
# 在 GDB 中：
thread apply all bt full
info sharedlibrary
```

core 分析需要保留崩溃时的二进制、动态库与调试符号。除了全部线程的调用栈，还应记录实际加载的库路径，避免系统 SDK 与 conda 中的同名库混用。提交复现时附上模型哈希、退出信号、发生阶段、重启间隔和首次内核错误；对外版本去掉设备地址、账号及业务输入。

# 三、退出时需要等待什么

在常规启用 GIL 的 CPython 中，原生线程调用 Python C API 前需要正确关联线程状态并持有 GIL。GIL 保护 Python 对象访问，模型会话和 TPU handle 仍需按运行库的线程约束管理。重复释放则需要从所有权入手：谁创建对象，谁负责销毁，停止流程与析构是否重复释放了同一资源。

正常退出时，先停止接收新请求，处理尚未执行的排队任务，再等待正在运行的原生调用和设备任务结束，最后释放资源。取消 HTTP 请求或外层等待任务，不会自动停止已经开始的原生推理；线程池 Future 的 `cancel()` 对正在执行的任务会返回 `False`。此时提前析构模型，会产生释放后访问的风险。

BMRuntime 官方示例先调用 `bmrt_destroy`，再调用 `bm_dev_free`。应用自行申请的设备缓冲区，也需在最后一次使用完成后由其所有者释放。使用封装类时按封装提供的销毁接口处理，避免再从外层重复释放底层 handle。

这些步骤适用于正常收尾。进程被强制结束后，设备任务与资源如何回收依赖驱动和固件，应用层析构无法兜底。重启服务前应检查设备是否还能完成最小任务；反复拉起服务会增加新日志和资源申请，使首次故障更难保留。

# 四、一些反思

这次排查最有价值的收敛，是把“运行中偶发异常”变成了与进程中断、重启相关的复现条件。接下来需要比较正常退出与强制终止，才能判断正常清理流程、驱动回收和设备重新初始化分别起了什么作用。

宿主程序修复与设备故障定位可以同时推进。修复 GIL、重复释放和动态库冲突，能够减少干扰，但只有在相同中断条件下重新测试，才能判断 TPU hang 是否也被消除。服务恢复运行与具体缺陷得到解释，应分别记录。

# 五、参考资料

- [Sophon demo](https://github.com/sophgo/sophon-demo)
- [BMRuntime 资源管理与执行接口](https://doc.sophgo.com/sdk-docs/v23.05.01/docs_latest_release/docs/tpu-runtime/reference_en/html/bmruntime/runtime.html)
- [bmrt_test 工具与 BModel 验证](https://doc.sophgo.com/sdk-docs/v23.07.01/docs_latest_release/docs/tpu-runtime/reference/html/bmrt_test/bmrt_test.html)
- [Python C API 线程状态与 GIL](https://docs.python.org/3/c-api/init.html#thread-state-and-the-global-interpreter-lock)
- [Python Future 取消行为](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Future.cancel)
- [Linux 信号与 SIGKILL](https://man7.org/linux/man-pages/man7/signal.7.html)
- [GDB 手册](https://sourceware.org/gdb/current/onlinedocs/gdb.html/)

