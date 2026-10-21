---
title: Rockchip RKNN 环境搭建与板端推理
created: 2026-06-10
updated: 2026-06-12
status: draft
blog: false
tags: [Rockchip, RK3576, RKNN, ONNX, 环境配置, 部署]
aliases: [/assets/tutorial/Rockchip_RKNN环境与推理.html]
---

# Rockchip RKNN 环境搭建与板端推理

RKNN 部署通常分为两端：开发主机使用 RKNN-Toolkit2 将 ONNX 编译为指定芯片的 `.rknn`，板端使用 RKNN Runtime 或 Toolkit-Lite2 加载模型并执行。主机不需要连接开发板才能完成离线转换；板端也不需要安装完整转换工具链。

本文基于 RK3576 的环境搭建记录，使用 RKNN-Toolkit2/Runtime 2.3.2、RKNPU Driver 0.9.8。开发主机为 x86_64 Linux、Python 3.10，板端为 aarch64 Linux。该组合已完成最小模型的 C++ 与 Lite2 推理验证；具体模型的精度、性能和算子支持仍需分别测试。

# 一、区分 SDK 中的几个组件

| 组件 | 安装位置 | 作用 |
| --- | --- | --- |
| RKNN-Toolkit2 | 开发主机 | 导入模型、配置前处理、构建与导出 `.rknn`，也提供调试能力 |
| RKNN-Toolkit-Lite2 | Rockchip 板端 Python 环境 | 通过 Python 加载、初始化与执行模型 |
| RKNN Runtime | 板端 | `librknnrt.so`，提供 C/C++ 推理接口 |
| RKNPU Driver | 板端内核 | 与 NPU 硬件交互 |
| rknn_server | 按需安装在板端 | 为主机连接板端的调试路径提供服务 |

板端没有 `rknn_server`，不影响“主机离线转换，再交给板端应用推理”的路线。C++ 编译需要 `rknn_api.h` 和链接库，运行时需要匹配的 Runtime 及驱动，不需要将头文件也作为运行依赖安装。

开发主机与板端的 Python 版本可以不同。现场板端 Python 3.12 venv 和 Python 3.10 conda 环境都完成了 Lite2 加载与推理；重要的是各自 wheel 的 Python ABI、CPU 架构和平台依赖匹配，不能把主机的 x86_64 Toolkit wheel 拷到 aarch64 板端使用。

# 二、建立固定版本的转换环境

## 2.1 安装与实例化检查

在 Linux 开发主机上创建独立环境：

```bash
conda create -n rknn-toolkit python=3.10 -y
conda activate rknn-toolkit

python -m pip install \
  rknn-toolkit2==2.3.2 \
  onnx==1.16.1 protobuf==4.25.4 \
  numpy==1.26.4 setuptools==80.10.2
python -m pip check
```

这组依赖来自已验证环境。旧 SDK 的依赖声明即使允许更高版本，也不代表后续发布的所有包都兼容其内部接口。安装后确认版本，并实际创建编译器对象：

```bash
python -m pip show rknn-toolkit2 onnx protobuf numpy setuptools
python - <<'PY'
from rknn.api import RKNN

rknn = RKNN()
try:
    print("RKNN compiler object created")
finally:
    rknn.release()
PY
```

仅 `import` 成功不够，部分依赖在实例化或构建时才使用。若还需要从 PT 导出 ONNX，再安装相互匹配的 PyTorch、torchvision 和模型导出依赖；已有 ONNX 的最小编译示例不需要调用 PyTorch。

## 2.2 两个依赖兼容问题

| 现场组合 | 问题 | 已验证的处理 |
| --- | --- | --- |
| setuptools 82.0.1＋RKNN 2.3.2 | SDK 使用的 `pkg_resources` 不可用 | 固定 setuptools 80.10.2 |
| ONNX 1.21.0＋RKNN 2.3.2 | SDK 调用的 `onnx.mapping` 接口不再可用 | 固定 ONNX 1.16.1 |

这类错误发生在转换器的 Python 依赖层，还没有到板端 Runtime。更换板端驱动或修改模型结构，无法修复主机包接口缺失。反过来，修正主机依赖并生成模型，也不能证明目标设备已经能够执行。

确认环境后保存版本清单，后续安装模型导出依赖时继续沿用这些约束，避免自动升级 ONNX、NumPy 等包：

```bash
python -m pip freeze > requirements-rknn-lock.txt
```

# 三、从最小 ONNX 到 RKNN

先用固定形状、非量化的小模型贯通转换与运行，再增加真实模型、INT8 和动态输入。历史冒烟使用 Identity 模型，日志确认了加载、查询和执行接口可用；全零输入下的全零输出还不足以检查数值、布局或输入是否真正生效。

下面用单通道 1×1 卷积保留一个实际计算，构造 $y=2x+1$，用非零输入核对结果。该示例用于复现检查，不作为新增的板端实测记录。

## 3.1 生成模型与参考数据

保存为 `make_smoke.py`，在主机端执行：

```python
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

x = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4) / 16
node = helper.make_node(
    "Conv", ["input", "weight", "bias"], ["output"],
    kernel_shape=[1, 1],
)
graph = helper.make_graph(
    [node], "rknn_smoke",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 4, 4])],
    [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 4, 4])],
    initializer=[
        numpy_helper.from_array(np.full((1, 1, 1, 1), 2, np.float32), "weight"),
        numpy_helper.from_array(np.ones((1,), np.float32), "bias"),
    ],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, "smoke.onnx")
np.save("input.npy", x)
np.save("expected.npy", 2 * x + 1)
```

```bash
python make_smoke.py
```

这里只使用单通道，便于检查数值链路；真实 RGB 模型仍需另测通道顺序和 NCHW/NHWC 布局。

## 3.2 调用 Toolkit2 编译

保存为 `convert_smoke.py`：

```python
from rknn.api import RKNN

def check(stage, result):
    if result != 0:
        raise RuntimeError(f"{stage} failed: {result}")

rknn = RKNN(verbose=True)
try:
    check("config", rknn.config(target_platform="rk3576"))
    check("load_onnx", rknn.load_onnx(model="smoke.onnx"))
    check("build", rknn.build(do_quantization=False))
    check("export_rknn", rknn.export_rknn("smoke-rk3576.rknn"))
finally:
    rknn.release()
```

```bash
python convert_smoke.py
```

实际转换由 Toolkit2 的模型导入、图优化和目标编译完成，`librknnrt.so` 不参与主机端的 ONNX 编译。`target_platform` 必须对应板端芯片，不能将为另一芯片构建的产物直接当作通用文件分发。

此处没有设置 mean/std，输入是已经准备好的浮点张量。不要把 YOLO 的 `/255` 配置加到这个数值探针中，否则参考公式也会改变。

# 四、准备板端 Runtime 与 Lite2

先确认系统架构、NPU 设备节点和已部署的 Runtime。Runtime 文件可能位于系统库目录或应用目录，路径以实际部署为准：

```bash
uname -m
cat /etc/os-release
ls -l /dev/dri/renderD*
ldconfig -p | grep rknn
```

设备节点存在，只说明系统暴露了相应接口；Runtime API 与 Driver 的实际版本，应通过应用初始化后的 `RKNN_QUERY_SDK_VERSION` 查询并记录。不要为了与主机包版本一致，直接覆盖 BSP 提供的 Runtime 或内核驱动，应先核对对应 SDK 的兼容要求。

从官方 SDK 的 Lite2 包目录选择匹配 aarch64、Python 版本和系统平台的 wheel，在板端虚拟环境安装。例如使用 Python 3.12 时：

```bash
python3.12 -m venv ~/venvs/rknn-lite
source ~/venvs/rknn-lite/bin/activate

# 替换为已下载、匹配 Python 3.12/aarch64 的官方 wheel。
RKNN_LITE_WHEEL=/path/to/rknn_toolkit_lite2.whl
python -m pip install "$RKNN_LITE_WHEEL" numpy==1.26.4
python -c "from rknnlite.api import RKNNLite; print('Lite2 import OK')"
```

现场的 `externally-managed-environment` 来自系统 Python 的包管理限制，不等于 cp312 wheel 不支持板端。使用 venv 或 conda 隔离即可，无需为此修改系统 Python 的管理策略。

# 五、加载模型并检查数值

将 `smoke-rk3576.rknn`、`input.npy`、`expected.npy` 拷到板端同一目录，再保存并运行 `infer_smoke.py`：

```python
import numpy as np
from rknnlite.api import RKNNLite

def check(stage, result):
    if result != 0:
        raise RuntimeError(f"{stage} failed: {result}")

rknn = RKNNLite()
try:
    check("load_rknn", rknn.load_rknn("smoke-rk3576.rknn"))
    check("init_runtime", rknn.init_runtime())
    x = np.load("input.npy")                     # NCHW
    x_nhwc = np.ascontiguousarray(x.transpose(0, 2, 3, 1))
    outputs = rknn.inference(inputs=[x_nhwc], data_format=["nhwc"])
    if outputs is None or len(outputs) != 1:
        raise RuntimeError("expected one output")
    expected = np.load("expected.npy")
    actual = np.asarray(outputs[0], dtype=np.float32)
    np.testing.assert_allclose(
        actual.reshape(-1), expected.reshape(-1), rtol=1e-3, atol=1e-3,
    )
    print("output:", actual.shape, actual.dtype)
    print("max abs error:", np.max(np.abs(actual.reshape(-1) - expected.reshape(-1))))
finally:
    rknn.release()
```

```bash
python infer_smoke.py
```

单通道模型允许这里按展平值比较，完整检测模型不能靠展平来忽略布局差异。正式测试还需同时核对 shape、dtype、输出顺序和语义。

现场 Identity 模型的日志也显示：内部输入和输出可为 FP16，输入布局为 NHWC、输出布局为 NCHW，浮点输出接口返回的缓冲区却按 FP32 展开。因此，模型内部的 `type/size` 与应用请求转换后的数据大小需要区分，不能按一份固定属性解释所有缓冲区。

# 六、C++ 部署需要补齐什么

头文件来自对应版本 SDK 的：

```text
rknpu2/runtime/Linux/librknn_api/include/rknn_api.h
```

正式运行程序可以沿用官方 C API 示例，关键调用顺序如下：

```text
rknn_init(model)
  → rknn_query(SDK_VERSION / IN_OUT_NUM / INPUT_ATTR / OUTPUT_ATTR)
  → 按实际输入属性准备数据，设置输入 type、fmt、size
  → rknn_inputs_set
  → rknn_run
  → rknn_outputs_get（按需 want_float=1）
  → 核对输出数值
  → rknn_outputs_release
  → rknn_destroy
```

已有推理源文件时，在板端编译与检查动态库解析，例如：

```bash
g++ -std=c++17 -O2 -I./include infer.cpp \
  -L./lib -lrknnrt -o infer
ldd ./infer | grep rknn
```

`infer.cpp` 由应用实现，`include/` 与 `lib/` 放匹配 SDK 的头文件和板端架构库。`-L` 只控制链接阶段的搜索路径，运行时还需让加载器找到同一份 Runtime。若板端只有 Runtime、没有编译工具，不影响运行预先构建的程序；本机编译或交叉编译还需要各自的编译器、开发头文件与匹配 sysroot。

# 七、接入真实模型

最小探针通过后，再转入 [CV 模型多平台编译与部署](CV_多平台编译与部署.md)中的 RKNN 路径。该文说明统一转换入口、YOLO 前处理融合和模型差异，本篇集中处理 SDK 环境与运行链路。

| 阶段 | 需要确认 |
| --- | --- |
| 浮点模型 | 输入 shape、RGB/BGR、布局、归一化是否与参考一致 |
| INT8 | 按 API 要求生成校准清单，固定 mean/std，检查任务精度 |
| 动态输入 | 编译时声明的合法形状与 Runtime 设置方式一致 |
| 输出与后处理 | 读取真实属性，核对量化参数、坐标还原、候选筛选 |
| 性能 | 分开统计预处理、前向、搬运和后处理 |

校准图片目录可以作为应用脚本的输入，但 Toolkit2 的 `build(dataset=...)` 接收校准清单文件，不能假设把目录路径直接传进去就会自动遍历。多输入模型还需按对应格式为每个样本列出全部输入。

`do_quantization=False` 表示不执行这一步量化，并不保证所有 I/O 都保持 FP32；输入输出属性以构建产物和 Runtime 查询为准。Identity 等小模型通过，也不代表 DEIM 的 GridSample 或其他复杂结构已经支持，相关分析见 [RKNN 上 DEIM 的采样语义与部署边界](../../topics/graphs-operators/RKNN_DEIM部署边界.md)。

# 八、参考资料

- [RKNN-Toolkit2 v2.3.2 SDK](https://github.com/airockchip/rknn-toolkit2/tree/v2.3.2)
- [RKNN-Toolkit2 官方文档目录](https://github.com/airockchip/rknn-toolkit2/tree/v2.3.2/doc)
- [Toolkit-Lite2 官方推理示例](https://github.com/airockchip/rknn-toolkit2/blob/v2.3.2/rknn-toolkit-lite2/examples/resnet18/test.py)
- [RKNN C API 头文件](https://github.com/airockchip/rknn-toolkit2/blob/v2.3.2/rknpu2/runtime/Linux/librknn_api/include/rknn_api.h)
- [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo)
