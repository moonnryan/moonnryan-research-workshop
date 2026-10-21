---
title: Qualcomm MNN 本地编译与模型部署
created: 2026-09-03
updated: 2026-09-09
status: draft
blog: true
tags: [Qualcomm, QCS9075, MNN, Hexagon, OpenCL, 量化, 部署]
aliases: [/assets/tutorial/Qualcomm_MNN编译与部署.html]
---

# Qualcomm MNN 本地编译与模型部署

MNN 适合把自己的模型与 runtime 一起编译、导出和部署。相较于依赖已有的设备专属 bundle，它提供了更灵活的模型接入与后端选择；代价是导出精度、算子覆盖和性能需要自己验证。本地能编译出模型，并不意味着运行效率已经达到高通专门优化的 QAIRT 路径。

本文基于 MNN 3.6.1、QCS9075 Linux 的构建与 Qwen3.5-2B 测试记录，区分 CPU、Adreno OpenCL、直接 Hexagon 与 QNN 四条路径。QAIRT 资产的服务部署另见 [Qualcomm GenieX 的 GGUF 与 QAIRT 部署](Qualcomm_GenieX部署.md)。

# 一、先选择模型资产与后端

MNN 的模型导出从 Hugging Face checkpoint 出发，生成 MNN 图、外置权重、tokenizer 和运行配置。VLM 还包含视觉图，不能只复制 `llm.mnn` 就认为模型完整。

```text
model/
├── config.json
├── llm_config.json
├── llm.mnn
├── llm.mnn.weight
├── tokenizer.mtok
├── visual.mnn             # 多模态模型
└── visual.mnn.weight
```

embedding 是否单独存储、tokenizer 文件名以及其他附属文件，以实际导出结果为准，保留完整目录。

| 后端 | 编译与运行条件 | 需要单独核对 |
| --- | --- | --- |
| CPU | ARM64 MNN runtime | CPU 指令集、线程与量化 kernel |
| OpenCL | 编入 OpenCL，板端具备 Adreno runtime/ICD | 模型是否支持、解量化与设备搬运 |
| 直接 Hexagon | MNN Hexagon 宿主库和匹配 DSP 算子库 | 架构、量化布局、算子回退 |
| QNN | QAIRT/QNN 依赖和对应模型转换 | SoC、context、图大小与形状支持 |

直接 Hexagon 调用 MNN 自己的 DSP 算子，不等于 QNN。`--generate_for_npu` 属于归档导出器的 QNN 转换链，也不是给 CPU/OpenCL 模型打开通用 HTP 加速的开关。

# 二、本地编译转换器和运行时

## 2.1 主机转换器

固定 MNN 源码版本，在开发主机编译本机可执行的 MNNConvert。下面从 MNN 源码根目录执行：

```bash
cmake -S . -B build-host \
  -DCMAKE_BUILD_TYPE=Release \
  -DMNN_BUILD_CONVERTER=ON \
  -DMNN_BUILD_TOOLS=ON
cmake --build build-host --parallel 8
```

主机转换器与板端 runtime 是两个产物。主机上的 x86_64 可执行文件用于转换，板端程序需编译为 aarch64；交叉编译时必须给后者配置目标 sysroot 与工具链。

## 2.2 ARM64 Linux 运行时

在 ARM64 Linux 本机编译，或已经配置好 ARM64 交叉工具链的 CMake 环境中，CPU/OpenCL 基线可使用以下选项：

```bash
cmake -S . -B build-runtime \
  -DCMAKE_BUILD_TYPE=Release \
  -DMNN_BUILD_SHARED_LIBS=ON \
  -DMNN_BUILD_LLM=ON \
  -DMNN_BUILD_LLM_OMNI=ON \
  -DMNN_BUILD_OPENCV=ON \
  -DMNN_OPENCL=ON \
  -DMNN_LOW_MEMORY=ON \
  -DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
  -DMNN_USE_SYSTEM_LIB=ON \
  -DMNN_SEP_BUILD=OFF
cmake --build build-runtime --parallel 8
```

以上开关对应归档版本，构建后保存 `llm_demo`、`llm_bench` 和实际依赖库。使用新版本时核对视觉功能与算子融合开关，不能仅因 CMake 接受参数就认为目标模块已经构建。

直接 Hexagon 还需分别编译 ARM64 host stub 和 DSP skeleton，再启用 `MNN_HEXAGON`。归档发布包使用 v73，包含宿主 `libMNN_htpops.so` 与架构匹配的 DSP 库。Linux 的 Hexagon SDK 工具链及辅助接口需要专门适配，单加 `MNN_HEXAGON=ON` 并不足够；使用官方后端文档和对应版本构建脚本，分别检查 ARM64/DSP ELF 与加载日志。

部署包还要满足目标 GLIBC/GLIBCXX，板端提供 OpenCL、FastRPC 与固件依赖。CPU 基线运行时，日志探测到 Hexagon 库不代表实际执行已经切换到 HTP，应核对最终后端和节点放置。

# 三、导出模型并固定量化配置

在独立 Python 导出环境中安装当前源码要求的依赖，先用 `llmexport.py --help` 确认选项。归档采用 Python 3.11，使用主机转换器将完整 HF checkpoint 导出为 MNN：

```bash
cd transformers/llm/export
python llmexport.py \
  --path /srv/models/Qwen3.5-2B \
  --dst_path /srv/exports/qwen35-2b-mnn-w4 \
  --export mnn \
  --quant_bit 4 --quant_block 64 \
  --lm_quant_bit 4 --lm_quant_block 64 \
  --visual_quant_bit 8 --visual_quant_block 128 \
  --sym \
  --mnnconvert /srv/tools/MNNConvert
```

这组参数对应已记录的标准 W4 路径，未启用 HQQ。视觉导出与模型识别由该版本对 Qwen3.5 的实现处理，其他 VLM 按各自导出入口确认，不能直接复制全部参数。

W4A16 表示权重量化与计算精度的组合，`quant_block=64` 是权重分组；C4 则是 Transformer 图融合布局，不是另一种四位量化。归档直接 Hexagon 路径要求相应的四位对称权重及融合形式，CPU、OpenCL 与 QNN 的支持条件应分别检查。

大词表模型还需关注 embedding 是否共享。拆出 BF16 embedding 会增加体积；对齐精度时可以建立独立浮点基线，但不要只为改变文件组织就让正式模型额外常驻一份大词表权重。

# 四、先验证 CPU，再切换后端

保留导出生成的 `config.json`，从 CPU 开始。后端配置的关键字段例如：

```json
{
  "backend_type": "cpu",
  "thread_num": 8,
  "precision": "low",
  "memory": "low",
  "max_new_tokens": 128
}
```

这些字段合并到原配置中，不替换模型文件、tokenizer 等完整配置。用固定短文本和图片分别验证：

```bash
./llm_demo /srv/models/qwen35-2b-mnn/config.json
```

归档 `llm_demo` 的图片提示使用 `<img>/srv/reference/image.jpg</img>` 形式，具体语法按版本确认。文本正确后再测图片，避免 tokenizer、视觉配置和量化问题混在一起。

之后分别保存 `opencl`、`hexagon` 配置副本，在其他参数一致时测试。QNN 需走其单独的在线构图或离线 context 生成路线，不通过修改一个后端名称就自动获得可部署的设备图。

# 五、HQQ 问题与单变量对照

归档 W4A16-HQQ 模型曾在 CPU 图片推理中连续输出重复字符。保持 W4、group 64、对称量化和 W8 视觉编码器，移除 `--hqq` 后重新导出，图片描述恢复正常。标准 W8 与 FP16 对照也正常。

这将故障缩小到当时的 HQQ 导出与执行组合，不能据此判定四位量化不可用，也不能将其扩展成 HQQ 算法的普遍缺陷。排查顺序是先固定模型与输入，在 CPU 比较浮点和量化，再切后端。

样例描述正常以后，仍需用固定解码参数与标注集检查任务结果，尤其是结构化输出的可解析率、字段语义及召回。随机采样下两次回答接近，不足以证明量化无精度损失。

# 六、性能与 QAIRT 的取舍

QCS9075、MNN 3.6.1 的 256×256 单图、84 个 prompt token 测试结果如下：

| 后端 | Vision / s | Prefill / token·s⁻¹ | Decode / token·s⁻¹ |
| --- | ---: | ---: | ---: |
| CPU W4A16 standard | 0.43 | 78.15 | 33.30 |
| 直接 Hexagon W4A16 standard | 0.39 | 59.03 | 15.27 |
| OpenCL W4A16 standard | 27.68 | 3.30 | 4.09 |

三条路径都得到正常图片描述。Hexagon 路径仍有 18 个 FusedLinearAttention 层回退，CPU 在这个样例中的优势不能解释成“高通 NPU 不适合该模型”，它反映的是当时 MNN 的覆盖与调度路径。

MNN 的便利主要在本地编译、自定义模型导出和多后端调试。对于已经提供设备专属资产的模型，高通 QAIRT 的专门优化更适合作为追求执行效率的优先路线；MNN 编译成功并不能代替这些优化。但上述 MNN 2B 与已测 QAIRT 4B 不是同一资产，不能据此给出跨框架加速倍率。选择时同时看任务质量、首响应、生成延迟、内存和维护成本。

# 七、参考资料

- [MNN 源码](https://github.com/alibaba/MNN)
- [MNN 模型导出与运行](https://github.com/alibaba/MNN/blob/master/transformers/README.md)
- [MNN Hexagon 后端](https://github.com/alibaba/MNN/blob/master/source/backend/hexagon/README.md)
- [MNN QNN 转换与部署](https://github.com/alibaba/MNN/wiki/npu)
