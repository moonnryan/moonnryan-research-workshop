---
title: CV 模型训练与微调
created: 2026-01-29
updated: 2026-10-09
status: draft
blog: false
tags: [CV, YOLO, DEIMv2, EdgeCrafter, NVIDIA, AMD, Ascend, 微调]
---

# CV 模型训练与微调

CV 模型的训练可以共享数据检查、任务记录与评估流程，模型入口和设备环境则需要分别配置。训练设备与部署设备是两组选择：在 NVIDIA GPU、AMD GPU 或昇腾上得到的权重，之后可以交给不同平台的编译工具。

本文以 DEIMv2、EdgeCrafter 的容器训练流程为主，保留 YOLO 的独立训练入口。三个模型的参数有相近含义，但不能直接互换命令。后续操作见 [ONNX 导出与验证](CV_ONNX导出与验证.md) 和 [多平台编译与部署](CV_多平台编译与部署.md)。

# 一、准备数据与微调起点

## 1.1 先固定划分与类别

检查图片可解码、标注在图像边界内、类别 ID 与名称对应。视频或连续采样数据按来源分组划分，避免相邻帧分别进入训练集和验证集。划分完成后保存数据版本，不因换模型重新随机划分。

YOLO 使用数据 YAML 与相应任务的标签；DEIMv2、EdgeCrafter 的训练插件直接引用 COCO 目录：

```text
dataset/
├── images/
│   ├── train/
│   └── val/
└── annotations/
    ├── instances_train.json
    └── instances_val.json
```

COCO 的 `file_name` 按训练入口约定相对数据集根解析。检测需要 bbox，实例分割需要 segmentation，姿态需要 keypoints 和对应关键点定义。更换文件后缀不能完成标签格式转换，数据转换应在训练前独立完成。

类别 ID 不必天然连续。转换时保存原始类别与模型类别索引的映射，训练、评估、导出和部署沿用同一份顺序。COCO 预训练类别与自定义类别也不能只靠“类别数相同”判断兼容。

## 1.2 区分微调、从头训练与恢复

| 模式 | 加载内容 | 启动时检查 |
| --- | --- | --- |
| 微调 | 预训练模型参数，按任务调整预测头 | 实际加载的权重、缺失键和不匹配键 |
| 从头训练 | 按配置初始化，明确是否仍加载骨干预训练 | 模型规格、初始化来源 |
| 恢复训练 | 模型、优化器、调度器与进度 | 恢复 epoch、学习率和优化器状态 |

不根据权重文件名判断训练模式。DEIMv2 的微调入口使用 tuning 权重，resume 使用完整训练 checkpoint；两者用途不同。恢复训练时还要固定数据版本与训练配置，防止“能继续运行”掩盖实验条件已经改变。

# 二、准备设备环境与任务目录

训练工程中，CV 插件与设备镜像分别组织：插件决定模型参数和训练入口，镜像提供 CUDA、ROCm 或 CANN/PyTorch 环境。当前 CV 插件为 `cv.deimv2` 与 `cv.edgecrafter`；YOLO 使用自己的 Ultralytics 环境。

| 训练设备 | 独立启动入口 | 需要核对 |
| --- | --- | --- |
| NVIDIA GPU | `run-train-gpu.sh` | 驱动、CUDA wheel、容器 GPU 映射 |
| AMD GPU | `run-train-amd.sh` | ROCm、GPU 架构、设备权限与扩展 |
| Ascend 910B | `run-train-910b.sh` | CANN、torch_npu、适配补丁与可见设备 |

版本固定到经过验证的镜像或依赖组合。某个 detect 模型训练通过，不等于同家族 segment、pose 或所有模型尺寸都已验证。归档的三平台短训练记录主要覆盖 DEIMv2-S 与 EdgeCrafter-S detect。

运行目录按任务隔离：

```text
workspace/<task-id>/
├── config/
├── staging/
├── output/
├── logs/
└── task.json
```

数据和基础权重可以按路径共享，配置、日志和产物属于具体任务。容器必须能访问数据与权重的实际路径；训练代码通过挂载更新时，还应记录代码版本，镜像 digest 本身不能完整描述一次实验。

# 三、运行一条 DEIMv2 微调任务

以下命令在训练工程的 `plugins/train-cv` 目录执行。先准备设备镜像，再确认资源挂载。当前 GPU 独立入口只把部署工程挂到 `/workspace`，不会自动挂载任意宿主绝对路径。文中使用 `/srv/models` 和 `/srv/datasets` 时，需要在实际启动模板的 `docker run` 中补上同路径只读挂载：

```bash
-v /srv/models:/srv/models:ro \
-v /srv/datasets:/srv/datasets:ro
```

这两行是 Docker 参数片段，不能单独作为命令运行。AMD、昇腾入口提供的附加挂载也有自己的范围，统一核对后再执行下面的命令；填写 `--models-dir` 或 `--datasets-dir` 本身不会创建 Docker 挂载。

```bash
./run-train-gpu.sh \
  --plugin cv.deimv2 \
  --weights /srv/models/deimv2_dinov3_s_coco.pth \
  --dataset /srv/datasets/coco-demo \
  --epochs 20 --batch-size 4 --workers 4 --amp 1 \
  --task-id deimv2-finetune-01 --gpus device=0 \
  --dry-run
```

先检查生成的计划是否指向正确的数据、权重、输出目录与设备，再移除 `--dry-run` 执行训练。该开关只跳过训练执行，外层启动脚本仍可能加载或准备镜像，不应把它当作整个命令绝对无副作用的保证。

将相同训练参数传给 AMD 或昇腾入口时，资源选择改用对应脚本的 `--devices`，例如：

```bash
./run-train-910b.sh \
  --plugin cv.deimv2 \
  --weights /srv/models/deimv2_dinov3_s_coco.pth \
  --dataset /srv/datasets/coco-demo \
  --epochs 20 --batch-size 4 --workers 4 --amp 1 \
  --task-id deimv2-npu-01 --devices 0
```

20 epoch、batch 4、workers 4 是操作示例。实际配置按数据规模、显存和模型选择。尤其要核对 warmup：如果训练总迭代数小于默认 warmup，短任务可能几乎一直处于学习率爬升阶段。一次冒烟主要验证环境和流程，不能据此判断已经完成有效微调。

训练日志至少应能确认预训练加载、实际 batch、学习率变化、验证指标和保存位置。权重路径出现在命令中，不能替代加载结果的检查。

# 四、替换模型时修改哪些步骤

## 4.1 EdgeCrafter 的任务与阶段配置

检测微调入口如下：

```bash
./run-train-gpu.sh \
  --plugin cv.edgecrafter --task detect \
  --weights /srv/models/ecdet_s.pth \
  --dataset /srv/datasets/coco-demo \
  --epochs 20 --batch-size 4 \
  --param noAugEpochs=2 \
  --task-id edgecrafter-detect-01 --gpus device=0
```

检测、分割和姿态使用各自的权重与标注，不能只改 `--task` 后继续传入检测数据。detect/segment 的配置要求 `1 ≤ noAugEpochs < epochs`，短任务需要同步调整无增强阶段与 warmup，不能照搬面向完整数据集的训练周期。

微调时明确传入 `--weights`。离线环境还需检查骨干等附属预训练资源是否已经准备好，避免训练开始后才触发下载。

## 4.2 YOLO 的训练入口

在匹配的 Ultralytics 环境准备：

```yaml
path: /srv/datasets/yolo-demo
train: images/train
val: images/val
names:
  0: object
```

```bash
yolo detect train model=yolo11s.pt data=data.yaml \
  imgsz=640 epochs=80 batch=16 device=0 \
  project=runs/train name=yolo11s-demo
```

训练环境与导出环境分别记录 Ultralytics、PyTorch 版本。pose 和 segment 使用对应模型及标签定义。昇腾训练还需要匹配的 PyTorch/torch_npu 和 YOLO 适配，不能将 GPU 命令直接视为 NPU 使用说明。

# 五、验证训练与保存产物

首先比较同一数据划分上的 AP、分割或关键点指标，再查看阈值附近、遮挡、小目标等样本。不同模型的验证前处理可以不同，但各自必须与其部署前处理保持一致，不能为了统一接口把所有模型都套用 YOLO Letterbox。

测量时把训练迭代、数据加载、验证和保存分开。AMD 验证阶段的精度路径、昇腾 DataLoader 的进程启动方式，都可能影响整任务时间；总时长变化不能直接归为训练算力变化。

保存以下材料：

| 材料 | 用途 |
| --- | --- |
| 配置快照、代码版本、镜像版本 | 重现实际训练条件 |
| 数据版本、类别映射、基础权重哈希 | 确认训练输入 |
| best/last checkpoint 与保存语义 | 分别用于模型选择和恢复 |
| 验证日志、指标与固定样例 | 建立导出前数值基线 |
| artifact 清单 | 记录产物位置与哈希 |

文件名叫 best，并不说明其中只有模型参数。导出前确认评估使用普通权重还是 EMA；仅保留推理权重后，不能再将它等同于可恢复训练的完整 checkpoint。

训练插件目前主要收集 checkpoint 与日志，不能仅因产物声明中列有 ONNX，就假定训练完成会自动导出。下一步显式执行 [ONNX 导出与验证](CV_ONNX导出与验证.md)。微调中的判断依据另见 [DEIM 微调中的配置、前处理与权重产物](../../topics/models-training/DEIM_微调与产物契约.md)。

# 六、参考资料

- [Ultralytics 训练](https://docs.ultralytics.com/modes/train/)
- [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2)
- [COCO 数据与评测](https://cocodataset.org/#detection-eval)
- [PyTorch 数据加载](https://docs.pytorch.org/docs/stable/data.html)
