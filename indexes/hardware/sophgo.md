---
title: Sophgo 条目索引
created: 2026-10-12
updated: 2026-10-13
status: draft
blog: false
tags: [Sophon, 索引]
---

# Sophgo 条目索引

## 一、教程

| 设备 | 内容 | 条目 |
| --- | --- | --- |
| BM1684X SoC | 环境、编译量化、Python/C++ 运行与验收 | [Qwen3-VL 与 YOLOv5 完整教程](../../assets/tutorial/Sophgo_模型编译部署教程.md) |

## 二、技术分析

| 设备 | 核心问题 | 条目 |
| --- | --- | --- |
| BM1688 | LoRA 原理、编译预留、加载切换与状态验收 | [LoRA 编译预留与动态加载](../../topics/models-training/Sophgo_LoRA部署.md) |
| SE8 / BM1684X | 子图接口与实际设备拓扑如何匹配？ | [Qwen3-VL 子图导出与部署边界](../../topics/graphs-operators/Sophgo_Qwen3-VL导出与部署.md) |
| BM1684X | 校准范围、误差传播与任务精度如何分别判断？ | [YOLOv5 校准与混合精度选择](../../topics/quantization-accuracy/Sophgo_校准与混合精度.md) |
| BM1684X | group size 为什么不能修复输出通道对齐？ | [Qwen3.5-2B W4 通道对齐问题](../../topics/quantization-accuracy/Sophgo_W4通道对齐.md) |

## 三、阅读约定

各篇直接以背景正文开场，再解释机制、证据和实践取舍，主题文章最后写“一些反思”。完整操作教程保留流程与验收即可。topics 仅使用一级章节标题，教程按需保留小节。伪代码采用论文式算法图；关键实现保留短代码；公式使用 Markdown 数学分隔符。算法图的可编辑 LaTeX 文件与图片同目录保存。
