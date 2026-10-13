# Moonnryan's Research Workshop

模型与系统工程方向的个人知识库，记录训练与推理、模型部署、计算图与算子、量化精度和性能优化中的问题、方法与思考。

这个仓库既保存可持续修订的技术笔记，也提供一个基于 VitePress 的博客界面。Markdown 是正文的唯一来源；文章中的关键代码、公式、表格、流程图和算法图都随仓库保存，可以脱离本地项目独立阅读。

## 当前内容

### Topics

- [Qwen3-VL 子图导出与多设备部署边界](topics/graphs-operators/Sophgo_Qwen3-VL导出与部署.md)
- [LoRA 编译预留与动态加载](topics/models-training/Sophgo_LoRA部署.md)
- [W4 权重量化中的通道对齐问题](topics/quantization-accuracy/Sophgo_W4通道对齐.md)
- [校准数据与混合精度选择](topics/quantization-accuracy/Sophgo_校准与混合精度.md)

### Tutorial

- [Sophgo 模型编译部署教程](assets/tutorial/Sophgo_模型编译部署教程.md)

当前内容仍处于 `draft` 阶段。文章会在补全校验、参考资料和表达后逐篇转为可发布状态。

## 仓库结构

```text
moonnryan-research-workshop/
├── topics/                 # 围绕一个技术问题展开的分析文章
│   ├── data-evaluation/
│   ├── models-training/
│   ├── graphs-operators/
│   ├── quantization-accuracy/
│   ├── inference-performance/
│   └── engineering-runtime/
├── assets/
│   ├── tutorial/           # 可复现的完整操作流程
│   ├── matrices/           # 平台、模型与工具链约束矩阵
│   └── media/              # 图片及其可编辑源文件
├── indexes/                # 按硬件或主题组织的导航
└── site/                   # VitePress 网站、构建脚本与主题
```

主题目录先直接放文章，只有在内容规模确实增长后再增加层级。教程和约束矩阵属于可复用资产，与围绕问题展开的主题文章分开维护。

## 文档元数据

每篇文档使用统一的 YAML frontmatter：

```yaml
---
title: 文章标题
created: 2026-10-12
updated: 2026-10-13
status: draft        # draft | ready
blog: true           # 是否作为博客候选
tags: [模型, 平台, 主题]
---
```

`blog` 表示文章是否适合公开展示，`status` 表示内容是否已完成。公开构建只收录同时满足 `blog: true` 和 `status: ready` 的文档；本地预览会显示全部草稿和教程。

## 本地浏览

网站使用 Node.js 运行。在仓库根目录执行：

```bash
cd site
npm ci
npm run dev
```

默认地址为 `http://127.0.0.1:5173`。同一局域网内预览可以使用：

```bash
npm run dev -- --host 0.0.0.0
```

常用命令：

| 命令 | 用途 |
| --- | --- |
| `npm run dev` | 预览全部内容，并监听 Markdown 与主题改动 |
| `npm run build:preview` | 构建包含草稿的审阅版本 |
| `npm run build` | 按发布条件生成公开版本 |
| `npm run check` | 检查导出路由、图片引用和 Mermaid 转换 |

构建结果位于 `site/dist/`。网站代码、生成规则和路径前缀设置见 [site/README.md](site/README.md)。

## 写作约定

- 文章从问题本身开始，说明关键机制、判断依据、适用条件和仍然存在的边界。
- Topics 的主标题不编号，章节统一使用 `# 一、…`、`# 二、…`；教程最多使用三级标题。
- 命令和关键实现使用代码块，长实现优先改写为伪代码；伪代码采用带输入输出、行号和控制结构的论文式算法图。
- 公式统一使用 `$…$` 与 `$$…$$`，流程关系优先使用 Mermaid，约束关系优先使用表格。
- 文末参考资料只列公网可访问的论文、项目、文档和 issue，正文不依赖本地源码或原始材料才能理解。

## License

本项目采用 [MIT License](LICENSE)。
