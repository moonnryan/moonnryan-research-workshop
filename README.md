# Moonnryan's Research Workshop

模型与系统工程方向的个人知识库，记录训练与推理、模型部署、计算图与算子、量化精度和性能优化中的问题、方法与思考。

这个仓库既保存可持续修订的技术笔记，也提供一个基于 VitePress 的博客界面。Markdown 是正文的唯一来源；文章中的关键代码、公式、表格、流程图和算法图都随仓库保存，可以脱离本地项目独立阅读。

## 当前内容

按工作时间整理了 2025 年 12 月至 2026 年 10 月的技术积累，共 41 篇：32 篇专题、8 篇教程和 1 份约束矩阵。相近记录合并为同一问题的完整篇章，后续验证与结论修正也归入原文。

- [按工作时间阅读全部内容](indexes/chronology.md)
- 按设备阅读：[Sophgo](indexes/hardware/sophgo.md)、[Intel](indexes/hardware/intel.md)、[Ascend](indexes/hardware/ascend.md)、[NVIDIA](indexes/hardware/nvidia.md)、[Qualcomm](indexes/hardware/qualcomm.md)、[Rockchip](indexes/hardware/rockchip.md)
- [CV 多平台部署约束矩阵](assets/matrices/CV_多平台部署约束.md)

内容覆盖 VLM 数据与 SFT、模型量化、计算图改写、异构执行、设备维护、视频时序、端侧会话和多平台训练。当前整理稿保留 `draft`，逐篇审阅后转为 `ready`。未完成的验证与研究假设会在文中说明。

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

`blog` 表示文章是否适合公开展示，`status` 表示是否已通过发布审阅。公开构建只收录同时满足 `blog: true` 和 `status: ready` 的文档；本地预览显示全部草稿、教程和矩阵。

`created` 采用对应工作的首次日期，`updated` 表示材料中后续验证的日期。无法确认原始实验日期时使用整理日期并在文中注明。新增材料前先检查同主题篇章，能补入原文的内容优先合并。

网页列表与正文中的 `#tag` 可点击，跳转到同标签文章；标签筛选可与主题和关键词组合，地址中的 `?tag=QNN` 可用于分享。公开网站的标签只检索已纳入公开构建的文章。

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
| `npm run index` | 按元数据更新时间与硬件索引 |

构建结果位于 `site/dist/`。网站代码、生成规则和路径前缀设置见 [site/README.md](site/README.md)。

## 写作约定

- 文章从问题本身开始，说明关键机制、判断依据、适用条件和仍然存在的边界。
- Topics 的主标题不编号，章节统一使用 `# 一、…`、`# 二、…`；教程最多使用三级标题。
- 命令和关键实现使用代码块，长实现优先改写为伪代码；伪代码采用带输入输出、行号和控制结构的论文式算法图。
- 公式统一使用 `$…$` 与 `$$…$$`，流程关系优先使用 Mermaid，约束关系优先使用表格。
- 文末参考资料只列公网可访问的论文、项目、文档和 issue，正文不依赖本地源码或原始材料才能理解。
- 对外文本去掉客户、人员、内部仓库、地址、账号、凭据和真实工作路径；示例统一使用公开模型与占位路径。
- 性能数据写明环境和计时边界，区分观察、假设和已验证结论；早期归因被推翻时更新原文，不继续引用旧结论。

## 持续更新

在对应主题中补充 Markdown，填写标题、工作日期和统一标签，再运行 `npm run index` 更新导航。使用本地预览检查文字、代码、公式和图表；准备发布的文章设为 `blog: true`、`status: ready`，执行公开构建与检查。

原始日记、聊天记录和未脱敏素材保存在仓库之外。公开仓库维护的是整理后的正文、资源及网站实现。

## License

本项目采用 [MIT License](LICENSE)。
