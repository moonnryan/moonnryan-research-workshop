# Research Workshop · Website

基于 VitePress 的个人知识库与博客界面，包含首页、文章列表、阅读页和关于页。原始 Markdown 保持在上级 `topics/`、`assets/tutorial/` 与 `assets/matrices/`，网页内容按构建模式自动生成。

## 本地预览

在本目录运行：

```bash
npm ci
npm run dev
```

开发服务器默认使用 `http://127.0.0.1:5173`。本地预览包含草稿及教程，编辑原 Markdown 后会重新生成页面并热更新。修改网站布局、样式或页面内容也会自动同步。

## 构建模式

```bash
# 公开构建：只纳入 blog: true 且 status: ready 的文章
npm run build

# 私有审阅构建：包含当前全部文章与教程
npm run build:preview

# 检查当前生成内容的路由和图片引用
npm run check

# 在本地查看刚才构建的结果
npm run preview
```

构建输出位于 `dist/`。当前内容保留 draft，公开构建只显示通过发布审阅的条目，本地与审阅模式可以查看全部整理稿。公开托管使用公开构建。

GitHub 项目页面等带路径前缀的部署，可以通过 `MRW_BASE` 设置基础路径，例如：

```bash
MRW_BASE=/moonnryan-research-workshop/ npm run build
```

将公开构建的输出目录部署到静态托管即可，浏览器不需要运行 Node.js 服务。

## 文件职责

| 位置 | 用途 |
| --- | --- |
| `app/config.mts` | VitePress、基础路径、数学渲染与标题配置 |
| `app/theme/` | 页面布局、组件与样式 |
| `pages/` | 首页、文章列表、关于页的入口 |
| `scripts/prepare-content.mjs` | 发布筛选、文章索引、Mermaid 转换与资源同步 |
| `scripts/dev.mjs` | 监听源文章与网站文件，自动更新预览 |
| `scripts/update-indexes.mjs` | 按工作日期和硬件标签更新仓库导航 |
| `content/` | 自动生成的网页内容，已忽略；不要在此直接修改文章 |

网站使用现有文件路径生成文章路由，主题来自文章所在目录。YAML 中的标题、标签和更新日期用于列表展示；公开筛选以 `blog` 和 `status` 为准。

公式采用 MathJax，Mermaid 通过客户端组件渲染，论文式算法直接使用现有 PNG。一级章节标题会进入文章目录，无需修改当前知识库的标题风格。

列表与正文标签链接到 `articles.html?tag=标签`。标签精确匹配，可继续叠加主题和关键词筛选；清除筛选会同步清除 URL 参数。增加或修改文章后，运行 `npm run index` 更新仓库里的时间与硬件索引。
