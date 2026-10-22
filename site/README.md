# Research Workshop · Website

基于 VitePress 的个人知识库与博客界面，包含首页、文章列表、阅读页和关于页。原始 Markdown 保持在上级 `topics/`，其中操作教程集中在 `topics/tutorial/`，网页内容按构建模式自动生成。

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

# 检查内容、图片引用与泡泡碰撞约束
npm run check

# 在本地查看刚才构建的结果
npm run preview
```

构建输出位于 `dist/`。公开构建只显示通过发布审阅的条目，本地与审阅模式可以查看全部整理稿。公开托管使用公开构建，检查脚本会阻止没有文章的公开版本部署。

## GitHub Pages

网站地址为 https://moonnryan.github.io/，对应仓库为 `moonnryan/moonnryan.github.io`。根域名使用默认的 `base: '/'`，本地文件夹可以继续使用原来的名称。

仓库 Settings → Pages → Source 选择 **GitHub Actions**。根目录的 `.github/workflows/deploy.yml` 会在推送到 `main` 后自动安装依赖、构建公开内容、检查并发布 `site/dist/`，也可以在 Actions 中手动运行。无需再配置 Jekyll 或 Static HTML，也不需要额外添加部署密钥。

在 Actions 中确认 **Deploy VitePress site to Pages** 的 build 和 deploy 均通过，再访问网站。后续只需修改源文件、提交并推送，网站会自动更新。

GitHub 项目页面等带路径前缀的部署，可以通过 `MRW_BASE` 设置基础路径，例如：

```bash
MRW_BASE=/moonnryan-research-workshop/ npm run build
```

将公开构建的输出目录部署到静态托管即可，浏览器不需要运行 Node.js 服务。

## 文件职责

| 位置 | 用途 |
| --- | --- |
| `app/config.mts` | VitePress、基础路径、数学渲染与标题配置 |
| `app/topic-groups.mjs` | 主题名称与旧分类的合并映射 |
| `app/theme/` | 页面布局、组件与样式 |
| `pages/` | 首页、文章列表、关于页的入口 |
| `scripts/prepare-content.mjs` | 发布筛选、文章索引、Mermaid 转换与资源同步 |
| `scripts/dev.mjs` | 监听源文章与网站文件，自动更新预览 |
| `scripts/update-indexes.mjs` | 按工作日期和硬件标签更新仓库导航 |
| `content/` | 自动生成的网页内容，已忽略；不要在此直接修改文章 |

网站使用现有文件路径生成文章路由，主题由文章所在目录与 `app/topic-groups.mjs` 共同决定。“模型与训练”“推理与性能”“数据与评测”在网站上合并为“训练与部署”，源目录和文章地址保留。YAML 中的标题、标签和更新日期用于列表展示；公开筛选以 `blog` 和 `status` 为准。

公式采用 MathJax，Mermaid 通过客户端组件渲染，论文式算法直接使用现有 PNG。一级章节标题会进入文章目录，无需修改当前知识库的标题风格。

文章列表每页 10 篇，分页作用于筛选后的结果；切换主题、标签或关键词时回到第一页。标签支持多选，再次点击取消，多个标签按并集匹配，并与主题、关键词叠加。筛选地址使用重复的 `tag` 参数，例如 `articles.html?topic=quantization-accuracy&tag=NVFP4&tag=INT8`。清除标签保留当前主题；清除全部筛选恢复全站列表。增加或修改文章后，运行 `npm run index` 更新仓库里的时间与硬件索引。

导航顺序为首页、主题、文章、关于。“主题”进入 `topics.html`，泡泡面积按文章数变化。位置与大小采用弹簧缓动，悬停时逐渐放大并由圆形碰撞约束推开邻居。点击后从原泡泡的位置缓动展开，返回时反向收起。标签泡泡按该主题内的文章数决定大小，以不规则圆形布局浮动，默认浅色填充，选中后变为深色；切换选择保留现有位置与速度。实体色“前往”按钮进入筛选后的文章列表，不选标签时进入该主题全部文章。

标签多时可在大泡泡内滚动，滚动条隐藏，内容始终裁切在圆内。没有关闭叉号，点击实际圆形外部或按 Esc 返回；键盘焦点由原生对话框管理。深色模式与移动端使用同一套数据，系统设置减少动态效果时停用浮动与悬停缩放，页面不可见时暂停模拟。`tests/bubble-physics.test.mjs` 检查密集排布、边界、悬停放大与碰撞分离，随 `npm run check` 执行。

文章末尾最多推荐三篇相关内容，综合共同标签与主题排序，并排除当前文章。出现较少的具体标签比通用平台标签权重更高。“更多”直接进入对应主题的文章列表。目录根据章节位置突出当前阅读章节，文章切换或图表高度变化后重新计算；窄屏继续隐藏侧栏，推荐卡片改为单列。新增文章后，主题统计、标签与相关推荐随构建自动更新。
