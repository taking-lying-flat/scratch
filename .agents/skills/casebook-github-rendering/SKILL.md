---
name: casebook-github-rendering
description: "设计 casebook 技术文档在 GitHub 仓库和 GitHub Pages 中的视觉呈现。用于 GitHub Markdown 排版、alerts、Mermaid、图表、响应式技术文章页面、静态站点生成和 Pages 构建；保持 casebook Markdown 为唯一内容源。不用于修改技术结论。"
---

# Casebook GitHub 与前端呈现

让同一份 `casebook/*.md` 在 GitHub 文件页直接可读，并可由前端构建为视觉完整的 GitHub Pages 技术站点。保留 Markdown 作为唯一内容源，页面构建只转换结构和样式，不复制或改写正文。

## GitHub 原生呈现

主动使用 GitHub alerts 建立视觉锚点，不设整篇数量上限。每个需要扫读者立即捕获的不变量、阶段结论、风险或修复建议都可以使用 alert；同一结论只高亮一次，相邻 alerts 之间保留论证正文。按语义使用 `[!NOTE]` 表示范围与背景、`[!TIP]` 表示已验证建议、`[!IMPORTANT]` 表示不变量与中心结论、`[!WARNING]` 表示错误结果或兼容性风险、`[!CAUTION]` 表示数据损坏、死锁、安全与正确性风险。

仅当调用链、时序、状态机或所有权难以用连续正文和表格准确表达，且图示能显著增加信息密度时使用 Mermaid；用户拒绝图示时全部改用正文、公式或表格。使用表格压缩版本、shape、配置和组件职责，使用仓库内图片呈现仅靠文本难以辨识的结构。图示和正文保持相同术语，并在图后用一个完整段落说明图示证明的结论。禁止使用 `<details>`、折叠正文或依赖 GitHub 文件页自定义 CSS/JavaScript 的结构。

## GitHub Pages 前端

仅在用户要求网页或前端呈现时创建站点。先检查仓库现有技术栈；已有站点则延续其框架，没有站点则选择依赖少、支持静态导出和 Markdown 扩展的方案。通过 GitHub Actions 构建并部署 Pages，不提交生成目录。构建器必须兼容 GitHub alerts、Mermaid、代码高亮、heading anchor、相对链接和项目子路径，任何 Markdown 内容更新都应自动进入站点。

页面采用高信息密度的学术技术风格：正文阅读宽度稳定，标题层级清楚，桌面端提供目录与源码入口，移动端保持单列阅读，代码和宽表允许横向滚动。Alert 组件同时使用图标、文字标签和颜色，确保信息不依赖颜色单独传达；深浅主题均保持足够对比度。首页按主题组织案例并展示标题、摘要、技术标签与更新时间，避免大幅 hero、营销文案、装饰性卡片和过度动画。

## 验证

分别检查 GitHub 文件页和本地静态构建：正文完整、alerts 类型正确、Mermaid 可渲染、代码块与表格不溢出、目录和 anchor 有效、相对链接在仓库路径与 Pages base path 下均可访问。发布、启用 Pages 或修改仓库设置需要用户明确授权。
