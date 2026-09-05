# 阅读页

GitHub 仓库保存源稿，GitHub Actions 构建，GitHub Pages 发布。HTML、样式、脚本、中文字体和公式均由同一 Pages 站点提供，阅读时不依赖外部 CDN。

- 阅读地址：<https://taking-lying-flat.github.io/scratch/rope/>
- 正文：`content/rope.md`
- 排版：`reader.css`、`template.html`
- 构建：`build.mjs`，使用 Markdown-it、MathJax 和 Highlight.js，在构建阶段生成公式与代码高亮。

修改源稿并推送到 `main` 后，`.github/workflows/pages.yml` 自动更新网页。行内公式使用 GitHub 的 `$` 加反引号语法，独立公式使用 `math` 代码块。长公式在源稿中使用 `aligned` 或 `gathered` 分行。

首次配置仓库时，在 **Settings → Pages → Build and deployment → Source** 选择 **GitHub Actions**。

本地预览（Node.js 24、Python 3）：

```sh
cd site
npm ci
npm run build
npm run preview
```

打开 <http://127.0.0.1:4173/rope/>。页面支持章节跳转、复制代码，以及通过浏览器打印保存 PDF。

`dist/` 和 `node_modules/` 不提交；构建后的页面作为 Pages artifact 发布。字体与相关依赖的许可随产物保存在 `assets/fonts/` 和 `assets/licenses/`。
