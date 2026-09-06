import { mkdir, readFile, writeFile, copyFile, cp, rm } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import MarkdownIt from 'markdown-it';
import hljs from 'highlight.js/lib/core';
import python from 'highlight.js/lib/languages/python';
import json from 'highlight.js/lib/languages/json';
import { mathjax } from '@mathjax/src/js/mathjax.js';
import { TeX } from '@mathjax/src/js/input/tex.js';
import { SVG } from '@mathjax/src/js/output/svg.js';
import { liteAdaptor } from '@mathjax/src/js/adaptors/liteAdaptor.js';
import { RegisterHTMLHandler } from '@mathjax/src/js/handlers/html.js';
import '@mathjax/src/js/util/asyncLoad/esm.js';
import '@mathjax/src/js/input/tex/base/BaseConfiguration.js';
import '@mathjax/src/js/input/tex/ams/AmsConfiguration.js';
import '@mathjax/src/js/input/tex/newcommand/NewcommandConfiguration.js';

const root = path.dirname(fileURLToPath(import.meta.url));
const output = path.join(root, 'dist');
const source = await readFile(path.join(root, 'content/rope.md'), 'utf8');
const markdown = new MarkdownIt({ html: false, typographer: false });
const escape = markdown.utils.escapeHtml;
hljs.registerLanguage('python', python);
hljs.registerLanguage('json', json);

markdown.inline.ruler.before('backticks', 'math_inline', (state, silent) => {
  if (!state.src.startsWith('$`', state.pos)) return false;
  const end = state.src.indexOf('`$', state.pos + 2);
  if (end === -1) throw new Error('Unclosed inline math delimiter');
  if (!silent) {
    const token = state.push('math_inline', 'span', 0);
    token.content = state.src.slice(state.pos + 2, end);
  }
  state.pos = end + 2;
  return true;
});

const adaptor = liteAdaptor({ fontSize: 18 });
RegisterHTMLHandler(adaptor);
const svg = new SVG({ fontCache: 'local', displayOverflow: 'overflow', linebreaks: { inline: false } });
const document = mathjax.document('', {
  InputJax: new TeX({
    packages: ['base', 'ams', 'newcommand'],
    formatError(_jax, error) { throw error; },
  }),
  OutputJax: svg,
});

const tokens = markdown.parse(source, {});
if (tokens[0]?.type !== 'heading_open' || tokens[0].tag !== 'h1') throw new Error('Expected document title');
const postTitle = tokens[1].content;
tokens.splice(0, 3);
let inlineCount = 0;
let displayCount = 0;

async function renderMath(token, display) {
  const node = await document.convertPromise(token.content, { display, em: 18, ex: 8.1, containerWidth: 780 });
  const html = adaptor.outerHTML(node);
  if (/data-mjx-error|data-mml-node="merror"/.test(html)) throw new Error(`Invalid formula: ${token.content}`);
  token.meta = { html };
  if (display) displayCount++;
  else inlineCount++;
}

for (const token of tokens) {
  if (token.type === 'fence' && token.info.trim() === 'math') await renderMath(token, true);
  for (const child of token.children ?? []) {
    if (child.type === 'math_inline') await renderMath(child, false);
  }
}

const sections = [
  { id: 'frequency', label: '角频率与底数', prefix: 'RoPE（Rotary' },
  { id: 'configuration', label: '模型配置与张量', fence: 'json' },
  { id: 'rotation', label: '二维旋转', prefix: 'RoFormer §3.2.1' },
  { id: 'relative-position', label: '完整矩阵与相对位置', prefix: 'RoFormer 式（15）' },
  { id: 'partial-rope', label: 'Partial RoPE', prefix: 'Qwen3.5 的 partial RoPE' },
  { id: 'implementation', label: '张量计算与源码', prefix: '对普通 RoPE' },
];

for (const section of sections) {
  const index = tokens.findIndex((token, i) => section.fence
    ? token.type === 'fence' && token.info === section.fence
    : token.type === 'paragraph_open' && tokens[i + 1]?.content.startsWith(section.prefix));
  if (index < 0) throw new Error(`Missing section: ${section.id}`);
  tokens[index].attrSet('id', section.id);
}

markdown.renderer.rules.math_inline = (items, i) =>
  `<span class="math-inline" role="math" aria-label="${escape(items[i].content)}">${items[i].meta.html}</span>`;
markdown.renderer.rules.fence = (items, i) => {
  const token = items[i];
  const language = token.info.trim();
  if (language === 'math') {
    return `<div class="equation" tabindex="0" role="math" aria-label="${escape(token.content)}">${token.meta.html}</div>\n`;
  }
  const id = token.attrGet('id');
  const name = language === 'json' ? 'config.json · text_config'
    : language === 'text' ? '张量维度'
    : token.content.includes('def rotate_half') ? 'rotate_half / apply_rotary_pos_emb'
    : 'compute_default_rope_parameters';
  const code = hljs.getLanguage(language)
    ? hljs.highlight(token.content, { language, ignoreIllegals: false }).value
    : escape(token.content);
  const label = { python: 'Python', json: 'JSON', text: 'Text' }[language] ?? language;
  const lineCount = token.content.replace(/\n$/, '').split('\n').length;
  const numbers = language === 'text' ? '' : `<span class="line-numbers" aria-hidden="true">${
    Array.from({ length: lineCount }, (_, line) => line + 1).join('\n')
  }</span>`;
  return `<figure class="code-block"${id ? ` id="${escape(id)}"` : ''}>
    <figcaption>
      <span class="window-controls" aria-hidden="true"><span></span><span></span><span></span></span>
      <span class="window-title" title="${escape(name)}">${escape(name)}</span>
      <span class="code-actions"><span class="code-language">${escape(label)}</span><button type="button" class="copy-button" hidden>复制</button></span>
    </figcaption>
    <pre tabindex="0">${numbers}<code class="language-${escape(language)}">${code}</code></pre>
  </figure>\n`;
};

const content = markdown.renderer.render(tokens, markdown.options, {});
const toc = sections.map(({ id, label }) => `<li><a href="#${id}">${label}</a></li>`).join('\n');
const template = await readFile(path.join(root, 'template.html'), 'utf8');
const siteUrl = 'https://taking-lying-flat.github.io/scratch/';
const postPath = 'posts/rope/';
const description = '从角频率、二维旋转矩阵到 Qwen3.5 的 partial RoPE：论文公式与 Transformers 源码的对应。';
const proseText = tokens.filter((token) => token.type === 'inline').map((token) => token.content).join(' ');
const readingMinutes = Math.max(1, Math.ceil(
  (proseText.match(/\p{Script=Han}/gu)?.length ?? 0) / 300 +
  (proseText.match(/[A-Za-z]+/g)?.length ?? 0) / 200 + displayCount * 0.3 + 2
));
const metadata = `<time datetime="2026-09-05">2026 年 9 月 5 日</time><span>约 ${readingMinutes} 分钟</span>`;
const files = ['reader.css', 'reader.js', 'theme.js', 'favicon.svg'];
const assetVersion = createHash('sha256');
for (const file of files) assetVersion.update(await readFile(path.join(root, file)));
const version = assetVersion.digest('hex').slice(0, 10);

function page({ title, description, route = '', body, type = 'website', pageClass = '' }) {
  const values = {
    TITLE: escape(title), DESCRIPTION: escape(description), TYPE: type,
    URL: `${siteUrl}${route}`, ROOT: '../'.repeat(route.split('/').filter(Boolean).length) || './',
    POSTS_CURRENT: route === '' ? ' aria-current="page"' : '',
    ARCHIVES_CURRENT: route === 'archives/' ? ' aria-current="page"' : '',
    PAGE_CLASS: pageClass, BODY: body,
  };
  let html = template.replace(/\{\{([A-Z_]+)\}\}/g, (_, key) => {
    if (!(key in values)) throw new Error(`Unknown template field: ${key}`);
    return values[key];
  });
  for (const file of files) html = html.replaceAll(`assets/${file}"`, `assets/${file}?v=${version}"`);
  if (/\{\{(?:TOC|CONTENT|BODY)\}\}|\$`|data-mjx-error|language-math/.test(html)) throw new Error('Unrendered content in output');
  return html;
}

const article = page({
  title: `${postTitle} · Casebook`, description, route: postPath, type: 'article', pageClass: 'post-page',
  body: `<article class="post-single">
    <header class="post-header">
      <h1>${escape(postTitle)}</h1>
      <div class="post-meta">${metadata}<span>taking-lying-flat</span></div>
    </header>
    <details class="toc">
      <summary>目录</summary>
      <nav aria-label="文章目录"><ol>${toc}</ol></nav>
    </details>
    <div class="prose">${content}</div>
    <footer class="post-footer">
      <div class="post-topics" aria-label="文章主题"><span>RoPE</span><span>Qwen3.5</span><span>Transformers</span></div>
      <div class="post-actions">
        <a href="#top">返回顶部 ↑</a>
      </div>
      <a class="back-link" href="../../">← 全部文章</a>
    </footer>
  </article>`,
});
const home = page({
  title: 'Casebook · 技术笔记', description: '关于模型、论文与源码的技术笔记。', pageClass: 'home-page',
  body: `<section aria-labelledby="post-list-title">
    <div class="post-list-heading">
      <h1 id="post-list-title">全部文章 <span>01</span></h1>
    </div>
    <article class="post-entry">
      <div class="entry-content">
        <h2><a href="${postPath}">${escape(postTitle)}</a></h2>
        <ul class="entry-topics" aria-label="文章主题"><li>Qwen3.5</li><li>Transformers</li></ul>
        <footer class="entry-footer">
          <div class="post-meta">${metadata}</div>
          <span class="entry-arrow" aria-hidden="true">↗</span>
        </footer>
      </div>
      <div class="entry-art" aria-hidden="true">
        <span class="art-caption">RoFormer / 01</span>
        <div class="rotation-figure">
          <span class="rotation-axis axis-x"></span><span class="rotation-axis axis-y"></span>
          <span class="rotation-ring ring-outer"></span><span class="rotation-ring ring-inner"></span>
          <span class="rotation-vector vector-start"></span><span class="rotation-vector vector-end"></span>
          <span class="rotation-origin"></span><span class="rotation-theta">θ</span>
        </div>
        <span class="art-footer">POSITION ENCODING</span>
      </div>
    </article>
  </section>`,
});
const archives = page({
  title: '归档 · Casebook', description: 'Casebook 技术笔记归档。', route: 'archives/', pageClass: 'archive-page',
  body: `<header class="page-header"><h1>归档</h1><p>1 篇文章</p></header>
    <section class="archive-year" aria-labelledby="year-2026">
      <h2 id="year-2026">2026 <span>1</span></h2>
      <div class="archive-month"><h3>九月</h3><article class="archive-entry">
        <h4><a href="../${postPath}">${escape(postTitle)}</a></h4>
        <div class="post-meta">${metadata}</div>
      </article></div>
    </section>`,
});

await rm(output, { recursive: true, force: true });
await mkdir(path.join(output, postPath), { recursive: true });
await mkdir(path.join(output, 'rope'), { recursive: true });
await mkdir(path.join(output, 'archives'), { recursive: true });
await mkdir(path.join(output, 'assets'), { recursive: true });
await writeFile(path.join(output, postPath, 'index.html'), article);
await writeFile(path.join(output, 'index.html'), home);
await writeFile(path.join(output, 'archives/index.html'), archives);
await writeFile(path.join(output, 'assets/math.css'), adaptor.cssText(svg.styleSheet(document)));
for (const file of files) await copyFile(path.join(root, file), path.join(output, 'assets', file));
const fonts = path.join(root, 'node_modules/@fontsource-variable/jetbrains-mono');
await cp(path.join(fonts, 'files'), path.join(output, 'assets/fonts/files'), { recursive: true });
await copyFile(path.join(fonts, 'index.css'), path.join(output, 'assets/fonts/index.css'));
await copyFile(path.join(fonts, 'LICENSE'), path.join(output, 'assets/fonts/LICENSE'));
await mkdir(path.join(output, 'assets/licenses'), { recursive: true });
await copyFile(path.join(root, 'node_modules/@mathjax/src/LICENSE'), path.join(output, 'assets/licenses/MathJax.txt'));
const mathFont = JSON.parse(await readFile(path.join(root, 'node_modules/@mathjax/mathjax-newcm-font/package.json'), 'utf8'));
await writeFile(path.join(output, 'assets/licenses/NewCM.txt'),
  `${mathFont.name} ${mathFont.version}\n${mathFont.repository.url}\nLicense: ${mathFont.license}\n\n` +
  await readFile(path.join(root, 'node_modules/@mathjax/src/LICENSE'), 'utf8'));
await copyFile(path.join(root, 'node_modules/highlight.js/LICENSE'), path.join(output, 'assets/licenses/highlight.js.txt'));
await copyFile(path.join(root, 'content/rope.md'), path.join(output, postPath, 'rope.md'));
await copyFile(path.join(root, 'content/rope.md'), path.join(output, 'rope/rope.md'));
await writeFile(path.join(output, '.nojekyll'), '');
await writeFile(path.join(output, 'sitemap.xml'), `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">${['', postPath, 'archives/'].map((route) => `<url><loc>${siteUrl}${route}</loc></url>`).join('')}</urlset>\n`);
await writeFile(path.join(output, 'rope/index.html'), `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="0;url=../${postPath}"><link rel="canonical" href="${siteUrl}${postPath}"><title>${escape(postTitle)} · Casebook</title></head>
<body><a href="../${postPath}">阅读 ${escape(postTitle)}</a><script>location.replace('../${postPath}' + location.search + location.hash);</script></body></html>\n`);
console.log(`Built home, archives, RoPE article, and legacy redirect: ${displayCount} display formulas, ${inlineCount} inline formulas. All assets are local.`);
