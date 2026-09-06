import { mkdir, readFile, writeFile, copyFile, cp, rm } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import MarkdownIt from 'markdown-it';
import hljs from 'highlight.js/lib/core';
import python from 'highlight.js/lib/languages/python';
import json from 'highlight.js/lib/languages/json';
import cpp from 'highlight.js/lib/languages/cpp';
import bash from 'highlight.js/lib/languages/bash';
import diff from 'highlight.js/lib/languages/diff';
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
const siteUrl = 'https://taking-lying-flat.github.io/blog/';
const posts = JSON.parse(await readFile(path.join(root, 'posts.json'), 'utf8'))
  .sort((a, b) => b.date.localeCompare(a.date));
// Repository-authored HTML tables are rendered alongside Markdown.
const markdown = new MarkdownIt({ html: true, typographer: false });
const escape = markdown.utils.escapeHtml;
for (const [name, language] of Object.entries({ python, json, cpp, bash, diff })) {
  hljs.registerLanguage(name, language);
}
const languageNames = { python: 'Python', json: 'JSON', cpp: 'C++ / CUDA', bash: 'Shell', diff: 'Diff', text: 'Text' };

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
let inlineCount = 0;
let displayCount = 0;

async function renderMath(token, display) {
  const node = await document.convertPromise(token.content, { display, em: 18, ex: 8.1, containerWidth: 1024 });
  const html = adaptor.outerHTML(node);
  if (/data-mjx-error|data-mml-node="merror"/.test(html)) throw new Error(`Invalid formula: ${token.content}`);
  token.meta = { html };
  if (display) displayCount++;
  else inlineCount++;
}

const inlineText = (token) => (token?.children ?? []).map((child) =>
  child.type === 'softbreak' || child.type === 'hardbreak' ? ' ' : child.content).join('');
const slugify = (label) => label.toLowerCase().replace(/[^\p{L}\p{N}\p{M}_\-\s]/gu, '').replace(/\s/g, '-');
const dateLabel = (date) => new Intl.DateTimeFormat('zh-CN', {
  dateStyle: 'long', timeZone: 'UTC',
}).format(new Date(`${date}T00:00:00Z`));
const metadata = (post) => `<time datetime="${post.date}">${dateLabel(post.date)}</time><span>约 ${post.readingMinutes} 分钟</span>`;

function removeManualToc(tokens) {
  for (let i = 0; i < tokens.length - 3; i++) {
    if (!['heading_open', 'paragraph_open'].includes(tokens[i].type)) continue;
    if (inlineText(tokens[i + 1]).trim() !== '目录') continue;
    if (!['bullet_list_open', 'ordered_list_open'].includes(tokens[i + 3].type)) continue;
    let end = i + 3;
    let depth = 0;
    do { depth += tokens[end++].nesting; } while (depth > 0 && end < tokens.length);
    tokens.splice(i, end - i);
    i--;
  }
}

function prepareCallouts(tokens) {
  const labels = { NOTE: '说明', TIP: '提示', IMPORTANT: '重要', WARNING: '注意', CAUTION: '注意' };
  for (let i = 0; i < tokens.length; i++) {
    if (tokens[i].type !== 'blockquote_open' || tokens[i + 1]?.type !== 'paragraph_open') continue;
    const inline = tokens[i + 2];
    const match = inline?.content.match(/^\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\](?:\n|$)/);
    if (!match) continue;
    tokens[i].attrSet('class', `callout callout-${match[1].toLowerCase()}`);
    tokens[i].meta = { label: labels[match[1]] };
    inline.content = inline.content.slice(match[0].length);
    inline.children = [];
    markdown.inline.parse(inline.content, markdown, {}, inline.children);
  }
}

markdown.renderer.rules.blockquote_open = (items, i, _options, _env, renderer) =>
  `<blockquote${renderer.renderAttrs(items[i])}>${items[i].meta?.label
    ? `<p class="callout-title">${escape(items[i].meta.label)}</p>` : ''}\n`;
const tableWrapper = '<div class="table-scroll" role="region" aria-label="表格" tabindex="0">';
markdown.renderer.rules.table_open = () => `${tableWrapper}<table>\n`;
markdown.renderer.rules.table_close = () => '</table></div>\n';
markdown.renderer.rules.html_block = (items, i) => items[i].content.includes('<table')
  ? `${tableWrapper}${items[i].content}</div>\n` : items[i].content;
markdown.renderer.rules.math_inline = (items, i) =>
  `<span class="math-inline" role="math" aria-label="${escape(items[i].content)}">${items[i].meta.html}</span>`;
markdown.renderer.rules.fence = (items, i, _options, env) => {
  const token = items[i];
  const language = token.info.trim().split(/\s+/)[0] || 'text';
  if (language === 'math') {
    return `<div class="equation" tabindex="0" role="math" aria-label="${escape(token.content)}">${token.meta.html}</div>\n`;
  }
  const id = token.attrGet('id');
  const label = languageNames[language] ?? language;
  const name = env.slug === 'rope'
    ? (language === 'json' ? 'config.json · text_config' : language === 'text' ? '张量维度'
      : token.content.includes('def rotate_half') ? 'rotate_half / apply_rotary_pos_emb' : 'compute_default_rope_parameters')
    : label;
  const code = hljs.getLanguage(language)
    ? hljs.highlight(token.content, { language, ignoreIllegals: true }).value : escape(token.content);
  const lineCount = token.content.replace(/\n$/, '').split('\n').length;
  const numbers = language === 'text' ? '' : `<span class="line-numbers" aria-hidden="true">${
    Array.from({ length: lineCount }, (_, line) => line + 1).join('\n')
  }</span>`;
  return `<figure class="code-block"${id ? ` id="${escape(id)}"` : ''}>
    <figcaption>
      <span class="window-title" title="${escape(name)}">${escape(name)}</span>
      <span class="code-actions"><span class="code-language">${escape(label)}</span><button type="button" class="copy-button" hidden>复制</button></span>
    </figcaption>
    <pre tabindex="0">${numbers}<code class="language-${escape(language)}">${code}</code></pre>
  </figure>\n`;
};

const ropeSections = [
  { id: 'frequency', prefix: 'RoPE（Rotary' },
  { id: 'configuration', fence: 'json' },
  { id: 'rotation', prefix: 'RoFormer §3.2.1' },
  { id: 'relative-position', prefix: 'RoFormer 式（15）' },
  { id: 'partial-rope', prefix: 'Qwen3.5 的 partial RoPE' },
  { id: 'implementation', prefix: '对普通 RoPE' },
];

for (const post of posts) {
  post.source = await readFile(path.join(root, post.file), 'utf8');
  post.route = `posts/${post.slug}/`;
  const tokens = markdown.parse(post.source, {});
  if (tokens[0]?.type !== 'heading_open' || tokens[0].tag !== 'h1') throw new Error(`Expected document title: ${post.file}`);
  post.title = inlineText(tokens[1]);
  post.titleId = slugify(post.title);
  tokens.splice(0, 3);
  removeManualToc(tokens);
  prepareCallouts(tokens);
  const usedIds = new Map([[post.titleId, 1]]);
  for (let i = 0; i < tokens.length; i++) {
    if (tokens[i].type !== 'heading_open') continue;
    const label = inlineText(tokens[i + 1]);
    const baseId = slugify(label);
    const occurrence = usedIds.get(baseId) ?? 0;
    usedIds.set(baseId, occurrence + 1);
    const id = occurrence ? `${baseId}-${occurrence}` : baseId;
    tokens[i].attrSet('id', id);
  }
  if (post.slug === 'rope') {
    for (const section of ropeSections) {
      const index = tokens.findIndex((token, i) => section.fence
        ? token.type === 'fence' && token.info === section.fence
        : token.type === 'paragraph_open' && tokens[i + 1]?.content.startsWith(section.prefix));
      if (index < 0) throw new Error(`Missing section: ${section.id}`);
      tokens[index].attrSet('id', section.id);
    }
  }
  const previousDisplays = displayCount;
  for (const token of tokens) {
    if (token.type === 'fence' && token.info.trim() === 'math') await renderMath(token, true);
    for (const child of token.children ?? []) {
      if (child.type === 'math_inline') await renderMath(child, false);
    }
  }
  const proseText = tokens.filter((token) => token.type === 'inline').map(inlineText).join(' ');
  post.readingMinutes = Math.max(1, Math.ceil(
    (proseText.match(/\p{Script=Han}/gu)?.length ?? 0) / 300 +
    (proseText.match(/[A-Za-z]+/g)?.length ?? 0) / 200 + (displayCount - previousDisplays) * 0.3 + 2
  ));
  const firstParagraph = tokens.findIndex((token, i) => token.type === 'paragraph_open' && tokens[i + 1]?.content);
  post.description = inlineText(tokens[firstParagraph + 1]).slice(0, 180) || post.title;
  post.content = markdown.renderer.render(tokens, markdown.options, { slug: post.slug });
}

const template = await readFile(path.join(root, 'template.html'), 'utf8');
const primer = path.join(root, 'node_modules/@primer/primitives');
const assets = new Map([
  ...['reader.css', 'reader.js', 'theme.js', 'favicon.svg'].map((file) => [file, path.join(root, file)]),
  ...['light', 'dark'].map((mode) => [`github-${mode}-tritanopia.css`,
    path.join(primer, `dist/css/functional/themes/${mode}-tritanopia.css`)]),
]);
const assetVersion = createHash('sha256');
for (const file of assets.values()) assetVersion.update(await readFile(file));
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
  for (const file of assets.keys()) html = html.replaceAll(`assets/${file}"`, `assets/${file}?v=${version}"`);
  return html;
}

const home = page({
  title: 'Blog · 技术笔记', description: '关于模型、论文与源码的技术笔记。', pageClass: 'home-page',
  body: `<section aria-labelledby="post-list-title">
    <div class="post-list-heading"><h1 id="post-list-title">全部文章 <span>${String(posts.length).padStart(2, '0')}</span></h1></div>
    <div class="post-list">${posts.map((post) => `<article class="post-entry">
      <h2><a href="${post.route}">${escape(post.title)}</a></h2>
      <footer class="entry-footer">
        <div class="entry-details">
          <div class="post-meta">${metadata(post)}</div>
          <ul class="entry-topics" aria-label="文章主题">${post.tags.map((tag) => `<li>${escape(tag)}</li>`).join('')}</ul>
        </div>
        <svg class="entry-arrow" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 12h14m-6-6 6 6-6 6"/></svg>
      </footer>
    </article>`).join('\n')}</div>
  </section>`,
});

const years = [...new Set(posts.map((post) => post.date.slice(0, 4)))];
const archives = page({
  title: '归档 · Blog', description: 'Blog 技术笔记归档。', route: 'archives/', pageClass: 'archive-page',
  body: `<header class="page-header"><h1>归档</h1><p>${posts.length} 篇文章</p></header>
    ${years.map((year) => {
      const yearPosts = posts.filter((post) => post.date.startsWith(year));
      const months = [...new Set(yearPosts.map((post) => post.date.slice(0, 7)))];
      return `<section class="archive-year" aria-labelledby="year-${year}">
        <h2 id="year-${year}">${year} <span>${yearPosts.length}</span></h2>
        ${months.map((month) => `<div class="archive-month">
          <h3>${new Intl.DateTimeFormat('zh-CN', { month: 'long', timeZone: 'UTC' }).format(new Date(`${month}-01T00:00:00Z`))}</h3>
          <div class="archive-entries">${yearPosts.filter((post) => post.date.startsWith(month)).map((post) => `<article class="archive-entry">
            <h4><a href="../${post.route}">${escape(post.title)}</a></h4>
            <div class="post-meta">${metadata(post)}</div>
          </article>`).join('\n')}</div>
        </div>`).join('\n')}
      </section>`;
    }).join('\n')}`,
});

await rm(output, { recursive: true, force: true });
await mkdir(path.join(output, 'archives'), { recursive: true });
await mkdir(path.join(output, 'assets'), { recursive: true });
await writeFile(path.join(output, 'index.html'), home);
await writeFile(path.join(output, 'archives/index.html'), archives);
for (const [index, post] of posts.entries()) {
  const previous = posts[index - 1];
  const next = posts[index + 1];
  const article = page({
    title: `${post.title} · Blog`, description: post.description, route: post.route, type: 'article', pageClass: 'post-page',
    body: `<article class="post-single">
      <header class="post-header">
        <h1 id="${escape(post.titleId)}">${escape(post.title)}</h1>
        <div class="post-meta">${metadata(post)}<span>taking-lying-flat</span></div>
      </header>
      <div class="prose">${post.content}</div>
      <footer class="post-footer">
        <div class="post-topics" aria-label="文章主题">${post.tags.map((tag) => `<span>${escape(tag)}</span>`).join('')}</div>
        <div class="post-actions"><a href="#top">返回顶部 ↑</a></div>
        <nav class="post-pagination" aria-label="文章翻页">
          ${previous ? `<a href="../${previous.slug}/"><span>← 上一篇</span>${escape(previous.title)}</a>` : ''}
          ${next ? `<a class="post-next" href="../${next.slug}/"><span>下一篇 →</span>${escape(next.title)}</a>` : ''}
        </nav>
        <a class="back-link" href="../../">← 全部文章</a>
      </footer>
    </article>`,
  });
  await mkdir(path.join(output, post.route), { recursive: true });
  await writeFile(path.join(output, post.route, 'index.html'), article);
}

await writeFile(path.join(output, 'assets/math.css'), adaptor.cssText(svg.styleSheet(document)));
for (const [name, file] of assets) await copyFile(file, path.join(output, 'assets', name));
const fonts = path.join(root, 'node_modules/@fontsource-variable/jetbrains-mono');
await cp(path.join(fonts, 'files'), path.join(output, 'assets/fonts/files'), { recursive: true });
await copyFile(path.join(fonts, 'index.css'), path.join(output, 'assets/fonts/index.css'));
await copyFile(path.join(fonts, 'LICENSE'), path.join(output, 'assets/fonts/LICENSE'));
await mkdir(path.join(output, 'assets/licenses'), { recursive: true });
await copyFile(path.join(primer, 'LICENSE'), path.join(output, 'assets/licenses/Primer.txt'));
await copyFile(path.join(root, 'node_modules/@mathjax/src/LICENSE'), path.join(output, 'assets/licenses/MathJax.txt'));
const mathFont = JSON.parse(await readFile(path.join(root, 'node_modules/@mathjax/mathjax-newcm-font/package.json'), 'utf8'));
await writeFile(path.join(output, 'assets/licenses/NewCM.txt'),
  `${mathFont.name} ${mathFont.version}\n${mathFont.repository.url}\nLicense: ${mathFont.license}\n\n` +
  await readFile(path.join(root, 'node_modules/@mathjax/src/LICENSE'), 'utf8'));
await copyFile(path.join(root, 'node_modules/highlight.js/LICENSE'), path.join(output, 'assets/licenses/highlight.js.txt'));
await writeFile(path.join(output, '.nojekyll'), '');
await writeFile(path.join(output, 'sitemap.xml'), `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">${['', 'archives/', ...posts.map((post) => post.route)].map((route) => `<url><loc>${siteUrl}${route}</loc></url>`).join('')}</urlset>\n`);
const rope = posts.find((post) => post.slug === 'rope');
await mkdir(path.join(output, 'rope'), { recursive: true });
await copyFile(path.join(root, rope.file), path.join(output, rope.route, 'rope.md'));
await copyFile(path.join(root, rope.file), path.join(output, 'rope/rope.md'));
await writeFile(path.join(output, 'rope/index.html'), `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="0;url=../${rope.route}"><link rel="canonical" href="${siteUrl}${rope.route}"><title>${escape(rope.title)} · Blog</title></head>
<body><a href="../${rope.route}">阅读 ${escape(rope.title)}</a><script>location.replace('../${rope.route}' + location.search + location.hash);</script></body></html>\n`);
console.log(`Built ${posts.length} articles, home, archives, and legacy redirect: ${displayCount} display formulas, ${inlineCount} inline formulas.`);
