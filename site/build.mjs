import { mkdir, readFile, writeFile, copyFile, cp } from 'node:fs/promises';
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
tokens.splice(0, 3);
let inlineCount = 0;
let displayCount = 0;

async function renderMath(token, display) {
  const node = await document.convertPromise(token.content, { display, em: 18, ex: 8.1, containerWidth: 880 });
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
  return `<figure class="code-block"${id ? ` id="${escape(id)}"` : ''}>
    <figcaption><span>${name}</span><button type="button" class="copy-button" hidden>复制</button></figcaption>
    <pre tabindex="0"><code class="language-${escape(language)}">${code}</code></pre>
  </figure>\n`;
};

const content = markdown.renderer.render(tokens, markdown.options, {});
const toc = sections.map(({ id, label }, i) =>
  `<li><a href="#${id}"><span class="toc-number">${String(i + 1).padStart(2, '0')}</span>${label}</a></li>`).join('\n');
const template = await readFile(path.join(root, 'template.html'), 'utf8');
const html = template.replaceAll('{{TOC}}', toc).replace('{{CONTENT}}', content);
if (/\{\{(?:TOC|CONTENT)\}\}|\$`|data-mjx-error|language-math/.test(html)) throw new Error('Unrendered content in output');

await mkdir(path.join(output, 'rope'), { recursive: true });
await mkdir(path.join(output, 'assets'), { recursive: true });
await writeFile(path.join(output, 'rope/index.html'), html);
await writeFile(path.join(output, 'assets/math.css'), adaptor.cssText(svg.styleSheet(document)));
for (const file of ['reader.css', 'reader.js']) await copyFile(path.join(root, file), path.join(output, 'assets', file));
const fonts = path.join(root, 'node_modules/@fontsource-variable/noto-serif-sc');
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
await copyFile(path.join(root, 'content/rope.md'), path.join(output, 'rope/rope.md'));
await writeFile(path.join(output, '.nojekyll'), '');
await writeFile(path.join(output, 'index.html'), `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="0;url=rope/"><title>Casebook · RoPE</title></head>
<body><a href="rope/">阅读 RoPE</a></body></html>\n`);
console.log(`Built RoPE: ${displayCount} display formulas, ${inlineCount} inline formulas. All assets are local.`);
