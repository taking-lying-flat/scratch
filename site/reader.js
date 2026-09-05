const status = document.querySelector('#copy-status');
if (navigator.clipboard?.writeText) {
  document.querySelectorAll('.copy-button').forEach((button) => {
    button.hidden = false;
    button.addEventListener('click', async () => {
      const code = button.closest('.code-block').querySelector('code').textContent;
      try {
        await navigator.clipboard.writeText(code);
        button.textContent = '已复制';
        status.textContent = '代码已复制';
      } catch {
        button.textContent = '请手动复制';
        status.textContent = '无法访问剪贴板，请选中代码手动复制';
      }
      setTimeout(() => { button.textContent = '复制'; }, 2000);
    });
  });
}

const printButton = document.querySelector('#print-button');
printButton.hidden = false;
printButton.addEventListener('click', async () => {
  await document.fonts.ready;
  window.print();
});

const links = [...document.querySelectorAll('.toc a')];
const sections = links.map((link) => document.querySelector(link.getAttribute('href')));
let scheduled = false;
function updateSection() {
  let current = sections[0];
  for (const section of sections) {
    if (section.getBoundingClientRect().top <= 180) current = section;
  }
  for (const link of links) {
    if (link.hash === `#${current.id}`) link.setAttribute('aria-current', 'location');
    else link.removeAttribute('aria-current');
  }
  scheduled = false;
}
addEventListener('scroll', () => {
  if (!scheduled) {
    scheduled = true;
    requestAnimationFrame(updateSection);
  }
}, { passive: true });
addEventListener('resize', updateSection);
document.fonts.ready.then(updateSection);
updateSection();

document.querySelectorAll('.mobile-toc a').forEach((link) => {
  link.addEventListener('click', () => { document.querySelector('.mobile-toc').open = false; });
});
