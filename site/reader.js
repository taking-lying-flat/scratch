const themeButton = document.querySelector('.theme-toggle');
if (themeButton) {
  const syncThemeLabel = () => {
    const dark = document.documentElement.dataset.colorMode === 'dark';
    const label = dark ? '切换到 Light Tritanopia' : '切换到 Dark Tritanopia';
    themeButton.setAttribute('aria-label', label);
    themeButton.title = `${dark ? 'Dark Tritanopia' : 'Light Tritanopia'} · ${label}`;
  };
  themeButton.hidden = false;
  syncThemeLabel();
  themeButton.addEventListener('click', () => {
    const theme = document.documentElement.dataset.colorMode === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = theme;
    document.documentElement.dataset.colorMode = theme;
    document.querySelector('meta[name="theme-color"]').content = getComputedStyle(document.documentElement).getPropertyValue('--bgColor-muted').trim();
    try { localStorage.setItem('blog-theme', theme); } catch {}
    syncThemeLabel();
  });
}

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
