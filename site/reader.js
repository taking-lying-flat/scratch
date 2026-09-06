const themeButton = document.querySelector('.theme-toggle');
if (themeButton) {
  const syncThemeLabel = () => {
    const label = document.documentElement.dataset.theme === 'dark' ? '切换为浅色' : '切换为深色';
    themeButton.setAttribute('aria-label', label);
    themeButton.title = label;
  };
  themeButton.hidden = false;
  syncThemeLabel();
  themeButton.addEventListener('click', () => {
    const theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = theme;
    document.querySelector('meta[name="theme-color"]').content = theme === 'dark' ? '#1d1e20' : '#ffffff';
    try { localStorage.setItem('casebook-theme', theme); } catch {}
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
