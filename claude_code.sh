# 1. 安装基础工具
sudo apt update
sudo apt install -y curl git build-essential

# 2. 安装 nvm，从 install.sh 开始
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.4/install.sh | bash

# 3. 让 nvm 生效
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"

# 4. 验证 nvm
nvm --version

# 5. 安装 Node.js LTS
nvm install --lts
nvm alias default 'lts/*'
nvm use default

# 6. 验证 node / npm
node -v
npm -v

# 7. 安装 Claude Code
npm install -g @anthropic-ai/claude-code

# 8. 验证 Claude Code
claude --version
claude doctor

# 9. 登录并启动
claude
