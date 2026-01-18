## 1. 基础配置与查看

```bash
# 看版本/安装路径/依赖
pip show <pkg>

# 配置清华源 (全局设置)
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 临时使用清华源安装
pip install <package-name> -i https://pypi.tuna.tsinghua.edu.cn/simple

# Conda 安装 xformers
conda install -c conda-forge xformers -y
```

## 2. Flash Attention 2 安装

### 常规安装
```bash
pip install packaging ninja
pip install flash-attn --no-build-isolation
```

### 源码编译安装
*适用于预编译包不兼容或需要特定版本的情况*
```bash
git clone --recursive https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git submodule update --init --recursive

# 更新构建依赖
pip install -U pip setuptools wheel
pip install -U packaging ninja

# 编译安装 (MAX_JOBS 限制并发防止内存溢出)
MAX_JOBS=4 pip install -v . --no-build-isolation
```

## 3. ms-swift 安装

```bash
# PyPI 安装
pip install ms-swift -U

# 源码安装 (开发模式，需在源码根目录执行)
pip install -e .

# 安装 Qwen-VL 等多模态所需依赖
pip install "qwen_vl_utils>=0.0.14" "decord" -U -i https://mirrors.aliyun.com/pypi/simple
```

## 4. 维护与初始化

### 缓存清理
```bash
# 清理包管理器缓存
pip cache purge
conda clean -a -y

# 清理编译缓存
rm -rf ~/.cache/pip ~/.cache/torch_extensions
```

### Conda Shell 初始化
```bash
/root/miniconda3/bin/conda init bash
source ~/.bashrc
```
