# 看版本/安装路径/依赖
pip show <pkg>                    
# 配置清华源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install <package-name> -i https://pypi.tuna.tsinghua.edu.cn/simple
conda install -c conda-forge xformers -y

# 安装 flash-attn 2
pip install packaging ninja
pip install flash-attn --no-build-isolation

# 源码安装
git clone --recursive https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git submodule update --init --recursive
pip install -U pip setuptools wheel
pip install -U packaging ninja
export MAX_JOBS=4 pip install -v . --no-build-isolation

# 安装 ms-swift
pip install ms-swift -U
# 源码安装
pip install -e .
pip install "qwen_vl_utils>=0.0.14" "decord" -U -i https://mirrors.aliyun.com/pypi/simple/

# 清理 pip 和 conda 缓存
pip cache purge
conda clean -a -y
rm -rf ~/.cache/pip ~/.cache/torch_extensions
