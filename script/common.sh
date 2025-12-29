# 配置清华源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.extra-index-url https://download.pytorch.org/whl/cu129

# 指定清华源安装 vllm
pip install <package-name> -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 flash-attn 2
pip install packaging ninja
pip install flash-attn --no-build-isolation

# 安装 ms-swift
pip install ms-swift -U
# 源码安装
pip install -e .
pip install "qwen_vl_utils>=0.0.14" "decord" -U -i https://mirrors.aliyun.com/pypi/simple/
