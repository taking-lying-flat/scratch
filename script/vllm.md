## 1. 环境变量配置

```bash
# 设置注意力计算后端 (可选: FLASH_ATTN, TORCH_SDPA, XFORMERS 等)
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# 采样是否使用 FlashInfer (0=关闭, 1=开启)
export VLLM_USE_FLASHINFER_SAMPLER=0

# CPU 线程控制
# 作用：限制 OpenMP 算子（MKL/OpenBLAS）仅使用 1 个线程
# 目的：防止多进程/多 Worker 模式下 CPU 负载爆炸
export OMP_NUM_THREADS=1
```
