## 环境变量配置

```bash
# 强制设置注意力计算后端 
# 可选值: FLASH_ATTN (推荐), TORCH_SDPA, XFORMERS, ROCM_FLASH
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# 采样是否使用 FlashInfer (0=关闭, 1=开启)
# 开启有助于提升采样阶段速度，但需额外安装 flashinfer 依赖
export VLLM_USE_FLASHINFER_SAMPLER=0

# CPU 线程控制 (防过载)
# 限制 OpenMP 算子 (MKL/OpenBLAS) 的线程数
# 严重建议设为 1，防止 vLLM 的每个 Worker 进程都抢占所有 CPU 核心导致死锁或性能下降
export OMP_NUM_THREADS=1
```
