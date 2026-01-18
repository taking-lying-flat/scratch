## 环境变量配置

```bash
# 强制设置注意力计算后端 
# 可选值: "TORCH_SDPA", "FLASH_ATTN", "XFORMERS", "ROCM_FLASH", "FLASHINFER", "FLASHMLA"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# 采样是否使用 FlashInfer (0=关闭, 1=开启)
# 开启有助于提升采样阶段速度，但需额外安装 flashinfer 依赖
export VLLM_USE_FLASHINFER_SAMPLER=0

# CPU 线程控制 (防过载)
# 限制 OpenMP 算子 (MKL/OpenBLAS) 的线程数
# 严重建议设为 1，防止 vLLM 的每个 Worker 进程都抢占所有 CPU 核心导致死锁或性能下降
export OMP_NUM_THREADS=1
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
nohup python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --max-model-len 4096 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.85 \
  --dtype bfloat16 \
  --max-num-seqs 32 \
  --disable-log-requests \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --chat-template-content-format string \
  --enable-lora \
  --lora-modules "${LORA_NAME}=${LORA_PATH}" \
  --max-loras 1 \
  --max-lora-rank 64 \
  > "$LOG" 2>&1 &
```
