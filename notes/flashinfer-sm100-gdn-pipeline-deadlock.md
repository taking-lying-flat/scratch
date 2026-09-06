# CUDA 同步故障：ZeRO-3 梯度竞态与 GDN Pipeline 死锁

> 两个表面症状完全不同、但本质都属于同步协议被破坏的 `GPU` 并发故障

| 案例 | 上游修复 | 影响范围 | 表面症状 | 故障类型 |
| --- | --- | --- | --- | --- |
| `DeepSpeed ZeRO-3` | [DeepSpeed#7898](https://github.com/deepspeedai/DeepSpeed/pull/7898) | `DeepSpeed 0.18.7` 及更早版本 | 第一次 `optimizer.step()` 后部分权重变成 `NaN` | 跨 `CUDA stream` 的 `RAW race` |
| `FlashInfer GDN` | [flashinfer#3581](https://github.com/flashinfer-ai/flashinfer/pull/3581) | `FlashInfer 0.6.12 / 0.6.13`、`SM100` | `prefill` 随机永久卡住 | `CUTLASS pipeline ownership violation` |

> [!IMPORTANT]
> 两个问题都不是普通的数值精度故障。`DeepSpeed` 等错梯度的生产 `stream`；`FlashInfer` 则让非 `owner` 参与 `pipeline` 的终止同步。前者破坏数据的 `happens-before` 关系，后者破坏 `mbarrier` 的 `phase/lifetime` 状态

## 目录

- [案例一：DeepSpeed ZeRO-3 梯度归约竞态](#案例一deepspeed-zero-3-梯度归约竞态)
- [案例二：FlashInfer GDN Pipeline 死锁](#案例二flashinfer-gdn-pipeline-死锁)
- [两个案例的共同模式](#两个案例的共同模式)

## 案例一：DeepSpeed ZeRO-3 梯度归约竞态

### TL;DR

> [!CAUTION]
> `DeepSpeed` 固定等待 `default_stream()`，但 `param.grad` 可能由 `non-default current_stream()` 上的 `backward kernel` 产生。因此 `reduce_and_partition_stream` 可能在梯度写完前就读取它，形成 `read-after-write race`，并把污染后的 `gradient shard` 带入第一次 `optimizer update`

```diff
 self.reduce_and_partition_stream.wait_stream(
-    get_accelerator().default_stream()
+    get_accelerator().current_stream()
 )
```

### 现象与复现结果

上游复现展示典型的随机污染特征：

| 配置项 | 复现环境 / 结果 |
| --- | --- |
| 模型 | `Qwen3-4B` |
| 硬件 | `7 × H200` |
| 软件 | `DeepSpeed 0.18.7`、`PyTorch 2.10.0` |
| 精度 | `BF16` |
| 修复前 | 第一次 `optimizer step` 后超过 15 万个 `NaN`，分布在 55 个 `weight layer` |
| 修复后 | 同一配置下 `NaN` 数量为 0 |

污染位置并不固定，可能出现在不同 `rank`、不同 `layer`，尤其容易影响 `attention projection`、`MLP weight` 等参数的 `gradient shard`。这种随机性正是 `CUDA kernel` 调度时序参与故障的信号

### ZeRO-3 的梯度生产与消费

当 `overlap_comm=True` 时，`ZeRO-3` 使用独立的 `reduce_and_partition_stream`，让 `backward` 计算与梯度通信重叠。某个参数的梯度完成后，`gradient hook` 会进入 `stage3.py` 的 `__add_grad_to_ipg_bucket()`：

```python
@torch.no_grad()
def __add_grad_to_ipg_bucket(self, param):
    if not get_accelerator().resolves_data_dependency():
        self.reduce_and_partition_stream.wait_stream(
            get_accelerator().default_stream()
        )

    bucket = self.ipg_buckets[self.get_param_comm_dtype(param)]
    # 随后读取 param.grad，复制到 contiguous gradient bucket，
    # 最终执行 reduce-scatter。
```

在这条路径里：

- `backward CUDA kernel` 是 `param.grad` 的 `producer`
- `reduce_and_partition_stream` 是 `gradient buffer` 的 `consumer`
- `consumer` 必须在 `producer` 完成写入后才能执行 `bucket copy` 和 `reduce-scatter`

### 错误的数据依赖

`PyTorch autograd` 不保证 `backward kernel` 一定运行在 `CUDA default stream`。它会记录 `forward op` 所在的 `stream`，并让对应的 `backward CUDA op` 在相应 `stream` 上执行。`gradient hook` 又在 `autograd backward` 的执行上下文中同步触发，因此 `hook` 内的 `current_stream()` 才是实际生产当前梯度的 `stream`

旧代码建立的是：

```text
default_stream ────────────────► reduce_and_partition_stream

backward_stream ── write(param.grad)
                   ↑
                   没有同步关系
```

真正需要的依赖是：

```text
current/backward stream:
    backward kernel
    write param.grad
          │
          │ wait_stream(current_stream)
          ▼
reduce_and_partition_stream:
    copy param.grad -> bucket
    reduce-scatter
```

> [!WARNING]
> `reduce_stream.wait_stream(default_stream)` 只能证明 `default stream` 已完成，不能证明真正写入 `param.grad` 的 `backward_stream` 已完成。等待错误的对象，等价于没有建立所需的数据依赖

### `Race` 如何污染权重

正常顺序应当是：

```text
backward kernel
  -> param.grad 写完
  -> copy gradient 到 bucket
  -> reduce-scatter
  -> optimizer.step()
```

存在 `bug` 时，两个 `stream` 可能并发访问同一块 `gradient buffer`：

```text
S1 / backward_stream                   reduce_and_partition_stream
        │                                          │
        ├── 正在写 param.grad                       │
        │                                          ├── 提前读 param.grad
        │                                          └── reduce-scatter
        └── 写入尚未完成
```

这是一种典型的 `read-after-write (RAW) data hazard`。`consumer` 可能读到尚未完整写入的数据，污染后的内容随后进入 `reduce-scatter`，形成错误的 `gradient partition`

### 为什么 `NaN` 在 `optimizer.step()` 后才出现

> [!NOTE]
> `Race` 发生在 `backward`，但首先损坏的是 `gradient`，而不是 `parameter`。`Adam/AdamW` 在 `optimizer.step()` 中消费错误梯度，更新一阶动量、二阶动量和 `parameter partition` 后，`NaN` 才真正进入权重

因此，从外部看起来像是“`optimizer` 第一步把权重算坏”，实际却是 `optimizer` 消费上一步已经被 `CUDA race` 污染的 `gradient shard`

### 修复为什么有效

`__add_grad_to_ipg_bucket()` 从 `autograd gradient hook` 中调用，因此 `current_stream()` 正是 `autograd` 为当前 `backward op` 设置的 `stream`：

- `backward` 位于 `default stream` 时，`current_stream()` 与旧行为等价
- `backward` 位于 `non-default stream` 时，它能建立真正的 `producer -> consumer` 依赖
- 修改不会牺牲 `overlap`，只会补齐缺失的 `stream synchronization`

> [!TIP]
> 并发语义可以概括为：不要猜 `producer` 位于哪个 `stream`，而应等待实际 `producer`。修复恢复 `backward_stream -> reduce_stream` 的 `happens-before` 边

## 案例二：FlashInfer GDN Pipeline 死锁

### TL;DR

> [!CAUTION]
> `SM100 GDN prefill kernel` 中，`o_store pipeline` 的合法 `producer` 是 `CG1`，但 `CG0` 也错误调用 `o_store_producer.tail()`。`CG0` 没有执行配套的 `acquire/commit`，其本地 `PipelineState` 与真实 `mbarrier phase` 不一致，最终等待一个不会再出现的 `barrier transition`

这个缺陷在 `non-null initial state` 与 `long-tail ragged varlen workload` 下更容易触发。局部 `warp group` 阻塞会进一步阻止整个 `CTA` 完成，最终表现为 `GPU kernel hang`

### `CTA` 的 `warp` 分工

一个 `FlashInfer SM100 GDN kernel CTA` 包含 12 个 `warp`，共 384 个线程。各 `warp` 的角色划分如下：

| `Warp` | 名称 | 大小 | 主要职责 |
| --- | --- | --- | --- |
| `warp 0–3` | `CG0 / Compute Group 0` | `4 warps = 128 threads` | 前处理、`QK/KK` 后处理、矩阵求逆等 |
| `warp 4–7` | `CG1 / Compute Group 1` | `4 warps = 128 threads` | `recurrent state`、`value/state` 更新、输出 `epilogue` 计算等 |
| `warp 8` | `MMA warp` | `1 warp` | 发起 7 个 `GEMM / Tensor Core MMA` |
| `warp 9` | `TMA load warp` | `1 warp` | 从显存加载 `Q/K/V` |
| `warp 10` | `gate/beta load warp` | `1 warp` | 加载 `gate`、`beta` |
| `warp 11` | `epilogue warp` | `1 warp` | 把最终输出 `O` 写回 `global memory` |

### `o_store pipeline` 的所有权

`o_store_producer` 描述的是从 `CG1` 到 `epilogue warp` 的输出 `pipeline`：

```text
CG1 / producer                          Epilogue / consumer

acquire()
   │
   │ 等待当前 slot 变为 EMPTY
   ▼
write O into SMEM
   │
commit()
   │
   ├──────── signal FULL ─────────────────► wait()
   │                                          │
   │                                      TMA store O
   │                                          │
   │              ◄──── signal EMPTY ─── release()
   │
acquire next slot
   ...
tail()
```

角色在代码中也被明确绑定：

```python
o_store_producer, o_store_consumer = pipeline.PipelineAsync.create(
    producer_group=cg_cg1,  # CG1
    consumer_group=cg_epi,  # warp 11
    ...
).make_participants()
```

因此，`CG1` 是合法 `producer`，负责 `acquire -> 写入 shared memory -> commit`；`epilogue warp` 是 `consumer`，负责 `wait -> 写回 global memory -> release`。`CG0` 从未参与这条 `pipeline` 的正常生产过程

### `CUTLASS PipelineState` 与 `tail()`

`CUTLASS pipeline` 是一套基于 `mbarrier` 的状态机，而不只是一个 `buffer` 封装：

| 操作 | 执行方 | 同步含义 |
| --- | --- | --- |
| `acquire` | `producer` | 等待当前 `stage` 变为 `empty`，取得可写 `slot` |
| `commit` | `producer` | 标记当前 `stage` 已生产完成，发出 `full signal` |
| `wait` | `consumer` | 等待 `stage` 变为 `full` |
| `release` | `consumer` | 消费完成，将 `stage` 重新标记为 `empty` |
| `tail` | `producer` | 退出前等待最后使用过的 `buffer` 恢复为 `empty` |

`PipelineState` 保存 `circular buffer` 当前的 `index` 和 `phase`。`tail()` 仍属于同步协议的一部分，必须基于该 `producer` 经过 `acquire/commit` 后形成的正确状态执行

### 错误的 `owner` 调用 `tail()`

`CG0` 的收尾代码错误包含：

```python
work = scheduler.get_current_work()
a_inv_ready_producer.tail()
qk_ready_producer.tail()
o_store_producer.tail()       # 错误：CG0 不是 owner
group_order_producer.tail()
```

正确修复是让 `CG0` 不再终止 `CG1` 所拥有的 `pipeline`：

```diff
 work = scheduler.get_current_work()
 a_inv_ready_producer.tail()
 qk_ready_producer.tail()
-o_store_producer.tail()
 group_order_producer.tail()
```

`CG1` 才是唯一应推进这条 `producer` 状态机并执行对应 `tail()` 的 `warp group`：

```text
CG0 ─── o_store_producer.tail()   ❌ 非 owner

CG1 ─── acquire
     ├─ commit
     └─ o_store_producer.tail()   ✅ 真正 owner
```

### 为什么会永久死锁

`CG0` 没有经历 `o_store pipeline` 的 `acquire/commit` 状态推进，却拿自己的本地 `state` 执行 `tail()`。它可能仍在等待旧 `index/phase` 对应的 `empty transition`，而 `CG1` 与 `epilogue` 已经沿正确 `phase` 继续推进，不会再产生 `CG0` 所等待的信号

```text
CG0 (warp 0-3)             CG1 (warp 4-7)          Epilogue (warp 11)
      │                           │                        │
      │                           ├── produce O ──────────►│
      │                           │   o_store pipeline     │
      │                           │                        │
      └── 错误调用 ───────────────┐
          o_store_producer.tail() │
                                  ▼
                      等待错误的 barrier phase
                                  │
                                  ▼
                           CG0 永久阻塞
                                  │
                                  ▼
                     CTA 无法完成，kernel hang
```

故障会逐层放大：

```text
CG0 等不到 mbarrier
  -> warp group 无法退出
  -> CTA 无法完成
  -> GDN kernel 不返回
  -> vLLM prefill 永久卡住
  -> 其他 tensor-parallel rank 继续等待
  -> rollout / 训练整体随机 hang
```

> [!WARNING]
> `tail()` 不是无害的析构函数。让非 `owner` 调用它，会用未经正常推进的 `PipelineState` 参与 `barrier lifetime synchronization`，足以让整个 `CTA` 永久无法 `retire`

### 正确的 `pipeline invariant`

> [!TIP]
> 一条 `warp-specialized pipeline` 的生命周期只能由其合法参与者维护：`producer` 推进 `acquire/commit/tail`，`consumer` 推进 `wait/release`。没有参与生产的 `warp group` 不得代替 `owner` 执行终止同步

## 两个案例的共同模式

| 维度 | `DeepSpeed ZeRO-3` | `FlashInfer GDN` |
| --- | --- | --- |
| 共享对象 | `param.grad / gradient buffer` | `o_store pipeline / mbarrier state` |
| 合法 `producer` | `autograd current/backward stream` | `CG1` |
| 合法 `consumer` | `reduce_and_partition_stream` | `epilogue warp` |
| 被破坏的规则 | `consumer` 等待错误的 `stream` | 非 `owner` 推进 `producer termination` |
| 并发后果 | 写入未完成时提前读取 | 等待不会再出现的 `barrier phase` |
| 外部症状 | 随机 `gradient/weight NaN` | `GPU kernel` 永久 `hang` |

> [!IMPORTANT]
> `GPU` 并发代码中最关键的两个问题是：**谁真正生产数据，以及谁拥有同步状态机。** 等待错误的 `producer` 会产生数据竞态；让错误的 `owner` 推进 `pipeline` 会产生永久等待
