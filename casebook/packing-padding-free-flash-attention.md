# `Packing`、`Padding-Free` 与 `FlashAttention Varlen` 的完整数据链路

> 从 `ms-swift` 的数据装箱与批处理，一直追踪到 `Transformers` 桥接层和 `FlashAttention` 内核

| 项目 | 内容 |
| --- | --- |
| 分析范围 | 常规 `ms-swift SFT/PT/RLHF` 的 `Transformers/FSDP` 路径，以及 `Qwen2-VL` 多模态模板 |
| 核心问题 | 多条样本如何在物理上连续存储，同时在注意力语义上保持完全隔离 |
| 数据侧入口 | `PackingDataset`、`Template.data_collator()`、`packing_row()` |
| 模型侧入口 | `Transformers._flash_attention_forward()` |
| 内核侧入口 | `flash_attn_varlen_func()`、`mha_varlen_fwd()`、`BlockInfo` |

> [!IMPORTANT]
> `packing`、`padding-free` 和 `FlashAttention varlen` 分别工作在数据集层、批处理层和注意力内核层。三者组合后形成的是“物理连续、逻辑分段”的训练输入，而不是一条新的长语义序列。

## 1. 先看完整结果：三条序列如何进入同一次前向

假设当前有三条已经完成模板编码的序列：

| 逻辑序列 | `input_ids` 长度 | 局部位置 |
| --- | ---: | --- |
| `A` | 5 | `0 1 2 3 4` |
| `B` | 3 | `0 1 2` |
| `C` | 4 | `0 1 2 3` |

普通批处理会把它们补齐到相同宽度：

```text
A A A A A
B B B PAD PAD
C C C C PAD
```

`padding-free` 则把有效位置直接拼成一行：

```text
input_ids:
A A A A A | B B B | C C C C

position_ids:
0 1 2 3 4 | 0 1 2 | 0 1 2 3

physical shape:
[1, 12]
```

边界随后被转换成：

```text
cu_seqlens_q = [0, 5, 8, 12]
cu_seqlens_k = [0, 5, 8, 12]
max_seqlen_q = 5
max_seqlen_k = 5
```

经过 `embedding` 和 `QKV projection` 后，`FlashAttention` 收到的物理缓冲区仍然是连续的：

```text
Q = [Q_A | Q_B | Q_C]
K = [K_A | K_B | K_C]
V = [V_A | V_B | V_C]
```

但 `cu_seqlens` 将它解释为三个独立区间：

| 逻辑序列 | `Q` 区间 | `K/V` 区间 |
| --- | --- | --- |
| `A` | `[0, 5)` | `[0, 5)` |
| `B` | `[5, 8)` | `[5, 8)` |
| `C` | `[8, 12)` | `[8, 12)` |

因此实际语义是：

```text
Attention(A)
Attention(B)
Attention(C)
```

而不是：

```text
Attention(A | B | C as one causal sequence)
```

> [!CAUTION]
> `position_ids` 重新从 0 开始，不会自行阻止跨样本注意力。真正把 `Q/K/V` 切成独立注意力问题的是 `cu_seqlens_q/k`；`position_ids` 在这里首先是生成边界的载体，同时还负责各样本自己的 `RoPE/mRoPE` 位置语义。

## 2. 三个概念必须分层理解

| 机制 | 所在层级 | 输入 | 输出 | 核心职责 |
| --- | --- | --- | --- | --- |
| `packing` | 数据集 / 调度层 | 样本索引与编码后长度 | 若干样本索引组 | 决定哪些完整样本组成一个 `pack` |
| `padding-free` | `collator` / 张量布局层 | 当前微批中的多条已编码序列 | 一条连续 `token stream` | 消除二维批次中的补齐位置，同时保留原序列边界 |
| `FlashAttention varlen` | 注意力实现层 | 连续 `Q/K/V` 与边界元数据 | 连续输出 | 按边界执行多个彼此独立的注意力问题 |

### 2.1 单独开启 `padding-free`

如果不开启 `packing`，一个 `DataLoader batch` 可能直接是：

```text
[A, B, C]
```

公共 `_data_collator()` 看到 `self.padding_free=True` 后调用：

```python
batch[:] = [self.packing_row(batch)]
```

于是三条序列被直接整理成一个物理行。

### 2.2 同时开启 `packing`

如果开启 `packing`，数据集先把样本组织为若干 `pack`。例如：

```text
PackingDataset item 0 = [A, C]
PackingDataset item 1 = [B, D]
```

当 `per_device_train_batch_size=2` 时，`DataLoader` 交给公共 `data_collator()` 的对象是两层列表：

```text
batch = [
    [A, C],
    [B, D],
]
```

`Template.data_collator()` 会先执行：

```python
if self.packing and isinstance(batch[0], list):
    batch = sum(batch, start=[])
```

列表因此变为：

```text
[A, C, B, D]
```

之后 `_data_collator()` 再通过 `packing_row()` 将四条序列物理拼接。源码见 [`Template.data_collator()`](https://github.com/taking-lying-flat/ms-swift/blob/ed1b2f3742296c4dbf1ddc553cb4ac43ea0c4165/swift/template/base.py#L1714-L1737)。

> [!NOTE]
> `PackingDataset` 产生的 `pack` 边界不是注意力边界。真正的注意力边界仍然是每条原始样本的边界。一个微批可以包含多个 `pack`，公共 `data_collator()` 会将其展开，但每条样本各自重置的 `position_ids` 仍会生成完整的 `cu_seqlens`。

## 3. `ms-swift` 的数据集层：`PackingDataset` 只产生组合关系

### 3.1 数据管线在哪里接入 `PackingDataset`

常规训练管线先用 `LazyLLMDataset` 包装非流式数据，再根据 `args.packing` 选择 `PackingDataset` 或 `IterablePackingDataset`：

```python
if not args.streaming and args.truncation_strategy != 'split':
    dataset = LazyLLMDataset(dataset, template.encode, ...)

if args.packing:
    packing_dataset_cls = (
        IterablePackingDataset if args.streaming else PackingDataset
    )
    dataset = packing_dataset_cls(
        template,
        dataset,
        packing_length=args.packing_length,
        packing_num_proc=args.packing_num_proc,
        packing_strategy=args.packing_strategy,
        ...,
    )
```

这说明装箱依据的是模板编码后的长度，而不是原始字符串长度。对纯文本，它对应最终的语言模型 `token` 数；对多模态，它需要反映视觉占位展开后最终进入语言主干的序列长度。管线入口见 [`sft.py`](https://github.com/taking-lying-flat/ms-swift/blob/ed1b2f3742296c4dbf1ddc553cb4ac43ea0c4165/swift/pipelines/train/sft.py#L125-L156)。

### 3.2 两种装箱策略

`calculate_matched_group()` 支持两种行为不同的策略：

```python
if strategy == 'sequential':
    # order-preserving next-fit
    ...

# default: best-fit-decreasing bin packing
sequences = binpacking.to_constant_volume(
    sequences,
    packing_length,
    weight_pos=1,
)
```

| 策略 | 行为 | 是否保持输入顺序 | 一个典型结果 |
| --- | --- | --- | --- |
| `binpack` | 按长度寻找更合适的组合 | 不保证 | `A=700, C=300` 可以组成同一个 `pack` |
| `sequential` | 维护一个开放 `pack`，下一个样本放不下时立即提交 | 保持 | 样本按原始顺序依次进入 `pack` |

默认 `binpack` 不要求只能合并相邻样本。例如原始顺序为：

```text
A=700, B=600, C=300, D=400
```

它可以形成：

```text
[A, C] = 1000
[B, D] = 1000
```

相关实现见 [`calculate_matched_group()`](https://github.com/taking-lying-flat/ms-swift/blob/ed1b2f3742296c4dbf1ddc553cb4ac43ea0c4165/swift/dataset/packing.py#L16-L47)。

### 3.3 `packed_idx` 才是这一层的主要产物

非流式 `PackingDataset` 的关键过程是：

```text
dataset['lengths']
  → (sample_index, sample_length)
  → calculate_matched_group(...)
  → packed_idx
  → __getitem__ 返回一组原始样本
```

源码中的 `__getitem__()` 很直接：

```python
def __getitem__(self, index):
    sequence = self.packed_idx[index]
    row = [self.dataset[i] for i in sequence]
    return row
```

因此这一层没有生成 `Q/K/V`，也没有执行 `torch.cat(input_ids)`。它只把“一个数据集条目”从单个样本变成样本列表。完整实现见 [`PackingDataset`](https://github.com/taking-lying-flat/ms-swift/blob/ed1b2f3742296c4dbf1ddc553cb4ac43ea0c4165/swift/dataset/packing.py#L50-L134)。

### 3.4 `packing_num_proc` 会影响组合边界

非流式实现先把全量 `lengths` 划分给多个装箱进程，每个进程在自己的长度分片内生成 `packed_idx`。这意味着不同的 `packing_num_proc` 不只改变预处理并发度，也可能改变最终样本组合；不同进程的分片之间不会共同寻找长度互补的样本。

对于要求严格保持全局顺序的 `sequential` 场景，源码注释建议使用 `packing_num_proc=1`。这属于数据顺序语义，不应与后面的注意力隔离混在一起。

### 3.5 `packing_length`、`max_seqlen` 与总 `token` 数不是同一个值

| 名称 | 所在层级 | 含义 |
| --- | --- | --- |
| `packing_length` | 数据集层 | 单个 `pack` 的目标容量，默认使用 `max_length` |
| `max_seqlen_q/k` | 注意力层 | 当前调用中最长的一条逻辑序列长度 |
| `total_q/total_k` | 注意力层 | 当前调用中所有逻辑序列的查询或键值 `token` 总数 |

如果一个微批包含两个 `pack`，物理总长度可以大于单个 `packing_length`；但 `max_seqlen` 仍然只由其中最长的原始逻辑序列决定。不能把 `packing_length` 直接作为 `FlashAttention` 的 `max_seqlen`。

## 4. `ms-swift` 的批处理层：`packing_row()` 到底拼了什么

### 4.1 `packing=True` 会启用 `padding_free`

当前 `SFTConfig` 的检查逻辑为：

```python
if self.padding_free or self.packing:
    if self.packing:
        feature = 'packing'
        self.padding_free = True
    else:
        feature = 'padding_free'

    supported_impls = [
        'flash_attn',
        'flash_attention_2',
        'flash_attention_3',
        'flash_attention_4',
    ]
```

所以当前常规 `ms-swift` 路径中：

```text
packing=True
  → padding_free=True
  → attention implementation 必须属于 FlashAttention 系列
```

这是 `ms-swift` 当前实现选择的后端契约。参数检查见 [`sft_args.py`](https://github.com/taking-lying-flat/ms-swift/blob/ed1b2f3742296c4dbf1ddc553cb4ac43ea0c4165/swift/arguments/sft_args.py#L185-L196)。

### 4.2 `packing_row()` 的字段级行为

`packing_row()` 不是只处理 `input_ids`。它先收集所有样本字段，再按字段类型选择拼接方式：

| 字段 | 处理方式 | 原因 |
| --- | --- | --- |
| `input_ids` | Python 列表顺序拼接 | 形成连续语言模型输入 |
| `labels` | 与 `input_ids` 同序拼接 | 保持逐 `token` 监督对齐 |
| `loss_scale` | 同序拼接 | 保持逐 `token` 权重对齐 |
| 一维 `position_ids` | 同序拼接 | 保留每条样本的位置重置 |
| 三维 `position_ids` | 沿最后一维 `torch.cat` | 支持多模态 `mRoPE` |
| `token_type_ids` | 同序拼接 | 保持类型标记与 `token` 对齐 |
| `mm_token_type_ids` | 沿最后一维 `torch.cat` | 保持多模态位置类型 |
| `channel` | 保留为逐样本列表 | 后续按原样本归组 |
| 图像、视频等多模态数据 | `_data_collator_mm_data()` 聚合 | 交给模板专用逻辑处理 |

核心源码如下：

```python
for key in keys:
    if key == 'position_ids' and is_3d_position_ids \
            or key in {'mm_token_type_ids'}:
        packed[key] = torch.cat([x.get(key) for x in row], dim=-1)
    elif key in {
        'input_ids', 'labels', 'loss_scale',
        'position_ids', 'token_type_ids'
    }:
        packed[key] = sum((x.get(key) or [] for x in row), start=[])
    elif key == 'channel':
        packed[key] = [x.get(key) for x in row]
```

若普通文本样本没有预先构造 `position_ids`，父类会根据每条样本的 `length` 分别生成局部位置：

```python
if 'position_ids' not in packed:
    packed['position_ids'] = sum(
        (list(range(x)) for x in length),
        start=[],
    )
```

完整实现见 [`packing_row()`](https://github.com/taking-lying-flat/ms-swift/blob/ed1b2f3742296c4dbf1ddc553cb4ac43ea0c4165/swift/template/base.py#L721-L742)。

### 4.3 `_data_collator()` 如何把逻辑批次变成一个物理行

当 `self.padding_free=True` 时：

```python
batch[:] = [self.packing_row(batch)]
assert 'position_ids' in batch[0]
```

接着它只保留一个物理行，并将字段转换为张量：

```python
assert len(batch) == 1
for key in ['input_ids', 'channel', ...]:
    value = batch[0].get(key)
    if value is not None:
        result[key] = value if key == 'channel' else [value]
```

这就是最终出现 `[1, total_tokens]` 的原因。物理 `batch dimension=1` 只描述张量布局，不代表逻辑序列数变成 1。

### 4.4 为什么这条路径没有普通 `attention_mask`

父类只有在 `not self.padding_free` 时才会根据 `seq_lens` 创建二维 `attention_mask`：

```python
if not self.padding_free and seq_lens:
    result['attention_mask'] = [
        torch.ones(seq_len, dtype=torch.int64)
        for seq_len in seq_lens
    ]
```

所以标准 `padding-free` 路径不会再构造一个 `[1, total_tokens]` 的全 1 掩码，也不会物化跨样本的块对角稠密掩码。它依赖 `position_ids/cu_seqlens` 进入变长注意力分支。相关逻辑见 [`_data_collator()`](https://github.com/taking-lying-flat/ms-swift/blob/ed1b2f3742296c4dbf1ddc553cb4ac43ea0c4165/swift/template/base.py#L1909-L2024)。

## 5. 从 `position_ids` 生成 `FlashAttention` 边界

### 5.1 `get_packed_seq_params()` 的实际输出

`ms-swift` 的实现为：

```python
def get_packed_seq_params(position_ids):
    assert position_ids.shape[0] == 1
    position_ids_f = position_ids.flatten()
    indices_q = torch.arange(
        position_ids_f.shape[0],
        device=position_ids_f.device,
        dtype=torch.int32,
    )

    cu_seqlens = torch.cat([
        indices_q[position_ids_f == 0],
        torch.tensor(
            position_ids_f.shape,
            device=position_ids_f.device,
            dtype=torch.int32,
        ),
    ])

    max_length = cu_seqlens.diff().max()
    return {
        'cu_seq_lens_q': cu_seqlens,
        'cu_seq_lens_k': cu_seqlens,
        'max_length_q': max_length,
        'max_length_k': max_length,
    }
```

源码见 [`transformers_utils.py`](https://github.com/taking-lying-flat/ms-swift/blob/ed1b2f3742296c4dbf1ddc553cb4ac43ea0c4165/swift/utils/transformers_utils.py#L347-L363)。

普通文本的父类 `collator` 不一定直接调用这个工具函数：它可以只把带重置的 `position_ids` 传给 `Transformers`，再由 `Transformers` 内部生成累计长度。部分多模态或模型专用路径则会在 `collator` 阶段直接调用 `get_packed_seq_params()`，把四个变长参数显式传给模型。两条路径最后生成的是同一种原生 `FlashAttention` 元数据。

对下面的位置序列：

```text
0 1 2 3 4 | 0 1 2 | 0 1 2 3
```

所有 `position_id == 0` 的索引是：

```text
[0, 5, 8]
```

追加总长度 12 后得到：

```text
[0, 5, 8, 12]
```

相邻值的差重新给出三条逻辑序列的长度：

```text
[5, 3, 4]
```

### 5.2 接口层与原生层的命名差异

| 所在层 | 参数名 |
| --- | --- |
| `ms-swift/Transformers` 模型参数 | `cu_seq_lens_q`、`cu_seq_lens_k` |
| `FlashAttention` 原生函数 | `cu_seqlens_q`、`cu_seqlens_k` |

二者描述的是同一组累计边界。`Transformers` 调用原生函数时会完成参数名映射：

```python
flash_varlen_fn(
    q,
    k,
    v,
    cu_seqlens_q=cu_seq_lens_q,
    cu_seqlens_k=cu_seq_lens_k,
    ...,
)
```

### 5.3 `position_ids` 与显式边界的两种入口

新版本 `Transformers` 支持两种进入 `varlen` 的方式：

1. 模型已经传入完整的 `cu_seq_lens_q/k` 和 `max_length_q/k`；
2. 模型只传入能表示多段序列的 `position_ids`，由框架内部构造边界。

框架内部的 `prepare_fa_kwargs_from_position_ids()` 使用每段序列共同的最小起始位置，而不是把起点硬编码为 0。这是为了兼容起始位置不为 0 的模型。它还明确通过 `cu_seqlens.diff()` 计算最长序列，因为多模态位置不一定是普通单调的一维编号。见 [`prepare_fa_kwargs_from_position_ids()`](https://github.com/huggingface/transformers/blob/36deb0b53ed0863f4b4dfdea23dcaec7f3df3701/src/transformers/modeling_flash_attention_utils.py#L458-L533)。

### 5.4 必须同时维护的边界信息

| 元数据 | 消费者 | 作用 |
| --- | --- | --- |
| `position_ids` | `RoPE/mRoPE` | 为每条样本提供自己的局部位置坐标 |
| `cu_seqlens_q/k` | `FlashAttention varlen` | 划分独立的 `Q/K/V` 区间 |
| `max_seqlen_q/k` | `FlashAttention` 调度与边界处理 | 告知本次调用中最长逻辑序列的长度 |
| `labels` | 因果语言模型损失 | 切断拼接边界上的错误预测目标 |
| `loss_scale` | 加权损失 | 保持逐 `token` 权重与拼接顺序一致 |

> [!WARNING]
> 只拼接 `input_ids` 而不按同一顺序拼接 `labels`、`loss_scale` 与位置元数据，会得到形状看似合法但训练语义已经错位的批次。

## 6. `Transformers` 桥接层：两条不同路径为什么都调用 `varlen`

### 6.1 `_flash_attention_forward()` 有三个分支

`Transformers` 的公共前向根据输入状态选择：

| 条件 | 使用的函数 | 前后处理 |
| --- | --- | --- |
| 存在二维 `attention_mask` | `flash_varlen_fn` | 先 `unpad`，计算后再 `pad` |
| 已有显式边界，或 `position_ids` 表示多段序列 | `flash_varlen_fn` | 直接展平并计算，不做恢复补齐 |
| 没有补齐，也没有多段边界 | 普通 `flash_fn` | 作为标准定长输入计算 |

源码中的判断为：

```python
is_fa_with_position_ids = _is_packed_sequence(
    position_ids,
    batch_size=query_states.size(0),
)
is_fa_with_varlen_kwargs = all(
    value is not None
    for value in (
        cu_seq_lens_q,
        cu_seq_lens_k,
        max_length_q,
        max_length_k,
    )
)
```

完整实现见 [`_flash_attention_forward()`](https://github.com/huggingface/transformers/blob/36deb0b53ed0863f4b4dfdea23dcaec7f3df3701/src/transformers/modeling_flash_attention_utils.py#L694-L830)。

### 6.2 普通补齐批次：`unpad → varlen → pad`

存在 `attention_mask` 时，输入仍是标准二维批次。例如：

```text
hidden_states.shape = [3, 5, hidden_size]
attention_mask =
1 1 1 1 1
1 1 1 0 0
1 1 1 1 0
```

`_get_unpad_data()` 依次生成：

| 产物 | 示例 | 用途 |
| --- | --- | --- |
| `seqlens_in_batch` | `[5, 3, 4]` | 每行有效长度 |
| `indices` | 所有非补齐位置在展平矩阵中的索引 | 从补齐张量中抽取有效 `token` |
| `cu_seqlens` | `[0, 5, 8, 12]` | 告诉 `varlen` 每条序列的边界 |
| `max_seqlen_in_batch` | `5` | 传给内核的最长序列长度 |

随后：

```text
padded Q/K/V
  → index_first_axis 抽取有效 token
  → compact Q/K/V
  → flash_varlen_fn
  → pad_fn 按 indices 散射回原二维形状
```

这条路径的 `varlen` 是 `FlashAttention` 对普通补齐批次的内部优化，模型外部仍保持原来的二维批次布局。相关实现见 [`_get_unpad_data()` 与 `_upad_input()`](https://github.com/huggingface/transformers/blob/36deb0b53ed0863f4b4dfdea23dcaec7f3df3701/src/transformers/modeling_flash_attention_utils.py#L351-L455)。

### 6.3 真正的 `padding-free`：已经紧凑，不再 `gather/scatter`

`ms-swift` 的 `padding-free` 输入已经是：

```text
query_states.shape = [1, total_tokens, num_q_heads, head_dim]
attention_mask = None
```

框架只需要把物理批次维压掉：

```python
q = query_states.reshape(-1, query_states.size(-2), query_states.size(-1))
k = key_states.reshape(-1, key_states.size(-2), key_states.size(-1))
v = value_states.reshape(-1, value_states.size(-2), value_states.size(-1))
```

然后直接调用：

```python
out = flash_varlen_fn(
    q,
    k,
    v,
    cu_seqlens_q=cu_seq_lens_q,
    cu_seqlens_k=cu_seq_lens_k,
    max_seqlen_q=max_length_q,
    max_seqlen_k=max_length_k,
    ...,
)
```

输出仍按原来的连续 `token` 顺序排列，再被视图恢复为 `[1, total_tokens, num_heads, head_dim]`。这里不需要 `pad_fn`，因为调用前根本没有补齐位置。

### 6.4 `position_ids` 不是注意力掩码

`_is_packed_sequence()` 只用 `position_ids` 判断当前是否存在多段递增序列，并据此选择 `varlen` 分支。真正进入原生函数的隔离信息已经转换成 `cu_seqlens`。

因此应准确区分：

```text
position_ids reset
  → 表示“这里可能开始一条新序列”

cu_seqlens
  → 表示“Q/K/V 的这两个下标之间是一条独立序列”

causal=True
  → 表示“每条独立序列内部不能读取未来位置”
```

## 7. `FlashAttention varlen` 的原生接口

### 7.1 `Q/K/V` 的实际形状

`flash_attn_varlen_func()` 接收：

| 参数 | 形状 | 含义 |
| --- | --- | --- |
| `q` | `[total_q, num_q_heads, head_dim]` | 所有逻辑查询序列的连续存储 |
| `k` | `[total_k, num_kv_heads, head_dim]` | 所有逻辑键序列的连续存储 |
| `v` | `[total_k, num_kv_heads, head_dim]` | 所有逻辑值序列的连续存储 |

`total_q` 是所有查询 `token` 的总数，不是最长序列长度；`total_k` 同理。普通 `decoder-only self-attention` 通常满足：

```text
total_q == total_k
cu_seqlens_q == cu_seqlens_k
```

接口仍将 `Q` 和 `K/V` 的边界分开，是因为它还支持查询长度与键值长度不同的注意力。

`num_kv_heads` 可以小于 `num_q_heads`。这对应 `MQA/GQA`：多个查询头共享较少的键值头，只要查询头数量能够被键值头数量整除。官方接口说明见 [`flash_attn_varlen_func()`](https://github.com/Dao-AILab/flash-attention/blob/0251105a2fb19d2957484b7f023cd8c115286ced/flash_attn/flash_attn_interface.py#L1391-L1482)。

### 7.2 边界参数逐项解释

| 参数 | 示例 | 准确含义 |
| --- | --- | --- |
| `cu_seqlens_q` | `[0, 3, 8, 10]` | `Q` 中每条逻辑序列的累计边界 |
| `cu_seqlens_k` | `[0, 3, 8, 10]` | `K/V` 中每条逻辑序列的累计边界 |
| `max_seqlen_q` | `5` | 本次调用最长的查询序列长度 |
| `max_seqlen_k` | `5` | 本次调用最长的键值序列长度 |

对长度 `[3, 5, 2]`：

```text
q:
| A A A | B B B B B | C C |

index:
0       3           8     10

cu_seqlens_q:
[0, 3, 8, 10]
```

必须始终成立：

```text
q.shape[0] == cu_seqlens_q[-1]
k.shape[0] == cu_seqlens_k[-1]
v.shape[0] == cu_seqlens_k[-1]
len(cu_seqlens_q) == logical_batch_size + 1
len(cu_seqlens_k) == logical_batch_size + 1
max_seqlen_q == max(diff(cu_seqlens_q))
max_seqlen_k == max(diff(cu_seqlens_k))
```

### 7.3 `causal`、序列隔离与 `window_size` 是三件事

| 约束 | 由谁决定 | 作用范围 |
| --- | --- | --- |
| 不同样本互不可见 | `cu_seqlens_q/k` | 逻辑序列之间 |
| 当前 `token` 不读取未来 `token` | `causal=True` | 每条逻辑序列内部 |
| 当前 `token` 只读取有限邻域 | `window_size=(left, right)` | 每条逻辑序列内部 |

例如：

```text
cu_seqlens = [0, 5, 8, 12]
causal = True
window_size = (4, 0)
```

其含义是：

- `A/B/C` 先由 `cu_seqlens` 分成三条序列；
- 每条序列内部使用因果方向；
- 每个查询最多读取自己和左侧 4 个位置；
- 窗口不会跨越 `A/B/C` 的边界。

`window_size` 最终被拆为 `window_size_left/right` 传入原生内核。`softcap`、`alibi_slopes`、`dropout_p` 等参数也会沿同一调用链传递，但它们不负责恢复样本边界。

原生包装层还包含一些与边界兼容、但不由 `packing` 产生的参数：

| 参数 | 作用 | 与训练 `packing` 的关系 |
| --- | --- | --- |
| `dropout_p` | 控制注意力概率随机失活 | 每个逻辑注意力问题内部应用 |
| `softmax_scale` | 指定注意力分数缩放 | 不改变逻辑边界 |
| `softcap` | 对注意力分数做上界软限制 | 不改变逻辑边界 |
| `alibi_slopes` | 添加按头或按批次的 `ALiBi` 偏置 | 必须与逻辑批次维匹配 |
| `block_table` | 使用分页 `K/V` 存储 | 常规训练自注意力通常不使用 |
| `leftpad_k` | 描述每条键值序列的左侧补齐偏移 | 标准 `padding-free self-attention` 通常为空 |
| `seqused_k` | 指定每条键值序列实际使用的长度 | 可进一步限制 `K/V` 有效范围 |
| `num_splits` | 控制部分前向调度策略 | 不提供样本边界 |

公共 `flash_attn_varlen_func()` 主要暴露训练常用参数；更底层的 `_flash_attn_varlen_forward()` 再把 `window_size` 拆成左右两个标量，并携带 `block_table/leftpad_k/seqused_k/num_splits` 调用扩展模块。

### 7.4 Python 到 CUDA 的调用链

```text
Transformers._flash_attention_forward
  → flash_attn_varlen_func
  → FlashAttnVarlenFunc.apply
  → _flash_attn_varlen_forward
  → flash_attn_gpu.varlen_fwd
  → C++ mha_varlen_fwd
  → set_params_fprop
  → CUDA forward kernel
```

Python 包装层会先确保 `q/k/v` 满足连续性要求，再把边界、最长序列、窗口和因果参数传入 `flash_attn_gpu.varlen_fwd`。入口见 [`_flash_attn_varlen_forward()`](https://github.com/Dao-AILab/flash-attention/blob/0251105a2fb19d2957484b7f023cd8c115286ced/flash_attn/flash_attn_interface.py#L153-L180)。

## 8. `FlashAttention` 内核如何读取逻辑边界

### 8.1 C++ 入口先建立参数契约

`mha_varlen_fwd()` 在启动内核前检查：

| 检查项 | 要求 |
| --- | --- |
| `q/k/v` 精度 | `fp16` 或 `bf16`，且三者一致 |
| `cu_seqlens_q/k` 类型 | `int32` |
| 设备 | `q/k/v` 与累计长度位于设备侧 |
| 连续性 | `q/k/v` 最后一维连续，累计长度张量连续 |
| 逻辑批次 | `batch_size = cu_seqlens_q.numel() - 1` |
| 头维 | 当前该入口要求不超过 256，且满足内核对齐条件 |
| `GQA/MQA` | `num_q_heads` 必须能被 `num_kv_heads` 整除 |

完成检查后，`set_params_fprop()` 将 `cu_seqlens_q/k` 的设备指针、`max_seqlen_q/k`、头数、步长与窗口参数写入 `Flash_fwd_params`。见 [`flash_api.cpp`](https://github.com/Dao-AILab/flash-attention/blob/0251105a2fb19d2957484b7f023cd8c115286ced/csrc/flash_attn/flash_api.cpp#L539-L653) 和 [参数设置](https://github.com/Dao-AILab/flash-attention/blob/0251105a2fb19d2957484b7f023cd8c115286ced/csrc/flash_attn/flash_api.cpp#L699-L720)。

### 8.2 `BlockInfo` 从累计长度恢复起点与实际长度

每个逻辑批次索引 `bidb` 都会构造一个 `BlockInfo`：

```cpp
sum_s_q = params.cu_seqlens_q[bidb];
sum_s_k = params.cu_seqlens_k[bidb];

actual_seqlen_q =
    params.cu_seqlens_q[bidb + 1] - sum_s_q;

actual_seqlen_k =
    params.cu_seqlens_k[bidb + 1] - sum_s_k;
```

`sum_s_q` 和 `sum_s_k` 是当前逻辑序列在连续缓冲区中的起点；相邻边界之差是实际长度。

随后指针偏移为：

```cpp
q_offset = sum_s_q * q_row_stride;
k_offset = sum_s_k * k_row_stride;
```

对 `[0, 5, 8, 12]`：

| `bidb` | `sum_s_q` | `actual_seqlen_q` | `q_offset` 指向 |
| ---: | ---: | ---: | --- |
| 0 | 0 | 5 | `A` 的第一个查询 |
| 1 | 5 | 3 | `B` 的第一个查询 |
| 2 | 8 | 4 | `C` 的第一个查询 |

这一步是“物理连续、逻辑隔离”的内核级实现，而不是概念类比。源码见 [`block_info.h`](https://github.com/Dao-AILab/flash-attention/blob/0251105a2fb19d2957484b7f023cd8c115286ced/csrc/flash_attn/src/block_info.h#L12-L45)。

### 8.3 查询块和键块都使用局部实际长度

前向内核创建 `BlockInfo` 后，首先检查当前查询块是否已经超过本序列的 `actual_seqlen_q`：

```cpp
if (m_block * kBlockM >= binfo.actual_seqlen_q) return;
```

键块上界则根据 `actual_seqlen_k` 计算；若启用 `causal` 或局部窗口，还会继续收紧可访问的键块范围。

这意味着：

- `max_seqlen_q/k` 为启动和模板选择提供本批上界；
- `actual_seqlen_q/k` 决定每条逻辑序列真正执行哪些块；
- `sum_s_q/k` 决定这些块从连续缓冲区的哪个位置开始；
- `causal/window` 只在当前逻辑切片内部收紧可见范围。

内核片段见 [`flash_fwd_kernel.h`](https://github.com/Dao-AILab/flash-attention/blob/0251105a2fb19d2957484b7f023cd8c115286ced/csrc/flash_attn/src/flash_fwd_kernel.h#L87-L105)。

### 8.4 输出为什么仍保持拼接顺序

`out` 与 `q` 使用相同的逻辑起点。内核写回第 `bidb` 条序列时，同样通过 `q_offset()` 找到它在连续输出缓冲区中的位置。因此输出布局为：

```text
O = [O_A | O_B | O_C]
```

标准 `padding-free` 路径只把它重新视为 `[1, total_tokens, ...]`，不会恢复成三行补齐张量。后续残差、`MLP` 与语言模型头继续按这条连续 `token` 维处理；需要逐样本语义的模块必须继续携带或正确使用边界。

### 8.5 新版 `CuTe` 内核仍使用同一契约

较新的 `FlashAttention CuTe` 实现中，`SeqlenInfoQK.create()` 同样执行：

```text
offset_q = cu_seqlens_q[batch_idx]
offset_k = cu_seqlens_k[batch_idx]
seqlen_q = cu_seqlens_q[batch_idx + 1] - offset_q
seqlen_k = cu_seqlens_k[batch_idx + 1] - offset_k
```

随后 `offset_batch_Q/K()` 通过这些值对 `ragged tensor` 做指针偏移。见 [`seqlen_info.py`](https://github.com/Dao-AILab/flash-attention/blob/0251105a2fb19d2957484b7f023cd8c115286ced/flash_attn/cute/seqlen_info.py#L83-L190)。

因此从经典 `FlashAttention-2` 到较新的 `CuTe/FlashAttention-4` 路径，`packing/padding-free` 所依赖的核心数据契约没有变化：连续 `Q/K/V` 加显式累计边界。

### 8.6 反向传播继续使用同一组边界

`FlashAttnVarlenFunc.forward()` 在需要梯度时，会连同 `q/k/v`、前向输出和 `softmax_lse` 一起保存 `cu_seqlens_q/k`。进入 `backward()` 后，原来的累计长度与 `max_seqlen_q/k` 被原样传给 `_flash_attn_varlen_backward()`：

```python
ctx.save_for_backward(
    q,
    k,
    v,
    out,
    softmax_lse,
    cu_seqlens_q,
    cu_seqlens_k,
    rng_state,
)

_wrapped_flash_attn_varlen_backward(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    dq,
    dk,
    dv,
    cu_seqlens_q,
    cu_seqlens_k,
    ctx.max_seqlen_q,
    ctx.max_seqlen_k,
    ...,
)
```

所以样本隔离不是只存在于前向。`dQ/dK/dV` 同样按照原逻辑序列边界计算并写回连续梯度缓冲区。源码见 [`FlashAttnVarlenFunc`](https://github.com/Dao-AILab/flash-attention/blob/0251105a2fb19d2957484b7f023cd8c115286ced/flash_attn/flash_attn_interface.py#L914-L1015)。

## 9. 从数据集到内核的逐阶段形状变化

以下表格把同一个 `[A=5, B=3, C=4]` 示例贯穿到底：

| 阶段 | 主要对象 | 形状 / 内容 | 边界保存位置 |
| --- | --- | --- | --- |
| 模板编码后 | 三个样本字典 | `A(5)`、`B(3)`、`C(4)` | 三个独立 Python 对象 |
| `PackingDataset` | 样本索引组 | 例如 `[[A,B], [C]]` | 列表嵌套关系 |
| 公共 `data_collator()` | 展开的样本列表 | `[A,B,C]` | 样本仍是独立字典 |
| `packing_row()` | 拼接字段 | `A|B|C` | 每条样本的 `position_ids` 重置 |
| `_data_collator()` | 张量批次 | `input_ids.shape=[1,12]` | `position_ids=[0..4|0..2|0..3]` |
| 边界构造 | 累计长度 | `[0,5,8,12]` | `cu_seq_lens_q/k` |
| `QKV projection` | 连续注意力输入 | `q.shape=[12,Hq,D]`、`k/v.shape=[12,Hkv,D]` | 边界作为额外参数传递 |
| C++ 参数层 | `Flash_fwd_params` | 指针、步长、最大长度 | `cu_seqlens_q/k` 设备指针 |
| CUDA 内核 | `BlockInfo(bidb)` | 起点与实际长度 | 相邻累计边界之差 |
| 输出 | 连续隐藏状态 | `O_A|O_B|O_C` | 物理顺序不变 |

完整链路为：

```text
raw sample
  → template.encode
  → encoded sample + length
  → PackingDataset groups indices
  → DataLoader returns one or more packs
  → Template.data_collator flattens nested pack lists
  → packing_row concatenates token-aligned fields
  → _data_collator creates one physical row
  → position reset points become cu_seqlens
  → model computes packed Q/K/V
  → Transformers selects flash_varlen_fn
  → FlashAttention validates metadata
  → BlockInfo offsets each logical sequence
  → causal/window attention runs inside each slice
  → output remains in the same packed token order
```

## 10. 因果语言模型的监督边界也必须同步处理

注意力隔离正确，不代表语言模型损失自然正确。标准因果语言模型会使用当前位置的输出预测下一个位置；物理拼接后，`A` 的最后一个位置紧邻 `B` 的第一个位置。

如果 `B` 的首个 `label` 没有被忽略，就会产生错误的：

```text
A_last → B_first
```

`ms-swift` 在每条样本独立编码完成时将该样本的第一个 `label` 设为 `-100`，`packing_row()` 随后按原样拼接所有 `labels`：

```text
tokens:
A0 A1 A2 | B0 B1 B2

labels:
-  A1 A2 | -  B1 B2
```

所以每条逻辑样本的开头都保留一个损失断点。相关处理见 [`base.py`](https://github.com/taking-lying-flat/ms-swift/blob/ed1b2f3742296c4dbf1ddc553cb4ac43ea0c4165/swift/template/base.py#L1545-L1572)。

这一点说明完整的 `padding-free` 正确性至少包含两套边界：

| 边界 | 表示形式 | 防止的问题 |
| --- | --- | --- |
| 注意力边界 | `cu_seqlens_q/k` | 不同样本互相读取隐藏状态 |
| 监督边界 | 每条样本首位置的 `label=-100` | 前一条样本末尾预测后一条样本开头 |

## 11. `Qwen2-VL`：多模态 `packing` 的完整路径

### 11.1 装箱长度必须对应最终语言主干序列

多模态样本在原始数据中可能只有一个 `<image>` 标记，但它不会只占语言主干中的一个位置。`Qwen2VLTemplate._encode()` 先调用视觉处理器得到 `image_grid_thw/video_grid_thw`，再根据 `merge_size` 计算需要展开的视觉占位数量：

```python
merge_length = processor.image_processor.merge_size ** 2

def _get_new_tokens(i):
    token_len = media_grid_thw[i].prod() // merge_length
    return [media_token] * token_len
```

随后 `_extend_tokens()` 同时扩展：

- `input_ids`；
- `labels`；
- `loss_scale`；
- 多模态位置掩码。

实现见 [`Qwen2VLTemplate._encode()`](https://github.com/taking-lying-flat/ms-swift/blob/ed1b2f3742296c4dbf1ddc553cb4ac43ea0c4165/swift/template/templates/qwen.py#L381-L426)。

因此，多模态 `packing` 的长度对象是：

```text
最终 LLM-visible text token
+ 最终 LLM-visible visual/video token position
+ 模板特殊 token
```

而不是原始文本长度，也不是视觉编码器内部未经合并的全部 `patch`。

### 11.2 必须先逐样本计算 `mRoPE`，再执行拼接

普通文本可以直接为每条样本生成 `0...L-1`。`Qwen2-VL` 的位置不仅包含文本顺序，还包含视觉时间、高度和宽度坐标。

因此它重写 `packing_row()`：

```python
def packing_row(self, row):
    for sample in row:
        sample_copy = sample.copy()
        sample_copy['input_ids'] = torch.tensor(
            sample_copy['input_ids']
        )[None]
        sample.update(self._get_position_ids(sample_copy))

    return super().packing_row(row)
```

顺序非常关键：

```text
sample A → get_rope_index(A)
sample B → get_rope_index(B)
sample C → get_rope_index(C)
           ↓
再沿 sequence 维拼接 A/B/C
```

如果先把 `A/B/C` 当成一条长序列再计算 `mRoPE`，后一条样本会继承前一条样本的位置状态，逻辑边界已经丢失。

### 11.3 为什么同时存在 `text_position_ids` 与 `mRoPE position_ids`

模型的 `get_rope_index()` 返回多维位置。模板再调用：

```python
def _concat_text_position_ids(position_ids):
    seq_len = position_ids.shape[-1]
    text_position_ids = torch.arange(
        seq_len,
        device=position_ids.device,
    ).expand(1, *position_ids.shape[1:])
    return torch.concat([text_position_ids, position_ids], dim=0)
```

这会在多维 `mRoPE` 位置前增加一层普通文本顺序位置：

```text
combined position planes
  ├── plane 0: text_position_ids
  └── remaining planes: mRoPE T/H/W positions
```

因为 `_concat_text_position_ids()` 是逐样本调用的，每条样本的 `text_position_ids` 都独立从 0 开始。拼接后第一层自然携带完整逻辑边界。实现见 [`_concat_text_position_ids()`](https://github.com/taking-lying-flat/ms-swift/blob/ed1b2f3742296c4dbf1ddc553cb4ac43ea0c4165/swift/template/base.py#L2343-L2347)。

### 11.4 `_data_collator()` 将两种位置用途拆开

多模态 `collator` 在父类完成拼接后执行：

```python
position_ids = result['position_ids']
result['position_ids'] = position_ids[1:]
result['text_position_ids'] = text_position_ids = position_ids[0]

result.update(get_packed_seq_params(text_position_ids))
```

于是：

| 输出字段 | 后续用途 |
| --- | --- |
| `position_ids` | 交给语言主干，对 `Q/K` 应用多维 `mRoPE` |
| `text_position_ids` | 标出每条多模态样本从哪里重新开始 |
| `cu_seq_lens_q/k` | 交给 `FlashAttention varlen` 隔离不同样本 |
| `max_length_q/k` | 提供最长逻辑多模态序列长度 |

对应实现见 [`Qwen2VLTemplate.packing_row()` 与 `_data_collator()`](https://github.com/taking-lying-flat/ms-swift/blob/ed1b2f3742296c4dbf1ddc553cb4ac43ea0c4165/swift/template/templates/qwen.py#L459-L512)。

### 11.5 不同 `Transformers` 版本怎样把文本边界送入注意力

`Qwen2VLTemplate` 对新旧 `Transformers` 采用两种桥接方式：

| 条件 | 边界传递方式 |
| --- | --- |
| `Transformers >= 4.53.0.dev` | `collator` 调用 `get_packed_seq_params(text_position_ids)`，显式加入 `cu_seq_lens_q/k` 和 `max_length_q/k` |
| 更早版本 | `forward_context()` 临时包装模型模块的 `_flash_attention_forward`，把 `text_position_ids` 注入 `position_ids` 参数 |

旧版本路径的核心包装为：

```python
def _flash_attention_forward(*args, **kwargs):
    kwargs['position_ids'] = position_ids
    return original_flash_attention_forward(*args, **kwargs)
```

其目的不是修改 `mRoPE`，而是确保旧模型实现调用 `FlashAttention` 时仍能看见用于分段的文本位置。上下文退出后，原始 `_flash_attention_forward` 会被恢复，避免永久修改模块函数。见 [`forward_context()`](https://github.com/taking-lying-flat/ms-swift/blob/ed1b2f3742296c4dbf1ddc553cb4ac43ea0c4165/swift/template/templates/qwen.py#L428-L438) 与 [`_patch_flash_attention_forward()`](https://github.com/taking-lying-flat/ms-swift/blob/ed1b2f3742296c4dbf1ddc553cb4ac43ea0c4165/swift/template/base.py#L2261-L2285)。

新版本路径不再依赖运行时包装，而是在 `collator` 中直接生成显式累计长度。两条路径的终点仍然都是 `Transformers` 的 `padding-free varlen` 分支。

### 11.6 视觉特征何时替换占位位置

在训练前向前，`_post_encode()` 先得到文本 `embedding`，再调用 `_get_inputs_embeds_hf()` 运行视觉模块并把视觉特征写入相应占位位置：

```text
packed input_ids
  → text embedding
  → visual encoder produces image/video embeddings
  → image/video token mask locates placeholder positions
  → masked_scatter replaces placeholders
  → packed multimodal inputs_embeds
```

这一步不改变已经建立的语言序列顺序和 `cu_seqlens`。视觉特征只是替换对应位置的向量；不同多模态样本在语言主干中仍由相同的累计长度隔离。相关入口见 [`_post_encode()`](https://github.com/taking-lying-flat/ms-swift/blob/ed1b2f3742296c4dbf1ddc553cb4ac43ea0c4165/swift/template/templates/qwen.py#L440-L450)。

### 11.7 多模态端到端链路

```text
image / video / text
  → Qwen2VLTemplate._encode
  → image_grid_thw / video_grid_thw
  → visual placeholder expands to LLM-visible positions
  → each sample has its final LLM sequence length
  → PackingDataset groups complete multimodal samples
  → Qwen2VLTemplate.packing_row
  → get_rope_index runs separately for every sample
  → text position plane + mRoPE T/H/W planes
  → concatenate samples along sequence dimension
  → _data_collator separates text_position_ids and mRoPE positions
  → text_position_ids generate cu_seqlens
  → _post_encode replaces visual placeholders with embeddings
  → Q/K receive mRoPE
  → FlashAttention varlen receives cu_seqlens
  → different multimodal samples stay isolated
```

> [!IMPORTANT]
> 这里被装箱的是最终进入语言主干的完整多模态序列。视觉编码器内部自己的 `patch sequence`、注意力实现和批处理方式属于另一条计算图，不能把语言主干的 `packing + FlashAttention varlen` 直接等同为视觉编码器也执行了相同装箱。

## 12. 混合架构不能只检查全注意力层

如果模型除了标准全注意力，还包含沿序列传播状态的模块，那么这些模块也必须理解相同边界。典型对象包括：

- `causal convolution`；
- `linear attention`；
- `gated delta rule`；
- 状态空间递推；
- 自定义 `sequence parallel`；
- 任何跨序列位置聚合的后处理。

例如把 `[A|B]` 直接交给不支持变长边界的 `causal convolution`，`B` 开头会读取 `A` 尾部。即使全注意力层已经正确调用 `FlashAttention varlen`，整个模型仍然存在跨样本污染。

当前分析版本中的 `Qwen3.5` 特殊路径会：

1. 从 `kwargs['cu_seq_lens_q']` 取得累计边界；
2. 把边界传给变长 `causal_conv1d`；
3. 把边界传给 `gated delta rule`；
4. 当存在多条逻辑序列但相应变长内核不可用时，拒绝回退到忽略边界的普通实现。

见 [内核选择约束](https://github.com/taking-lying-flat/ms-swift/blob/ed1b2f3742296c4dbf1ddc553cb4ac43ea0c4165/swift/model/models/qwen.py#L1235-L1247)、[`causal_conv1d` 调用](https://github.com/taking-lying-flat/ms-swift/blob/ed1b2f3742296c4dbf1ddc553cb4ac43ea0c4165/swift/model/models/qwen.py#L1250-L1259) 与 [`gated delta rule` 边界传递](https://github.com/taking-lying-flat/ms-swift/blob/ed1b2f3742296c4dbf1ddc553cb4ac43ea0c4165/swift/model/models/qwen.py#L1336-L1381)。

这给出一个更严格的支持标准：

> [!WARNING]
> 命令行接受 `--packing true` 或 `--attn_impl flash_attn`，只证明配置检查通过；只有所有跨序列传播信息的模块都消费同一套边界，才能证明整个模型支持 `padding-free`。

## 13. 最终理解

整条机制可以按下面五层理解：

| 层 | 做了什么 | 保留下来的关键信息 |
| --- | --- | --- |
| 模板编码 | 把原始文本或多模态输入变成最终语言序列 | 每条样本的 `input_ids/labels/length` |
| `PackingDataset` | 根据长度选择样本组合 | 原始样本仍是独立对象 |
| `padding-free collator` | 将多个对象拼成一个物理行 | 每条样本重置的位置和监督边界 |
| `Transformers` | 将位置边界映射为原生变长参数 | `cu_seqlens/max_seqlen` |
| `FlashAttention` | 按累计边界偏移指针并执行局部注意力 | 每条逻辑序列独立的起点与实际长度 |

> [!TIP]
> `packing` 决定“哪些完整样本一起进入一个数据项”；`padding-free` 决定“这些样本怎样连续存储”；`cu_seqlens` 决定“连续存储中的逻辑边界”；`FlashAttention varlen` 则让内核真正按照这些边界执行彼此独立的注意力。

## 源码版本

| 项目 | 分析版本 | 关键源码 |
| --- | --- | --- |
| `ms-swift` | [`ed1b2f374`](https://github.com/taking-lying-flat/ms-swift/tree/ed1b2f3742296c4dbf1ddc553cb4ac43ea0c4165) | `packing.py`、`template/base.py`、`template/templates/qwen.py` |
| `Transformers` | [`36deb0b53`](https://github.com/huggingface/transformers/tree/36deb0b53ed0863f4b4dfdea23dcaec7f3df3701) | `modeling_flash_attention_utils.py` |
| `FlashAttention` | [`0251105a2`](https://github.com/Dao-AILab/flash-attention/tree/0251105a2fb19d2957484b7f023cd8c115286ced) | `flash_attn_interface.py`、`flash_api.cpp`、`block_info.h`、`seqlen_info.py` |
