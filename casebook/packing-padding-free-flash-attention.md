# `Packing`、`Padding-Free` 与 `FlashAttention Varlen`

## 1. Mental Model：物理连续，逻辑不连续

`padding-free` 描述的是**LLM token 的物理布局**。传统 batch 把不同长度的样本补齐到统一宽度；padding-free 则删除这些补齐位置，只保留有效 token，并把它们紧凑地放进同一条 flat token stream。它减少的是 padding 对显存、带宽和算力的占用，本身并不定义 token 之间能否互相 attention

`packing` 描述的是**完整样本如何组合**。它以模板编码后的 LLM token 数为长度，将若干条能够放入同一预算的 sample 组成一个 pack；packing 改变的是 DataLoader 一次交付哪些样本，不会把这些样本合并成一条新的训练语义；同一个 pack 中的 sample 仍是独立序列

`FlashAttention varlen` 描述的是**kernel 如何解释连续的 Q/K/V 内存**。它不要求重新构造带 padding 的二维 batch，而是读取 `cu_seqlens`，为每个 logical sequence 恢复独立的起点和长度。同一块连续 buffer 因而可以承载多个彼此隔离的 attention problem

> [!IMPORTANT]
> padding-free 只建立物理连续性，`cu_seqlens` 才重新建立逻辑不连续性

假设 `A`、`B`、`C` 的 LLM token 长度分别为 5、3、4，packing 与 padding-free 得到的物理布局及其逻辑解释如下

```text
sample space        A: len=5        B: len=3       C: len=4
packing plan        pack_0 = [A, B, C]

physical offset     0              5        8             12
                    │              │        │              │
flat token memory   ├─ A A A A A ──┼─ B B B ┼─ C C C C ───┤
position_ids        │  0 1 2 3 4   │  0 1 2 │  0 1 2 3   │
cu_seqlens          └───────[ 0,       5,       8,      12 ]

kernel view         bidb=0: Q/K/V[0:5]      → Attention(A)
                    bidb=1: Q/K/V[5:8]      → Attention(B)
                    bidb=2: Q/K/V[8:12]     → Attention(C)

output memory       ├──── O_A ─────┼── O_B ──┼──── O_C ────┤
```

`A_last` 与 `B_first` 在 flat memory 中相邻，只表示它们的地址连续，不表示二者具有语言模型意义上的前后继关系；kernel 对每个 `bidb` 分别取得 `cu_seqlens[bidb]` 与 `cu_seqlens[bidb + 1]`，由这两个累计偏移确定当前 Q/K/V slice，因此计算的是三次独立的 attention，输出再按原来的物理顺序连续写回

三个约束相互独立

- 物理布局；`input_ids`、`labels`、`position_ids` 等 token-aligned metadata 使用同一 flatten 顺序；Q/K/V 与输出共享长度为 `T` 的连续 token 轴
- 逻辑边界；`cu_seqlens = [0, 5, 8, 12]` 把连续地址解释为 `[0, 5)`、`[5, 8)`、`[8, 12)`；每个区间拥有独立的 Q/K/V base pointer 与 actual sequence length
- 序列内可见性；`causal=True` 只在当前 `cu_seqlens` slice 内限制未来 token；causal 规则不会推断 sample boundary 或修复错误切片

> [!NOTE]
> `position_ids` reset 可以作为构造边界的上游信号，但真正进入 varlen kernel 并隔离内存访问的是 `cu_seqlens`

从数学语义看，这等价于 block-diagonal causal attention；从实现看，FlashAttention 不会物化 `[T, T]` block-diagonal mask，而是直接把每个 `bidb` 的 Q/K/V 地址重定位到对应 slice；若累计边界从 `[0, 5, 8, 12]` 变成 `[0, 5, 12]`，flat memory 没有发生任何变化，kernel 看到的逻辑问题却从 `A / B / C` 变成 `A / (B+C)`，causal attention 随后只会在这个已经合并的 slice 内正确执行

## 2. ms-swift：Packing 与 Padding-Free 如何落地

在 ms-swift 中，packing 与 padding-free 共用同一条 flat-token 实现，但两者介入的阶段不同：`padding_free` 直接把当前 DataLoader batch 中的完整 sample 压成一行；`packing` 则先根据编码后的长度重新规划 sample 组合，再把组合结果交给相同的 padding-free collator

> [!IMPORTANT]
> 在 ms-swift 中启用 `packing` 会同时启用 `padding_free`，两者都要求使用 FlashAttention 实现

这个约束在参数初始化阶段直接执行；`packing=True` 会先设置 `padding_free=True`；如果 `attn_impl` 不属于 FlashAttention family，训练会在进入数据管线前抛出异常，而不是回退到 SDPA 或 eager attention

```python
def _check_padding_free(self):
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
        if self.attn_impl not in supported_impls:
            raise ValueError(
                f'The "{feature}" feature requires '
                'a flash attention implementation.'
            )
```

原因不是 FlashAttention 单纯更快，而是 flatten 之后已经不存在传统的 `[B, S]` padded layout；attention 必须接收 sample boundary，把一条物理 token 轴重新解释成多个 varlen sequence；ms-swift 选择的执行 contract 就是 FlashAttention varlen

非流式训练首先执行 `template.encode`。因此 packing 使用的长度不是原始文本字符数或 tokenizer 之前的估计值，而是模板处理结束后的 LLM token 长度；多模态样本也已经完成 LLM-visible 占位符展开

```python
if not args.streaming and args.truncation_strategy != 'split':
    dataset = LazyLLMDataset(
        dataset,
        template.encode,
        strict=args.strict,
        random_state=args.data_seed,
    )

if args.packing:
    dataset = PackingDataset(
        template,
        dataset,
        packing_length=args.packing_length,
        packing_num_proc=args.packing_num_proc,
        packing_strategy=args.packing_strategy,
        ...
    )
```

`PackingDataset` 只解决“哪些完整 sample 放在一起”。它从 `dataset['lengths']` 建立 `(sample_index, encoded_length)`，在 `packing_length` 预算内生成 `packed_idx`，但不会在 dataset 层拼接 `input_ids`、构造 `position_ids` 或生成 Q/K/V

- `binpack`；默认使用 best-fit-decreasing，提高每个 pack 的 token 填充率；允许改变 sample 的组合顺序
- `sequential`；使用 order-preserving next-fit；若要求单一全局顺序，需要同时设置 `packing_num_proc=1`

```python
data = [
    (i + offset, sum(length) if isinstance(length, list) else length)
    for i, length in enumerate(lengths)
]

sequences, input_data = calculate_matched_group(
    input_data,
    packing_length,
    is_finished=is_finished,
    strategy=packing_strategy,
)

def __getitem__(self, index):
    sequence = self.packed_idx[index]
    return [self.dataset[i] for i in sequence]
```

> [!NOTE]
> `PackingDataset` 内部保存的是 sample index list，`__getitem__()` 只按索引取回完整 sample；真正的 token 拼接发生在 collator

DataLoader 取出的一个 packing item 仍是 `List[Dict]`；`data_collator()` 先展开 batch 中的 pack list，随后只要 `padding_free=True`，就调用一次 `packing_row()`，把所有完整 sample 压成一个 packed row

```text
encoded dataset
  ├─ sample A: length=5
  ├─ sample B: length=3
  ├─ sample C: length=4
  └─ sample D: length=6

PackingDataset
  ├─ pack 0: [A, C]
  └─ pack 1: [B, D]

data_collator
  └─ [A, C, B, D]
       → packing_row
       → one flat row: [A | C | B | D]
```

```python
def data_collator(self, batch, *, padding_to=None):
    if self.packing and isinstance(batch[0], list):
        batch = sum(batch, start=[])

    return self._data_collator(batch, padding_to=padding_to)

def _data_collator(self, batch, *, padding_to=None):
    if self.padding_free:
        batch[:] = [self.packing_row(batch)]
        assert 'position_ids' in batch[0]
```

`packing_row()` 才执行 token 级 flatten。普通一维字段使用相同 sample 顺序拼接；三维 `position_ids` 与 `mm_token_type_ids` 沿最后一个 sequence dimension 拼接；缺少显式 `position_ids` 时，则为每条 sample 分别生成从 0 开始的局部位置

```python
def packing_row(self, row):
    packed = {}
    length = [sample['length'] for sample in row]

    for key in keys:
        if key == 'position_ids' and is_3d_position_ids \
                or key == 'mm_token_type_ids':
            packed[key] = torch.cat(
                [sample.get(key) for sample in row],
                dim=-1,
            )
        elif key in {
            'input_ids', 'labels', 'loss_scale',
            'position_ids', 'token_type_ids',
        }:
            packed[key] = sum(
                (sample.get(key) or [] for sample in row),
                start=[],
            )

    if 'position_ids' not in packed:
        packed['position_ids'] = sum(
            (list(range(seq_len)) for seq_len in length),
            start=[],
        )

    return packed
```

> [!IMPORTANT]
> `packing_row()` 的核心不是 concatenate，而是所有 token-aligned metadata 必须保持同一个 physical permutation

flatten 后总 token 数为 `T` 时，下列字段必须指向同一条 token 轴

- `len(input_ids) == T`
- `len(labels) == T`
- `len(loss_scale) == T`，字段存在时
- `len(token_type_ids) == T`，字段存在时
- `position_ids.shape[-1] == T`
- `mm_token_type_ids.shape[-1] == T`，字段存在时

packing 也不会自动创造 next-token loss boundary；ms-swift 在每条 sample 的编码阶段已经将首个 label 设为 `-100`，`packing_row()` 只是保留这些 sample-local ignore position；loss 执行 shift 后，前一条 sample 的最后一个 logit 对齐到的是下一条 sample 的 `-100`

```text
input_ids        A0 A1 A2 | B0 B1
labels           -1 A1 A2 | -1 B1       # -1 表示 -100
shifted target   A1 A2 -1 | B1 -1
                         ↑
                  A_last 不监督 B_first
```

> [!NOTE]
> pack boundary 只服务于数据组合；进入模型的是 sample boundary；多个 pack 被 collator 展开后，各条 sample 仍分别重启局部位置并保留 loss boundary

## 3. Boundary：`position_ids` 如何编译成 `cu_seqlens`

padding-free flatten 删除了原来的 batch dimension，因此下游不能再从 tensor shape 判断每条 sample 的起止位置；ms-swift 在拼接时保留 sample-local `position_ids`：每条文本 sample 分别从 0 开始计数，reset point 由此成为可编译的 boundary signal

> [!IMPORTANT]
> `position_ids` reset 只是边界的上游表示，真正交给 varlen attention 的执行元数据是累计边界 `cu_seqlens`

```text
flat token index   0 1 2 3 4 | 5 6 7 | 8 9 10 11
position_ids       0 1 2 3 4 | 0 1 2 | 0 1  2  3
reset index        0           5       8
append total       0           5       8          12
cu_seqlens         [0, 5, 8, 12]
sequence lengths      5  3  4
max_seqlen             5
```

这里发生的是一次 metadata compilation，而不是 attention 计算；reset index 给出每条 sample 的物理起点，末尾追加 total token count 后，相邻累计值之差才得到各条 sample 的实际长度

ms-swift 的显式转换函数把 `position_id == 0` 当作 boundary sentinel，主要供需要单独 boundary carrier 的模板使用

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

普通文本模型也可以把 reset 后的 `position_ids` 交给 Transformers 通用路径，由它在 `position_ids.min()` 的位置恢复边界。使用共同最小值而不是硬编码 0，是为了兼容合法起点不是 0 的位置编码

```python
position_ids = position_ids.reshape(-1)
indices_q = (position_ids == position_ids.min()).nonzero().view(-1)

cu_seq_lens_q = torch.cat((
    indices_q.to(dtype=torch.int32, device=position_ids.device),
    torch.tensor(
        position_ids.size(),
        dtype=torch.int32,
        device=position_ids.device,
    ),
))

max_length_q = cu_seq_lens_q.diff().max()
```

> [!NOTE]
> 两条路径寻找 reset point 的方式不同，但最终都必须生成同一种 cumulative boundary contract

累计边界必须同时满足以下关系

- 边界数组；dtype 为 `int32`；首值为 0；末值等于 flat token 总数；logical batch 为 `B` 时长度为 `B + 1`；相邻值严格递增
- Q/K/V 对齐；`q.shape[0] == cu_q[-1]`；`k.shape[0] == v.shape[0] == cu_k[-1]`；self-attention 通常要求 `cu_q == cu_k`；cross-attention 的第 `b` 个 Q 与 K/V slice 必须属于同一 sample
- 最大长度；`max_length_q = diff(cu_q).max()`；`max_length_k = diff(cu_k).max()`；它们只描述 launch 与 tile 上界，不承担 sample 分段语义

## 4. Transformers：如何进入 Varlen

Transformers 的公共 FlashAttention bridge 会根据输入 metadata 在三条互斥路径中选择；真正重要的不是配置里写了 `flash_attn`，而是 padding-free 输入最终必须命中 `flash_varlen_fn`

> [!IMPORTANT]
> 选择 FlashAttention 只确定 kernel family，进入 varlen branch 才真正恢复 logical batch 并隔离 sample

- padded batch；存在二维 `attention_mask`；`_upad_input()` 删除 padding；`flash_varlen_fn` 完成计算；`pad_fn()` 将输出散射回 padded shape
- padding-free packed batch；没有二维 padding mask；显式提供完整 `cu_seq_lens_q/k + max_length_q/k`，或提供可识别的 packed `position_ids`；Q/K/V flatten 后直接调用 `flash_varlen_fn`；输出只做 view
- dense no-padding batch；没有 padding mask 或 boundary metadata；调用普通 `flash_fn`；整个 sequence dimension 被解释为一条序列

Transformers 用 `all(...)` 检查四个显式 varlen 参数是否同时存在。只传 `cu_seq_lens_q/k` 而缺少 `max_length_q/k` 不会构成完整的显式 varlen contract

```python
is_fa_with_position_ids = _is_packed_sequence(
    position_ids,
    batch_size=query_states.size(0),
)
is_fa_with_varlen_kwargs = all(
    kwarg is not None
    for kwarg in (
        cu_seq_lens_q,
        cu_seq_lens_k,
        max_length_q,
        max_length_k,
    )
)

if attention_mask is not None:
    q, k, v, indices_q, cu_lens, max_lens = _upad_input(...)
    out_unpad = flash_varlen_fn(q, k, v, ...)
    out = pad_fn(out_unpad, indices_q, batch_size, query_length)

elif is_fa_with_varlen_kwargs or is_fa_with_position_ids:
    ...

else:
    out = flash_fn(query_states, key_states, value_states, ...)
```

ms-swift padding-free 输入通常仍保留外层 shape `[1, T, H, D]`。当显式边界已经由 collator 准备好时，Transformers 只移除这个长度为 1 的伪 batch dimension，把三组状态变成原生 varlen API 需要的 `[T, H, D]`

```text
query_states     [1, T_q, H_q, D] ── reshape ──► [T_q, H_q, D]
key_states       [1, T_k, H_k, D] ── reshape ──► [T_k, H_k, D]
value_states     [1, T_k, H_k, D] ── reshape ──► [T_k, H_k, D]
                                                    │
cu_q / cu_k ────────────────────────────────────────┤
max_q / max_k ──────────────────────────────────────┤
                                                    ▼
                                          flash_varlen_fn
                                                    │
                                                    ▼
output             [1, T_q, H_q, D] ◄─── view ─── [T_q, H_q, D]
```

```python
elif is_fa_with_varlen_kwargs or is_fa_with_position_ids:
    if cu_seq_lens_q is None or cu_seq_lens_k is None:
        q, k, v, cu_lens, max_lens = _prepare_from_posids(
            query_states,
            key_states,
            value_states,
            position_ids,
        )
        (cu_seq_lens_q, cu_seq_lens_k) = cu_lens
        (max_length_q, max_length_k) = max_lens
    else:
        q = query_states.reshape(-1, query_states.size(-2), query_states.size(-1))
        k = key_states.reshape(-1, key_states.size(-2), key_states.size(-1))
        v = value_states.reshape(-1, value_states.size(-2), value_states.size(-1))

    out = flash_varlen_fn(
        q,
        k,
        v,
        cu_seqlens_q=cu_seq_lens_q,
        cu_seqlens_k=cu_seq_lens_k,
        **flash_kwargs(
            max_seqlen_q=max_length_q,
            max_seqlen_k=max_length_k,
        ),
    )
    out = out.view(
        query_states.size(0),
        -1,
        out.size(-2),
        out.size(-1),
    )
```

> [!NOTE]
> direct padding-free 路径没有 `pad_fn()`；varlen 输出的 token 顺序必须与输入 flat stream 完全一致

## 5. FlashAttention Kernel：从 Boundary 到地址域

FlashAttention varlen 不会先物化 `[T, T]` block-diagonal mask，再逐元素屏蔽跨样本访问；`cu_seqlens` 在 score 计算之前就参与 global-memory address rebasing，把 flat Q/K/V 划成多个具有独立 base pointer 与 extent 的地址域

> [!IMPORTANT]
> varlen 的第一层隔离是重新定位 Q/K/V 指针与张量长度，不是在全局 attention matrix 上补一层 mask

对 grid 中的 logical batch index `bidb`，`BlockInfo` 分别读取 Q 与 K 的累计起点，并通过相邻边界之差得到当前序列的实际长度。下面展开的是本文对应的训练分支，即 Q/K 使用 cumulative boundary，且没有 left padding 与 KV cache

```cpp
template<bool Varlen = true>
struct BlockInfo {
    template<typename Params>
    __device__ BlockInfo(const Params& params, const int bidb)
        : sum_s_q(params.cu_seqlens_q[bidb])
        , sum_s_k(params.cu_seqlens_k[bidb])
        , actual_seqlen_q(
              params.cu_seqlens_q[bidb + 1] - sum_s_q)
        , actual_seqlen_k(
              params.cu_seqlens_k[bidb + 1] - sum_s_k) {}

    template<typename index_t>
    __device__ index_t q_offset(
            index_t batch_stride,
            index_t row_stride,
            int bidb) const {
        return uint32_t(sum_s_q) * row_stride;
    }

    template<typename index_t>
    __device__ index_t k_offset(
            index_t batch_stride,
            index_t row_stride,
            int bidb) const {
        return uint32_t(sum_s_k) * row_stride;
    }
};
```

累计 token offset 乘以 row stride 后，得到当前 logical sequence 在 global memory 中的基地址；kernel 随后用这个 base 与 `actual_seqlen` 构造局部 tensor view

```text
flat Q/K/V memory

0                   5             8                  12
│                   │             │                   │
├─────── A ─────────┼──── B ──────┼──────── C ────────┤
│                   │             │                   │
│ bidb=0            │ bidb=1      │ bidb=2            │
│ base = ptr + 0*s  │ base=ptr+5*s│ base = ptr + 8*s  │
│ extent = 5        │ extent = 3  │ extent = 4        │
```

```cpp
Tensor mQ = make_tensor(
    make_gmem_ptr(
        reinterpret_cast<Element*>(params.q_ptr)
        + binfo.q_offset(
            params.q_batch_stride,
            params.q_row_stride,
            bidb)),
    make_shape(binfo.actual_seqlen_q, params.h, params.d),
    make_stride(params.q_row_stride, params.q_head_stride, _1{})
);

Tensor mK = make_tensor(
    make_gmem_ptr(
        reinterpret_cast<Element*>(params.k_ptr)
        + binfo.k_offset(
            params.k_batch_stride,
            params.k_row_stride,
            bidb)),
    make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
    make_stride(params.k_row_stride, params.k_head_stride, _1{})
);
```

Q/K tensor view 同时确定 base pointer 与可访问行数，之后才计算 tile 范围

- launch 范围；`max_seqlen_q` 决定 grid 为最长 Q sequence 启动多少个 `m_block`
- per-sequence Q 范围；`actual_seqlen_q` 使短序列上多余的 Q block 立即退出
- per-sequence K 范围；`actual_seqlen_k` 给出当前 Q block 能枚举的 K block 上界
- sequence-local visibility；causal 或 local window 在当前 K tile 范围内继续收紧 `n_block_min/max`

```cpp
const int num_m_block =
    (params.seqlen_q + kBlockM - 1) / kBlockM;

if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

int n_block_max = cute::ceil_div(
    binfo.actual_seqlen_k,
    kBlockN
);

if (Is_causal || Is_local) {
    n_block_max = std::min(
        n_block_max,
        cute::ceil_div(
            (m_block + 1) * kBlockM
                + binfo.actual_seqlen_k
                - binfo.actual_seqlen_q
                + params.window_size_right,
            kBlockN
        )
    );
}
```

> [!NOTE]
> `cu_seqlens` 先定义 sequence address domain，`actual_seqlen` 再定义合法 tile domain，causal/window 最后只在该 domain 内限制可见性

输出写回复用 Q-side 的 `q_offset`，因此 `O` 与输入 Q 保持相同的 flat token ordering

```cpp
Tensor mO = make_tensor(
    make_gmem_ptr(
        reinterpret_cast<Element*>(params.o_ptr)
        + binfo.q_offset(
            params.o_batch_stride,
            params.o_row_stride,
            bidb)),
    make_shape(binfo.actual_seqlen_q, params.h, params.d),
    make_stride(params.o_row_stride, params.o_head_stride, _1{})
);
```

## 6. Qwen3.5：LLM packing 与多模态边界

多模态 packing 关注的不是原始文本长度、图像分辨率、视频帧数或音频时长，而是各模态经过 encoder 与下采样后，**最终等价映射到 LLM 序列中的 token 数**。文本 token 直接占据 LLM position；图像、视频和音频则通过 `<|image_pad|>`、`<|video_pad|>` 与 `<|audio_pad|>` 等占位符，把下采样后的特征数量展开成对应数量的 LLM token 槽位

> [!IMPORTANT]
> packing 计数的是各模态占位符按下采样结果展开后，在 LLM 序列中实际占据的 token 数

当前 ms-swift 的 Qwen3.5 模板处理 image 与 video，因此实际参与这条路径的是 `<|image_pad|>` 和 `<|video_pad|>`；`<|audio_pad|>` 使用相同的长度映射原则，但属于支持音频输入的其他多模态模板

ms-swift 的非流式训练管线先用 `template.encode` 得到编码后的样本，再构造 `PackingDataset`：

```python
if not args.streaming and args.truncation_strategy != 'split':
    dataset = LazyLLMDataset(
        dataset,
        template.encode,
        strict=args.strict,
        random_state=args.data_seed,
    )

if args.packing:
    dataset = PackingDataset(
        template,
        dataset,
        packing_length=args.packing_length,
        packing_strategy=args.packing_strategy,
        ...
    )
```

因此，`PackingDataset` 读取的 `lengths` 已经是模板编码后的长度。对 Qwen3.5，`Qwen3_5Template` 继承 `Qwen3VLTemplate._encode()`。图像处理器先产生 `image_grid_thw`，模板再根据 spatial merge 后的网格大小计算图像在 LLM 序列中应占多少个位置：

```python
merge_length = processor.image_processor.merge_size**2

def _get_new_tokens(i):
    if media_type == 'images':
        token_len = media_grid_thw[i].prod() // merge_length
        return [media_token] * token_len
    else:
        return splited_tokens[i]

input_ids, labels, loss_scale, mm_mask = self._extend_tokens(
    input_ids,
    labels,
    loss_scale,
    idx_list,
    _get_new_tokens,
    mm_mask=mm_mask,
)
```

`_extend_tokens()` 用这组新 token 替换原来的单个媒体标记，并同步扩展所有 token-aligned metadata。视觉位置不参与语言建模监督，所以对应的 labels 被写成 `-100`；`mm_mask` 则标记这些位置属于多模态区域：

```python
new_tokens = get_new_tokens(i)
token_len = len(new_tokens)

input_ids = (
    input_ids[:idx + added_tokens_len]
    + new_tokens
    + input_ids[added_tokens_len + idx + 1:]
)
labels = (
    labels[:idx + added_tokens_len]
    + [-100] * token_len
    + labels[added_tokens_len + idx + 1:]
)
mm_mask = (
    mm_mask[:idx + added_tokens_len]
    + [True] * token_len
    + mm_mask[added_tokens_len + idx + 1:]
)
```

图像的 LLM-visible 长度由 `grid_thw.prod() // merge_size²` 决定；视频路径则直接使用 processor 展开后的 token group。两条路径都在 `PackingDataset` 分组之前完成，所以 packing 面对的是已经确定长度的完整多模态 LLM 序列

> [!NOTE]
> Packing 只重排 LLM token 序列，不处理视觉 encoder 内部的 patch 序列

Qwen3.5 中存在两种不同的序列

- 视觉 encoder 内部处理的 patch / grid 序列
- LLM token 序列；模板根据 grid metadata 预留视觉 token 槽位；packing 只处理这一层；视觉 embedding 在 forward 前计算并写回槽位

```python
input_ids = inputs['input_ids']
inputs_embeds = base_model.model.embed_tokens(input_ids)

inputs_embeds = self._get_inputs_embeds_hf(
    inputs_embeds,
    inputs,
    model.visual,
    self.processor,
    model.config,
)
```

`_get_inputs_embeds_hf()` 运行视觉模块后，根据 image/video token mask 执行 `masked_scatter`：

```python
image_embeds = visual(pixel_values, grid_thw=image_grid_thw)

image_mask = (
    input_ids == config.image_token_id
).unsqueeze(-1).expand_as(inputs_embeds)

inputs_embeds = inputs_embeds.masked_scatter(
    image_mask,
    image_embeds.to(inputs_embeds.dtype),
)
```

这里的 scatter 只替换向量，不改变 token 数量和顺序。也就是说，多模态“先展开”准确地说是：先展开视觉特征在 LLM 序列中的位置，再按这个最终 LLM 长度 packing；实际视觉特征可以稍后生成，但其数量必须与预留的视觉 token 槽位严格一致

> [!IMPORTANT]
> mRoPE 表示多模态几何位置，独立的 text position plane 表示 sample boundary

视觉位置展开后，每条样本仍需独立计算 mRoPE。Qwen 模板覆盖了 `packing_row()`，先对 pack 中的每条 sample 调用模型的 `get_rope_index()`，然后才交给父类沿 sequence 维拼接：

```python
def packing_row(self, row):
    for sample in row:
        sample_copy = sample.copy()
        sample_copy['input_ids'] = torch.tensor(
            sample_copy['input_ids']
        )[None]

        if 'mm_token_type_ids' in sample_copy:
            sample_copy['mm_token_type_ids'] = (
                sample_copy['mm_token_type_ids'][None]
            )

        sample.update(self._get_position_ids(sample_copy))

    return super().packing_row(row)
```

`get_rope_index()` 返回 temporal、height 和 width 三层 mRoPE 坐标；这些坐标服务于 Q/K 的旋转位置编码，却不适合作为 sample boundary：视觉网格中的坐标可以重复，多个合法位置也可能同时等于 0；若直接查找 reset 点，会把样本内部的视觉区域误切成多段

因此，ms-swift 在每条样本的三层 mRoPE 前额外增加一层严格递增的文本顺序坐标：

```python
@staticmethod
def _concat_text_position_ids(position_ids):
    seq_len = position_ids.shape[-1]
    text_position_ids = torch.arange(
        seq_len,
        device=position_ids.device,
    ).expand(1, *position_ids.shape[1:])

    return torch.concat(
        [text_position_ids, position_ids],
        dim=0,
    )
```

对单条样本，这个张量的四层含义是：

- plane 0；`0, 1, 2, ...`；表示 sample-local sequence order；packing 后用于恢复 sample boundary
- planes 1..3；分别表示 temporal、height、width mRoPE coordinates

因为 `_concat_text_position_ids()` 在拼接前逐 sample 执行，plane 0 会在每条样本开头重新从 0 计数；父类 `packing_row()` 沿最后一维拼接后，它自然成为完整的 boundary carrier；collator 随后将顺序坐标与 mRoPE 坐标重新拆开：

```python
position_ids = result['position_ids']

result['position_ids'] = position_ids[1:]
result['text_position_ids'] = (
    text_position_ids
) = position_ids[0]

result.update(
    get_packed_seq_params(text_position_ids)
)
```

最终，`position_ids` 只保留三层 mRoPE，交给语言模型计算旋转位置；独立的 `text_position_ids` 用于生成 `cu_seq_lens_q/k` 与 `max_length_q/k`。几何位置与 attention boundary 因而具有不同的数据来源和消费者，不会因为视觉坐标的重复而互相干扰

> [!IMPORTANT]
> Qwen3.5 的同一组 sample boundaries 必须同时约束 full attention 与 linear attention

Qwen3.5 decoder layer 根据 `layer_types` 在 `full_attention` 与 `linear_attention` 之间选择：

```python
if self.block_type == 'linear_attention':
    hidden_states = self.linear_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        **kwargs,
    )
elif self.block_type == 'full_attention':
    hidden_states, _ = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        **kwargs,
    )
```

full-attention 层将 `cu_seq_lens_q/k` 传给 FlashAttention varlen；Gated DeltaNet 路径同样消费累计边界，并把它传给 recurrent/chunk kernel：

```python
core_attn_out, last_recurrent_state = (
    torch_chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=recurrent_state,
        cu_seqlens=kwargs.pop(
            'cu_seq_lens_q',
            None,
        ),
        ...
    )
)
```

同一组 boundary 同时隔离 full-attention 的 Q/K/V slice 与 linear-attention 的递归状态；前者限定 attention domain，后者阻止 recurrent state 跨 sample 传播

完整的数据顺序分为三个阶段

- 数据展开；`template.encode` 处理 raw text / image / video；media placeholder 展开为 LLM-visible token slots；确定每条 sample 的最终 LLM 长度
- Packing 与位置构造；`PackingDataset` 组合完整 sample；逐 sample 计算 mRoPE 与 text boundary plane；所有 token-aligned metadata 按相同顺序 flatten
- 模型计算；visual encoder 生成 embedding；`masked_scatter` 写入预留槽位；full attention 与 linear attention 消费同一组 sample boundaries
