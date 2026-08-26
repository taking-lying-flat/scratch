# 从 `Packing` 到 `FlashAttention Varlen`：`Padding-Free` 训练的数据布局与内核语义

> 基于 `ms-swift`、`Transformers` 与 `FlashAttention` 源码的端到端分析

| 层级 | 核心对象 | 解决的问题 | 不负责什么 |
| --- | --- | --- | --- |
| 数据集层 | `packing` | 选择哪些完整样本进入同一个近定长工作单元 | 不拼接 `Q/K/V`，不定义注意力可见性 |
| 批处理层 | `padding-free` | 将多条变长序列压紧为连续的 `token stream` | 不会仅凭 `torch.cat` 自动隔离样本 |
| 框架桥接层 | `position_ids → cu_seqlens` | 把逻辑边界转换为 `FlashAttention` 的变长接口元数据 | `position_ids` 本身不是注意力掩码 |
| 内核层 | `flash_attn_varlen_func` | 在连续存储上执行彼此独立的注意力问题 | 不恢复数据集顺序，也不决定训练样本分组 |

> [!IMPORTANT]
> `packing`、`padding-free` 与 `FlashAttention varlen` 不是三个同义词，而是同一条流水线上的三种不同变换：`packing` 优化样本调度，`padding-free` 优化物理布局，`varlen` 内核依据显式边界恢复逻辑隔离。

## 结论

设一个批次包含长度为 $L_0,L_1,\ldots,L_{n-1}$ 的完整序列。`ms-swift` 的标准训练路径可概括为：

```text
样本及其 token 长度
  → PackingDataset 进行装箱
  → data_collator 展开 pack
  → packing_row 拼接 input_ids / labels / position_ids
  → position_ids 的起点生成 cu_seqlens
  → Transformers 选择 flash_attn_varlen_func
  → FlashAttention 按 cu_seqlens 切分 Q/K/V
  → 每个逻辑序列独立执行 causal attention
```

最终送入内核的不是带有大量补齐位置的 $[B,S_{max},H,D]$，而是紧凑的：

$$
Q\in\mathbb{R}^{T\times H_q\times D},\qquad
K,V\in\mathbb{R}^{T\times H_{kv}\times D},\qquad
T=\sum_iL_i.
$$

同时传入：

$$
\mathrm{cu\_seqlens}=[0,L_0,L_0+L_1,\ldots,T].
$$

对于第 $i$ 条序列，内核只计算：

$$
O_i=\operatorname{softmax}\!\left(
\frac{Q_iK_i^{\mathsf T}}{\sqrt D}+M_i^{\mathrm{causal}}
\right)V_i,
\qquad O=\operatorname{concat}(O_0,\ldots,O_{n-1}).
$$

这里不存在 $Q_iK_j^{\mathsf T}\;(i\ne j)$ 的跨样本项。语义上等价于一个 `block-diagonal causal mask`，但实现上不会物化这个稠密矩阵。

> [!CAUTION]
> 仅把多条序列拼成一条长序列，再调用普通 `causal attention`，会产生跨样本信息泄漏。正确实现必须把边界以 `cu_seqlens` 或等价的变长元数据交给内核。

## 1. 三个概念的严格分工

### 1.1 `Packing` 是样本调度问题

`packing` 的输入是样本长度及索引，输出是若干索引组。例如：

| 样本 | 长度 |
| --- | ---: |
| `A` | 700 |
| `B` | 600 |
| `C` | 300 |
| `D` | 400 |

当 `packing_length=1024` 时，一种可行装箱为 `[A,C]` 与 `[B,D]`。这一阶段只确定“哪些样本同行”，没有执行任何张量拼接。

`ms-swift` 的 `calculate_matched_group()` 提供两种策略：

| `packing_strategy` | 算法语义 | 顺序性质 | 适用侧重 |
| --- | --- | --- | --- |
| `binpack` | 调用 `to_constant_volume` 的近似 `best-fit-decreasing` 装箱 | 允许按长度重排 | 提高填充率，降低工作量方差 |
| `sequential` | 单开放箱的 `next-fit` 贪心策略 | 保持输入顺序 | 顺序采样、时序数据或可复现边界 |

源码中，`PackingDataset` 先读取预处理得到的 `lengths`，生成 `packed_idx`；`__getitem__()` 返回 `[sample_i, sample_j, ...]`，而不是已经拼好的张量。随后公共 `data_collator()` 将一个微批中的这些列表展开，再进入真正的张量整理阶段。参见 [`packing.py`](https://github.com/taking-lying-flat/ms-swift/blob/ed1b2f3742296c4dbf1ddc553cb4ac43ea0c4165/swift/dataset/packing.py#L16-L47)、[`PackingDataset`](https://github.com/taking-lying-flat/ms-swift/blob/ed1b2f3742296c4dbf1ddc553cb4ac43ea0c4165/swift/dataset/packing.py#L50-L134) 与 [`data_collator()`](https://github.com/taking-lying-flat/ms-swift/blob/ed1b2f3742296c4dbf1ddc553cb4ac43ea0c4165/swift/template/base.py#L1714-L1737)。

`packing_length` 默认回落到 `max_length`。因此它首先是一个数据层的 `token-budget`：每个 `pack` 不拆分完整序列，并尽量接近该预算；它不是模型看到的逻辑序列长度，也不是 `FlashAttention` 的 `max_seqlen`。数据管线的包装位置见 [`sft.py`](https://github.com/taking-lying-flat/ms-swift/blob/ed1b2f3742296c4dbf1ddc553cb4ac43ea0c4165/swift/pipelines/train/sft.py#L125-L156)。

### 1.2 `Padding-Free` 是张量布局问题

在普通动态补齐中，长度 `[5,3,4]` 会形成一个 $[3,5]$ 的二维批次：

| 行 | 物理布局 | 有效长度 |
| --- | --- | ---: |
| `A` | `A A A A A` | 5 |
| `B` | `B B B PAD PAD` | 3 |
| `C` | `C C C C PAD` | 4 |

`padding-free` 将其改写为一行连续存储：

```text
input_ids     = [A A A A A | B B B | C C C C]
position_ids  = [0 1 2 3 4 | 0 1 2 | 0 1 2 3]
physical shape = [1, 12]
```

`ms-swift` 的 `_data_collator()` 在 `self.padding_free=True` 时调用 `packing_row(batch)`，把 `input_ids`、`labels`、`loss_scale`、`position_ids` 与 `token_type_ids` 沿序列维拼接；若样本尚无 `position_ids`，则为每条原始序列分别生成 `0...L_i-1`。实现见 [`packing_row()`](https://github.com/taking-lying-flat/ms-swift/blob/ed1b2f3742296c4dbf1ddc553cb4ac43ea0c4165/swift/template/base.py#L721-L742) 与 [`_data_collator()`](https://github.com/taking-lying-flat/ms-swift/blob/ed1b2f3742296c4dbf1ddc553cb4ac43ea0c4165/swift/template/base.py#L1909-L2024)。

这一步同时解释了两个容易误判的事实：

- 物理 `batch dimension` 变成 1，不代表逻辑批次只有一条样本；逻辑批次大小由边界数量决定。
- `position_ids` 的重置同时服务于位置编码和边界发现，但真正限制注意力可见性的仍是后续 `cu_seqlens`。

在当前 `ms-swift` 中，`packing=True` 会进一步令 `padding_free=True`，并要求注意力实现属于 `flash_attn`、`flash_attention_2/3/4`。这是当前实现的后端约束，而不是“所有无补齐算法在数学上只能由 `FlashAttention` 实现”。参数检查见 [`_check_padding_free()`](https://github.com/taking-lying-flat/ms-swift/blob/ed1b2f3742296c4dbf1ddc553cb4ac43ea0c4165/swift/arguments/sft_args.py#L185-L196)。

### 1.3 `FlashAttention Varlen` 是边界感知的计算问题

对上面的 `[5,3,4]`，框架生成：

```text
cu_seqlens_q = cu_seqlens_k = [0, 5, 8, 12]  # int32
max_seqlen_q = max_seqlen_k = 5
```

其中第 $i$ 个逻辑序列占用半开区间：

$$
[\mathrm{cu}_i,\mathrm{cu}_{i+1}).
$$

`ms-swift` 的 `get_packed_seq_params()` 查找每条序列的位置起点，并追加总 `token` 数；字段在模型侧命名为 `cu_seq_lens_q/k`，传入 `FlashAttention` 时映射为其原生参数 `cu_seqlens_q/k`。参见 [`transformers_utils.py`](https://github.com/taking-lying-flat/ms-swift/blob/ed1b2f3742296c4dbf1ddc553cb4ac43ea0c4165/swift/utils/transformers_utils.py#L347-L363)。

> [!NOTE]
> `cu_seq_lens_q` 是 `Transformers/ms-swift` 接口层的拼写，`cu_seqlens_q` 是 `FlashAttention` 原生接口的拼写。二者描述同一种 `cumulative sequence lengths`，不要将命名差异误解成两份不同元数据。

## 2. `ms-swift → Transformers → FlashAttention` 的真实调用链

### 2.1 数据侧产生边界

以普通自注意力为例，数据侧完成四项工作：

1. `PackingDataset` 根据长度重组完整样本。
2. `data_collator()` 展开 `pack`，`packing_row()` 拼接字段。
3. 每条样本的首位置保持相同起点，通常为 0。
4. `get_packed_seq_params()` 将起点转换为 `cu_seq_lens_q/k` 与 `max_length_q/k`。

这些字段会由 `pre_forward_hook()` 保留并交给模型。它不是在每层重复做数据预处理，而是把一份批次级边界元数据传给所有需要它的层。参见 [`pre_forward_hook()`](https://github.com/taking-lying-flat/ms-swift/blob/ed1b2f3742296c4dbf1ddc553cb4ac43ea0c4165/swift/template/base.py#L1648-L1664)。

### 2.2 `Transformers` 选择 `varlen` 分支

`Transformers` 的 `_flash_attention_forward()` 区分三条路径：

| 输入状态 | 路径 | 行为 |
| --- | --- | --- |
| 存在二维 `attention_mask` | `unpad → varlen → pad` | 先从补齐批次抽取有效 `token`，计算后再散射回原形状 |
| 已给出 `cu_seq_lens_*`，或 `position_ids` 表示多段序列 | `padding-free varlen` | 将 `Q/K/V` 展平后直接调用 `flash_varlen_fn` |
| 无补齐且无分段 | 普通 `flash_fn` | 将整个序列作为单一注意力问题 |

在 `padding-free` 分支中，框架将：

```python
q = query_states.reshape(-1, num_heads_q, head_dim)
k = key_states.reshape(-1, num_heads_kv, head_dim)
v = value_states.reshape(-1, num_heads_kv, head_dim)

out = flash_varlen_fn(
    q, k, v,
    cu_seqlens_q=cu_seq_lens_q,
    cu_seqlens_k=cu_seq_lens_k,
    max_seqlen_q=max_length_q,
    max_seqlen_k=max_length_k,
    causal=True,
)
```

完整分支与注释明确指出：选择 `flash_varlen_fn` 的目的之一就是阻止 `cross-example attention`。见 [`Transformers` 的 `_flash_attention_forward()`](https://github.com/huggingface/transformers/blob/36deb0b53ed0863f4b4dfdea23dcaec7f3df3701/src/transformers/modeling_flash_attention_utils.py#L694-L830)。若模型未显式传入累计长度，框架也能从分段 `position_ids` 构造它们，见 [`prepare_fa_kwargs_from_position_ids()`](https://github.com/huggingface/transformers/blob/36deb0b53ed0863f4b4dfdea23dcaec7f3df3701/src/transformers/modeling_flash_attention_utils.py#L458-L533)。

从架构上看，优先在 `collator` 生成显式 `cu_seqlens` 更稳健：边界在数据仍保持结构化时就是已知量，无需在模型内部再次推断；这也与 `Transformers` 源码中的注释一致。

### 2.3 原生接口的契约

`flash_attn_varlen_func()` 的核心接口如下：

| 参数 | 形状 / 类型 | 语义 |
| --- | --- | --- |
| `q` | `[total_q, nheads_q, headdim]` | 所有查询序列的紧凑存储 |
| `k`, `v` | `[total_k, nheads_kv, headdim]` | 所有键值序列的紧凑存储 |
| `cu_seqlens_q` | `[batch+1]`, `int32` | 每条查询序列的累计边界 |
| `cu_seqlens_k` | `[batch+1]`, `int32` | 每条键值序列的累计边界 |
| `max_seqlen_q/k` | 标量 | 本批逻辑序列的最大长度，用于调度与临时空间 |
| `causal` | `bool` | 在每个逻辑序列内部施加因果约束 |

官方接口文档见 [`flash_attn_interface.py`](https://github.com/Dao-AILab/flash-attention/blob/0251105a2fb19d2957484b7f023cd8c115286ced/flash_attn/flash_attn_interface.py#L1391-L1482)。原生入口还检查 `cu_seqlens` 必须位于设备侧、连续且为 `int32`，并令 `batch_size = numel(cu_seqlens_q) - 1`，见 [`flash_api.cpp`](https://github.com/Dao-AILab/flash-attention/blob/0251105a2fb19d2957484b7f023cd8c115286ced/csrc/flash_attn/flash_api.cpp#L570-L653)。

## 3. 为什么连续存储仍能做到逻辑隔离

`FlashAttention` 的 `BlockInfo` 给出了内核层面的直接证据。对于第 `bidb` 条序列，它读取：

```cpp
sum_s_q         = cu_seqlens_q[bidb];
sum_s_k         = cu_seqlens_k[bidb];
actual_seqlen_q = cu_seqlens_q[bidb + 1] - sum_s_q;
actual_seqlen_k = cu_seqlens_k[bidb + 1] - sum_s_k;
```

随后 `q_offset()` 与 `k_offset()` 把全局连续缓冲区的指针移动到本序列起点。实现见 [`block_info.h`](https://github.com/Dao-AILab/flash-attention/blob/0251105a2fb19d2957484b7f023cd8c115286ced/csrc/flash_attn/src/block_info.h#L12-L45)。

因此对 `cu_seqlens=[0,5,8,12]`：

| `bidb` | `Q/K/V` 区间 | `actual_seqlen` | 可访问对象 |
| ---: | --- | ---: | --- |
| 0 | `[0,5)` | 5 | 仅 `A` |
| 1 | `[5,8)` | 3 | 仅 `B` |
| 2 | `[8,12)` | 4 | 仅 `C` |

前向内核会根据 `actual_seqlen_q` 提前丢弃越界的查询块，再根据 `actual_seqlen_k` 和局部因果条件计算可访问的键块上界。换言之，`causal` 三角形是在每个切片的局部坐标中建立，而不是在长度 12 的全局坐标中建立。见 [`flash_fwd_kernel.h`](https://github.com/Dao-AILab/flash-attention/blob/0251105a2fb19d2957484b7f023cd8c115286ced/csrc/flash_attn/src/flash_fwd_kernel.h#L87-L105)。

同一数据契约在较新的 `CuTe` 实现中仍然成立：`SeqlenInfoQK.create()` 使用相邻累计边界之差计算实际长度，并通过 `offset_batch_Q/K()` 对不规则张量进行指针偏移。见 [`seqlen_info.py`](https://github.com/Dao-AILab/flash-attention/blob/0251105a2fb19d2957484b7f023cd8c115286ced/flash_attn/cute/seqlen_info.py#L83-L147)。这说明 `packing/padding-free` 依赖的是稳定的变长接口契约，而不是某一代内核的偶然实现细节。

## 4. 训练正确性不只取决于注意力边界

### 4.1 `position_ids` 与 `cu_seqlens` 的职责不同

| 元数据 | 主要消费者 | 正确性职责 |
| --- | --- | --- |
| `position_ids` | `RoPE/mRoPE` 与边界构造逻辑 | 每条样本使用自己的局部位置坐标 |
| `cu_seqlens` | `FlashAttention varlen` | 划分 `Q/K/V`，禁止跨样本注意力 |
| `labels` | 因果语言模型损失 | 禁止在拼接边界上产生跨样本预测目标 |
| `loss_scale` | 加权损失 | 与拼接后的 `labels/token` 一一对应 |

仅重置 `position_ids` 不能阻止普通注意力读取前一个样本；仅提供 `cu_seqlens` 也不能自动保证损失函数不会把前一序列末尾的 `logit` 用来预测后一序列首 `token`。

### 4.2 拼接边界必须切断语言模型损失

常见因果语言模型损失使用：

$$
\mathrm{logits}_{t}\longrightarrow\mathrm{labels}_{t+1}.
$$

当 `A|B` 被物理拼接时，若 `B` 的首个 `label` 仍有效，就会错误训练 `A` 的末尾去预测 `B` 的开头。`ms-swift` 在每条样本编码完成后将 `labels[0]` 设为 `-100`，随后 `packing_row()` 原样拼接各样本的 `labels`，所以每个逻辑边界都保留一个忽略位置。相关处理见 [`base.py`](https://github.com/taking-lying-flat/ms-swift/blob/ed1b2f3742296c4dbf1ddc553cb4ac43ea0c4165/swift/template/base.py#L1545-L1572)。

正确语义为：

```text
tokens:  [A0 A1 A2 | B0 B1 B2]
labels:  [-  A1 A2 |  - B1 B2]
                    ^
             不建立 A2 → B0 的训练目标
```

### 4.3 边界元数据的不变量

一个可靠实现至少应在进入模型前验证：

```text
cu_seqlens.dtype == int32
cu_seqlens[0] == 0
cu_seqlens[-1] == total_tokens
all(diff(cu_seqlens) > 0)          # 不接受空逻辑序列时
max(diff(cu_seqlens)) == max_seqlen
len(cu_seqlens_q) == len(cu_seqlens_k)
labels[cu_seqlens[:-1]] == -100    # causal LM 场景
```

对自注意力，`Q/K/V` 通常共享同一组边界；对交叉注意力，`Q` 与 `K/V` 可以有不同的累计长度，但逻辑批次数必须一致。

## 5. 多模态场景：边界必须在最终 `LLM-visible sequence` 上定义

多模态 `packing` 的长度单位不是原始图片数量，也不是视觉编码器内部的原始 `patch` 数，而是最终进入语言主干的统一序列长度：

```text
image / video
  → vision processor / encoder
  → LLM-visible visual token positions

text
  → text token positions

visual positions + text positions
  → 每条样本自己的 multimodal sequence
  → packing
  → padding-free flatten
  → language backbone 的 FlashAttention varlen
```

以 `Qwen2-VL` 模板为例：

1. 每条样本先按 `image_grid_thw` 与合并比例展开视觉占位位置。
2. `packing_row()` 在样本仍彼此独立时调用模型的 `get_rope_index()`，得到各自的多维 `mRoPE position_ids`。
3. 多条样本沿序列维拼接；`T/H/W` 位置用于旋转 `Q/K`。
4. `text_position_ids` 单独提取，用来生成 `cu_seq_lens_q/k`，从而隔离语言主干中的不同多模态样本。

对应实现见 [`Qwen2VLTemplate.packing_row()` 与 `_data_collator()`](https://github.com/taking-lying-flat/ms-swift/blob/ed1b2f3742296c4dbf1ddc553cb4ac43ea0c4165/swift/template/templates/qwen.py#L459-L512)。

| 元数据平面 | 用途 |
| --- | --- |
| `text_position_ids` | 提取样本起点，生成 `cu_seqlens` |
| `mRoPE T/H/W` | 为文本与视觉位置构造多维旋转坐标 |
| `image_grid_thw` / `video_grid_thw` | 描述视觉网格及视觉 `token` 展开关系 |
| `inputs_embeds` | 将视觉特征替换到语言序列的占位位置 |

> [!NOTE]
> 这里的 `FlashAttention varlen` 约束的是语言主干中的统一多模态序列。视觉编码器内部是否进行自己的 `packing` 或变长注意力，是另一条计算图，不能由语言主干的 `cu_seqlens` 自动推导。

## 6. 超出标准全注意力的模型需要额外审计

“注意力层已经使用 `varlen`”并不自动证明整个模型支持 `padding-free`。任何沿序列传播状态的模块都必须理解同一组边界，包括：

- `causal convolution`；
- `linear attention` 或状态空间递推；
- 自定义 `sequence parallel` 切分；
- 在序列维执行的池化、归一化或后处理。

例如混合架构若含有 `causal convolution`，把 `[A|B]` 当成单序列卷积会让 `B` 的开头读取 `A` 的尾部，即使全注意力层已完全隔离。当前 `ms-swift` 的 `Qwen3.5` 特殊路径会把同一份 `cu_seq_lens_q` 传入变长 `causal_conv1d` 与 `gated delta rule` 内核；当批次包含多条逻辑序列而相应变长内核不可用时，它会直接拒绝回退到不理解边界的普通实现。见 [内核选择与约束](https://github.com/taking-lying-flat/ms-swift/blob/ed1b2f3742296c4dbf1ddc553cb4ac43ea0c4165/swift/model/models/qwen.py#L1235-L1247)、[`causal_conv1d` 调用](https://github.com/taking-lying-flat/ms-swift/blob/ed1b2f3742296c4dbf1ddc553cb4ac43ea0c4165/swift/model/models/qwen.py#L1250-L1259) 与 [边界传递位置](https://github.com/taking-lying-flat/ms-swift/blob/ed1b2f3742296c4dbf1ddc553cb4ac43ea0c4165/swift/model/models/qwen.py#L1336-L1381)。

因此，判断一个模型是否真正支持 `packing/padding-free`，应逐模块检查“是否跨序列维传播信息”，不能只检查命令行是否接受 `--attn_impl flash_attn`。

## 7. 推荐的验证方法

### 7.1 数值等价性

构造两条短序列 `A`、`B`，关闭随机性并使用相同权重，比较：

```text
reference = concat(model(A), model(B))
packed    = model(A | B, cu_seqlens=[0, len(A), len(A)+len(B)])
```

检查有效位置的前向输出、损失和反向梯度在所用精度容差内一致。只检查最终 `loss` 不够，因为跨样本泄漏可能在聚合后被掩盖。

### 7.2 边界扰动测试

固定 `B`，只改变 `A` 的内容。正确实现必须满足：

```text
packed_output_B(A₁ | B) == packed_output_B(A₂ | B)
```

该测试直接验证 `B` 是否读取了 `A`。还应覆盖长度 1、不同起始位置编号、多模态位置和极端长短组合。

## 8. 工程判断

`ms-swift` 当前把 `packing` 自动绑定到 `padding-free`，这一选择的合理性在于：数据层装箱只提供组合关系，真正高效且语义正确的执行仍依赖边界感知的紧凑布局。完整设计可以压缩为四条原则：

1. `packing` 只重排完整样本，不切断样本内部语义。
2. `padding-free` 只改变物理存储，不改变逻辑批次。
3. `cu_seqlens` 是注意力隔离的权威边界，不能用全局因果掩码替代。
4. 所有跨序列传播状态的模块与损失函数都必须共享同一套边界语义。

> [!TIP]
> 最准确的一句话不是“`packing` 把短文本拼成长文本”，而是：`packing` 重组训练调度，`padding-free` 压紧存储，`FlashAttention varlen` 在连续缓冲区上恢复一组彼此独立的注意力问题。

## 源码版本

| 项目 | 分析版本 | 关键入口 |
| --- | --- | --- |
| `ms-swift` | [`ed1b2f374`](https://github.com/taking-lying-flat/ms-swift/tree/ed1b2f3742296c4dbf1ddc553cb4ac43ea0c4165) | `PackingDataset`、`packing_row()`、`get_packed_seq_params()` |
| `Transformers` | [`36deb0b53`](https://github.com/huggingface/transformers/tree/36deb0b53ed0863f4b4dfdea23dcaec7f3df3701) | `_flash_attention_forward()` |
| `FlashAttention` | [`0251105a2`](https://github.com/Dao-AILab/flash-attention/tree/0251105a2fb19d2957484b7f023cd8c115286ced) | `flash_attn_varlen_func()`、`BlockInfo` |
