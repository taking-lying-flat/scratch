# ms-swift 模型构造与运行时故障分析

**目录**

- [统一分析框架](#统一分析框架)
- [Qwen3-Omni 非持久 Buffer 在 Meta Materialization 后失效](#qwen3-omni-非持久-buffer-在-meta-materialization-后失效)
- [关闭 `padding_free` 后 Qwen3.5 GDN 显存暴涨](#关闭-padding_free-后-qwen35-gdn-显存暴涨)
- [Transformers 5.2–5.3 中尾批 `batch=1` 触发 M-RoPE 与 FA2 metadata 越界](#transformers-5253-中尾批-batch1-触发-m-rope-与-fa2-metadata-越界)
- [`enable_audio_output` 在 Thinker-Talker 架构中的作用](#enable_audio_output-在-thinker-talker-架构中的作用)

## 统一分析框架

```text
模型目录 / 环境变量 / CLI
        │
        ▼
Config resolution
        │  决定模型分支与模块拓扑
        ▼
Module construction
        │  创建 Parameter、persistent buffer、non-persistent buffer
        ▼
post_init() / initialize_weights()
        │  建立构造完成后的确定状态
        ▼
Checkpoint loading
        │  只恢复 checkpoint 中实际序列化的内容
        ▼
Template / Dataset / Collator
        │  生成 input_ids、position_ids、attention_mask、cu_seqlens
        ▼
Forward / Attention / GDN kernel
        │  消费 tensor 与 runtime metadata
        ▼
Generation
        │  EOS、PAD、stopping criteria 决定终止
        ▼
Output
```

<div align="center">
<table>
  <thead>
    <tr>
      <th align="center">生命周期阶段</th>
      <th align="center">该阶段拥有的状态</th>
      <th align="center">必须保持的不变量</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>Config resolution</code></td><td>模型类型、子模块开关、RoPE 和输出模态配置</td><td>配置只能创建 checkpoint 实际支持的模块拓扑</td></tr>
    <tr><td><code>Construction / post_init</code></td><td>Parameter 与由公式生成的 buffer</td><td>构造结束后的所有运行时状态均已确定且可复现</td></tr>
    <tr><td><code>Checkpoint loading</code></td><td>持久 Parameter 与 persistent buffer</td><td>不能假设 checkpoint 会恢复 <code>persistent=False</code> 状态</td></tr>
    <tr><td><code>Collation</code></td><td>Padding、packing、M-RoPE、sequence boundary</td><td>tensor token 数与 metadata 描述的 token 数一致</td></tr>
    <tr><td><code>Kernel execution</code></td><td>Q/K/V、GDN state、workspace、<code>cu_seqlens</code></td><td>shape、layout、dtype 与 offset 必须闭合</td></tr>
    <tr><td><code>Generation</code></td><td>停止 token 与 decode state</td><td>调用哪一层 <code>generate()</code>，就由哪一层负责终止参数</td></tr>
  </tbody>
</table>
</div>

> [!IMPORTANT]
> Checkpoint 加载完整性与运行时数值正确性属于不同的验证维度。不存在 `missing key` 表明 checkpoint 中已序列化的 Parameter 与 persistent buffer 完成匹配，但该结论不覆盖 non-persistent buffer、runtime metadata、batch layout 和 generation stopping configuration。类似地，OOM 的失败分配点与 CUDA 异步异常的同步报告点均属于错误观测位置，不能直接等同于资源增长或非法访问的起始位置；根因定位仍需沿上游状态转换、tensor shape 和内存分配链路进行追踪

## Qwen3-Omni 非持久 Buffer 在 Meta Materialization 后失效

**触发条件与故障现象。** 当 ms-swift 通过 `mcore-bridge` 构造 Qwen3-Omni，并使 `_from_config()` 创建的 Hugging Face AudioEncoder 经历 `meta device → to_empty()/empty materialization → checkpoint loading` 时，受影响版本会触发该问题。checkpoint 中的持久参数可以全部加载，`missing_keys` 也不会包含 `audio_tower.positional_embedding`；错误直到含音频样本的 batch 进入 AudioEncoder forward 才参与计算

**失败根因。** `SinusoidsPositionEmbedding.__init__()` 在 `meta` device 上执行时，只能得到具有正确 shape 与 dtype 的 meta tensor，并未生成可供计算使用的真实正弦数值；后续 `to_empty()` 或等价 materialization 只为它分配未初始化 storage。该 tensor 又以 `persistent=False` 注册，不进入 `state_dict()`，所以 checkpoint loader 既不会写入它，也不会将它报告为 missing key。受影响版本的模型专用 `_init_weights()` 没有在物化后重新计算这张位置表，未初始化值因此一直保留到 AudioEncoder forward

**直接结果。** 失效的是 AudioEncoder 的绝对正弦位置表，不是 RoPE。`to_empty()` 保留 tensor shape，因此不会产生 shape mismatch；未初始化数值会污染 `audio_features` 及随后写入 LLM 的 `inputs_embeds`，表现为错误输出，含非有限值时还可能出现 `NaN loss`

ms-swift 使用 `mcore-bridge` 后端时，先将 Hugging Face config 转换为 `mcore_bridge.ModelConfig`，随后构造 Megatron 模型。当前入口位于 `swift/megatron/model/utils.py`：

```python
def get_mcore_model(args, hf_config):
    bridge_backend = args.bridge_backend
    if bridge_backend == "megatron-bridge":
        return _get_megatron_bridge_model(args, hf_config)

    config = get_mcore_model_config(args, hf_config)
    models = _get_mcore_model(config)
    return models
```

在自定义多模态模型注册路径中，Megatron 主干与 Hugging Face 音频/视觉子模块并非一次性从 checkpoint 构造完成。Qwen3-Omni 由 `Qwen3OmniBridge + Qwen3Omni_Vit + Qwen3VLLoader` 组合注册；其中 `Qwen3Omni_Vit.prepare_model()` 通过 `_from_config()` 创建 `audio_tower` 与 `visual`，桥接层随后再单独映射 checkpoint 权重。AudioEncoder 因而经历以下顺序：

```text
ms-swift get_mcore_model()
        ↓
mcore-bridge 解析自定义 ModelMeta
        ↓
构造 Megatron LLM 与 Hugging Face multimodal submodule
        ↓
Qwen3OmniMoeAudioEncoder._from_config(audio_config)
        ↓
SinusoidsPositionEmbedding.__init__()
        ↓
在 meta device 上按公式构造 positional_embedding 的 shape/dtype
        ↓
register_buffer(..., persistent=False)
        ↓
to_empty() / empty materialization 分配未初始化 storage
        ↓
桥接 checkpoint 中的持久参数；该 buffer 不在其中
        ↓
AudioEncoder forward 消费错误的位置表
```

关键状态不是普通训练权重，而是由 config 与确定公式生成的 `non-persistent buffer`。Transformers 的 `SinusoidsPositionEmbedding` 在构造函数中计算正弦位置表：

```python
class SinusoidsPositionEmbedding(nn.Module):
    def __init__(self, length, channels, max_timescale=10000):
        super().__init__()
        self.length = length
        self.channels = channels
        self.max_timescale = max_timescale
        position_embedding = (
            self.compute_default_singular_positional_embedding()
        )
        self.register_buffer(
            "positional_embedding",
            position_embedding,
            persistent=False,
        )

    def compute_default_singular_positional_embedding(self):
        log_increment = np.log(self.max_timescale) / (self.channels // 2 - 1)
        inv_timescales = torch.exp(
            -log_increment * torch.arange(self.channels // 2).float()
        )
        scaled_time = (
            torch.arange(self.length)[:, np.newaxis]
            * inv_timescales[np.newaxis, :]
        )
        return torch.cat(
            [torch.sin(scaled_time), torch.cos(scaled_time)],
            dim=1,
        )
```

该 buffer 在 AudioEncoder 中不是用于旋转 Q/K，而是在卷积特征进入 encoder layer 前直接执行逐元素加法：

```python
positional_embedding = (
    self.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
    .unsqueeze(0)
    .to(padded_embed.dtype)
)
padded_embed = padded_embed + positional_embedding
hidden_states = torch.index_select(
    padded_embed.reshape(-1, padded_embed.shape[-1]),
    0,
    valid_indices,
)
```

> [!IMPORTANT]
> `positional_embedding` 是由配置与确定性公式生成的非可学习 buffer，而不是参与梯度更新的 Parameter。`persistent=False` 并不表示该 tensor 不会被物化，而是表示其数值不写入 `state_dict()`。在 meta-device 路径中，`to_empty()` 仍会为它分配真实 storage，但不会恢复正弦位置表的有效数值；checkpoint 又不包含该 key，因此 loader 无法加载或报告这项状态。最终缺失的不是 tensor 对象、shape 或 device，而是该 buffer 应当承载的确定性数值

```text
meta 构造只保留 shape 与 dtype
        ↓
empty materialization 分配未初始化 storage
        ↓
checkpoint 不含该 key
        ↓
load_state_dict() 不加载，也不报告 missing key
        ↓
首次 AudioEncoder forward 读取错误位置表
```

**这些初始化函数的调用关系如下：**

- `__init__()` 创建模型结构、Parameter 和 buffer，AudioEncoder 在构造末尾主动调用 `post_init()`

- `post_init()` 完成模型构造的收尾工作，然后调用 `init_weights()`；它只属于构造阶段，不会在 checkpoint 加载完成后自动再执行一次

- `init_weights()` 决定当前是否可以写入初始值。普通 device 上会继续调用 `initialize_weights()`，meta device 上因为没有真实 storage 而跳过数值初始化

- `initialize_weights()` 遍历模型中的各个子模块，把需要初始化的模块交给 `_initialize_weights()`

- `_initialize_weights()` 跳过已经从 checkpoint 得到有效权重的模块，其余模块再交给模型自己的 `_init_weights(module)`

- `_init_weights(module)` 执行真正的数值写入。通用实现负责 Linear、Conv、Embedding 等常见模块，模型专用实现负责 `SinusoidsPositionEmbedding` 这类特殊 buffer

- checkpoint 加载完成后走的是 `_initialize_missing_keys() → initialize_weights() → _initialize_weights() → _init_weights()`，而不是重新调用 `post_init()`。受影响版本虽然进入这条初始化链，但 `_init_weights()` 中没有 `SinusoidsPositionEmbedding` 分支，因此未初始化的位置表没有被恢复

**正确修复位置。** 当前 Transformers 的 `Qwen3OmniMoePreTrainedModel._init_weights()` 已显式重建两个状态：

```python
@torch.no_grad()
def _init_weights(self, module):
    super()._init_weights(module)

    if isinstance(module, Qwen3OmniMoeCode2Wav):
        init.copy_(
            module.code_offset,
            torch.arange(module.config.num_quantizers).view(1, -1, 1)
            * module.config.codebook_size,
        )
    elif isinstance(module, SinusoidsPositionEmbedding):
        position_embeddings = (
            module.compute_default_singular_positional_embedding()
        )
        init.copy_(module.positional_embedding, position_embeddings)
```

这比在 ms-swift 或 mcore-bridge 中按字段名手工修补更稳定，因为状态所有权仍属于模型定义本身：

```text
__init__() 生成确定值
        ↓
post_init() 统一执行模型初始化协议
        ↓
model-specific _init_weights() 重建 non-persistent state
        ↓
checkpoint 加载持久参数
        ↓
普通 from_pretrained 与 bridge _from_config 路径得到相同最终状态
```

Whisper 类 AudioEncoder、由 config 生成的 position table、RoPE cache、codebook offset 和其他 deterministic buffer 都应按同一原则审查：如果某个状态不进入 `state_dict()`，就必须验证所有构造路径结束后它仍然正确

## 关闭 `padding_free` 后 Qwen3.5 GDN 显存暴涨

**显存增长来源。** Qwen3.5 同时包含 Full Attention 与大量 Gated DeltaNet 层。关闭 `padding_free` 后，显存上涨首先来自参与前向和反向计算的 token 数改变，不应优先归因于 recurrent state 泄漏

假设一个 micro-batch 中四条样本的实际长度为 `2000、8000、16000、32000`：

```text
padding_free = true

input_ids.shape = [1, 58000]
cu_seqlens = [0, 2000, 10000, 26000, 58000]
实际参与计算：58000 token


padding_free = false

input_ids.shape = [4, 32000]
四条样本全部补齐到 32000
实际参与计算：128000 token
```

同一个 micro-batch 的 token 数由 `58000` 增加到 `128000`。普通 Transformer 会增加 activation；GDN 还会为每个 token 产生 Q/K/V、gate、beta、decay 与 chunk-state 相关中间结果，训练阶段又必须为 backward 保留必要张量，因此差距会被多层累计。当前 Megatron collator 在非 `padding_free` 路径还会构造 `[B,1,S,S]` 的布尔 causal mask，长序列下这部分 `B × S²` 开销也必须计入，不能把全部增量只归因于 GDN activation

```text
关闭 padding_free
        ↓
计算规模从 sum(actual_seq_len)
变为 batch_size × max_seq_len
        ↓
padding token 进入每一层 GDN 与 FFN
        ↓
activation + workspace + backward saved tensor 同时增长
        ↓
后续某个普通 buffer 分配成为最终 OOM 触发点
```

FLA 的 variable-length GDN 接口本身能够接收 flatten 后的序列与 `cu_seqlens`。当前 ms-swift 的 Qwen3.5 patch 也只在真正使用 FLA kernel 时把边界传入：

```python
chunk_kwargs = {
    "g": g,
    "beta": beta,
    "initial_state": None,
    "output_final_state": cache_params is not None,
    "use_qk_l2norm_in_kernel": True,
}

if (
    cu_seqlens is not None
    and selected_chunk_gated_delta_rule is chunk_gated_delta_rule
):
    chunk_kwargs["cu_seqlens"] = cu_seqlens

core_attn_out, last_recurrent_state = (
    selected_chunk_gated_delta_rule(
        query,
        key,
        value,
        **chunk_kwargs,
    )
)
```

相邻 offset 确定每条样本的独立边界，kernel 不需要把不同样本的 recurrent state 串联起来。当前 ms-swift 还会在 packed batch 包含多条序列、但 FLA/causal-conv kernel 不可用时直接拒绝执行，而不是静默使用不理解 `cu_seqlens` 的 fallback：

```python
if (
    _has_multiple_sequences(cu_seqlens)
    and (causal_conv1d is None or chunk_gated_delta_rule is None)
):
    raise ImportError(
        "Qwen3.5 linear attention packing/padding-free with "
        "multiple sequences requires flash-linear-attention"
    )
```

Megatron 配置层对 `padding_free` 的自动降级也不是因为 GDN。`swift/megatron/model/utils.py` 当前只对不兼容的 `unfused` Attention backend 关闭该模式：

```python
def _check_padding_free(args, config):
    if not args.padding_free:
        return

    attention_backend = config.attention_backend.name
    if attention_backend == "unfused":
        logger.warning(
            "Attention backend unfused is not supported in "
            "padding-free mode. Setting args.padding_free to False."
        )
        args.padding_free = False
```

在相关 OOM trace 中，错误最终报告在类似 `v_new = torch.empty_like(u)` 的位置。源码显示 `v_new` 默认与 `u` 同尺寸，但该行是否只是最后一次申请仍应结合显存 trace 判断：

```text
前面各层已经保留大量 activation
        ↓
当前 GDN 已创建 w、u、A 与 chunk state h
        ↓
v_new 再申请一份与 u 同尺寸的 tensor
        ↓
剩余显存不足，错误在这一行暴露
```

正确处理原则如下：

- 当前 MCore、FLA 与 causal-conv 路径能够完整传递 sequence boundary 时，保持 `padding_free=true`

- 确认 batch 已 flatten 为 `[1, sum(Li), ...]`，并验证 `cu_seqlens[-1] == sum(Li)`

## Transformers 5.2–5.3 中尾批 `batch=1` 触发 M-RoPE 与 FA2 metadata 越界

**版本边界。** Transformers `5.2.0` 与 `5.3.0` 仍受 [issue #44910](https://github.com/huggingface/transformers/issues/44910) 影响，修复见 [PR #44399](https://github.com/huggingface/transformers/pull/44399)，并已进入 `5.4.0`

**尾批触发条件。** Qwen3.5 是 Hybrid 模型：GDN 层执行 recurrent update，Full Attention 层仍可能进入 FlashAttention 2。旧版 Qwen3.5 调用链中的 M-RoPE `position_ids` 为 `[3,B,S]`；Full Attention 已经通过 `position_embeddings` 应用 RoPE，但 DecoderLayer 又把三维 `position_ids` 作为 `**kwargs` 继续传给通用 Attention backend

```python
hidden_states, _ = self.self_attn(
    hidden_states=hidden_states,
    attention_mask=attention_mask,
    position_ids=position_ids,
    past_key_values=past_key_values,
    position_embeddings=position_embeddings,
    **kwargs,
)
```

Transformers FA2 的 packed-sequence 探测包含一个关键条件 `batch_size == 1`：

```python
def _is_packed_sequence(position_ids, batch_size):
    if position_ids is None:
        return False

    increasing_position_sequences = (
        torch.arange(
            position_ids.shape[1],
            device=position_ids.device,
        )
        + position_ids.min()
    )
    return batch_size == 1 and (
        increasing_position_sequences - position_ids
    ).abs().sum().bool()
```

前面的完整 batch 即使携带错误维度，也会被第一个条件挡住：

```text
position_ids.shape = [3, 8, S]
query batch_size = 8
        ↓
batch_size == 1 为 False
        ↓
不进入 position_ids → packed metadata 路径
```

尾批只有一条样本时，同一个潜伏的数据解释错误突然被激活：

```text
position_ids.shape = [3, 1, S]
query batch_size = 1
        ↓
_is_packed_sequence() 返回 True
        ↓
三维 M-RoPE position_ids 被误作二维 packed position metadata
```

**错误 metadata 的形成。** FA2 的 helper 会先将 `position_ids` flatten，再根据值为 `0` 的位置构造 sequence start：

```python
def prepare_fa_kwargs_from_position_ids(position_ids):
    position_ids = position_ids.reshape(-1)
    indices_q = (position_ids == 0).nonzero().view(-1)
    cu_seq_lens_q = torch.cat(
        (
            indices_q.to(dtype=torch.int32),
            torch.tensor(
                position_ids.size(),
                dtype=torch.int32,
                device=position_ids.device,
            ),
        )
    )
    return cu_seq_lens_q
```

当 `S=256` 时，实际 Q 只有 256 个 token，但三条 M-RoPE 轴 flatten 后包含 768 个位置：

```text
真实 query
q.shape = [256, num_heads, head_dim]
q token count = 256

错误输入 metadata
position_ids.shape = [3, 1, 256]
position_ids.reshape(-1).shape = [768]
cu_seqlens_q = [0, 256, 512, 768]

关键不变量被破坏
cu_seqlens_q[-1] = 768
q.shape[0] = 256
```

FlashAttention varlen kernel 信任 `cu_seqlens` 描述的边界，因而可能按照 768 个 token 访问只有 256 行的 Q/K/V，结果是错误 attention、非法显存访问或 CUDA 越界。该次复现还观察到 logits 退化、持续输出异常字符和 EOS 未及时出现；这些是错误 attention 的后续表现，不是源码保证发生的固定症状。由于 CUDA 异步执行，异常也可能在后续 `.tolist()`、tensor copy 或其他同步点才被报告

```text
尾批 B=1
        ↓
FA2 packed-sequence 探测被激活
        ↓
3D M-RoPE position_ids 被误作 packed metadata
        ↓
cu_seqlens[-1] != q token count
        ↓
FA2 varlen kernel 越界或产生错误 attention
        ↓
logits / EOS 行为退化
        ↓
CUDA 异步错误在后续同步点暴露
```

**修复原则。** M-RoPE 的三维 `position_ids` 属于 rotary-position 语义，不应自动成为 FA2 的 packed-sequence metadata。稳健修复可以位于以下任一明确边界，但不应仅通过固定 `drop_last=true` 隐藏触发条件：

- Full Attention 已通过 `position_embeddings` 应用 RoPE 后，不再把三维 `position_ids` 传给通用 FA packed detector

- `_is_packed_sequence()` 仅接受协议规定的二维 `[1,S]` position metadata，遇到 `[3,1,S]` 时直接拒绝推断

- collator 显式生成独立的 `cu_seqlens_q/cu_seqlens_k`，Attention backend 不再从 M-RoPE tensor 猜测 sequence boundary

- 在调用 varlen kernel 前执行 fail-fast 检查：

```python
if cu_seqlens_q is not None:
    # q 是即将送入 varlen kernel 的 flattened query operand
    if int(cu_seqlens_q[-1]) != q.shape[0]:
        raise ValueError(
            "FlashAttention metadata mismatch: "
            f"cu_seqlens_q[-1]={int(cu_seqlens_q[-1])}, "
            f"query_tokens={q.shape[0]}"
        )
```

**上游修复。** Qwen3.5 TextModel 在进入 DecoderLayer 前已经分别持有三维 M-RoPE `position_ids` 与二维 `text_position_ids`。`5.2.0` 和 `5.3.0` 错误地继续传递前者；`5.4.0` 起改为传递后者：

```python
# Transformers 5.2.0 / 5.3.0
hidden_states = decoder_layer(
    hidden_states,
    position_embeddings=position_embeddings,
    position_ids=position_ids,       # [3, B, S]
    ...,
)

# Transformers >= 5.4.0
hidden_states = decoder_layer(
    hidden_states,
    position_embeddings=position_embeddings,
    position_ids=text_position_ids,  # [B, S]
    ...,
)
```

## `enable_audio_output` 在 Thinker-Talker 架构中的作用

**配置开关决定模块拓扑。** `enable_audio_output` 控制 Qwen3-Omni 顶层模型的模块组成。Transformers 在 `__init__()` 中读取该配置，并据此决定是否创建 Talker 和 Code2Wav：

```python
class Qwen3OmniMoeForConditionalGeneration(
    Qwen3OmniMoePreTrainedModel,
    GenerationMixin,
):
    def __init__(self, config):
        super().__init__(config)
        self.thinker = (
            Qwen3OmniMoeThinkerForConditionalGeneration
            ._from_config(config.thinker_config)
        )
        self.has_talker = config.enable_audio_output
        if self.has_talker:
            self.enable_talker()
        self.post_init()

    def enable_talker(self):
        self.talker = (
            Qwen3OmniMoeTalkerForConditionalGeneration
            ._from_config(self.config.talker_config)
        )
        self.code2wav = Qwen3OmniMoeCode2Wav._from_config(
            self.config.code2wav_config
        )
```

`enable_audio_output=true` 对应完整的 Thinker-Talker 拓扑，模型包含 Thinker、Talker 与 Code2Wav；`enable_audio_output=false` 对应 Thinker-only 拓扑

```text
读取 config.json
        ↓
创建 Thinker
        ↓
检查 enable_audio_output
        ├── false → 模型仅包含 Thinker
        │
        └── true  → enable_talker()
                        ├── 创建 Talker
                        └── 创建 Code2Wav
```

ms-swift 的 `Qwen3OmniLoader.get_config()` 在模型创建前读取 `ENABLE_AUDIO_OUTPUT`：

```python
class Qwen3OmniLoader(ModelLoader):
    def get_config(self, model_dir: str):
        self._check_qwen_omni_utils()
        config = super().get_config(model_dir)
        enable_audio_output = get_env_args(
            "ENABLE_AUDIO_OUTPUT", bool, None
        )
        if enable_audio_output is not None:
            config.enable_audio_output = enable_audio_output
        return config
```

仅训练或使用 Thinker 时，可以在构造前关闭 Talker 与 Code2Wav：

```bash
ENABLE_AUDIO_OUTPUT=false swift megatron sft ...
```

直接使用 Transformers 时可以在加载参数中覆盖该配置：

```python
model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    model_path,
    enable_audio_output=False,
)
```
