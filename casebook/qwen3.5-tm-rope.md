# Interleaved M-RoPE 解析

## 1. RoPE

RoPE（Rotary Position Embedding）通过旋转 Q/K 特征，使 attention 内积包含相对位置信息。设旋转维度为偶数 $`d_r`$，频率底数为 $`\beta`$，位置为 $`p`$；第 $`i`$ 个二维子空间的角频率与旋转角为

```math
\omega_i=\beta^{-2i/d_r},\qquad
\phi_{p,i}=p\omega_i,\qquad
i=0,\ldots,\frac{d_r}{2}-1.
```

- $`\beta`$ 对应 `rope_theta`，源码变量名为 `base`，是生成频率序列的底数
- 当 $`\beta>1`$ 时，频率以公比 $`\beta^{-2/d_r}`$ 递减：$`\omega_0=1`$，后续频率逐渐降低。增大 `base` 会拉长 $`i>0`$ 各分量的周期 $`2\pi/\omega_i`$

- `inv_freq[i]` 表示第 $`i`$ 对旋转特征每单位位置的相位增量，实际旋转角为 `position_id * inv_freq[i]`

<table align="center" width="720">
<tr><td align="left" width="720">

```json
{
  "text_config": {
    "hidden_size": 5120,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "max_position_embeddings": 262144,
    "rope_parameters": {
      "rope_type": "default",
      "rope_theta": 10000000,
      "partial_rotary_factor": 0.25,
      "mrope_interleaved": true,
      "mrope_section": [11, 11, 10]
    }
  }
}
```

</td></tr>
</table>

源码采用 `[B,H,T,d_h]` 布局：`B` 为 batch size，`T` 为当前输入长度，Q/K 的 head 数分别为 `24/4`；`head_dim` 为 $`d_h=256`$，旋转比例 $`r=0.25`$，故 $`d_r=\lfloor d_h r\rfloor=64`$

<table align="center" width="720">
<tr><td align="left" width="720">

```text
Q / K            [B, 24, T, 256] / [B, 4, T, 256]
position_ids     [B, T]
inv_freq         [32]
相位 Φ           [B, T, 32]
cos / sin        [B, T, 64] → unsqueeze(1) → [B, 1, T, 64]
```

</td></tr>
</table>

先由 `Qwen3_5TextRotaryEmbedding.compute_default_rope_parameters()` 计算 32 个频率：

```python
base = config.rope_parameters["rope_theta"]
partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
dim = int(head_dim * partial_rotary_factor)

attention_factor = 1.0
inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
return inv_freq.to(device), attention_factor
```

- `arange(0,64,2)/64` 生成指数 $`[0,1/32,\ldots,31/32]`$，得到 `inv_freq` 为 $`[1,10^{-7/32},\ldots,10^{-217/32}]`$
- 频率按旋转维度 `64` 直接计算，而非从全维 RoPE 中截取

RoFormer §3.2.1 式（13）先给出二维形式。只写 Q 分支，以 $`\mathbf q_m`$ 表示旋转前的 Q、$`\widetilde{\mathbf q}_m`$ 表示旋转后的 Q：

```math
\widetilde{\mathbf q}_m=f_q(\mathbf x_m,m)
=
\begin{bmatrix}
\cos(m\omega)&-\sin(m\omega)\\
\sin(m\omega)&\cos(m\omega)
\end{bmatrix}
\begin{bmatrix}
W_q^{(11)}&W_q^{(12)}\\
W_q^{(21)}&W_q^{(22)}
\end{bmatrix}
\begin{bmatrix}x_m^{(1)}\\x_m^{(2)}\end{bmatrix}
=R(m\omega)\mathbf q_m,
\qquad \mathbf q_m=W_q\mathbf x_m.
```

- 从右向左计算：输入 $`\mathbf x_m`$ 先经 $`W_q`$ 投影得到 $`\mathbf q_m`$，再乘旋转矩阵得到 $`\widetilde{\mathbf q}_m`$。角频率统一记为 $`\omega`$，对应前文某一通道对的 $`\omega_i`$；$`m\omega`$ 是位置乘频率

- 对应到 Qwen3.5，取一个 token、一个 head 传入 RoPE 的 Q 向量 `q`（已完成投影和 Q norm）。设位置 $`p=1`$，`q[0]=1`、`q[32]=2`；这对通道的频率 $`\omega_0=1`$，因此：

```math
\begin{bmatrix}\widetilde q_0\\\widetilde q_{32}\end{bmatrix}
=
\begin{bmatrix}\cos 1&-\sin 1\\\sin 1&\cos 1\end{bmatrix}
\begin{bmatrix}1\\2\end{bmatrix}
\approx
\begin{bmatrix}-1.142640\\1.922076\end{bmatrix}.
```

- 结果已经是旋转后 Q 的第 0、32 维。其余 31 对同样计算，再接回未旋转的后 192 维，得到完整的 $`\widetilde{\mathbf q}_p=\widehat R_p\mathbf q_p\in\mathbb R^{256}`$；$`\widehat R_p`$ 的完整矩阵见下文。K 同理，随后用旋转后的 Q/K 计算 attention 内积。同一 token 的各 head 复用 cos/sin，分别旋转各自的 Q/K 数值

RoFormer 式（15）对全部 $`d`$ 维旋转。令 $`F=d/2`$、$`\phi_i=p\beta^{-2i/d}`$，以从 0 开始的索引写为：

```math
R_p^{(d)}=
\begin{pmatrix}
\cos\phi_0&-\sin\phi_0&0&0&\cdots&0&0\\
\sin\phi_0& \cos\phi_0&0&0&\cdots&0&0\\
0&0&\cos\phi_1&-\sin\phi_1&\cdots&0&0\\
0&0&\sin\phi_1& \cos\phi_1&\cdots&0&0\\
\vdots&\vdots&\vdots&\vdots&\ddots&\vdots&\vdots\\
0&0&0&0&\cdots&\cos\phi_{F-1}&-\sin\phi_{F-1}\\
0&0&0&0&\cdots&\sin\phi_{F-1}& \cos\phi_{F-1}
\end{pmatrix}
\in\mathbb R^{d\times d}.
```

- 每个 $`2\times2`$ 块作用于 Q 的相邻分量 $`(q_{2i},q_{2i+1})`$，K 同理；整体为 $`\operatorname{diag}(R(\phi_0),\ldots,R(\phi_{F-1}))`$。式（14）定义 $`\widetilde{\mathbf q}_m=R_m^{(d)}W_q\mathbf x_m`$、$`\widetilde{\mathbf k}_n=R_n^{(d)}W_k\mathbf x_n`$；记投影结果为 $`\mathbf q_m,\mathbf k_n`$，由 $`R_m^\top R_n=R_{n-m}`$ 得到相对位置内积：

```math
\begin{aligned}
\widetilde{\mathbf q}_m^\top\widetilde{\mathbf k}_n
&=(R_m\mathbf q_m)^\top(R_n\mathbf k_n)\\
&=\mathbf q_m^\top R_m^\top R_n\mathbf k_n\\
&=\mathbf q_m^\top R_{n-m}\mathbf k_n.
\end{aligned}
```

Qwen3.5 的 partial RoPE 是分块正交变换：前 64 维参与旋转，后 192 维由单位映射保留。令 $`D_c(p)=\operatorname{diag}(\cos\phi_{p,0},\ldots,\cos\phi_{p,31})`$、$`D_s(p)=\operatorname{diag}(\sin\phi_{p,0},\ldots,\sin\phi_{p,31})`$，其中 $`\phi_{p,i}=p\beta^{-2i/64}`$，则

```math
\widehat R_p=
\left(
\begin{array}{cc|c}
D_c(p)&-D_s(p)&0\\
D_s(p)& D_c(p)&0\\\hline
0&0&I_{192}
\end{array}
\right),\qquad
\widetilde{\mathbf q}_p=\widehat R_p\mathbf q_p,
\qquad
\widetilde{\mathbf k}_p=\widehat R_p\mathbf k_p.
```

- 左上角的 $`64\times64`$ 旋转块记为 $`\mathcal R_p`$，通过固定的通道置换与论文的块对角矩阵对应。旋转在 attention 内积之前分别作用于 Q 和 K，只在各自的成对通道内作线性组合，保持向量范数，不含平移项；旋转通道与保留通道之间没有交叉混合。

- 完整 head 在添加 mask 前的 attention score 因而分解为：

```math
s_{mn}=\frac{
(\mathbf q_m^{\mathrm{rot}})^\top \mathcal R_{n-m}\mathbf k_n^{\mathrm{rot}}
+(\mathbf q_m^{\mathrm{pass}})^\top\mathbf k_n^{\mathrm{pass}}
}{\sqrt{d_h}}.
```

- 第一项通过 $`\mathcal R_{n-m}`$ 引入相对位置，第二项是保留通道的普通内积；两项共同构成 attention score，缩放分母仍为 $`\sqrt{d_h}=\sqrt{256}`$。完整的 256 维 K 均写入 cache

对普通 RoPE，位置张量为 $`P\in\mathbb Z^{B\times T}`$，其中 $`P_{b,t}`$ 保存第 $`b`$ 个样本中第 $`t`$ 个 token 的位置编号。每个位置与 32 个频率分别相乘，得到该 token 的 32 个旋转角：

```math
\Phi_{b,t,i}=P_{b,t}\omega_i,\qquad
\Phi\in\mathbb R^{B\times T\times(d_r/2)},
\qquad
C=[\cos\Phi,\cos\Phi],\quad S=[\sin\Phi,\sin\Phi].
```

- $`\Phi`$ 保存旋转角，本例形状为 `[B,T,32]`；$`C`$、$`S`$ 保存对应的 cos/sin 系数。方括号表示沿最后一维拼接，使 $`C/S`$ 的形状成为 `[B,T,64]`：第 $`i`$、$`i+32`$ 个通道属于同一旋转对，因而需要相同的系数。`unsqueeze(1)` 再将系数变为 `[B,1,T,64]`，供同一 token 的所有 Q/K head 复用。

- 固定一个 token 和一个 head，记 $`p=P_{b,t}`$。源码将二维旋转矩阵的乘法展开为逐元素乘加，第 $`i`$、$`i+32`$ 个通道的输出为：

```math
\begin{aligned}
\widetilde q_i&=q_i\cos(p\omega_i)-q_{i+32}\sin(p\omega_i),\\
\widetilde q_{i+32}&=q_i\sin(p\omega_i)+q_{i+32}\cos(p\omega_i),
\qquad i=0,\ldots,31.
\end{aligned}
```

- `rotate_half(q_rot)` 在这两个位置分别提供 $`-q_{i+32}`$ 和 $`q_i`$；乘以 sin 后，再加上 `q_rot * cos`，就得到上式。这里代码中的 `cos`、`sin` 对应广播后的 $`C`$、$`S`$，K 的计算相同

```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed
```

## 2. Mental Model：token 序号、空间坐标和频率槽

理解 M-RoPE，需要分清三个不同的索引。

| 索引 | 回答的问题 | 示例 |
| --- | --- | --- |
| token 序号 | 当前 token 位于输入序列的第几列？ | 第 4 列，zero-based index 为 `4` |
| `T / H / W` 坐标 | 当前 token 在多模态位置空间中在哪里？ | `(2, 3, 2)` |
| 频率槽 `j` | Q/K 的哪一对旋转维度使用哪种频率？ | 槽 `j=1` 选取 H 坐标 |

普通文本的位置是一维的。M-RoPE 将同一个位置复制到三路，例如位置 `5` 表示为 `(5, 5, 5)`；无论某个频率槽选择 T、H 还是 W，最后得到的角度都相同，因此退化为该配置下的一维 partial RoPE。

图像 token 则来自二维网格。同一张图像的 T 坐标相同，H/W 分别表示行列；展平之后，相邻 token 可以共享其中一个坐标，却仍然占用不同的序列位置和 KV cache 条目。

```text
序列 token index     0   1  │  2   3   4   5  │  6   7   8
输入类型             text  │  image: 2 × 2   │    text
                           │                │
M-RoPE T             0   1 │  2   2   2   2 │  4   5   6
M-RoPE H             0   1 │  2   2   3   3 │  4   5   6
M-RoPE W             0   1 │  2   3   2   3 │  4   5   6
```

这个例子有 9 个实际 token，但最大 RoPE 坐标是 6，后续文本从位置 7 继续。两者的差保存在 `rope_deltas = 7 - 9 = -2` 中。

> [!IMPORTANT]
> M-RoPE 压缩的是视觉段占据的坐标跨度。图像的 4 个 LLM token 仍然是 4 个 token；位置编号变小不会减少 attention 的 token 数，也不会把 4 个 KV cache 条目合成 2 个。

完整调用链可以拆为两个阶段：先给 token 分配坐标，再根据坐标旋转 Q/K。

```text
input_ids + mm_token_type_ids + image/video_grid_thw
                         │
                         ▼
Qwen3_5Model.get_rope_index()
    └─ get_vision_position_ids()
                         │
                         ├─ position_ids: [3, B, S]
                         └─ rope_deltas:  [B, 1]
                         │
                         ▼
Qwen3_5TextModel.forward()
    └─ Qwen3_5TextRotaryEmbedding.forward()
           ├─ 三路坐标 × inv_freq → [3, B, S, F]
           ├─ cos/sin → 各 [3, B, S, F]
           └─ recomposition_frequencies() → 各 [B, S, R]
                         │
                         ▼
full-attention 层：Qwen3_5Attention.forward()
    └─ apply_rotary_pos_emb(Q, K, cos, sin)
           ├─ 旋转 Q/K 的前 R 维
           ├─ K/V 写入 cache
           └─ attention(Q_rotated, K_rotated, V)
```

这里 `B` 是 batch size，`S` 是本次处理的 token 数，`R` 是实际旋转维度，`F=R/2` 是独立频率数。`3` 表示坐标轴数量，与 batch size、attention head 数量无关。

## 3. 视觉位置：从 patch grid 到 LLM 坐标

### 3.1 两个同名函数处于不同阶段

本文使用的是 `Qwen3_5Model.get_vision_position_ids()`，它在视觉 token 已经进入语言模型序列时分配三路坐标。

同一文件还从 `transformers.vision_utils` 导入了另一个 `get_vision_position_ids`，供 `Qwen3_5VisionModel` 内部构造视觉编码器的位置。两者的调用对象、输入语义和输出布局不同，阅读代码时需要看完整的调用位置。

`grid_thw` 表示视觉 patch embedding 之后的网格，不是原图的像素尺寸。进入语言模型之前还要按 `spatial_merge_size` 做空间合并。模型方法保留了时间合并与时间间距参数，关键运算如下，省略 docstring：

```python
llm_grid_t, llm_grid_h, llm_grid_w = (
    grid_thw[0].item() // temp_merge_size,
    grid_thw[1].item() // spatial_merge_size,
    grid_thw[2].item() // spatial_merge_size,
)

position_temporal = torch.arange(llm_grid_t, device=device) * time_interval
position_height = torch.arange(llm_grid_h, device=device) + start_position
position_width = torch.arange(llm_grid_w, device=device) + start_position

T_grid, H_grid, W_grid = torch.meshgrid(
    position_temporal, position_height, position_width, indexing="ij"
)
vision_position_ids = torch.stack([T_grid, H_grid, W_grid], dim=0).reshape(3, -1)
vision_position_ids[0] += start_position
```

用 `s=start_position`、`τ=time_interval` 表示偏移和时间间距，每个合并后网格点 `(t, h, w)` 的坐标是：

```text
T = s + t × τ
H = s + h
W = s + w
```

T 轴的 `start_position` 在乘以 `time_interval` 之后才加上。因此计算的是 `s + t × τ`，不会把整段起点也乘上时间间距。

`meshgrid(..., indexing="ij")` 再 `reshape` 的展平顺序是 W 最快、H 次之、T 最慢；输出为 `[3, llm_grid_t × llm_grid_h × llm_grid_w]`。

### 3.2 一个完整的图像例子

设图像 patch grid 是 `[1, 4, 4]`，`spatial_merge_size=2`，本段从 RoPE 位置 `s=2` 开始：

```text
patch grid             [1, 4, 4]   → 16 个视觉 patch
LLM grid               [1, 2, 2]   →  4 个语言模型视觉 token

position_temporal      [0]
position_height        [2, 3]
position_width         [2, 3]

展平后的 token          image_0   image_1   image_2   image_3
局部网格坐标 (h, w)       (0,0)     (0,1)     (1,0)     (1,1)

T                          2         2         2         2
H                          2         2         3         3
W                          2         3         2         3
```

第 3 个图像 token，即 `image_2`，得到 `(T,H,W)=(2,3,2)`。后面的频率和旋转例子继续使用这个 token。

这里有两次独立变化：视觉模型的空间合并把 16 个 patch 变为 4 个 LLM token；M-RoPE 再让这 4 个 token 使用跨度为 2 的空间坐标。后者没有继续压缩 token 数。

## 4. 全序列位置：`get_rope_index()` 如何拼接各模态

### 4.1 模态类型来自 processor

`get_rope_index()` 接收 `input_ids` 和与之逐 token 对齐的 `mm_token_type_ids`，Qwen3.5 使用以下三种类型：

| `mm_token_type_ids` | 本模型中的含义 | 位置分配 |
| --- | --- | --- |
| `0` | 普通文本及非视觉占位 token | 三轴使用相同的连续位置 |
| `1` | 图像占位 token | 读取一项 `image_grid_thw` |
| `2` | 视频占位 token | 读取一项拆帧后的 `video_grid_thw` |

Processor 的通用基类还定义了 audio type `3`，但 **Qwen3.5 的 `grid_iters` 只有 `1` 和 `2`**，不存在本模型的音频位置分支。通用字段支持某种类型，不等于所有模型都实现了该类型。

模态标注依据 image/video token ID 匹配生成。`<|vision_start|>`、`<|vision_end|>` 和视频时间戳文本不属于视觉占位 token，走 type `0` 的顺序编号。参见 `create_mm_token_type_ids()`。

### 4.2 视频先拆成逐帧视觉段

Qwen3.5 复用 `Qwen3VLProcessor`。其 `replace_video_token()` 为每个时间网格单元生成如下内容：

```text
<0.5 seconds><|vision_start|>[本帧 video token ...]<|vision_end|>
<1.5 seconds><|vision_start|>[本帧 video token ...]<|vision_end|>
```

时间字符串会经过 tokenizer，不能把整个 `<0.5 seconds>` 当作必然只有一个 token。这里的“帧”是视频预处理得到的时间网格单元；`grid_thw[0]` 已经反映 temporal patching，不应再次当作原始视频总帧数。

与这种序列格式对应，模型先把一个视频的网格拆开：

```python
if video_grid_thw is not None:
    video_grid_thw = torch.repeat_interleave(
        video_grid_thw, video_grid_thw[:, 0], dim=0
    )
    video_grid_thw[:, 0] = 1
```

```text
原 video_grid_thw         [[2, 4, 4]]
拆分后                   [[1, 4, 4],
                          [1, 4, 4]]

输入分组                 text → frame_0 → text → frame_1 → text
每帧 LLM 网格            [1, 2, 2]，假设 spatial_merge_size=2
```

> [!IMPORTANT]
> `get_vision_position_ids()` 虽然提供 `time_interval` 参数，但 Qwen3.5 的 `get_rope_index()` 调用没有传入它，使用默认值 `1`；拆帧后局部 T 长度又是 `1`，因此每帧视觉段的 T 全部等于该段的 `current_pos`。视频的实际时间通过时间戳文本进入序列，不能把本实现写成“按秒数乘时间间距，直接生成整段视频的 T 坐标”。

不同帧仍有不同的全局 T 坐标，因为前面的文本和视觉段会推进 `current_pos`。只是这个数值不是该帧的秒数。

### 4.3 去掉 padding，再按连续模态分组

模型先按 `attention_mask` 去掉当前样本的 padding，再用 `itertools.groupby` 合并连续的相同模态类型：

```python
current_input_ids = current_input_ids[attention_mask[batch_idx].bool()]
input_token_type = input_token_type[attention_mask[batch_idx].bool()]

for key, group in itertools.groupby(
    enumerate(input_token_type.tolist()), lambda x: x[1]
):
    group = list(group)
    start_index = group[0][0]
    end_index = group[-1][0] + 1
    input_type_group.append((key, start_index, end_index))
```

上面的 mask 两行仅在 `attention_mask is not None` 时执行。分组结果中的索引对应去掉 padding 后的有效序列。

```text
mm_token_type_ids     [0, 0, 1, 1, 1, 1, 0, 0, 0]
input_type_group      [(0, 0, 2), (1, 2, 6), (0, 6, 9)]
                       text      image       text
```

图像和视频各自有一个 grid iterator，按 batch 行及行内 token 的出现顺序消费。每个连续视觉段对应一项网格，所以 processor 展开的占位 token 数、模态标注和 grid 顺序必须一致；正常模板中的视觉起止标记也会分隔相邻视觉段。

### 4.4 `current_pos` 记录下一个坐标起点

每条 batch 样本都从 `current_pos=0` 开始。源码的两种分配分支如下：

```python
if modality_type == 0:
    text_len = end_idx - start_idx
    llm_pos_ids_list.append(
        torch.arange(text_len, device=input_ids.device)
        .view(1, -1).expand(3, -1) + current_pos
    )
    current_pos += text_len
else:
    grid_thw = next(grid_iters[modality_type])
    vision_position_ids = self.get_vision_position_ids(
        current_pos, grid_thw, 1, spatial_merge_size,
        device=input_ids.device,
    )
    llm_pos_ids_list.append(vision_position_ids)
    current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size
```

在有效且可整除的空间网格下，令 `h=H/merge_size`、`w=W/merge_size`：

| 分组 | 占用的实际 token 数 | `current_pos` 增量 |
| --- | --- | --- |
| 长度为 `n` 的文本 | `n` | `n` |
| 一张图像，LLM 网格 `[1,h,w]` | `h × w` | `max(h,w)` |
| 拆分后的一帧视频，LLM 网格 `[1,h,w]` | `h × w` | `max(h,w)` |

`current_pos` 是下一段在坐标空间中的起点，不能用它替代已经处理的 token 数。

### 4.5 拼接结果与 delta

各段沿 token 轴拼接，随后把有效位置写回原来的 batch/padding 布局：

```python
llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)

if attention_mask is not None:
    position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = (
        llm_positions.to(position_ids.device)
    )
else:
    position_ids[:, batch_idx] = llm_positions.to(position_ids.device)

mrope_position_deltas.append(
    llm_positions.max() + 1 - len(current_input_ids)
)
mrope_position_deltas = torch.tensor(
    mrope_position_deltas, device=input_ids.device
).unsqueeze(1)
```

`position_ids` 初始化为零，padding 列没有被写入，仍然为零。返回值的实际形状是：

```text
position_ids                 [3, B, S]
mrope_position_deltas         [B, 1]
```

源码 docstring 把 delta 的形状简写为 `(batch_size)`，但函数末尾的 `unsqueeze(1)` 明确产生 `[B,1]`，应以执行代码为准。

继续使用第 1 节的完整例子：

```text
段                    text 前缀      image 2×2      text 后缀
token 数                   2              4              3
current_pos             0 → 2          2 → 4          4 → 7

有效 token 数 N         2 + 4 + 3 = 9
最大位置 M              6
下一位置 P              M + 1 = 7
rope_delta              P - N = -2
```

## 5. 位置变为角度：`Qwen3_5TextRotaryEmbedding`

### 5.1 旋转维度由 `partial_rotary_factor` 决定

`compute_default_rope_parameters()` 根据 `head_dim` 和 `partial_rotary_factor` 计算参与旋转的维度：

```python
base = config.rope_parameters["rope_theta"]
partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
head_dim = getattr(config, "head_dim", None) or (
    config.hidden_size // config.num_attention_heads
)
dim = int(head_dim * partial_rotary_factor)

inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
```

以下数值示例设置为：

```text
head_dim                    D = 256
partial_rotary_factor       r = 0.25
rotary_dim                  R = int(D × r) = 64
独立频率数                  F = R / 2 = 32
rope_theta                  θ = 10000
mrope_section                   [11, 11, 10]，由 rotary 类在缺省时取用
```

对第 `j` 个频率槽：

```text
inv_freq[j] = θ^(-2j/R),    j = 0, 1, ..., F-1
```

这 32 个逆频率从高到低排列，每个频率最终对应旋转子空间中的两个通道。三路坐标共用同一组逆频率，不会各自重新生成一套 H/W 频率表。

`rope_type="default"` 使用上述公式；其他类型通过 `ROPE_INIT_FUNCTIONS` 选择初始化规则。

### 5.2 张量展开与矩阵乘法

`forward()` 将输入的三路位置分别乘上逆频率：

```python
inv_freq_expanded = (
    self.inv_freq[None, None, :, None].float()
    .expand(3, position_ids.shape[1], -1, 1)
)
position_ids_expanded = position_ids[:, :, None, :].float()

freqs = (
    inv_freq_expanded.float() @ position_ids_expanded.float()
).transpose(2, 3)
```

```text
position_ids                 [3, B, S]
inv_freq                     [F]

inv_freq_expanded             [3, B, F, 1]
position_ids_expanded         [3, B, 1, S]
矩阵乘积                     [3, B, F, S]
transpose(2, 3)              [3, B, S, F]
```

这里的矩阵乘法是坐标向量和逆频率向量的外积，不会把 T/H/W 混加在一起。每一项的数学含义是：

```text
freqs[a, b, s, j] = position_ids[a, b, s] × inv_freq[j]
```

虽然变量名叫 `freqs`，乘过位置之后它已经是旋转角度，也可以理解为相位。对图像 token `(2,3,2)`：

```text
T 路角度    [2ω₀, 2ω₁, 2ω₂, 2ω₃, ...]
H 路角度    [3ω₀, 3ω₁, 3ω₂, 3ω₃, ...]
W 路角度    [2ω₀, 2ω₁, 2ω₂, 2ω₃, ...]
```

源码在关闭 autocast 的区域内用 float32 计算这些角度和后面的 cos/sin，再把输出转回 `x.dtype`。`x` 在这个 rotary 模块中主要提供设备和输出 dtype；角度本身由 `position_ids`、配置及 `inv_freq` 决定。

## 6. Interleaved M-RoPE：每个频率槽选择哪条坐标轴

### 6.1 保持频率槽 `j`，替换该槽的坐标来源

`recomposition_frequencies()` 分别作用于三路 cos/sin，选择坐标轴并复制结果：

```python
def recomposition_frequencies(self, freq):
    freqs_thw = freq[0]
    for dim, offset in enumerate((1, 2), start=1):
        length = self.mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_thw[..., idx] = freq[dim, ..., idx]
    return torch.cat((freqs_thw, freqs_thw), dim=-1)
```

首先取 T 路作为结果，再将指定槽位替换为 H/W 路。三角函数逐元素计算，因此先选择相位再计算 cos/sin，与源码先计算 cos/sin 再选择坐标轴等价。对于频率槽 `j`，所用逆频率始终是 `inv_freq[j]`。

坐标选择沿**频率轴**进行，形状从 `[3,B,S,F]` 变为 `[B,S,F]`，最后沿特征轴复制为 `[B,S,2F]`；token 顺序不变。

### 6.2 `[11,11,10]` 的实际槽位布局

当 `F=32`、`mrope_section=[11,11,10]` 时：

```text
H: slice(1, 11×3, 3)    → 1, 4, 7, ..., 28, 31       共 11 个
W: slice(2, 10×3, 3)    → 2, 5, 8, ..., 26, 29       共 10 个
T: 未被覆盖的槽位       → 0, 3, 6, ..., 27, 30       共 11 个

完整布局                 (T H W) × 10 + (T H)
```

| 频率槽 `j` | 坐标轴 | 输出角度 |
| --- | --- | --- |
| `0` | T | `T × inv_freq[0]` |
| `1` | H | `H × inv_freq[1]` |
| `2` | W | `W × inv_freq[2]` |
| `3` | T | `T × inv_freq[3]` |
| `4` | H | `H × inv_freq[4]` |
| `5` | W | `W × inv_freq[5]` |
| `...` | `...` | `...` |
| `29` | W | `W × inv_freq[29]` |
| `30` | T | `T × inv_freq[30]` |
| `31` | H | `H × inv_freq[31]` |

> [!NOTE]
> 函数体显式读取 `mrope_section[1]` 和 `[2]` 来覆盖 H/W，T 占据剩余槽位；`[11,11,10]` 配合 32 个频率槽的结尾为 `TH`。

T/H/W 对应的相位沿递减的频率序列交错分布；下面按等价的相位选择给出数值。

### 6.3 `(T,H,W)=(2,3,2)` 的具体数值

继续使用 `R=64`、`θ=10000`：

| `j` | `inv_freq[j]`，约值 | 选择的坐标 | 交错后的角度 `φ[j]`，约值 |
| --- | --- | --- | --- |
| `0` | `1.000000` | `T=2` | `2.000000` |
| `1` | `0.749894` | `H=3` | `2.249683` |
| `2` | `0.562341` | `W=2` | `1.124683` |
| `3` | `0.421697` | `T=2` | `0.843393` |
| `4` | `0.316228` | `H=3` | `0.948683` |
| `5` | `0.237137` | `W=2` | `0.474275` |

对于文本 token，T/H/W 相等，同一槽位无论选取哪一条轴都得到 `p × inv_freq[j]`。M-RoPE 对文本退化为一维 RoPE，原因就在这一步的坐标相等。
