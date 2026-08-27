# Gemma 4 NVFP4 MoE 空输出

> [!CAUTION]
> **Issue：** [vLLM #51525：Gemma 4 NVFP4 MoE 在 CUTLASS 路径返回 PAD token 或空内容](https://github.com/vllm-project/vllm/issues/51525)
>
> **ROOT CAUSE — `VLLM_CUTLASS` W4A4 消费未完成完整性校验的 per-expert activation scale**
>
> ModelOpt calibration 按模型的真实 MoE routing 收集 activation statistics。低频 expert 若在校准语料中没有接收到 token，对应的 gate/up 或 down projection 就不会形成有效统计量；exporter 随后不会为该 expert 写出 `w13_input_scale` 或 `w2_input_scale`。Packed FP4 weight 及其 weight scale 仍然存在，因此 checkpoint 可以通过只关注权重数量和 shape 的检查
>
> vLLM 的 `ModelOptNvFp4FusedMoE` 使用 `torch.empty` 分配完整的 per-expert scale tensor，loader 只覆盖 checkpoint 中实际存在的条目。加载完成后，代码没有在 backend format conversion 前验证所有 expert slot 都已写入。选择 `VLLM_CUTLASS` 时，format conversion 继续对 `w13_input_scale` 和 `w2_input_scale` 取倒数，生成 `a1_gscale` 和 `a2_gscale`；缺失 slot 中的未初始化 bit pattern 因而被转换为 `inf`、`NaN` 或任意有限垃圾值，并被当作合法量化参数传给 kernel
>
> Router 命中缺项 expert 后，CUTLASS 路径首先将 `a1_gscale[expert_idx]` 传给 `scaled_fp4_experts_quant`。CUDA kernel 用该值计算每 16 个 activation 的 E4M3 block scale，并把归一化结果编码为 E2M1；错误 scale 会在进入第一个 `w13` expert GEMM 前直接破坏 activation。SiLU/mul 之后，第二次量化再由 `a2_gscale[expert_idx]` 驱动，并进入 `w2` expert GEMM。污染随后返回 transformer residual path，继续影响后续 layer、final norm、LM head logits 和 sampling。GPU kernel、worker 和 HTTP server 均可正常结束，因此请求仍返回 HTTP 200，但 token 可能全部为 PAD 或被解码为空内容


| 项目 | 已确认结论 |
| --- | --- |
| 复现模型 | `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4` |
| 已确认环境 | RTX 5090 / SM120，vLLM `0.24.0` |
| 失败路径 | 显式选择 `VLLM_CUTLASS` NVFP4 MoE，执行 `W4A4 expert computation` |
| checkpoint 缺陷 | 12 个 layer 中共有 25 个 `expert activation scale` 条目缺失 |
| loader 缺陷 | `per-expert scale` 由 `torch.empty` 分配，加载结束后没有验证 `checkpoint coverage` |
| 外部症状 | 请求耗尽 `max_tokens`，返回 PAD token、`content: null` 或空字符串 |

## NVFP4

ModelOpt 是 NVIDIA 的量化工具及量化 checkpoint 格式，不是一种数据类型。`bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4` 的量化配置包含 `producer=modelopt`、`quant_algo=NVFP4` 和 `group_size=16`：原始 BF16 模型先由 NVIDIA ModelOpt 执行 PTQ，再将量化权重及其 scale 按 ModelOpt 的字段约定保存到 Hugging Face checkpoint。vLLM 启动参数 `--quantization modelopt` 不会在服务启动时重新量化模型；该参数用于声明 checkpoint 的序列化格式，使 loader 按 ModelOpt 的字段名称、shape 和 layout 加载已有量化数据。ModelOpt、NVFP4 和执行 backend 分属不同层级

<div align="center">
<table>
  <thead>
    <tr>
      <th align="center">名称</th>
      <th align="center">所属层级</th>
      <th align="center">职责</th>
    </tr>
  </thead>
  <tbody>
    <tr><td align="center"><strong>Gemma 4 MoE</strong></td><td align="center">模型结构</td><td>决定 expert、router、gate/up/down projection 和每个 token 的执行路径</td></tr>
    <tr><td align="center"><strong>ModelOpt</strong></td><td align="center">量化工具与 schema</td><td>执行 PTQ、calibration 和 export，同时规定量化 checkpoint 的序列化字段</td></tr>
    <tr><td align="center"><strong>NVFP4</strong></td><td align="center">数值格式</td><td>规定 E2M1 数据以及两级 scaling factor</td></tr>
    <tr><td align="center"><strong>CUTLASS / Marlin</strong></td><td align="center">运行时 backend</td><td>vLLM 加载 checkpoint 后实际执行 GEMM</td></tr>
  </tbody>
</table>
</div>

```text
google/gemma-4-26B-A4B-it
        │  BF16 权重
        ▼
NVIDIA ModelOpt 执行 PTQ
        │  导出 NVFP4 W4A4 checkpoint
        ▼
bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4
        │  vLLM 按 ModelOpt schema 加载
        ▼
ModelOptNvFp4FusedMoE
        │
        ├── VLLM_CUTLASS  → W4A4 执行  → 需要 activation input_scale
        │
        └── MARLIN        → W4A16 执行 → 不消费 activation input_scale
```

Gemma 4 的 expert weight 原本是 fused 3D tensor，而不是 128 组普通 `nn.Linear`。该模型的量化脚本先通过 ModelOpt plugin 将 expert 展开为可量化的 Linear，完成量化后再把导出键名转换为 vLLM FusedMoE 所需的形式。模型卡记录的 calibration 配置是 ModelOpt `0.43`、CNN/DailyMail `4096` 个样本、`seq_len=1024`，并保留由真实 router 决定的 natural expert routing；vision encoder 不参与量化

ModelOpt 负责生成 checkpoint，vLLM 不会在启动时重新 calibration。对 MoE expert，checkpoint 中与 NVFP4 计算直接相关的数据可以归纳为：

<div align="center">
<table>
  <thead>
    <tr>
      <th align="center">数据</th>
      <th align="center">checkpoint 内容</th>
      <th align="center">粒度</th>
      <th align="center">产生时间</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Weight FP4</td><td><code>w13_weight</code>、<code>w2_weight</code></td><td>两个 E2M1 值打包进一个 byte</td><td>ModelOpt export</td></tr>
    <tr><td>Weight block scale</td><td><code>w13_weight_scale</code>、<code>w2_weight_scale</code></td><td>K 维每 16 个 weight 一个 E4M3 scale</td><td>ModelOpt export</td></tr>
    <tr><td>Weight global scale</td><td><code>w13_weight_scale_2</code>、<code>w2_weight_scale_2</code></td><td>每个 expert/projection 一个 FP32 scale</td><td>ModelOpt calibration/export</td></tr>
    <tr><td>Activation global scale</td><td><code>w13_input_scale</code>、<code>w2_input_scale</code></td><td>每个 expert/projection 一个 FP32 scale</td><td>ModelOpt calibration/export</td></tr>
    <tr><td>Activation FP4 与 block scale</td><td>不在 checkpoint 中</td><td>当前请求路由到 expert 后动态生成</td><td>vLLM runtime</td></tr>
  </tbody>
</table>
</div>

NVFP4 是面向 NVIDIA Blackwell Tensor Core 的 4-bit 浮点量化格式。其数值主体采用 FP4 E2M1：最高 1 bit 为符号位，中间 2 bit 为指数位，最低 1 bit 为尾数位。`6` 是 E2M1 的最大有限幅值，全部编码如下：

```text
0000 ->  0.0    0001 ->  0.5    0010 ->  1.0    0011 ->  1.5
0100 ->  2.0    0101 ->  3.0    0110 ->  4.0    0111 ->  6.0
1000 -> -0.0    1001 -> -0.5    1010 -> -1.0    1011 -> -1.5
1100 -> -2.0    1101 -> -3.0    1110 -> -4.0    1111 -> -6.0
```

设 `s` 为符号位，`e` 为两位无符号指数，`m` 为一位尾数，指数 bias 为 `1`。E2M1 的解码规则为：

```math
\operatorname{E2M1}(s,e,m)=(-1)^s
\begin{cases}
0.5m, & e=0 \\
2^{e-1}(1+0.5m), & e>0
\end{cases}
```

当 `e=0` 时，`m=0` 和 `m=1` 分别产生带符号零和 `0.5`；当 `e>0` 时，两位指数依次给出 `1、2、4` 的基准量级，一位尾数再区分每个量级内的两个可表示值

vLLM 的 NVFP4 activation kernel 不使用手写分段公式选择 E2M1 编码，而是在 CUDA C++ 中调用 Blackwell PTX 转换指令。下面是 `csrc/libtorch_stable/quantization/fp4/nvfp4_utils.cuh` 中实际执行转换和打包的代码：

```cpp
// Convert 4 float2 values into 8 e2m1 values (represented as one uint32_t).
__device__ __forceinline__ uint32_t fp32_vec8_to_e2m1(
    float2 (&array)[4]) {
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32 byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32 byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}\n"
      : "=r"(val)
      : "f"(array[0].x), "f"(array[0].y),
        "f"(array[1].x), "f"(array[1].y),
        "f"(array[2].x), "f"(array[2].y),
        "f"(array[3].x), "f"(array[3].y));
  return val;
}

__device__ __forceinline__ uint32_t pack_fp4(float2 (&v)[4]) {
  return fp32_vec8_to_e2m1(v);
}
```

`cvt.rn.satfinite.e2m1x2.f32` 是真正完成数值选择的指令：`rn` 表示 round-to-nearest、ties-to-even，`satfinite` 将超出 E2M1 范围的输入饱和到最大有限值，`e2m1x2.f32` 表示从两个 FP32 输入生成两个 E2M1 4-bit 值。四条转换指令共生成 8 个 E2M1 值，每两个值装入一个 byte；`mov.b32` 再把 4 个 byte 合并为一个 `uint32_t`。后面的 `cvt_warp_fp16_to_fp4` 在完成 block scaling 后调用 `pack_fp4`，因此上面的编码表就是由硬件转换指令实施的，而不是由软件逐项查表。NVFP4 不直接用这组离散值拟合整个 tensor，而是采用两级 scale：

<div align="center">
<table>
  <thead>
    <tr>
      <th align="center">第一层 scaling factor</th>
      <th align="center">第二层 scaling factor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><strong>Micro-block scale</strong><br><code>FP8 E4M3</code></td>
      <td align="center"><strong>Tensor-level global scale</strong><br><code>FP32</code></td>
    </tr>
    <tr>
      <td>每连续 <code>16</code> 个元素共享一个 scale</td>
      <td>整个 tensor；在 MoE 中细化到 expert/projection</td>
    </tr>
    <tr>
      <td>Weight：<code>weight_scale</code><br>Activation：runtime 动态生成</td>
      <td>Weight：<code>weight_scale_2</code><br>Activation：<code>input_scale</code></td>
    </tr>
  </tbody>
</table>
</div>

第一层 block scale 描述 16 元素 micro-block 的局部动态范围。FP4 E2M1 主要保留 block 内部的相对数值，E4M3 block scale 将该 block 恢复到相应量级。第二层 global scale 描述 tensor 级量化范围，用于把各 block scale 映射到 E4M3 可表示的范围，并在 GEMM 输出缩放时恢复整体数值尺度。E2M1 的最大有限值为 `6`，E4M3 的最大有限值为 `448`，二者对应的组合范围为 `2688`；ModelOpt 保存的 global scale 以这一范围为基础，并可结合 calibration headroom 调整

FP4 数据不能脱离 scale 独立解释。Weight 是静态参数，ModelOpt 在 PTQ/export 阶段把 packed FP4 weight、E4M3 block scale 和 FP32 global scale 一并写入 checkpoint。Activation 随请求内容和 MoE routing 改变，checkpoint 只保存 calibration 得到的 FP32 global scale；FP4 activation 和 E4M3 block scale 由 vLLM 在每次 GEMM 前动态生成

> [!IMPORTANT]
> **`input_scale` 对应第二层 FP32 activation global scale。**
>
> 它不是 16 元素 micro-block 使用的 E4M3 block scale，也不是某次请求即时统计得到的 scale；它由 calibration 确定、写入 checkpoint，并在 MoE 中按 expert/projection 使用。动态 block quantization 不能替代 `input_scale`

从原始 activation 到 Tensor Core operand 的运行时处理顺序如下：

```text
BF16/FP16 hidden state
        │  router 已确定 expert_idx
        ▼
读取该 expert/projection 的 FP32 input_scale
        │  vLLM 取倒数，得到传给量化 kernel 的 SFScaleVal
        ▼
每 16 个元素计算绝对值最大值
        │  结合 SFScaleVal 和 E2M1 最大值 6
        ▼
生成并舍入为 FP8 E4M3 block scale
        │  用 block scale 归一化当前 16 个值
        ▼
舍入到 {-6, -4, -3, -2, -1.5, -1, -0.5, 0, ..., 6}
        │  两个 E2M1 值打包为一个 byte
        ▼
FP4 activation + FP8 block scale 进入 W4A4 GEMM
        │  GEMM scaling 同时纳入 FP32 activation/weight global scale
        ▼
恢复到正确的输出数值尺度
```

E4M3 舍入是两级缩放不可省略的原因。global scale 先把不同 block 的局部范围映射到 E4M3 的有效表示区间；block scale 完成舍入后，再与 FP32 global scale 一起恢复整体量级。错误的 `input_scale` 会改变 E4M3 block scale 的舍入或裁剪结果，并进一步改变 E2M1 编码。Weight 侧执行同样的数值重建，只是其 E2M1 数据、E4M3 block scale 和 FP32 global scale 已由 ModelOpt 写入 checkpoint。实际 CUTLASS kernel 会融合 block scaling、Tensor Core GEMM 与 global scaling，但上述数据依赖保持不变

下面是 vLLM 的关键实现，省略地址计算和边界处理。源码文件使用 CUDA intrinsic、template 和 `if constexpr`，属于 CUDA C++，因此代码块标记为 `cpp`，不是 C。实现位于 `csrc/libtorch_stable/quantization/fp4/nvfp4_utils.cuh` 的 `cvt_warp_fp16_to_fp4`：

```cpp
auto localMax = __habs2(vec.elts[0]);

#pragma unroll
for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
  localMax = __hmax2(localMax, __habs2(vec.elts[i]));
}

if constexpr (CVT_FP4_NUM_THREADS_PER_SF == 2) {
  localMax = __hmax2(
      __shfl_xor_sync(0xffffffffu, localMax, 1), localMax);
}

float vecMax = float(__hmax(localMax.x, localMax.y));

// Compute and store the E4M3 micro-block scale.
SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
__nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
reinterpret_cast<__nv_fp8_e4m3&>(fp8SFVal) = tmp;
SFValue = float(tmp);
if (SFout) *SFout = fp8SFVal;

// Derive the normalization factor used before E2M1 packing.
outputScale = SFValue != 0.0f
    ? reciprocal_approximate_ftz(
          SFValue * reciprocal_approximate_ftz(SFScaleVal))
    : 0.0f;

fp2Vals[i].x *= outputScale;
fp2Vals[i].y *= outputScale;
return pack_fp4(fp2Vals);
```

`vecMax` 是当前 16 元素 block 的绝对值最大值；`SFScaleVal` 是 checkpoint activation global scale 的倒数；转换为 `__nv_fp8_e4m3` 后的 `SFValue` 是该 block 的 E4M3 scale；`outputScale` 是写入 E2M1 前使用的归一化因子；`pack_fp4` 负责最终的 E2M1 舍入与打包

MoE 版本不会对所有 token 使用同一个 global scale，而是按照 router 已选中的 `expert_idx` 读取对应条目。关键代码位于 `csrc/libtorch_stable/quantization/fp4/nvfp4_experts_quant.cu`：

```cpp
float const SFScaleVal =
    SFScale == nullptr ? 1.0f : SFScale[expert_idx];

out_pos = cvt_warp_fp16_to_fp4<
    Type, CVT_FP4_NUM_THREADS_PER_SF, UE8M0_SF>(
        quant_input, SFScaleVal, sf_out);
```

vLLM 的 MoE format conversion 在进入该内核前分别对 `w13_input_scale` 和 `w2_input_scale` 取倒数，生成 `a1_gscale` 和 `a2_gscale`。第一次 expert GEMM 前使用 `a1_gscale` 量化输入；SiLU/mul 后再使用 `a2_gscale` 量化中间 activation，然后执行第二次 expert GEMM

- **PTQ（Post-Training Quantization）：** 不重新训练基础模型，通过 calibration statistics 和量化规则生成低精度 checkpoint。本案例由 ModelOpt 对原始 BF16 Gemma 4 执行 PTQ

- **Calibration：** 使用代表性样本收集 activation range/statistics，以确定运行时不能仅由 weight 推导的量化参数。本案例中的自然 MoE routing 决定哪些 expert 能获得 `input_scale` 统计

- **Static weight quantization：** Weight 在 export 阶段完成量化，packed data 与 scale 固定写入 checkpoint。本案例对应 `w13/w2 weight`、`weight_scale` 和 `weight_scale_2`

- **Dynamic activation quantization：** 运行时依据当前 activation 生成低精度数据及局部 scale。本案例在每次 expert GEMM 前生成 FP4 activation 和 E4M3 block scale

- **Granularity：** 一个 scale 覆盖的数据范围，例如 per-tensor、per-channel、per-group、per-block 或 per-expert。NVFP4 micro-block 覆盖 16 个元素，global activation scale 则细化到 expert/projection

- **Packing：** 将多个低位宽元素编码进 byte 或 machine word；它是存储布局，不是另一种量化算法。本案例中两个 E2M1 值占一个 byte，8 个值由 `pack_fp4` 组成一个 `uint32_t`

- **Backend / kernel：** 消费量化权重与 scale 并执行 GEMM 的运行时实现。本案例中 CUTLASS 执行 W4A4，Marlin 执行不消费 activation `input_scale` 的路径

## 矩阵融合

Transformer checkpoint 中的权重名称首先描述逻辑 projection，而 serving backend 关注的是如何把共享输入的 projection 合并为更少、更大的 GEMM。矩阵融合通常不改变模型函数；它把多个输入相同、输出彼此独立的线性层沿 output dimension 拼接，执行一次 GEMM 后再切分结果。量化不会改变这一逻辑结构，主要改变 weight 的 dtype、packing、scale tensor 和 kernel-specific layout

```text
Hidden state X
      │
      ├── Attention
      │     Wq + Wk + Wv
      │          │  沿 output dimension 拼接
      │          ▼
      │        Wqkv ── 单次 GEMM ── split(Q, K, V)
      │                                  │
      │                            Attention + KV Cache
      │
      ├── Gated FFN
      │     Wgate + Wup
      │          │  沿 output dimension 拼接
      │          ▼
      │        W13 ── 单次 GEMM ── split(gate, up)
      │                                  │
      │                            activation × up
      │                                  │
      │                              W2 GEMM
      │
      └── SSM / Gated DeltaNet
            QKV + Z ── in_proj_qkvz GEMM ── split
            B + A   ── in_proj_ba GEMM   ── split
                                           │
                                           ▼
                              conv + recurrent state update
```

<div align="center">
<table>
  <thead>
    <tr>
      <th align="center">模块</th>
      <th align="center">逻辑权重</th>
      <th align="center">常见融合权重</th>
      <th align="center">融合后的主要计算</th>
    </tr>
  </thead>
  <tbody>
    <tr><td align="center"><strong>Attention</strong></td><td><code>q_proj</code>、<code>k_proj</code>、<code>v_proj</code></td><td><code>qkv_proj</code></td><td>一次 GEMM，随后切分 Q/K/V</td></tr>
    <tr><td align="center"><strong>Dense FFN</strong></td><td><code>gate_proj</code>、<code>up_proj</code></td><td><code>gate_up_proj</code> / <code>w13</code></td><td>一次 GEMM，随后 activation/mul</td></tr>
    <tr><td align="center"><strong>MoE FFN</strong></td><td>每个 expert 的 gate/up/down</td><td><code>w13 [E,2I,H]</code>、<code>w2 [E,H,I]</code></td><td>按 expert 组织 Grouped GEMM</td></tr>
    <tr><td align="center"><strong>SSM / GDN</strong></td><td><code>in_proj_qkv</code>、<code>in_proj_z</code>、<code>in_proj_b</code>、<code>in_proj_a</code></td><td><code>in_proj_qkvz</code>、<code>in_proj_ba</code></td><td>两个 projection GEMM，随后 convolution 与 recurrent update</td></tr>
  </tbody>
</table>
</div>

以 PyTorch 常见的 weight layout 为例，线性层权重保存为 `[out_features, in_features]`。只要若干 projection 使用同一个输入，就可以沿第 0 维拼接 weight；融合改变的是 GEMM 的输出宽度，而不是输入维度

Attention 中，Q、K、V 共享 hidden state `X`。`q_proj`、`k_proj`、`v_proj` 沿输出维拼成 `qkv_proj`，原来的三个 projection GEMM 由一个更宽的 GEMM 取代，输出再按 Q、K、V 的真实宽度切分

标准 MHA 中三个 projection 的输出宽度通常相同；GQA 中 K/V 的 head 数少于 Q，因此 `Wqkv` 的 shape 是 `[Q_dim + K_dim + V_dim, H]`，不能机械地写成 `[3H,H]`。GEMM 输出按各自真实宽度切分，随后进入 attention kernel；K/V 还会写入随 sequence length 增长的 KV Cache

下面是 vLLM `Gemma4Attention` 的关键计算路径，省略 KV-sharing 等条件分支。`QKVParallelLinear` 表示融合并按 tensor parallel 切分的 QKV projection；`split` 只恢复三个逻辑视图，随后 attention kernel 和输出 projection 仍是不同计算阶段

```python
# __init__: q_proj、k_proj、v_proj 被组织成一个 fused projection
self.qkv_proj = QKVParallelLinear(
    hidden_size,
    head_dim,
    total_num_heads,
    total_num_kv_heads,
    quant_config=quant_config,
)
self.o_proj = RowParallelLinear(
    total_num_heads * head_dim,
    hidden_size,
    quant_config=quant_config,
)

# forward: 一个 QKV GEMM，切分后进入 attention，再做输出 GEMM
qkv, _ = self.qkv_proj(hidden_states)
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

q = q.unflatten(-1, (self.num_heads, self.head_dim))
q = self.q_norm(q).flatten(-2, -1)
k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
k = self.k_norm(k).flatten(-2, -1)
v = v.unflatten(-1, (self.num_kv_heads, self.head_dim))
v = self.v_norm(v).flatten(-2, -1)

q, k = self.rotary_emb(positions, q, k)
attn_output = self.attn(q, k, v)
output, _ = self.o_proj(attn_output)
```

这条路径的计算特征是“宽 QKV GEMM + attention kernel + 输出 GEMM”。Projection fusion 减少 Q、K、V 的独立 kernel launch 和 hidden-state 读取次数，但 Q/K normalization、RoPE、KV Cache 写入、softmax attention 与 `o_proj` 仍保留各自的数值语义

Gated FFN 的核心矩阵是 gate、up 和 down。Gate 与 up 共享输入且输出宽度相同，因此可融合为 `w13 [2I,H]`；`w1` 通常表示 gate，`w3` 表示 up，`w2 [H,I]` 表示 down projection。运行时链路如下：

```text
hidden state [M,H]
        │
        ▼
w13 GEMM：输出 [M,2I]
        │  按 I + I 切分
        ├──────────────┐
        ▼              ▼
gate [M,I]          up [M,I]
        │              │
        └── SiLU + multiply
                    │
                    ▼
               w2 GEMM
                    │
                    ▼
output [M,H]
```

vLLM `Gemma4MLP` 直接把这种结构表示为 `MergedColumnParallelLinear + act_and_mul + RowParallelLinear`：

```python
self.gate_up_proj = MergedColumnParallelLinear(
    hidden_size,
    [intermediate_size] * 2,
    bias=False,
    quant_config=quant_config,
)
self.down_proj = RowParallelLinear(
    intermediate_size,
    hidden_size,
    bias=False,
    quant_config=quant_config,
)
self.act_fn = get_act_and_mul_fn(hidden_activation)

def forward(self, x):
    gate_up, _ = self.gate_up_proj(x)  # 一个 GEMM 产生 gate 和 up
    x = self.act_fn(gate_up)           # split + activation + multiply
    x, _ = self.down_proj(x)           # down GEMM
    return x
```

原来的 gate GEMM 和 up GEMM 因而合并为一次较大的 `w13` GEMM。输出切成 gate 与 up 后，activation 和 elementwise multiply 可以继续在 epilogue 或独立 fused kernel 中完成，随后执行 `w2` down GEMM。现代 gated FFN 的主干因此是“两次大 GEMM，中间一次 activation/mul”

MoE 不是另一套 FFN 结构，而是在 fused gated FFN 上增加 expert dimension。若共有 `E` 个 expert，则 `w13` 的逻辑 shape 为 `[E,2I,H]`，`w2` 为 `[E,H,I]`。Router 把 token 重排为每个 expert 的输入；不同 expert 获得的 token 行数由当前请求动态决定，所以运行时面对的不是一个规则大矩阵，而是一组输入高度不同的 GEMM problem：

```text
router output
      │
      ├── expert 0：M₀ 行 token ── 使用 w13[0]
      ├── expert 1：M₁ 行 token ── 使用 w13[1]
      ├── expert 2：M₂ 行 token ── 使用 w13[2]
      └── ...
              │
              ▼
     一次提交 Grouped GEMM problem list
```

逐 expert 单独 launch 会产生较高调度开销，因此 CUTLASS、FlashInfer 等 backend 通常使用 Grouped GEMM，一次提交当前 route 中所有 expert problem。这里同时存在两种不同的融合：`w1 + w3 → w13` 是 weight/projection fusion；多个 expert GEMM 共同提交是 grouped scheduling。矩阵融合还需要区分四个实现层次：

<div align="center">
<table>
  <thead>
    <tr>
      <th align="center">层次</th>
      <th align="center">发生的变化</th>
      <th align="center">本案例中的实例</th>
    </tr>
  </thead>
  <tbody>
    <tr><td align="center">Projection fusion</td><td>共享输入的 weight 沿 output dimension 拼接</td><td><code>w1 + w3 → w13</code></td></tr>
    <tr><td align="center">Physical layout conversion</td><td>packed weight 和 scale 转换为 backend 要求的排列</td><td>ModelOpt layout → CUTLASS/Marlin layout</td></tr>
    <tr><td align="center">Grouped scheduling</td><td>把不同 expert 的变长 GEMM problem 一次提交</td><td>按 <code>M_e</code> 组织 MoE GEMM</td></tr>
    <tr><td align="center">Operator/epilogue fusion</td><td>把 activation、multiply、quantization 等邻接算子并入同一 kernel</td><td>SiLU + mul + 第二次 NVFP4 quantization</td></tr>
  </tbody>
</table>
</div>

在当前 NVFP4 MoE 路径中，`w13_input_scale` 对应进入第一组 fused gate/up GEMM 的 activation global scale；`w2_input_scale` 对应 SiLU/mul 输出进入 down GEMM 前的 activation global scale。CUTLASS 的完整执行次序可以压缩为：

```text
router / token shuffle
        ↓
使用 a1_gscale 量化 activation
        ↓
Grouped w13 GEMM
        ↓
SiLU + multiply + 使用 a2_gscale 再量化
        ↓
Grouped w2 GEMM
        ↓
expert reduce / unpermute
```

这解释了为什么缺失 `input_scale` 会沿两次 expert GEMM 传播：scale 缺项不是单独某个标量的显示错误，而是破坏了 fused MoE 矩阵链路中 GEMM operand 的构造。Marlin W4A16 仍使用 `w13/w2` 的矩阵融合结构，但 activation 保持 BF16/FP16，因而不进入上述两次 FP4 activation quantization

**SSM / Gated DeltaNet 的 projection fusion。** Qwen3.5 MoE 中的 Gated DeltaNet 是 sequence mixer，MoE 是同一 transformer layer 中的 FFN。两者相互独立：Gated DeltaNet 负责 token 序列的信息混合与状态更新，MoE router 负责为 FFN 选择 expert。vLLM 将该模块实现为 `QwenGatedDeltaNetAttention`，文件位于 `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`。代码沿用 `Attention` 接口并把状态放在 `kv_cache` 容器中，但其核心不是 QK softmax attention；它执行 causal depthwise convolution 和 gated delta recurrent update。本文将其归入广义 SSM/固定状态递归路径

<div align="center">
<table>
  <thead>
    <tr>
      <th align="center">配置字段</th>
      <th align="center">含义</th>
      <th align="center">决定的 tensor 维度</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>linear_num_key_heads</code></td><td>Q/K head 数量</td><td><code>key_dim = num_k_heads × head_k_dim</code></td></tr>
    <tr><td><code>linear_num_value_heads</code></td><td>V 与 recurrent state 的 head 数量</td><td><code>value_dim = num_v_heads × head_v_dim</code></td></tr>
    <tr><td><code>linear_key_head_dim</code></td><td>每个 Q/K head 的宽度</td><td>决定 recurrent state 的 key axis</td></tr>
    <tr><td><code>linear_value_head_dim</code></td><td>每个 V head 的宽度</td><td>决定输出与 recurrent state 的 value axis</td></tr>
    <tr><td><code>linear_conv_kernel_dim</code></td><td>Causal depthwise convolution 的窗口宽度</td><td>每条序列只需保存 <code>kernel_size - 1</code> 个历史位置的 convolution state</td></tr>
  </tbody>
</table>
</div>

Hugging Face checkpoint 将 `in_proj_qkv`、`in_proj_z`、`in_proj_b` 和 `in_proj_a` 分别保存。vLLM 在加载 Qwen3.5 时执行两组 stacked mapping：Q/K/V/Z 合并为 `in_proj_qkvz`，B/A 合并为 `in_proj_ba`。对应的实际映射为：

```python
orig_to_new_stacked = {
    ".in_proj_qkv": (".in_proj_qkvz", (0, 1, 2)),
    ".in_proj_z":   (".in_proj_qkvz", 3),
    ".in_proj_b":   (".in_proj_ba", 0),
    ".in_proj_a":   (".in_proj_ba", 1),
}
```

因此，checkpoint 的四组逻辑 projection 在 vLLM runtime 中成为两个 `MergedColumnParallelLinear` GEMM，而不是一个包含所有分支的 `in_proj`。`in_proj_qkvz` 的四段输出宽度依次为 `key_dim`、`key_dim`、`value_dim`、`value_dim`；`in_proj_ba` 的两段输出宽度均为 `num_v_heads`。B/A 单独组成较窄的 projection，使其可以采用与主 QKVZ projection 不同的 tensor-parallel 和量化约束；例如某些 Marlin 配置下，vLLM 会复制较窄的 BA projection，而不是继续切分到每个 TP rank

```text
hidden states [tokens, hidden_size]
        │
        ├── in_proj_qkvz GEMM ── split ── Q │ K │ V │ Z
        │                                  │   │   │    └─ output gate
        │                                  └───┴───┐
        │                                          ▼
        │                              causal depthwise conv + SiLU
        │                                          │
        │                                  reshape Q / K / V heads
        │                                          │
        └── in_proj_ba GEMM ─── split ── B │ A     │
                                          │   │     │
                                          ▼   ▼     ▼
                                      beta  decay  gated delta rule
                                                     │
                                              recurrent state update
                                                     │
                                                     ▼
                                          RMSNormGated(core, Z)
                                                     │
                                               out_proj GEMM
```

下面第一段精简自 vLLM `QwenGatedDeltaNetAttention.forward_cuda`。它保留了实际的 fused projection 和 custom-op 边界：

```python
# 两个 fused projection GEMM
mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
ba, _ = self.in_proj_ba(hidden_states)

# Qwen3.5 checkpoint/runtime layout: [q, k, v, z] 与 [b, a]
qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
z_size = self.value_dim // self.tp_size
mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
z = z.reshape(z.size(0), -1, self.head_v_dim)
b, a = self.split_ba(ba)

# custom op 内部执行 causal conv 与 recurrent update
core_attn_out = torch.zeros(
    (hidden_states.size(0), self.num_v_heads // self.tp_size, self.head_v_dim),
    dtype=hidden_states.dtype,
    device=hidden_states.device,
)
torch.ops.vllm.qwen_gdn_attention_core(
    mixed_qkv,
    b.contiguous(),
    a.contiguous(),
    core_attn_out,
    layer_name=_encode_layer_name(self.prefix),
)

# Z 只在 recurrent core 之后参与门控归一化
output = self._output_projection(core_attn_out, z)
```

custom op 内部的计算特征可压缩为下面的控制流；代码省略 batch 维变换、sequence metadata 和 speculative decoding 分支。Prefill 对一段序列使用 chunk kernel，decode 对单步 token 原地更新已有 state：

```python
# Q/K/V 先经过共享的 depthwise causal convolution
mixed_qkv = causal_conv1d_fn(
    mixed_qkv.transpose(0, 1),
    conv_weights,
    self.conv1d.bias,
    activation="silu",
    conv_states=conv_state,
).transpose(0, 1)

# 一次 fused preparation 完成 split/reshape、QK L2Norm 和门控参数生成
q, k, v, g, beta = fused_post_conv_prep(
    conv_output=mixed_qkv,
    a=a,
    b=b,
    A_log=self.A_log,
    dt_bias=self.dt_bias,
    num_k_heads=self.num_k_heads // self.tp_size,
    head_k_dim=self.head_k_dim,
    head_v_dim=self.head_v_dim,
    apply_l2norm=True,
    output_g_exp=False,
)

if is_prefill:
    core, last_state = self.chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta,
        initial_state=ssm_state,
        output_final_state=True,
    )
else:
    core, last_state = fused_sigmoid_gating_delta_rule_update(
        q=q, k=k, v=v,
        a=a, b=b,
        A_log=self.A_log,
        dt_bias=self.dt_bias,
        initial_state=ssm_state,
        inplace_final_state=True,
    )
```

Q/K/V 只共享 projection 和 convolution，并不进入 softmax attention。Causal depthwise convolution 对每个 QKV channel 独立处理短程历史，再由 `torch.split` 和 reshape 恢复 head layout。`Z` 不进入 convolution 或 recurrent state；它保留到 recurrent core 输出之后，作为 `RMSNormGated` 的输出门控输入

`B` 经过 sigmoid 后形成 `beta`，控制 delta update 写入 recurrent state 的强度。`A` 与可训练参数 `A_log`、`dt_bias` 共同生成负的 decay log-gate，指数化后得到小于 1 的遗忘系数。`A_log` 决定每个 value head 的基础衰减速率，`dt_bias` 调整输入相关时间尺度。vLLM 将这些步骤放入 `fused_post_conv_prep` 或 recurrent update kernel，从而避免单独物化所有门控中间 tensor

当 `num_v_heads` 大于 `num_k_heads` 时，多个 value head 共享同一组 Q/K head，这就是 grouped-value attention（GVA）。逻辑上可以把 Q/K head 重复到与 V head 数量一致；实际 kernel 可以依据 head-group 映射直接读取共享 Q/K，未必需要在显存中执行 `repeat_interleave`

Prefill 与 decode 使用不同的状态更新方式。Prefill 需要并行处理一段 token，vLLM 调用 `chunk_gated_delta_rule`，并把最后一个 recurrent state 写回缓存。Decode 通常每条序列只新增一个 token，因而调用 `fused_sigmoid_gating_delta_rule_update` 原地更新 state。缓存由两部分组成：`conv_state` 保存 depthwise convolution 所需的固定窗口，`ssm_state` 保存每个 value head 的 recurrent matrix。二者的尺寸由模型维度和 convolution kernel width 决定，不随已生成序列长度线性增长

<div align="center">
<table>
  <thead>
    <tr>
      <th align="center">维度</th>
      <th align="center">Softmax Attention</th>
      <th align="center">Qwen3.5 Gated DeltaNet</th>
    </tr>
  </thead>
  <tbody>
    <tr><td align="center"><strong>序列混合</strong></td><td>QK score、softmax、value aggregation</td><td>Causal convolution 与 gated delta recurrent update</td></tr>
    <tr><td align="center"><strong>历史状态</strong></td><td>保存每个历史 token 的 K/V</td><td>保存固定窗口 <code>conv_state</code> 与固定尺寸 <code>ssm_state</code></td></tr>
    <tr><td align="center"><strong>缓存增长</strong></td><td>随 sequence length 增长</td><td>不随 sequence length 线性增长</td></tr>
    <tr><td align="center"><strong>输入投影</strong></td><td>常见 <code>qkv_proj</code></td><td><code>in_proj_qkvz</code> 与 <code>in_proj_ba</code></td></tr>
    <tr><td align="center"><strong>输出路径</strong></td><td>Attention output → <code>o_proj</code></td><td>Recurrent output + Z gate → <code>RMSNormGated</code> → <code>out_proj</code></td></tr>
  </tbody>
</table>
</div>

量化必须沿 operator boundary 判断。`in_proj_qkvz`、`in_proj_ba` 和 `out_proj` 是 Linear/GEMM，可以由 `quant_config` 替换为低精度 weight 或 W/A kernel；`conv1d`、`A_log`、`dt_bias`、门控预处理和 recurrent state update 是另一组算子，不会仅因 projection 使用 W4、W8 或 A4、A8 就自动采用相同格式。vLLM 的 `_output_projection` 还允许编译器在启用 `fuse_norm_quant` 时融合 `RMSNormGated` 与后续 quantization，但这属于 operator fusion，不是 checkpoint weight fusion

这里的 `kv_cache` 只是 vLLM 统一缓存接口的容器名称。Gated DeltaNet 实际存放的是 `conv_state` 与 `ssm_state`，不是 Attention 的历史 K/V tensor，因此“FP8 KV Cache”不能直接推导出 Gated DeltaNet recurrent state 也使用 FP8；二者必须分别配置和验证
