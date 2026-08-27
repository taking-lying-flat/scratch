# Gemma 4 NVFP4 MoE 空输出

**目录**

- [故障现象与根因](#gemma-4-nvfp4-moe-空输出)
- [NVFP4](#nvfp4)
- [矩阵融合](#矩阵融合)
- [Kernel Backend](#kernel-backend)
  - [Linear backend](#linear-backend)
  - [MoE backend](#moe-backend)
  - [Attention backend](#attention-backend)
  - [NVFP4 backend 选择](#nvfp4-backend-选择)
- [原因总结](#原因总结)

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
> **`input_scale` 对应第二层 FP32 activation global scale**
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

这条路径的计算特征是 `宽 QKV GEMM + attention kernel + 输出 GEMM`。`Projection fusion` 减少 Q、K、V 的独立 `kernel launch` 和 `hidden state` 读取次数，但 Q/K normalization、RoPE、KV Cache 写入、softmax attention 与 `o_proj` 仍保留各自的数值语义

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

原来的 gate GEMM 和 up GEMM 因而合并为一次较大的 `w13` GEMM。输出切成 gate 与 up 后，activation 和 elementwise multiply 可以继续在 epilogue 或独立 fused kernel 中完成，随后执行 `w2` down GEMM。现代 gated FFN 的主干因此是`两次大 GEMM，中间一次 activation/mul`

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

这解释为什么缺失 `input_scale` 会沿两次 expert GEMM 传播：scale 缺项不是单独某个标量的显示错误，而是破坏 fused MoE 矩阵链路中 GEMM operand 的构造。Marlin W4A16 仍使用 `w13/w2` 的矩阵融合结构，但 activation 保持 BF16/FP16，因而不进入上述两次 FP4 activation quantization

**SSM / Gated DeltaNet 的 projection fusion** Qwen3.5 MoE 中的 Gated DeltaNet 是 sequence mixer，MoE 是同一 transformer layer 中的 FFN。两者相互独立：Gated DeltaNet 负责 token 序列的信息混合与状态更新，MoE router 负责为 FFN 选择 expert。vLLM 将该模块实现为 `QwenGatedDeltaNetAttention`，文件位于 `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`。代码沿用 `Attention` 接口并把状态放在 `kv_cache` 容器中，但其核心不是 QK softmax attention；它执行 causal depthwise convolution 和 gated delta recurrent update。本文将其归入广义 SSM/固定状态递归路径

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

精简 vLLM `QwenGatedDeltaNetAttention.forward_cuda`。它保留实际的 fused projection 和 custom-op 边界：

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

量化必须沿 `operator boundary` 判断。`in_proj_qkvz`、`in_proj_ba` 和 `out_proj` 是 Linear/GEMM，可以由 `quant_config` 替换为低精度 weight 或 W/A kernel；`conv1d`、`A_log`、`dt_bias`、门控预处理和 recurrent state update 是另一组算子，不会仅因 projection 使用 W4、W8 或 A4、A8 就自动采用相同格式。vLLM 的 `_output_projection` 还允许编译器在启用 `fuse_norm_quant` 时融合 `RMSNormGated` 与后续 quantization，但这属于 operator fusion，不是 checkpoint weight fusion

## Kernel Backend

量化格式规定 `weight`、`activation` 和 `scale` 如何编码；`backend` 决定由哪套实现消费这些数据；具体 `kernel` 才是在 GPU 上执行的函数。vLLM 将三类计算分别配置：`--linear-backend` 选择普通量化 Linear 的 GEMM backend，`--moe-backend` 选择带 expert 维度的 MoE kernel，`--attention-config` 中的 `backend` 选择 Q/K/V 完成投影后的 Attention 实现

```text
模型 checkpoint
  │  quantization metadata + packed weight + scale
  ▼
逻辑算子
  ├── 普通 Quantized Linear
  │     qkv_proj / o_proj
  │     gate_up_proj / down_proj
  │     GDN in_proj / out_proj
  │          │
  │          └── --linear-backend
  │                    ▼
  │              Linear GEMM kernel
  │
  ├── Attention
  │     Q/K/V + attention metadata + KV Cache
  │                    │
  │                    └── --attention-config.backend
  │                              ▼
  │                    FlashAttention / FlashInfer
  │                    Triton / FlexAttention / ...
  │
  └── Fused MoE experts
        router 产生 top-k expert 与 routing weight
                    │
                    └── --moe-backend
                              ▼
                    token layout / permutation
                    expert GEMM 1
                    activation / quantization
                    expert GEMM 2
                    unpermute / weighted reduce
```

<div align="center">
<table>
  <thead>
    <tr>
      <th align="center">配置</th>
      <th align="center">作用域</th>
      <th align="center">典型对象</th>
      <th align="center">不负责的部分</th>
    </tr>
  </thead>
  <tbody>
    <tr><td align="center"><code>--linear-backend</code></td><td>普通 quantized Linear GEMM</td><td><code>qkv_proj</code>、<code>o_proj</code>、Dense FFN、GDN projection</td><td>不选择 Attention softmax kernel，也不选择 MoE expert Grouped GEMM</td></tr>
    <tr><td align="center"><code>--attention-config.backend</code></td><td>Attention dataflow</td><td>Q/K/V、KV Cache、mask、prefill/decode Attention</td><td>不执行 QKV/<code>o_proj</code> Linear，也不选择 MoE expert GEMM</td></tr>
    <tr><td align="center"><code>--moe-backend</code></td><td>MoE expert computation</td><td><code>w13</code>、<code>w2</code> 以及 expert token dataflow</td><td>不控制普通二维 Linear，也不改变 checkpoint 的量化算法</td></tr>
  </tbody>
</table>
</div>

vLLM 在 `vllm/config/kernel.py` 中以两个 `Literal` 定义 Linear 与 MoE 的用户级 backend。下面是这两类的完整枚举；Attention 使用后文单独介绍的 `AttentionBackendEnum`。枚举表示 CLI 可以接受的名称集合，不表示某个 backend 能处理全部 quantization scheme、GPU 架构、shape 和 parallel configuration

```python
MoEBackend = Literal[
    "auto",
    "triton",
    "batched_triton",
    "deep_gemm",
    "deep_gemm_mega_moe",
    "cutlass",
    "flashinfer_trtllm",
    "flashinfer_cutlass",
    "flashinfer_cutedsl",
    "flashinfer_b12x",
    "b12x",
    "marlin",
    "humming",
    "triton_unfused",
    "aiter",
    "flydsl",
    "hpc",
    "emulation",
]

LinearBackend = Literal[
    "auto",
    "cutlass",
    "flashinfer_cutlass",
    "flashinfer_cutedsl",
    "flashinfer_trtllm",
    "flashinfer_cudnn",
    "flashinfer_b12x",
    "b12x",
    "marlin",
    "humming",
    "triton",
    "deep_gemm",
    "torch",
    "aiter",
    "machete",
    "fbgemm",
    "conch",
    "exllama",
    "emulation",
    "xpu",
    "xpu_woq",
]
```

CLI 参数在 `vllm/engine/arg_utils.py` 中直接写入 `KernelConfig`，并把连字符统一替换为下划线：

```python
moe_backend_kwargs = kernel_kwargs["moe_backend"]
moe_backend_kwargs["type"] = lambda s: s.lower().replace("-", "_")
kernel_group.add_argument("--moe-backend", **moe_backend_kwargs)

linear_backend_kwargs = kernel_kwargs["linear_backend"]
linear_backend_kwargs["type"] = lambda s: s.lower().replace("-", "_")
kernel_group.add_argument("--linear-backend", **linear_backend_kwargs)
```

<div align="center">
<table>
  <thead>
    <tr>
      <th align="center">Backend family</th>
      <th align="center">Linear</th>
      <th align="center">MoE</th>
      <th align="center">常见定位</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>cutlass</code></td><td align="center">✓</td><td align="center">✓</td><td>vLLM 基于 NVIDIA CUTLASS 的量化 GEMM，常用于 FP8、FP4 Tensor Core 路径</td></tr>
    <tr><td><code>flashinfer_cutlass</code></td><td align="center">✓</td><td align="center">✓</td><td>FlashInfer 封装的 CUTLASS Linear/MoE kernel</td></tr>
    <tr><td><code>flashinfer_cutedsl</code></td><td align="center">✓</td><td align="center">✓</td><td>FlashInfer CuTeDSL kernel；面向新架构和 block-scaled 低精度格式</td></tr>
    <tr><td><code>flashinfer_trtllm</code></td><td align="center">✓</td><td align="center">✓</td><td>FlashInfer 接入 TensorRT-LLM / TRTLLM-GEN kernel</td></tr>
    <tr><td><code>flashinfer_b12x</code> / <code>b12x</code></td><td align="center">✓</td><td align="center">✓</td><td>面向 SM12x 的 FP4/FP8 路径；前者经 FlashInfer，后者为 native B12X</td></tr>
    <tr><td><code>marlin</code></td><td align="center">✓</td><td align="center">✓</td><td>低比特 weight-only GEMM；activation 通常保持 BF16/FP16</td></tr>
    <tr><td><code>triton</code></td><td align="center">✓</td><td align="center">✓</td><td>Triton 实现；便于针对 quantization scheme 和 shape 定制，MoE 另有 batched/unfused 变体</td></tr>
    <tr><td><code>deep_gemm</code></td><td align="center">✓</td><td align="center">✓</td><td>DeepGEMM，主要面向 FP8 block-quantized GEMM；MoE 另有 mega-MoE 变体</td></tr>
    <tr><td><code>humming</code></td><td align="center">✓</td><td align="center">✓</td><td>Mixed-precision kernel family</td></tr>
    <tr><td><code>aiter</code></td><td align="center">✓</td><td align="center">✓</td><td>AMD ROCm AITer；<code>flydsl</code> 是另一条 ROCm MoE 路径</td></tr>
    <tr><td><code>torch</code></td><td align="center">✓</td><td align="center">—</td><td>PyTorch native scaled-mm 等 Linear 实现</td></tr>
    <tr><td><code>emulation</code></td><td align="center">✓</td><td align="center">✓</td><td>反量化或 QDQ 后使用 BF16/FP16 GEMM 验证 correctness，不以生产性能为目标</td></tr>
    <tr><td><code>machete</code> / <code>fbgemm</code> / <code>conch</code> / <code>exllama</code> / <code>xpu</code></td><td align="center">✓</td><td align="center">—</td><td>特定 mixed-precision、CPU/GPU 或 Intel XPU Linear 实现</td></tr>
  </tbody>
</table>
</div>

### Linear backend

`--linear-backend` 的源码定义是 `Backend for quantized linear layer GEMM kernels`。它控制普通 `quantized Linear` 的 GEMM 实现，不控制未量化 Linear 的常规矩阵乘法，也不控制 `Attention kernel`。一个模型可以同时包含多种 `quantized Linear layer`，因此 vLLM 不是把一个 `backend` 无条件套到全模型，而是按每个 `layer` 的 `quantization type` 建立候选 `kernel class`

`vllm/model_executor/kernels/linear/__init__.py` 中的 `_LINEAR_BACKEND_KERNEL_MAP` 把用户级名称映射到实际 kernel class 集合。下面保留主要 family 的映射结构；每个集合内部还会按 INT8、FP8、FP4、MX format 等继续细分

```python
_LINEAR_BACKEND_KERNEL_MAP = {
    "cutlass": {
        CutlassInt8ScaledMMLinearKernel,
        CutlassFP8ScaledMMLinearKernel,
        CutlassFp8BlockScaledMMKernel,
        CutlassW4A8LinearKernel,
        CutlassNvFp4LinearKernel,
    },
    "flashinfer_cutlass": {
        FlashInferFP8ScaledMMLinearKernel,
        FlashInferFp8DeepGEMMDynamicBlockScaledKernel,
        FlashInferCutlassMxfp8LinearKernel,
        FlashInferCutlassNvFp4LinearKernel,
        FlashInferMxFp4LinearKernel,
    },
    "marlin": {
        MarlinFP8ScaledMMLinearKernel,
        MarlinLinearKernel,
        MarlinMxfp8LinearKernel,
        MarlinNvFp4LinearKernel,
        MarlinMxFp4LinearKernel,
    },
    "triton": {
        TritonInt8ScaledMMLinearKernel,
        TritonFp8BlockScaledMMKernel,
        TritonW4A16LinearKernel,
    },
    "deep_gemm": {DeepGemmFp8BlockScaledMMKernel},
    "torch": {
        PerTensorTorchFP8ScaledMMLinearKernel,
        ChannelWiseTorchFP8ScaledMMLinearKernel,
        RowWiseTorchFP8ScaledMMLinearKernel,
        BlockWiseTorchFP8ScaledMMLinearKernel,
    },
    "emulation": {
        EmulationMxfp8LinearKernel,
        EmulationNvFp4LinearKernel,
        EmulationMxfp6LinearKernel,
        EmulationMxfp4LinearKernel,
    },
}
```

backend 过滤的核心代码如下。`auto` 保留该 layer type 的完整候选列表；显式 backend 只保留映射表中属于该 backend 的 class。当前源码对混合量化模型做容错：如果指定的 backend 对某一种 layer type 根本没有实现，则记录 warning，并仅对该 layer 恢复正常自动选择

```python
def _filter_kernels_by_backend(backend, kernels):
    backend_kernels = _LINEAR_BACKEND_KERNEL_MAP.get(backend, set())
    return [k for k in kernels if k in backend_kernels]

def _resolve_backend_kernels(kernels, layer_desc):
    linear_backend = _get_linear_backend()
    if linear_backend == "auto":
        return kernels

    filtered = _filter_kernels_by_backend(linear_backend, kernels)
    if not filtered:
        logger.warning_once(
            "--linear-backend=%s was requested, but no %s kernel exists "
            "for %s layers; falling back to normal kernel selection",
            linear_backend,
            linear_backend,
            layer_desc,
        )
        return kernels
    return filtered
```

经过过滤后，quantization-specific initializer 按优先级检查候选项。`is_supported()` 判断当前平台、compute capability 和依赖是否可用；`can_implement(config)` 判断当前 layer 的 dtype、shape、scale granularity 等配置是否满足 kernel 约束；第一个同时通过两项检查的 class 被实例化

```python
possible = _resolve_backend_kernels(possible, layer_desc)

failure_reasons = []
for kernel_cls in possible:
    if kernel_cls.__name__ in envs.VLLM_DISABLED_KERNELS:
        continue

    is_supported, reason = kernel_cls.is_supported()
    if not is_supported:
        failure_reasons.append(f"{kernel_cls.__name__}: {reason}")
        continue

    can_implement, reason = kernel_cls.can_implement(config)
    if not can_implement:
        failure_reasons.append(f"{kernel_cls.__name__}: {reason}")
        continue

    return kernel_cls(config)
```

因此，`--linear-backend auto` 不是在启动时对所有 kernel 做在线 benchmark，而是根据 quantization type 取得有序候选列表，再选择第一个满足当前部署条件的实现。明确指定 backend 的价值主要是限制候选 family、复现实验路径或排除某个自动选择结果

### MoE backend

`--moe-backend` 的源码定义是 `Backend for MoE expert computation kernels`。普通 Linear 只需要处理一个规则矩阵乘，而 MoE 还必须处理 `router` 产生的动态 `token-to-expert mapping`。每个 `expert` 获得的 token 数不同，所以 `backend` 通常需要构造多个不同 M 维的 `GEMM problem`，或者把 token padding 成 `batched expert format`

vLLM 把 expert 实现分为 modular 与 monolithic 两类。`FusedMoEExpertsModular` 接收已经计算完成的 `topk_ids/topk_weights`，其抽象边界是 permute → experts → unpermute；`FusedMoEExpertsMonolithic` 直接接收 `router_logits`，允许 backend 把 routing 与 expert computation 融合在一起。两者的核心接口差异可以从 `vllm/model_executor/layers/fused_moe/modular_kernel.py` 直接看到：

```python
class FusedMoEExpertsModular(FusedMoEExperts):
    @staticmethod
    def is_monolithic() -> bool:
        return False

    def apply(
        self,
        output,
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation,
        global_num_experts,
        expert_map,
        a1q_scale,
        a2_scale,
        workspace13,
        workspace2,
        expert_tokens_meta,
        apply_router_weight_on_input,
    ) -> None:
        raise NotImplementedError


class FusedMoEExpertsMonolithic(FusedMoEExperts):
    @staticmethod
    def is_monolithic() -> bool:
        return True

    def apply(
        self,
        hidden_states,
        w1,
        w2,
        router_logits,
        activation,
        global_num_experts,
        expert_map,
        a1q_scale,
        apply_router_weight_on_input,
        **routing_args,
    ):
        raise NotImplementedError
```

因此，`MoE backend` 不能缩写成一个 `X @ W`。即使 `router` 本身在 `backend` 外执行，`modular expert class` 仍可能负责 `token permutation`、`activation quantization`、`Grouped GEMM`、activation、第二次 quantization/GEMM、`unpermute` 和 `routing-weight reduction`。`Monolithic class` 还能进一步接管 `top-k routing`

MoE 的选择逻辑不是一张对所有格式通用的静态 support matrix。FP8、NVFP4、MXFP4、MXFP8、INT8、WNA16 等 quantization method 分别维护自己的 backend oracle，例如 `select_fp8_moe_backend`、`select_nvfp4_moe_backend`、`select_mxfp4_moe_backend` 和 `select_wna16_moe_backend`。这些 selector 共同采用以下模式：

```python
if config.moe_backend != "auto":
    requested_backend = map_backend(config.moe_backend)
    return validate_or_raise(requested_backend, layer_config)

for backend in format_specific_priority:
    for experts_cls in backend_to_experts_classes(backend):
        supported, reason = experts_cls.is_supported_config(
            experts_cls,
            layer_config,
            weight_quant_key,
            activation_quant_key,
            activation_format,
        )
        if supported:
            return backend, experts_cls

raise NotImplementedError("No backend supports this deployment configuration")
```

与 Linear 当前按 layer 容错回退的行为不同，显式指定一个不支持当前 MoE configuration 的 backend 通常会在 selector 中抛错。`auto` 才会继续尝试下一候选项。即使启动时已经选择 primary backend，部分实现仍可能针对特殊 shape 在 runtime 使用其内部 fallback，因此日志中应同时核对 backend 名称、具体 expert class 和最终 kernel 路径

### Attention backend

`Attention backend` 位于 QKV projection 之后。`qkv_proj` 与 `o_proj` 仍由普通 Linear 路径执行；Attention backend 接收已经形成的 Q、K、V，以及 sequence length、block table、mask、KV Cache layout 等 metadata，负责 prefill/decode 阶段的注意力计算和 KV Cache 访问。因此 `--linear-backend cutlass` 与 `--attention-config '{"backend":"FLASH_ATTN"}'` 可以同时生效，二者控制的不是同一组 kernel

当前源码在 `vllm/v1/attention/backends/registry.py` 中通过 `AttentionBackendEnum` 注册实现。下面保留与 NVIDIA GPU 常规 Attention、`SDPA`、`FlexAttention` 以及 MLA 直接相关的主要条目：

```python
class AttentionBackendEnum(Enum):
    FLASH_ATTN = (
        "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
    )
    TRITON_ATTN = (
        "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"
    )
    FLASHINFER = (
        "vllm.v1.attention.backends.flashinfer.FlashInferBackend"
    )
    FLEX_ATTENTION = (
        "vllm.v1.attention.backends.flex_attention.FlexAttentionBackend"
    )
    TORCH_SDPA = ""  # this tag is only used for ViT

    FLASHINFER_MLA = (
        "vllm.v1.attention.backends.mla.flashinfer_mla.FlashInferMLABackend"
    )
    TRITON_MLA = (
        "vllm.v1.attention.backends.mla.triton_mla.TritonMLABackend"
    )
    FLASH_ATTN_MLA = (
        "vllm.v1.attention.backends.mla.flashattn_mla.FlashAttnMLABackend"
    )
    FLASHMLA = (
        "vllm.v1.attention.backends.mla.flashmla.FlashMLABackend"
    )
```

<div align="center">
<table>
  <thead>
    <tr>
      <th align="center">Backend</th>
      <th align="center">主要执行方式</th>
      <th align="center">在 vLLM 中的定位</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>FLASH_ATTN</code></td><td>FlashAttention FA2/FA3/FA4 kernel</td><td>常规 decoder、encoder、cross-attention 与 sliding-window 的主要 NVIDIA 路径；支持 paged KV Cache，具体能力受 FA 版本约束</td></tr>
    <tr><td><code>FLASHINFER</code></td><td>FlashInfer prefill/decode kernel</td><td>另一套 NVIDIA paged-attention 实现；在部分 GPU 架构、batch shape 或 KV Cache format 下优先</td></tr>
    <tr><td><code>TRITON_ATTN</code></td><td>Triton Attention kernel</td><td>实现灵活，常作为 FlashAttention/FlashInfer 之外的候选或特定功能路径</td></tr>
    <tr><td><code>FLEX_ATTENTION</code></td><td><code>torch.compile(torch.nn.attention.flex_attention)</code></td><td>通过 <code>score_mod</code> 与 <code>BlockMask</code> 表达复杂 mask；当前支持 decoder 和 encoder-only，KV Cache layout 与 dtype 有明确约束</td></tr>
    <tr><td><code>TORCH_SDPA</code></td><td><code>torch.nn.functional.scaled_dot_product_attention</code></td><td>当前注册表明确标注仅用于 ViT / multimodal encoder；它不是普通 paged decoder 的候选 backend</td></tr>
    <tr><td><code>*_MLA</code></td><td>FlashInfer、FlashAttention、FlashMLA、Triton 等 MLA kernel</td><td>面向 Multi-head Latent Attention；与普通 MHA/GQA 使用不同候选集</td></tr>
  </tbody>
</table>
</div>

对非 MLA 的 NVIDIA decoder，`auto` 不是简单调用 PyTorch `SDPA`。`vllm/platforms/cuda.py` 中的常规候选优先级如下：SM100 的 causal Attention 优先 FlashInfer；其他 CUDA 情况通常优先 FlashAttention。每个 class 随后通过 `validate_configuration()` 检查 compute capability、head size、dtype、KV Cache dtype、block size、sliding window、attention type 与并行配置

```python
def _get_backend_priorities(
    use_mla,
    device_capability,
    num_heads=None,
    kv_cache_dtype=None,
    use_non_causal=False,
):
    if not use_mla:
        if device_capability.major == 10 and not use_non_causal:
            return [
                AttentionBackendEnum.FLASHINFER,
                AttentionBackendEnum.FLASH_ATTN,
                AttentionBackendEnum.TRITON_ATTN,
                AttentionBackendEnum.FLEX_ATTENTION,
                AttentionBackendEnum.TURBOQUANT,
            ]
        return [
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.FLASHINFER,
            AttentionBackendEnum.TRITON_ATTN,
            AttentionBackendEnum.FLEX_ATTENTION,
            AttentionBackendEnum.TURBOQUANT,
        ]
```

显式指定 backend 时，vLLM 只验证该实现；配置不兼容会在启动时抛错。未指定时，selector 收集所有通过验证的候选并选择优先级最高者，而不是对它们执行在线 benchmark：

```python
if selected_backend is not None:
    backend_class = _get_attn_backend_class(selected_backend)
    invalid_reasons = backend_class.validate_configuration(
        device_capability=device_capability,
        **attn_selector_config._asdict(),
    )
    if invalid_reasons:
        raise ValueError(
            f"Selected backend {selected_backend} is not valid for "
            f"this configuration. Reason: {invalid_reasons}"
        )
    return _backend_cls_path(backend_class)

valid_backends, invalid_reasons = cls.get_valid_backends(...)
selected_candidate = min(
    valid_backends,
    key=lambda candidate: candidate.priority,
)
return _backend_cls_path(selected_candidate.backend_class)
```

三个容易混淆的执行入口如下。FlashAttention 直接消费 paged KV Cache、block table 与 variable-length metadata；FlexAttention 通过编译后的 `flex_attention` 消费 `BlockMask` 和 `score_mod`；ViT 的 Torch SDPA wrapper 则把 Q/K/V 调整为 `[B,H,T,D]` 后调用 PyTorch SDPA

```python
# FlashAttention decoder path
flash_attn_varlen_func(
    q=query[:num_actual_tokens],
    k=key_cache,
    v=value_cache,
    out=output[:num_actual_tokens],
    cu_seqlens_q=cu_seqlens_q,
    seqused_k=seqused_k,
    block_table=block_table,
    causal=causal,
    window_size=sliding_window_size,
    softmax_scale=self.scale,
    fa_version=self.vllm_flash_attn_version,
)

# FlexAttention decoder / encoder-only path
out = flex_attention_compiled(
    query,
    key_tensor,
    value_tensor,
    attn_metadata.transformed_score_mod,
    attn_metadata.block_mask,
    self.scale,
    enable_gqa=enable_gqa,
    kernel_options=kernel_options,
)

# TORCH_SDPA: ViT / multimodal encoder path
q, k, v = (
    einops.rearrange(x, "b s h d -> b h s d")
    for x in (q, k, v)
)
output = F.scaled_dot_product_attention(
    q,
    k,
    v,
    dropout_p=0.0,
    scale=scale,
    enable_gqa=enable_gqa,
)
```

常规 decoder 可通过 `--attention-config '{"backend":"FLASH_ATTN"}'` 或 `--attention-config '{"backend":"FLEX_ATTENTION"}'` 显式选择；`auto` 对应不设置 `backend`。多模态视觉 encoder 使用独立的 `--mm-encoder-attn-backend TORCH_SDPA`。当前配置还支持 `backend_per_kind`，可让 full attention、sliding-window 或 MLA 等不同 KV Cache group 使用不同实现。Attention backend 只影响 Attention dataflow，不改变本案例的 NVFP4 Linear/MoE activation scale，因此不能修复缺失 `input_scale` 的 W4A4 expert GEMM

### NVFP4 backend 选择

NVFP4 进入 vLLM 后仍然分别选择 Linear 与 MoE backend。普通 Linear 由 `ModelOptNvFp4LinearMethod` 调用 `init_nvfp4_linear_kernel()`；MoE 由 `ModelOptNvFp4FusedMoE` 调用 `select_nvfp4_moe_backend()`。两条路径共享 NVFP4 checkpoint 语义，但候选 class、format conversion 和执行 dataflow 不同

普通 NVFP4 Linear 在 CUDA 上使用下面的有序候选列表。`--linear-backend auto` 选择第一个通过 `is_supported()` 和 `can_implement()` 的 class；显式 `--linear-backend` 会先按前面的 `_LINEAR_BACKEND_KERNEL_MAP` 过滤列表

```python
_POSSIBLE_NVFP4_KERNELS = {
    PlatformEnum.CUDA: [
        FlashInferCuteDslNvFp4LinearKernel,
        FlashInferCutlassNvFp4LinearKernel,
        FlashInferB12xNvFp4LinearKernel,
        CutlassNvFp4LinearKernel,
        MarlinNvFp4LinearKernel,
        FlashInferTrtllmNvFp4LinearKernel,
        FlashInferCudnnNvFp4LinearKernel,
        FbgemmNvFp4LinearKernel,
        B12xNvFp4LinearKernel,
        EmulationNvFp4LinearKernel,
        HummingNvFp4LinearKernel,
    ],
    PlatformEnum.ROCM: [EmulationNvFp4LinearKernel],
}

class ModelOptNvFp4LinearMethod(LinearMethodBase):
    def __init__(self, quant_config):
        self.kernel = init_nvfp4_linear_kernel()
```

若最终选择 `CutlassNvFp4LinearKernel`，加载结束后先将 block scale swizzle，并对 packed weight 做 CUTLASS padding；forward 中再动态量化输入并执行一次 FP4 GEMM

```python
def process_weights_after_loading(self, layer):
    layer.weight_scale = torch.nn.Parameter(
        swizzle_blockscale(layer.weight_scale.data), requires_grad=False
    )
    padded_weight, padding_cols = pad_nvfp4_weight_for_cutlass(layer.weight.data)
    layer.weight = torch.nn.Parameter(padded_weight, requires_grad=False)
    layer.weights_padding_cols = padding_cols

def apply_weights(self, layer, x, bias=None):
    output_size = layer.output_size_per_partition
    output_shape = [*x.shape[:-1], output_size]
    padding_bytes = getattr(layer, "weights_padding_cols", 0)

    x_fp4, x_blockscale = scaled_fp4_quant(
        x,
        layer.input_global_scale_inv,
        is_sf_swizzled_layout=True,
        backend="cutlass",
        padded_n=x.shape[-1] + padding_bytes * 2,
    )
    out = cutlass_scaled_fp4_mm(
        x_fp4,
        layer.weight,
        x_blockscale,
        layer.weight_scale,
        layer.alpha,
        x.dtype,
    )
    out = slice_nvfp4_output(out, output_size)
    if bias is not None:
        out = out + bias
    return out.view(*output_shape)
```

MoE 侧首先由 ModelOpt quantization method 声明 weight 与 activation 的 quantization key。Activation-quantized NVFP4 checkpoint 使用 `kNvfp4Static` weight 和 `kNvfp4Dynamic` activation；W4A16 checkpoint 则把 `activation_key` 设为 `None`

```python
class ModelOptNvFp4FusedMoE(FusedMoEMethodBase):
    def __init__(self, quant_config, moe_config):
        self.use_a16 = quant_config.quant_method == "W4A16_NVFP4"
        self.nvfp4_backend, self.experts_cls = select_nvfp4_moe_backend(
            config=moe_config,
            weight_key=kNvfp4Static,
            activation_key=None if self.use_a16 else kNvfp4Dynamic,
        )
```

用户级 `MoEBackend` 随后在 `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py` 中映射到 NVFP4 内部 enum。CLI 应写 `--moe-backend cutlass`；`VLLM_CUTLASS` 是内部名称，不是 CLI value

```python
class NvFp4MoeBackend(Enum):
    B12X = "B12X"
    FLASHINFER_TRTLLM = "FLASHINFER_TRTLLM"
    FLASHINFER_CUTLASS = "FLASHINFER_CUTLASS"
    FLASHINFER_CUTEDSL = "FLASHINFER_CUTEDSL"
    FLASHINFER_CUTEDSL_BATCHED = "FLASHINFER_CUTEDSL_BATCHED"
    FLASHINFER_B12X = "FLASHINFER_B12X"
    VLLM_CUTLASS = "VLLM_CUTLASS"
    MARLIN = "MARLIN"
    HUMMING = "HUMMING"
    EMULATION = "EMULATION"

mapping = {
    "b12x": NvFp4MoeBackend.B12X,
    "cutlass": NvFp4MoeBackend.VLLM_CUTLASS,
    "flashinfer_trtllm": NvFp4MoeBackend.FLASHINFER_TRTLLM,
    "flashinfer_cutlass": NvFp4MoeBackend.FLASHINFER_CUTLASS,
    "flashinfer_cutedsl": NvFp4MoeBackend.FLASHINFER_CUTEDSL,
    "flashinfer_b12x": NvFp4MoeBackend.FLASHINFER_B12X,
    "marlin": NvFp4MoeBackend.MARLIN,
    "humming": NvFp4MoeBackend.HUMMING,
    "emulation": NvFp4MoeBackend.EMULATION,
}
```

<div align="center">
<table>
  <thead>
    <tr>
      <th align="center">CLI value</th>
      <th align="center">内部 NVFP4 backend</th>
      <th align="center">主要 expert class</th>
      <th align="center">Activation 路径</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>cutlass</code></td><td><code>VLLM_CUTLASS</code></td><td><code>CutlassExpertsFp4</code></td><td>W4A4，消费 per-expert activation global scale</td></tr>
    <tr><td><code>flashinfer_cutlass</code></td><td><code>FLASHINFER_CUTLASS</code></td><td><code>FlashInferExperts</code></td><td>NVFP4 activation quantization</td></tr>
    <tr><td><code>flashinfer_cutedsl</code></td><td><code>FLASHINFER_CUTEDSL</code></td><td><code>FlashInferCuteDSLExperts</code></td><td>FP4；batched format 使用 batched class</td></tr>
    <tr><td><code>flashinfer_trtllm</code></td><td><code>FLASHINFER_TRTLLM</code></td><td><code>TrtLlmNvFp4ExpertsMonolithic</code> 优先</td><td>可融合 routing 与 experts</td></tr>
    <tr><td><code>flashinfer_b12x</code></td><td><code>FLASHINFER_B12X</code></td><td><code>FlashInferB12xExperts</code></td><td>SM12x FP4</td></tr>
    <tr><td><code>b12x</code></td><td><code>B12X</code></td><td><code>B12xExperts</code></td><td>Native SM12x FP4</td></tr>
    <tr><td><code>marlin</code></td><td><code>MARLIN</code></td><td><code>MarlinExperts</code></td><td>不执行 FP4 activation quantization</td></tr>
    <tr><td><code>humming</code></td><td><code>HUMMING</code></td><td>Humming grouped/indexed experts</td><td>Mixed precision</td></tr>
    <tr><td><code>emulation</code></td><td><code>EMULATION</code></td><td><code>Nvfp4QuantizationEmulationTritonExperts</code></td><td>QDQ/reference</td></tr>
  </tbody>
</table>
</div>

NVFP4 MoE 的 `auto` 优先级由 `select_nvfp4_moe_backend` 明确给出。它不是所有 MoE 格式的通用顺序，只适用于当前 NVFP4 selector。显式 backend 只检查指定项；`auto` 才按下列次序逐项执行 `is_supported_config`。`b12x` 与 `flashinfer_b12x` 虽然存在于用户枚举和内部映射中，但没有进入该源码快照的自动候选列表，需要显式选择

```python
AVAILABLE_BACKENDS = [
    NvFp4MoeBackend.FLASHINFER_TRTLLM,
    NvFp4MoeBackend.FLASHINFER_CUTEDSL,
    NvFp4MoeBackend.FLASHINFER_CUTEDSL_BATCHED,
    NvFp4MoeBackend.FLASHINFER_CUTLASS,
    NvFp4MoeBackend.VLLM_CUTLASS,
    NvFp4MoeBackend.MARLIN,
    NvFp4MoeBackend.HUMMING,
    NvFp4MoeBackend.EMULATION,
]

if config.moe_backend != "auto":
    requested_backend = map_nvfp4_backend(config.moe_backend)
    return _return_or_raise(
        requested_backend,
        config,
        weight_key,
        activation_key,
        activation_format,
    )

for backend in AVAILABLE_BACKENDS:
    for kernel_cls in backend_to_kernel_cls(backend):
        supported, reason = kernel_cls.is_supported_config(
            kernel_cls,
            config,
            weight_key,
            activation_key,
            activation_format,
        )
        if supported:
            return backend, kernel_cls
```

选择完成后，`convert_to_nvfp4_moe_kernel_format` 才执行 backend-specific layout conversion。CUTLASS/FlashInfer 分支保留并转换 activation scale；Marlin 分支把 `a13_scale/a2_scale` 设为 `None`，随后转换 weight-only layout

```python
if nvfp4_backend == NvFp4MoeBackend.B12X:
    tensors = prepare_nvfp4_moe_layer_for_b12x(...)

elif (
    nvfp4_backend in FLASHINFER_NVFP4_MOE_BACKENDS
    or nvfp4_backend == NvFp4MoeBackend.VLLM_CUTLASS
):
    tensors = prepare_nvfp4_moe_layer_for_fi_or_cutlass(...)

elif nvfp4_backend == NvFp4MoeBackend.MARLIN:
    a13_scale = None
    a2_scale = None
    tensors = prepare_nvfp4_moe_layer_for_marlin(...)
```

若选择 `--moe-backend cutlass`，最终 class 是 `CutlassExpertsFp4`。下面是 `vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py` 中 `run_cutlass_moe_fp4` 的核心执行代码：

```python
# 根据 top-k routing 构造每个 expert 的 problem size 与 row map
ops.get_cutlass_moe_mm_data(
    topk_ids,
    expert_offsets,
    problem_sizes1,
    problem_sizes2,
    a_map,
    c_map,
    e,
    n,
    k,
    blockscale_offsets,
    is_gated=activation.is_gated,
)

# token permutation + 第一次 activation quantization
a = ops.shuffle_rows(a, a_map)
rep_a_fp4, rep_a_blockscale = ops.scaled_fp4_experts_quant(
    a,
    a1_gscale,
    expert_offsets,
    blockscale_offsets,
    topk_ids.size(1),
)

# Grouped expert GEMM 1: w13 / gate-up
ops.cutlass_fp4_moe_mm(
    c1,
    rep_a_fp4,
    w1_fp4,
    rep_a_blockscale,
    w1_blockscale,
    w1_alphas,
    problem_sizes1,
    expert_offsets[:-1],
    blockscale_offsets[:-1],
)

# SiLU + multiply + 第二次 activation quantization
int_fp4, int_blockscale = ops.silu_and_mul_scaled_fp4_experts_quant(
    c1,
    a2_gscale,
    expert_offsets,
    blockscale_offsets,
    topk_ids.size(1),
)

# Grouped expert GEMM 2: w2 / down
ops.cutlass_fp4_moe_mm(
    c3,
    int_fp4,
    w2_fp4,
    int_blockscale,
    w2_blockscale,
    w2_alphas,
    problem_sizes2,
    expert_offsets[:-1],
    blockscale_offsets[:-1],
)

# unpermute + routing-weighted reduction
c3 = ops.shuffle_rows(c3, c_map)
output.copy_(
    (
        c3.view(m, topk_ids.size(1), k)
        * topk_weights.view(m, topk_ids.size(1), 1).to(a.dtype)
    ).sum(dim=1)
)
```

NVFP4 选择链不是 `模型是 NVFP4，所以固定使用某一个 kernel`，而是 `quantization method → Linear/MoE 分流 → 用户 backend 或 auto priority → support check → backend-specific format conversion → concrete kernel class`。`cutlass` MoE 最终执行完整的 `W4A4 expert dataflow`，并两次消费 `per-expert activation scale`；`marlin` 仍计算相同的 `w13/w2` 逻辑网络，但 `format conversion` 会移除 `activation scale`，activation 保持 BF16/FP16。缺失 `input_scale` 因而只会破坏需要它的 `W4A4 backend`，而不会同样破坏 Marlin 路径

## 原因总结

`input_scale` 对 NVFP4 W4A4 是必需的数值参数。Activation 在进入 FP4 GEMM 前仍为 BF16/FP16，vLLM 必须读取当前 `expert/projection` 的 `input_scale`，构造 `a_gscale = 1 / input_scale`，再动态生成 FP4 activation。正常的 `input_scale` 必须是正的有限数；若缺失 slot 中的未初始化值恰为 `0`，取倒数后会得到 `inf`，若其 bit pattern 对应其他任意值，则可能得到错误的有限值、`NaN` 或 `inf`。[vLLM #45212](https://github.com/vllm-project/vllm/issues/45212) 已在同类缺项中观察到 affected expert 的 global scale 为 `inf`，并把 `NaN` 定位到第一个 MoE GEMM 输出

本 checkpoint 的直接缺陷来自 ModelOpt calibration 使用真实 MoE routing。部分低频 `expert` 在校准期间没有接收到 token，因此没有形成相应的 `activation statistics`，导出结果也缺少这些 `expert` 的 `w13_input_scale` 或 `w2_input_scale`。vLLM 又以 `torch.empty()` 分配完整的 `per-expert scale buffer`，checkpoint loader 只填写实际存在的条目，加载结束后没有验证 128 个 `expert slot` 是否全部写入。这使本应在启动阶段报告的 checkpoint 完整性错误转化为运行期的 `silent correctness failure`

只有 router 命中缺项 `expert` 时，未初始化 scale 才进入计算。因此故障具有 `prompt-dependent` 特征：调整 token 顺序、system role 或 prompt 内容会改变 routing，输出可能随之从正常答案变成全部 PAD token 或空内容。服务进程、GPU kernel 和 HTTP 协议处理仍可正常完成，所以服务存活、吞吐率与 HTTP 200 均不能证明模型输出具有数值有效性

这里必须区分 NVFP4 activation 的两级 scale。checkpoint 中缺失的是 `s_global`，即 ModelOpt calibration 得到的 `per-expert activation global scale`，在 vLLM 中对应 `input_scale`。第二级 `s_block` 并未因 checkpoint 缺字段而缺失；它是 vLLM 根据当前 activation 在运行时为每 16 个元素动态生成的 FP8 E4M3 `micro-block scale`。两者与 FP4 数据的关系可简记为 `x ≈ q_fp4 × s_block × s_global`

```text
checkpoint 中的 per-expert input_scale / activation global scale
        │
        ▼
a_gscale = 1 / input_scale
        │
        ▼
当前 BF16/FP16 activation
        │
        ▼
运行时为每 16 个元素计算 FP8 E4M3 block scale
        │
        ▼
生成 FP4 activation + FP8 block scales
        │
        ▼
NVFP4 W4A4 GEMM
```

CUTLASS MoE 的第一次量化把 `a1_gscale` 传给 `scaled_fp4_experts_quant()`，生成 `rep_a_fp4` 与 `rep_a_blockscale`，随后执行 `w13` GEMM。SiLU/mul 后，第二次量化再使用 `a2_gscale`，由 `silu_and_mul_scaled_fp4_experts_quant()` 生成 `int_fp4` 与 `int_blockscale`，随后执行 `w2` GEMM。错误的 global scale 因而会先破坏 activation 的 block-scale 计算和 E2M1 编码，再沿两次 expert GEMM 传播到 hidden state、后续 layer、LM head 与最终 token

这也解释了 `--moe-backend marlin` 正常而 `--moe-backend cutlass` 失败。CUTLASS 的 NVFP4 MoE 路径执行 W4A4，activation 同样量化为 FP4，所以实际消费 `w13_input_scale` 与 `w2_input_scale`。Marlin 走 weight-only W4A16，activation 保持 BF16/FP16；`format conversion` 直接将 `a13_scale` 和 `a2_scale` 置为 `None`，因此不会读取这些缺失的 activation scale

```text
ModelOpt calibration 未覆盖所有 MoE expert
        ↓
checkpoint 缺少部分 w13_input_scale / w2_input_scale
        ↓
vLLM 使用 torch.empty()，且未校验 per-expert coverage
        ↓
CUTLASS W4A4 获得错误的 a1_gscale / a2_gscale
        ↓
activation block scale 与 FP4 编码被错误 global scale 驱动
        ↓
两次 expert GEMM 污染 hidden state
        ↓
命中缺项 expert 的 prompt 返回 PAD token / 空内容
```

Weight 侧的 E4M3 block scale 是 checkpoint 中静态保存的 `w13_weight_scale` / `w2_weight_scale`，与 activation 侧运行时生成的 block scale 不同。目前确认缺失的是部分 expert 的 `w13_input_scale` / `w2_input_scale`，没有证据表明相应的 `w13_weight_scale` / `w2_weight_scale` 也缺失。相关评论建议同时验证 `weight_scale_2`，目的是让 loader 对全部必要的 `per-expert scale` 执行 fail-fast 完整性检查，并不表示该 Gemma 4 checkpoint 已确认缺少 weight scale。Linux SM120 复现、12 个 layer 共缺少 25 个 activation scale 条目，以及 layer 0 缺少 expert `40、42、82、98` 的证据见 [issue #51525 的 Linux 复现与根因评论](https://github.com/vllm-project/vllm/issues/51525#issuecomment-5229322893)
