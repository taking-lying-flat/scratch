# Gemma 4 NVFP4 MoE 空输出：从量化背景到缺失 Scale 的完整分析

> [!CAUTION]
> `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4` 在 vLLM 的 activation-quantized NVFP4 MoE 路径中可能返回 HTTP 200，同时生成 PAD token 或空内容。服务存活、吞吐率和响应状态只说明请求完成了协议处理，无法证明模型输出具有数值有效性。

| 项目 | 已确认结论 |
| --- | --- |
| 复现模型 | `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4` |
| 已确认环境 | RTX 5090 / SM120，vLLM `0.24.0`；Windows 首报，Linux 后续复现 |
| 失败路径 | 显式选择 `VLLM_CUTLASS` NVFP4 MoE，执行 W4A4 expert computation |
| 对照路径 | 仅将 MoE backend 切换为 Marlin，执行 weight-only W4A16 |
| checkpoint 缺陷 | 12 个 layer 中共有 25 个 expert activation scale 条目缺失；layer 0 缺少 expert `40、42、82、98` 的 `input_scale` |
| loader 缺陷 | per-expert scale 由 `torch.empty` 分配，加载结束后没有验证 checkpoint coverage |
| 外部症状 | 请求耗尽 `max_tokens`，返回 PAD token、`content: null` 或空字符串 |
| 修复状态 | fail-fast 修复 [PR #45320](https://github.com/vllm-project/vllm/pull/45320) 截至 2026-08-27 仍处于开放状态 |

## 目录

- [阅读目标与结论](#阅读目标与结论)
- [故障如何被定位](#故障如何被定位)
- [ModelOpt、NVFP4、checkpoint 与 backend](#modeloptnvfp4checkpoint-与-backend)
- [PTQ 与 calibration 在做什么](#ptq-与-calibration-在做什么)
- [W4A16、W8A8、W4A4 与 KV Cache](#w4a16w8a8w4a4-与-kv-cache)
- [NVFP4：从 E2M1 到两级 scale](#nvfp4从-e2m1-到两级-scale)
- [如何阅读 Transformer 中的矩阵](#如何阅读-transformer-中的矩阵)
- [从 Dense FFN 到 MoE](#从-dense-ffn-到-moe)
- [MoE 路由为何会制造 calibration 空洞](#moe-路由为何会制造-calibration-空洞)
- [NVFP4 checkpoint 中保存了什么](#nvfp4-checkpoint-中保存了什么)
- [vLLM 的 Linear backend 与 MoE backend](#vllm-的-linear-backend-与-moe-backend)
- [复现边界](#复现边界)
- [缺失 scale 如何传播成空输出](#缺失-scale-如何传播成空输出)
- [Marlin 对照为何能够通过](#marlin-对照为何能够通过)
- [修复策略与验证边界](#修复策略与验证边界)
- [从本案例可以推广的方法](#从本案例可以推广的方法)
- [结论](#结论)

## 阅读目标与结论

这篇案例以一个静默空输出 Bug 为主线，同时回答五组背景问题：低精度 checkpoint 如何产生；`W4A4` 与 `W4A16` 描述了什么；NVFP4 为什么同时需要 FP4 数据、FP8 block scale 和 FP32 global scale；Dense FFN 如何演化为带 expert 维的 MoE；vLLM 为什么能对普通 Linear 与 MoE 分别选择 backend。掌握这些关系后，`input_scale` 的缺失就能从一个孤立字段错误还原为完整的量化数据依赖问题。

根因包含两个相互衔接的完整性缺口。ModelOpt calibration 使用真实 MoE routing，部分低频 expert 没有接收到 calibration token，因此没有形成相应的 activation statistics，导出的 checkpoint 缺少 per-expert `input_scale`。vLLM 随后使用 `torch.empty` 创建 scale tensor，只写入 checkpoint 中存在的条目，且在 backend format conversion 前没有执行 completeness validation。路由命中缺项 expert 后，错误 global scale 进入 W4A4 activation quantization，数值污染沿两次 expert GEMM 传播到 hidden state 和最终 token。

Linux SM120 复现确认 12 个 layer 共缺少 25 个 expert activation scale 条目，并将 layer 0 的缺项定位为 `40、42、82、98`。相同请求在 `VLLM_CUTLASS` 下返回 48 个 PAD token，切换到 Marlin 后得到正确答案；证据见 [issue #51525 的 Linux 复现与根因评论](https://github.com/vllm-project/vllm/issues/51525#issuecomment-5229322893)。更早的 [issue #45212](https://github.com/vllm-project/vllm/issues/45212) 已在同类缺项上观察到 global scale 为 `inf`，并将 NaN 定位到第一个 MoE GEMM 输出。

> [!IMPORTANT]
> NVFP4 W4A4 MoE 的加载不变量是：每个可能被 activation-quantized backend 消费的 expert scale slot 均已从 checkpoint 加载，shape 与 projection 匹配，数值为正有限数。启动成功和少量 prompt 通过均无法覆盖这一不变量。

## 故障如何被定位

### 从服务健康到语义错误

原始现象容易被运行状态掩盖。Server 可以正常启动，kernel 能够 launch，请求返回 HTTP 200，吞吐统计也持续更新；异常集中在 completion 内容，表现为 `finish_reason: "length"`、`content: null` 或连续 PAD token。由此首先可以确定进程生命周期、HTTP 协议和主生成循环仍在工作，故障属于输出正确性范围。

语义健康检查需要验证生成内容、token 分布和停止原因。只检查端口、HTTP 状态、吞吐率或 GPU 利用率，会把静默数值污染判定为健康服务。该案例在 apparently healthy traffic 下运行约 2.5 小时才被发现，说明 serving 系统需要加入固定 prompt、确定性解码与期望结果校验。

### 单变量 backend 对照

失败分支显式设置 `--linear-backend cutlass --moe-backend cutlass`，成功分支保持模型、prompt、sampler、KV Cache、linear backend 和其他运行参数一致，只将 `--moe-backend` 改为 `marlin`。这个对照把差异限制在 MoE expert computation，attention 与普通量化 Linear 没有随实验变化。

Marlin 分支通过，说明 checkpoint 中足以支持 weight-only expert GEMM 的数据仍可使用；CUTLASS W4A4 失败则把调查重点收敛到 activation quantization 所需的元数据。该结论仍需 checkpoint inspection 与源码 tracing 支撑，单独的 A/B 实验只能定位依赖边界。

### Prompt dependence 指向路由覆盖

原始报告发现 system/developer role 和长周期 token 序列更容易触发空输出，保持 token 集合并改变顺序后又可能恢复。角色、长度和周期性没有成为已确认的 kernel 约束；它们共同改变 hidden state，继而改变 router 对 expert 的选择。

当缺项只分布在少数 expert 上时，prompt A 可能完全避开这些 expert，prompt B 则可能在任意一个 MoE layer 命中缺项。输出因 prompt 改变而呈现间歇性，实际控制变量是执行过程中形成的 expert route。Linux checkpoint inspection 找到缺失 expert id 后，这一解释获得了直接证据。

> [!NOTE]
> Prompt 是路由输入，缺失 expert 是故障状态。周期性、role 和 token 数量在现有实验中充当 route-shaping factor，尚未被证明具有独立的 kernel 级触发语义。

## ModelOpt、NVFP4、checkpoint 与 backend

本案例中的几个名称位于不同抽象层。ModelOpt 是 NVIDIA 的模型优化与量化工具；NVFP4 是数值表示和缩放规范；Hugging Face checkpoint 是量化产物的持久化载体；vLLM ModelOpt loader 负责解释字段并转换 layout；CUTLASS、FlashInfer 与 Marlin 则执行运行时 kernel。将这些层次分开后，`--quantization modelopt` 和 `--moe-backend cutlass` 的含义就不会混淆。

| 名称 | 抽象层 | 当前案例中的职责 |
| --- | --- | --- |
| NVIDIA ModelOpt | PTQ、calibration 与导出工具 | 在校准数据上运行模型，生成量化权重和 activation statistics |
| NVFP4 | 低精度数值与 scale 规范 | 定义 E2M1 FP4、E4M3 block scale 和 FP32 global scale |
| ModelOpt checkpoint schema | 参数命名与序列化约定 | 保存 packed weight、weight scale 和 activation `input_scale` |
| vLLM ModelOpt loader | 加载与 backend format conversion | 将 checkpoint 字段写入 layer/expert tensor 并转换为 kernel layout |
| Linear/MoE backend | GPU 执行实现 | 完成 quantization、GEMM、activation 与结果归并 |

模型配置中的 `producer=modelopt` 表明量化产物由 ModelOpt 生成，`quant_algo=NVFP4` 指定数值格式，`group_size=16` 指定 micro-block 粒度。vLLM 启动参数 `--quantization modelopt` 选择对应 checkpoint loader；启动期间直接消费已经导出的量化参数，不会重新运行原始 calibration corpus。

该模型的配置为 hidden size `2816`、`30` 个 transformer layer、MoE intermediate size `704`；每个 MoE layer 配置 `128` 个 expert，并为每个 token 选择 `8` 个 expert；总参数约 `25.2B`，单 token 激活参数约 `3.8B`。模型结构和部署要求见 [模型卡](https://huggingface.co/bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4)。

> [!IMPORTANT]
> “量化格式”和“执行 backend”是两次选择。checkpoint 先确定数据采用何种低精度表示，vLLM 再根据模型、硬件和配置选择能够消费该表示的 kernel 实现。backend 接受某个 layout 只说明接口匹配，输出正确性仍取决于全部量化元数据有效。

## PTQ 与 calibration 在做什么

Post-Training Quantization（PTQ）在训练完成后确定低精度表示。以 ModelOpt 的 PyTorch 流程为例，工具先把可量化 module 转换为带 quantizer 的模型，在少量训练或评估样本上执行 forward loop，收集 activation range，再计算量化参数并导出部署 checkpoint。ModelOpt 文档说明其 PyTorch 量化阶段使用 fake quantization 模拟低精度数值，实际显存与吞吐收益由下游部署框架中的真实低精度 kernel 实现，参见 [ModelOpt PyTorch Quantization Guide](https://nvidia.github.io/Model-Optimizer/guides/_pytorch_quantization.html)。

weight 与 activation 对 calibration 的依赖不同。weight 在模型加载后已经全部可见，工具可以直接扫描每个 weight tensor，求取 block 或 tensor 级统计量。activation 取决于输入样本和执行路径，工具只能记录 calibration forward loop 实际产生的数值。某个 module 没有在 calibration 中执行，或某个 MoE expert 始终没有收到 token，对应 activation observer 就不会获得有效样本。

Calibration 的目的在于选择量化范围。范围过窄会让大值 clipping，范围过宽会减少离散表示对常用区间的分辨率。max calibration 常以观测到的绝对最大值为依据，MSE calibration 会在候选范围中最小化量化前后的误差。ModelOpt 还提供面向 NVFP4 activation global scale 的 headroom calibration，从 per-block activation amax 分布中计算 per-tensor global scale，参见 [`nvfp4_act_headroom_calibrate` 文档](https://nvidia.github.io/Model-Optimizer/reference/generated/modelopt.torch.quantization.model_calib.html)。

MoE 为 calibration 增加了一项 coverage 要求。Dense module 只要执行一次就能观察到一批 activation；MoE layer 中的 128 个 expert 分别拥有独立权重和输入分布，top-k router 每个 token 只激活其中 8 个。Calibration corpus 需要覆盖“layer × expert × projection”组合，样本数量本身无法直接保证覆盖率。

## W4A16、W8A8、W4A4 与 KV Cache

`WnAm` 中的 `W` 表示 GEMM weight，`A` 表示进入 GEMM 的 activation，数字表示各自位宽。位宽没有编码整数或浮点语义：`A8` 可以采用 INT8，也可以采用 FP8 E4M3；`A4` 在本案例中采用 NVFP4 E2M1。判断实际数值行为时，需要同时读取位宽、format、scale granularity 和 accumulation dtype。

| 执行方式 | Weight | GEMM activation | 常见目的 | 当前案例中的角色 |
| --- | --- | --- | --- | --- |
| W4A16 | 4-bit packed weight | BF16/FP16 | 降低权重显存和读取带宽，保留高精度 activation | Marlin 对照路径 |
| INT8 W8A8 | signed INT8 | signed INT8 | 使用整数 Tensor Core 与整数 scale schema | 概念对照 |
| FP8 W8A8 | FP8 E4M3/E5M2 | FP8 E4M3/E5M2 | 在较大动态范围下执行低精度 GEMM | 概念对照 |
| NVFP4 W4A4 | FP4 E2M1 | runtime 动态生成的 FP4 E2M1 | 在 Blackwell 上进一步压缩 operand 并提高 Tensor Core 吞吐 | CUTLASS 失败路径 |

W4A16 常被称为 weight-only 量化。模型权重以 4-bit 形式保存，activation 保持 BF16/FP16，kernel 可以在计算时解码或按专用布局直接参与 GEMM。该路径仍然需要 weight scale，但没有 activation FP4 quantization，因而不消费本案例缺失的 `input_scale`。

W4A4 同时量化 weight 与 GEMM activation。层间 hidden state 通常保持 BF16/FP16，activation 在进入低精度 GEMM 前被动态转换为 FP4，GEMM 输出再进入较高精度的后续算子。`A4` 描述 kernel operand，无法推出整个网络长期以 FP4 保存 hidden state。

KV Cache 构成另一条量化轴。它保存 attention 历史 token 的 K/V tensor，显存占用随 batch size 和 context length 增长；`--kv-cache-dtype fp8` 控制这部分缓存的存储精度。该参数不会选择 MoE 的 A4/A16，也不会生成 `w13_input_scale` 或 `w2_input_scale`。因此复现中的 “NVFP4 W4A4 + FP8 KV Cache” 同时包含 expert GEMM 量化与 attention cache 量化，两者解决不同资源约束。

> [!NOTE]
> 本案例的实验变量是 MoE activation 精度及其 backend。FP8 KV Cache 在失败和成功分支中保持一致，不能解释两条分支的输出差异。

## NVFP4：从 E2M1 到两级 scale

### E2M1 的 4-bit 表示

E2M1 由 1 个符号位、2 个指数位和 1 个尾数位组成。四个位只能形成 16 种编码，有限幅值集合为 `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`，另有对应负值。与 BF16/FP16 相比，E2M1 的表示集合极其稀疏；它依靠 scale 把不同 tensor 或 block 的真实数值范围映射到这些离散值。

| 正数编码 | 数值 | 负数编码 | 数值 |
| --- | ---: | --- | ---: |
| `0000` | 0.0 | `1000` | -0.0 |
| `0001` | 0.5 | `1001` | -0.5 |
| `0010` | 1.0 | `1010` | -1.0 |
| `0011` | 1.5 | `1011` | -1.5 |
| `0100` | 2.0 | `1100` | -2.0 |
| `0101` | 3.0 | `1101` | -3.0 |
| `0110` | 4.0 | `1110` | -4.0 |
| `0111` | 6.0 | `1111` | -6.0 |

下面的最小解码函数展示了 bit pattern 与离散值的对应关系。实际 GPU kernel 使用 packed representation 和硬件指令，函数只用于理解 E2M1 语义。

```python
def decode_e2m1(bits: int) -> float:
    sign = (bits >> 3) & 0b1
    exponent = (bits >> 1) & 0b11
    mantissa = bits & 0b1
    magnitudes = (
        (0.0, 0.5),
        (1.0, 1.5),
        (2.0, 3.0),
        (4.0, 6.0),
    )
    value = magnitudes[exponent][mantissa]
    return -value if sign else value
```

### 为什么需要 scale

假设一个 block 的真实值位于 `[-0.02, 0.03]`，直接映射到 E2M1 会让大量值靠近零；另一个 block 可能位于 `[-120, 90]`，直接映射则会严重 clipping。Scale 把各自的数值范围归一化到 E2M1 能表达的区间，反量化时再恢复量级。量化误差主要来自 rounding、clipping 和 scale 粒度；更小的 block 能让 scale 更贴合局部分布，同时增加 scale 存储与处理成本。

NVIDIA NVFP4 使用两级缩放。每 16 个 FP4 元素共享一个 E4M3 FP8 block scale，整个 tensor 还使用一个 FP32 global scale。重建关系可以写成：

$$
\hat{x}_i = q_{i,\mathrm{E2M1}}\,
             s_{\mathrm{block}(i),\mathrm{E4M3}}\,
             s_{\mathrm{global},\mathrm{FP32}}
$$

E4M3 block scale 负责局部动态范围，FP32 global scale 负责 tensor 级归一化，使 block scale 能落入 E4M3 的有效范围。两级结构、16 元素 micro-block 和 E2M1 值域由 [NVIDIA NVFP4 技术说明](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) 给出。

### Weight scale 与 activation scale

| 数据侧 | FP4 主体 | FP8 block scale | FP32 global scale | 生成时机 |
| --- | --- | --- | --- | --- |
| Weight | packed E2M1 weight | checkpoint 中静态保存 | checkpoint 中的 `weight_scale_2` | PTQ/export |
| Activation | 由当前 BF16/FP16 输入动态生成 | runtime 按当前 activation 生成 | checkpoint 中的 `input_scale` | global scale 来自 calibration，block scale 来自 runtime |

weight 是固定参数，PTQ 可以提前生成 FP4 weight、block scale 和 global scale。activation 随请求、token 和 expert routing 变化，vLLM 在 expert GEMM 前动态生成 FP4 activation 与 FP8 block scale；这个动态过程仍然需要离线 calibration 写入的 global `input_scale`。在此语境中，“dynamic activation quantization”描述 runtime block quantization，“calibrated activation scale”描述预先确定的 tensor 级范围，两者同时存在。

一次 W4A4 GEMM 的近似关系可以写成：

$$
A \approx Q_A S_{A,\mathrm{block}}s_{A,\mathrm{global}},
\qquad
W \approx Q_W S_{W,\mathrm{block}}s_{W,\mathrm{global}}
$$

$$
Y = AW^\mathsf{T}
\approx
(Q_A S_{A,\mathrm{block}})
(Q_W S_{W,\mathrm{block}})^\mathsf{T}
s_{A,\mathrm{global}}s_{W,\mathrm{global}}
$$

这里的乘号表示按相应 block 或 tensor 施加 scale，真实 kernel 会根据 layout 融合 quantization、Tensor Core GEMM 和 scaling。当前 checkpoint 缺失的是部分 expert 的 `s_A,global`，即 activation `input_scale`；现有证据没有显示对应 packed weight 或 weight block scale 同时缺失。Runtime 即使成功计算 `S_A,block`，也无法恢复缺少校准依据的 `s_A,global`。

> [!IMPORTANT]
> Activation block scale 的动态生成无法替代 `input_scale`。前者描述当前 16 元素 block 的局部分布，后者确定整个 activation tensor 的全局量化范围；任一层级错误都会改变 FP4 映射。

## 如何阅读 Transformer 中的矩阵

定位量化问题时，checkpoint 名称需要映射回模型计算。Attention、Dense FFN、MoE 与 SSM/GDN 都包含大规模 Linear，但权重命名、融合方式和状态管理不同。量化通常保留逻辑矩阵关系，同时改变 dtype、packing、scale tensor 和 backend-specific layout。

| 模块 | 常见逻辑权重 | serving 中的常见融合 | 后续计算或状态 |
| --- | --- | --- | --- |
| Attention | `q_proj/k_proj/v_proj/o_proj` | `qkv_proj + o_proj` | attention kernel 与 KV Cache |
| Dense gated FFN | `gate_proj/up_proj/down_proj` | `gate_up_proj + down_proj` | activation、逐元素乘法、第二次 GEMM |
| MoE gated FFN | 每个 expert 一组 gate/up/down | `w13 [E,2I,H]` 与 `w2 [E,H,I]` | routing、Grouped GEMM、weighted reduce |
| SSM/GDN | `in_proj`、`out_proj`、`conv1d`、`A_log`、`dt_bias` 等 | 更大的 input projection fusion | convolution 与 recurrent state update |

Attention 的 Q、K、V 投影共享输入 `X`。Serving 框架常把 `W_q`、`W_k`、`W_v` 沿输出维拼成 `W_qkv`，一次 `XW_qkv^T` 产生 Q/K/V，再 split 给 attention kernel。Grouped Query Attention 中 K/V 维度可能小于 Q，融合矩阵的输出维应写为 `Q_dim + K_dim + V_dim`。`o_proj` 将 attention 输出映射回 hidden size。

Dense gated FFN 同样利用共享输入融合 gate 与 up。设 hidden size 为 `H`，intermediate size 为 `I`，逻辑权重 shape 为 `W_gate [I,H]`、`W_up [I,H]`、`W_down [H,I]`。推理框架通常保存或构造 `gate_up [2I,H]`，一次 GEMM 得到两块 `[M,I]` activation，经过 SiLU/GELU 与逐元素乘法后，再由 down projection 映射回 `H`。

SSM/GDN 模型常把 q/k/v、gate 和状态参数的输入 projection 合并为更大的 `in_proj`，随后进入 convolution 与 recurrent update。该类路径有助于理解 `linear-backend` 的覆盖范围，但 Gemma 4 当前故障位于 MoE expert computation，没有进入 SSM/GDN kernel。

## 从 Dense FFN 到 MoE

### Dense gated FFN

对 `M` 个 token 组成的输入 `X [M,H]`，gated FFN 可以写成：

$$
G = XW_{\mathrm{gate}}^\mathsf{T},\qquad
U = XW_{\mathrm{up}}^\mathsf{T}
$$

$$
Z = \operatorname{SiLU}(G)\odot U,\qquad
Y = ZW_{\mathrm{down}}^\mathsf{T}
$$

Gate 和 up 的输入完全相同，因此可以把两组权重拼接为 `W_{\mathrm{gate\_up}} [2I,H]`，将前两次 GEMM 合并为一次。量化后，逻辑 shape 保持不变，物理 weight 变为 packed FP4/INT4/FP8，并增加配套 scale。

### MoE 增加 expert 维

MoE 为 FFN 增加 expert 维 `E`。每个 expert 都有独立的 gate、up 和 down 权重，融合后常表示为：

```text
w13: [E, 2I, H]    # gate/up fused weight
w2:  [E, H, I]     # down projection weight
```

命名中的 `w13` 通常表示将 `w1`（gate）和 `w3`（up）沿输出维融合，`w2` 表示 down projection。对于 expert `e`，计算仍然采用 `w13[e]` 完成第一组 GEMM，执行 activation 与逐元素乘法，再以 `w2[e]` 完成第二组 GEMM。

### Router、Top-k 与 Grouped GEMM

Router 为每个 token 计算 expert score，再选取 top-k expert。设 token `t` 的 hidden state 为 `x_t`，router 概率为 `p_{t,e}`，被选集合为 `S_t`，MoE 输出可抽象为：

$$
y_t = \sum_{e\in S_t} p_{t,e}\,F_e(x_t)
$$

同一个 token 会被复制或引用到多个 selected expert。Runtime 先根据 expert id 排列 token，expert `e` 接收的 token 数记为 `M_e`，随后执行一组 shape 不同的 GEMM：`[M_e,H] × [2I,H]^T`。逐个 expert launch kernel 会产生较高调度开销，因此 CUTLASS、Triton、FlashInfer 等实现通常使用 Grouped GEMM 或 batched expert layout 批量处理多个 expert，最后执行 unpermute 与 weighted reduce。

这种执行方式引入了两类稀疏性。计算稀疏性来自每个 token 只激活 `k` 个 expert；数据稀疏性来自不同 expert 接收的 token 数和分布差异显著。前者降低单 token 激活参数量，后者使 calibration coverage 成为独立的正确性条件。

## MoE 路由为何会制造 calibration 空洞

对 dense layer，calibration forward loop 只要执行该 layer，observer 就能看到输入 batch。对 MoE layer，observer 还受 router 控制。可以为每个 layer `l` 和 expert `e` 定义覆盖计数：

$$
C_{l,e} =
\#\{t \mid e \in S_t \text{ at layer } l\}
$$

`C_{l,e}=0` 表示 calibration corpus 中没有 token 被该 layer 的 router 分配给 expert `e`。该 expert 的 weight 仍然完整存在，因为所有参数都包含在原始模型中；它的 activation observer 没有样本，因而无法从数据估计 `input_scale`。

Gemma 4 的每个 MoE layer 有 128 个 expert，每个 token 激活 8 个。即使 calibration 总 token 数看似充足，router 也可能长期偏向部分 expert；自然语言主题、语言分布、序列位置和前层 hidden state 都会影响选择。Coverage 因此需要按 layer 和 expert 显式统计，不能用样本数间接推断。

本案例公开诊断认为缺失项对应 calibration 中从未被路由的低频 expert。整个 checkpoint 在 12 个 layer 中共有 25 个缺失 expert activation scale 条目，layer 0 缺少 `40、42、82、98`。这组缺项规模足以解释两种现象：大量普通 prompt 可以通过；特定 prompt 在任意 layer 命中缺项后出现静默污染。

> [!WARNING]
> MoE checkpoint 的 parameter coverage 与 calibration coverage 是两个集合。权重文件包含全部 expert，只能证明参数存在；W4A4 inference 还要求所有可路由 expert 都拥有有效的 activation calibration metadata。

## NVFP4 checkpoint 中保存了什么

NVFP4 checkpoint 需要保存数值主体及其 scale。具体键名会随 exporter 和 fused layout 变化，vLLM 的 ModelOpt NVFP4 MoE 路径可以按下表理解：

| 逻辑字段 | 典型 vLLM 名称 | 粒度 | 来源 | 用途 |
| --- | --- | --- | --- | --- |
| FP4 expert weight | `w13_weight`、`w2_weight` | layer × expert × packed matrix | PTQ/export | Tensor Core GEMM operand |
| Weight block scale | `w13_weight_scale`、`w2_weight_scale` | weight micro-block | PTQ/export | 重建局部 weight 范围 |
| Weight global scale | `w13_weight_scale_2`、`w2_weight_scale_2` | expert/projection | PTQ/export | weight tensor 级范围 |
| Activation global scale | `w13_input_scale`、`w2_input_scale` | expert/projection | calibration/export | runtime FP4 activation quantization |
| Activation block scale | runtime temporary tensor | routed activation micro-block | inference runtime | 重建当前 activation 的局部范围 |

`w13_input_scale[e]` 描述进入 expert `e` 第一组 gate/up GEMM 的 activation global range，`w2_input_scale[e]` 描述进入 down projection 的中间 activation global range。`w13` 可能按 gate/up shard 保存多个 scale，`w2` 通常对应单组 down scale；loader 必须同时验证 expert 维、projection/shard 维和数值域。

W4A4 所需的基本完整性条件可以写成：

```text
for every routable expert e:
    isfinite(w13_input_scale[e]) and w13_input_scale[e] > 0
    isfinite(w2_input_scale[e])  and w2_input_scale[e]  > 0
```

Weight scale 也应满足相同的正有限值约束。当前 Gemma 4 证据集中在 activation `input_scale` 缺失；修复方案同时校验 weight global scale，是为了建立通用 loader invariant，并不表示已经确认该 checkpoint 的 weight scale 同样缺项。

## vLLM 的 Linear backend 与 MoE backend

vLLM 把普通量化 Linear GEMM 与 MoE expert computation 分成两个选择面。`--linear-backend` 服务于 attention 的 `qkv_proj/o_proj`、dense FFN 的 `gate_up/down` 等二维 Linear；`--moe-backend` 服务于带 expert 维的计算，并可能联合处理 token permutation、activation quantization、两次 Grouped GEMM、非线性激活与 weighted reduce。

当前 `KernelConfig` 将两者分别定义为 “quantized linear layer GEMM kernels” 和 “MoE expert computation kernels”，并分别维护 backend 枚举，源码见固定提交中的 [`kernel.py` lines 235–280](https://github.com/vllm-project/vllm/blob/4a6a3272e8d75518efe0a6f9393eb504f3ed2ee0/vllm/config/kernel.py#L235-L280)。

| Backend | 主要实现定位 | 与当前案例的关系 |
| --- | --- | --- |
| `cutlass` | vLLM 基于 NVIDIA CUTLASS 的量化 GEMM/MoE kernel | 显式选择后进入已复现的 NVFP4 W4A4 路径 |
| `flashinfer_cutlass` | FlashInfer 集成的 CUTLASS expert kernel | 原始机器的 `auto` 选择结果，未进入报告中的显式 CUTLASS 路径 |
| `flashinfer_trtllm` | FlashInfer 接入 TRTLLM-GEN kernel | backend 枚举背景，当前 checkpoint 未完成对照验证 |
| `flashinfer_cutedsl` | 面向 FP4 等格式的 CuTeDSL 路径 | backend 枚举背景，当前 checkpoint 未完成对照验证 |
| `triton` | Triton fused MoE 实现 | 具有不同 layout 与 kernel 组织，当前 checkpoint 未完成对照验证 |
| `deep_gemm` | 面向 block-quantized FP8 的 DeepGEMM 路径 | 精度格式不同，用于理解 backend 范围 |
| `marlin` | weight-only 低比特 kernel | 当前 W4A16 成功对照，不消费 activation `input_scale` |
| `emulation` | 反量化与参考计算路径 | 适合 correctness 对照，生产性能并非其目标 |

`auto` 会根据模型、量化格式、GPU 架构和已注册实现分别选择 Linear 与 MoE backend。原始 Windows 环境的 MoE `auto` 选择 `FLASHINFER_CUTLASS`，显式设置 `--moe-backend cutlass` 才进入 `VLLM_CUTLASS`。因此该问题没有发生在首报环境的默认 backend 选择上；显式 backend 与 checkpoint capability 的组合形成了配置边界。

> [!NOTE]
> 其他 activation-quantized backend 在数据依赖上也可能受到缺失 `input_scale` 影响，这是从 scale 消费关系得到的推论。公开实验只完成了 CUTLASS 与 Marlin 对照，其他 backend 的具体结果需要逐一验证。

## 复现边界

原始报告使用 native Windows、vLLM `0.24.0`、PyTorch `2.11.0+cu130`、CUDA 13.0 和 RTX 5090 Laptop。模型以 `--quantization modelopt` 加载，KV Cache 采用 FP8，Linear 与 MoE 都显式选择 CUTLASS。失败请求使用 temperature 0，耗尽 48 个 completion token 后返回 `finish_reason: "length"` 与空内容；切换 MoE backend 后，同一请求在 8 个 token 内输出正确答案。完整命令、请求、响应与 role/periodicity 实验见 [issue #51525](https://github.com/vllm-project/vllm/issues/51525)。

两条实验分支的最小差分为：

```diff
- --quantization modelopt --linear-backend cutlass --moe-backend cutlass
+ --quantization modelopt --linear-backend cutlass --moe-backend marlin
```

原始报告来自 native Windows，随后 Linux RTX 5090 / SM120 使用相同 vLLM 版本和 checkpoint 复现了 CUTLASS PAD 输出，并完成 checkpoint inspection。Linux 结果排除了 Windows/WDDM 单平台解释。现有证据仍集中于该 Gemma 4 checkpoint、SM120 和 CUTLASS/Marlin 两个对照后端；SM100、其他 checkpoint 和其他 W4A4 backend 需要独立实验。

模型卡明确要求 `--moe-backend marlin`。这一部署说明提供了可用配置边界，无法替代 loader 对量化元数据的验证：框架接受显式 CUTLASS、完成启动并返回 HTTP 200 时，仍应在不兼容或缺项处给出确定性错误。

## 缺失 scale 如何传播成空输出

### 第一阶段：calibration 没有覆盖全部 expert

ModelOpt 在真实 MoE routing 下收集 activation statistics。部分 expert 的覆盖计数为零，observer 没有形成有效 `input_scale`，exporter 最终没有写出对应 checkpoint key。Checkpoint 仍然包含这些 expert 的量化 weight，因此常规参数数量检查可能通过。

### 第二阶段：`torch.empty` 隐藏了缺项

截至 vLLM `main` 提交 [`4a6a3272e8d75518efe0a6f9393eb504f3ed2ee0`](https://github.com/vllm-project/vllm/commit/4a6a3272e8d75518efe0a6f9393eb504f3ed2ee0)，`ModelOptNvFp4FusedMoE.create_weights()` 对 `w13_weight_scale_2`、`w2_weight_scale_2`、`w13_input_scale` 和 `w2_input_scale` 使用 `torch.empty` 分配，见 [`modelopt.py` lines 1507–1540](https://github.com/vllm-project/vllm/blob/4a6a3272e8d75518efe0a6f9393eb504f3ed2ee0/vllm/model_executor/layers/quantization/modelopt.py#L1507-L1540)。

`torch.empty` 只申请存储，不初始化元素。Checkpoint loader 会覆盖实际存在的 key，缺失 expert slot 保留 allocator 中原有 bit pattern。该内容可能表现为零、NaN、负数、极大值，也可能偶然表现为正有限数；进程、显存复用和加载顺序都可能改变结果。

加载结束后，`process_weights_after_loading()` 直接把四组 scale 传入 backend format conversion，调用前没有 completeness validation，见 [`modelopt.py` lines 1542–1579](https://github.com/vllm-project/vllm/blob/4a6a3272e8d75518efe0a6f9393eb504f3ed2ee0/vllm/model_executor/layers/quantization/modelopt.py#L1542-L1579)。缺失状态因此从序列化层穿过 loader，进入 kernel 配置。

> [!WARNING]
> 对 `torch.empty` 结果执行零值检查或 finite 检查无法证明字段已加载。未初始化 slot 可能偶然包含正有限数。确定性 completeness check 需要在加载前写入不会出现在合法 scale 中的 sentinel，并在加载后检查 sentinel 是否残留。

### 第三阶段：倒数把错误 scale 转为量化参数

普通 W4A4 backend conversion 计算 `a1_gscale = 1 / a13_scale` 和 `a2_gscale = 1 / a2_scale`，见 [`nvfp4.py` lines 511–576](https://github.com/vllm-project/vllm/blob/4a6a3272e8d75518efe0a6f9393eb504f3ed2ee0/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py#L511-L576)。零 scale 会产生 `inf`；NaN 会继续传播；任意正有限垃圾值可以绕过朴素数值检查，同时把 activation 映射到错误范围。

倒数操作位于 kernel 接触 activation 之前，因此有效 scale 的约束应在此处之前验证。等到 GEMM 输出出现 NaN 再检查，只能捕获部分表现，且错误信息已经失去 layer、parameter 和 expert id 等最有价值的诊断上下文。

### 第四阶段：CUTLASS 两次消费 activation global scale

CUTLASS MoE 根据 router 结果排列 token，调用 `scaled_fp4_experts_quant(a, a1_gscale, ...)` 产生第一组 FP4 activation 与 runtime block scale，再执行 `w13` expert GEMM。SiLU/mul 形成中间 activation 后，第二次动态量化使用 `a2_gscale`，随后执行 `w2` GEMM。两处量化都消费 per-expert global scale，源码见 [`cutlass_moe.py` lines 620–661](https://github.com/vllm-project/vllm/blob/4a6a3272e8d75518efe0a6f9393eb504f3ed2ee0/vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py#L620-L661)。

| 执行阶段 | 输入 | 使用的 activation global scale | 运行时产生的数据 | 缺项后果 |
| --- | --- | --- | --- | --- |
| Token routing/permutation | BF16/FP16 hidden state | 无 | 按 expert 排列的 token | 决定是否命中缺项 expert |
| 第一次 activation quantization | expert input | `a1_gscale` / `w13_input_scale` | FP4 activation 与 FP8 block scale | `w13` 输入范围失真 |
| `w13` GEMM 与 SiLU/mul | FP4 activation、FP4 weight | 已融合到量化/GEMM 语义 | 中间 activation | 数值污染进入 expert state |
| 第二次 activation quantization | SiLU/mul 输出 | `a2_gscale` / `w2_input_scale` | 第二组 FP4 activation 与 FP8 block scale | `w2` 输入范围失真 |
| `w2` GEMM 与 reduce | FP4 activation、FP4 weight | 已融合到量化/GEMM 语义 | MoE hidden state | 污染返回 transformer residual path |

Issue #45212 在 affected expert 上观察到 global scale 为 `inf`，并确认 NaN 出现在第一个 MoE GEMM 输出。当前 Gemma 4 复现还指出缺项经 `torch.empty` 保留未初始化数据，因此具体 bit pattern 和首次异常阶段可能随运行变化；稳定结论是 scale completeness 已经在 kernel 前被破坏。

### 第五阶段：生成循环完成，内容失效

数值污染进入 MoE hidden state 后会影响后续 layer、final norm、LM head logits 和 sampling。GPU kernel 仍可能正常结束，worker 仍会向 engine 返回 tensor，API server 也能组装合法 JSON。Serving 控制面没有异常时，请求最终表现为协议成功和语义失败。

Prompt dependence 也在此处闭环。路由避开所有缺项 expert 时，每个被消费的 scale 都有效，请求可以输出正常内容；任一 layer 命中缺项 expert 后，污染进入 residual stream，后续 token 又会在受污染 hidden state 上继续解码。一次短 smoke prompt 通过，无法证明其他 route 安全。

## Marlin 对照为何能够通过

Marlin 采用 NVFP4 weight-only W4A16 路径：weight 保持 FP4，activation 保持 BF16/FP16。vLLM 在 Marlin backend conversion 中将 `a13_scale` 与 `a2_scale` 设为 `None`，随后构造仅包含 weight scale 的 W4A16 quant config，源码见 [`nvfp4.py` lines 448–467](https://github.com/vllm-project/vllm/blob/4a6a3272e8d75518efe0a6f9393eb504f3ed2ee0/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py#L448-L467) 和 [`nvfp4.py` lines 539–550](https://github.com/vllm-project/vllm/blob/4a6a3272e8d75518efe0a6f9393eb504f3ed2ee0/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py#L539-L550)。

这条路径不会执行 activation FP4 quantization，缺失 `w13_input_scale` 与 `w2_input_scale` 因而不进入 Marlin 数据依赖。Marlin 对照恢复了当前复现的正确输出，同时证明 packed weight 与必要 weight scale 至少能够支持已测请求。

Marlin 成功没有补全 checkpoint，也没有为 W4A4 accuracy 提供证据。它是 backend-aware workaround 和定位工具；使用该 checkpoint 部署时应遵循模型卡显式选择 Marlin，并核对启动日志中的最终 MoE backend。

> [!TIP]
> 当前可用部署配置是显式使用 Marlin，并以固定 prompt 执行语义回归。长期修复仍需补全 checkpoint 或让 loader 在 W4A4 backend 启动前拒绝缺项。

## 修复策略与验证边界

### Checkpoint 生成端

Calibration 应记录每个 MoE layer 的 per-expert token count，并在 export 前验证所有 W4A4 activation quantizer 已产生有效 scale。扩充 corpus 可以提高自然路由覆盖，强制 all-expert calibration、peer scale 同步或其他补全策略则会改变统计语义，需要由 ModelOpt recipe 明确规定并通过 accuracy evaluation。导出器应将缺少 scale 视为不完整产物，错误信息包含 layer、expert、projection 和缺项类型。

### vLLM loader

确定性方案是在分配 checkpoint-loaded scale 时填入 NaN sentinel。Loader 覆盖实际存在的 slot 后，validator 在任何比较、倒数或 backend conversion 前检查 `isfinite(scale) && scale > 0`，并报告残留 sentinel 对应的 expert id。该设计把 prompt-dependent 静默污染转换为启动时的可操作错误。

[PR #45320](https://github.com/vllm-project/vllm/pull/45320) 已实现四组 per-expert scale 的 NaN sentinel 初始化与 backend-aware validation。Weight global scale 在所有 backend 上校验；activation scale 在实际消费它们的 backend 上校验，Marlin W4A16 获得豁免。PR 的 focused tests 已通过，当前 `main` 快照仍保留 `torch.empty` 行为。

### Backend capability

Backend selection 应检查 checkpoint schema 与实际 scale dependency。W4A4 backend 需要完整 activation scale 时，应在 kernel 初始化前声明并验证；W4A16 backend 可以跳过未消费的 activation field。显式 backend 选择仍应执行 capability check，避免用户通过配置进入已知无效组合。

### 恢复模式

公开方案评论记录了 per-projection median filling：只填充缺失 activation scale，在当前 26B checkpoint 上恢复了 3/3 复现请求。该实验说明统计填充可用于受控恢复，无法证明全模型 accuracy。恢复模式需要显式启用，输出 repaired slot 和 accuracy warning，并在修复后再次执行正有限值校验；缺失 weight scale 应直接拒绝加载。实现讨论见 [issue #51525 的方案评论](https://github.com/vllm-project/vllm/issues/51525#issuecomment-5229382341)。

| 验证层级 | 必需检查 | 能证明的范围 |
| --- | --- | --- |
| Schema | 所有必需 key、shape、expert id 与 shard 完整 | checkpoint 结构满足 loader 合同 |
| 数值 | 所有 scale 为正有限数，sentinel 无残留 | 量化元数据通过基本数值约束 |
| Backend | W4A4/W4A16 按实际消费关系验证字段 | 当前 kernel 不会读取缺失 scale |
| Kernel | 第一、第二 GEMM 输出无 NaN/Inf | 已测 shape 与 route 的数值传播有效 |
| 模型 | 固定 prompt、长序列、多语言、routing coverage、accuracy benchmark | 已测数据分布下的端到端质量 |

> [!IMPORTANT]
> 默认策略应为 fail-fast。缺失 activation scale 表示 calibration evidence 缺失；统计填充只能作为显式恢复模式，并需要独立的模型级 accuracy evaluation。缺失 weight scale 应始终作为硬错误。

## 从本案例可以推广的方法

第一项方法是把差分实验当作数据依赖探针。只替换 MoE backend，保持 Linear、KV Cache、sampler 与请求一致，可以把调查从整个模型收敛到 expert computation。随后比较 W4A4 与 W4A16 的 operand 和 scale dependency，就能提出可验证的 checkpoint 假设。

第二项方法是为稀疏模型显式定义 coverage。MoE、条件计算、稀疏 attention 和按需 adapter 都可能出现“参数存在、校准路径未执行”的分离。Calibration 报告应包含 route coverage，checkpoint schema 应表达字段是否必需，loader 则应验证实际 backend 所需集合。

第三项方法是避免让未初始化内存承载加载状态。`torch.empty` 适合即将被完整覆盖的高性能 buffer；checkpoint 允许部分 key 时，未覆盖 slot 必须通过 bitmap、sentinel 或显式加载记录表示。数值本身无法可靠推断字段是否曾经加载。

第四项方法是区分运行健康与语义健康。Serving 系统需要同时监控进程、kernel、协议和模型输出。固定 deterministic prompt、停止原因、非空文本、token 分布和 NaN/Inf telemetry 能够覆盖 HTTP 200 无法发现的静默正确性问题。

## 结论

Gemma 4 NVFP4 MoE 空输出源于 sparse calibration coverage、checkpoint completeness 和 loader validation 三个环节的连续失效。ModelOpt 没有为从未被路由的低频 expert 生成 activation `input_scale`；vLLM 以 `torch.empty` 接收不完整字段并继续完成 backend conversion；CUTLASS W4A4 在两次 activation quantization 中消费错误 global scale，最终把数值污染送入 MoE hidden state。Marlin W4A16 不消费 activation scale，因此构成有效对照和当前 workaround。

从背景知识看，这个 Bug 同时连接了 PTQ calibration、E2M1 FP4、NVFP4 两级缩放、Transformer projection fusion、MoE routing、Grouped GEMM、checkpoint schema 与 backend capability。可靠修复需要在 exporter 和 loader 两端建立 per-expert completeness invariant，并在 kernel 接触量化元数据前完成确定性验证。
