# ms-swift ChatML 渲染

## Qwen3.5 推理模式

Qwen3.5 是 `hybrid-thinking` 模型。同一套模型和 ChatML 模板同时支持 `thinking` 与 `non-thinking`；两种模式的区别不是切换模型结构，也不是禁止某些 token，而是模型开始生成前，assistant 位置已经预填哪一种 `response prefix`

```text
thinking
<|im_start|>assistant
<think>
模型从思考区继续生成

non-thinking
<|im_start|>assistant
<think>

</think>

模型从答案区直接生成
```

> [!IMPORTANT]
> `is_thinking` 声明模板具备什么能力；`enable_thinking` 选择本次生成的起始前缀；`add_non_thinking_prefix` 规范训练数据中已经存在的普通 assistant 回答。三者位于不同阶段，不能视为同一个开关

### 1. 从 `TemplateMeta` 到实际 `response prefix`

Qwen3.5 在 `swift/template/templates/qwen.py` 中注册为 thinking 模板，并同时提供两种 assistant 前缀：

```python
register_template(
    QwenTemplateMeta(
        MLLMTemplateType.qwen3_5,
        template_cls=Qwen3_5Template,
        default_system=None,
        thinking_prefix='<think>\n',
        non_thinking_prefix='<think>\n\n</think>\n\n',
        agent_template='qwen3_5',
        is_thinking=True,
    )
)
```

这段注册代码只定义模板协议，不选择某一次请求的运行模式：

| 字段 | Qwen3.5 的值 | 模板语义 |
| --- | --- | --- |
| `is_thinking` | `True` | 模板属于 thinking 类型 |
| `thinking_prefix` | `<think>\n` | 思考区已经打开，模型继续生成推理过程 |
| `non_thinking_prefix` | `<think>\n\n</think>\n\n` | 思考区为空且已经关闭，模型直接生成答案 |

> [!IMPORTANT]
> `is_thinking=True` 不表示每次推理都必须思考。Qwen3.5 同时注册非空的 `non_thinking_prefix`，因此它是 hybrid-thinking，而不是只能进入 thinking 的纯思考模板

实例字段的赋值语句本身看不到默认值，因为默认值定义在参数层和构造函数入口。`swift/arguments/base_args/template_args.py` 中，用户没有显式传参时的初始值是：

```python
@dataclass
class TemplateArguments:
    response_prefix: Optional[str] = None
    enable_thinking: Optional[bool] = None
    add_non_thinking_prefix: bool = True
```

这些参数经过 `get_template()` 原样传入 `Template.__init__()`。`swift/template/base.py` 中的构造函数定义为：

```python
class Template(ProcessorMixin):
    def __init__(
        self,
        processor: Processor,
        template_meta: 'TemplateMeta',
        default_system: Optional[str] = None,
        max_length: Optional[int] = None,
        *,
        # 省略与 thinking 无关的关键字参数
        response_prefix: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
        preserve_thinking: Optional[bool] = None,
        add_non_thinking_prefix: bool = True,
    ) -> None:
        ...
```

这里的 `template_meta` 没有默认值，它是构造模板时必须提供的协议对象。对于 Qwen3.5，该对象就是前面 `register_template()` 注册的 `QwenTemplateMeta`；其 `is_thinking=True`、`thinking_prefix` 和 `non_thinking_prefix` 均来自注册代码。构造函数随后复制这份模板元数据，并把尚未解析的 `enable_thinking=None` 转换成实际布尔值：

```python
template_meta = deepcopy(template_meta)

if enable_thinking is None:
    enable_thinking = (
        template_meta.is_thinking
        and not template_meta.non_thinking_prefix
    )

self.response_prefix = response_prefix
self.template_meta = template_meta
self.enable_thinking = enable_thinking
self.add_non_thinking_prefix = add_non_thinking_prefix
```

因此，Qwen3.5 实例化完成后的四个字段为：

| 实例字段 | 参数入口默认值 | Qwen3.5 的实际初值 |
| --- | --- | --- |
| `self.response_prefix` | `None` | `None` |
| `self.template_meta` | 无；构造函数必传 | Qwen3.5 注册项的副本 |
| `self.enable_thinking` | `None` | 根据模板元数据解析为 `False` |
| `self.add_non_thinking_prefix` | `True` | `True` |

Qwen3.5 的计算结果是：

```text
template_meta.is_thinking               = True
bool(template_meta.non_thinking_prefix) = True

True and not True
        ↓
enable_thinking = False
```

所以当前版本的默认语义是：**模板支持 thinking，但一次未显式配置的生成默认走 non-thinking**

> [!NOTE]
> `ms-swift v4.3.0` 使用 `enable_thinking = template_meta.is_thinking`，Qwen3.5 默认开启 thinking；从 `v4.3.1` 开始改成 `is_thinking and not non_thinking_prefix`，hybrid-thinking 模型默认关闭 thinking。关键实验应显式设置 `--enable_thinking true/false`，不要依赖版本默认值

请求到达后，`_get_enable_thinking()` 先读取单条样本的 `chat_template_kwargs`，没有提供时才回退到模板实例的全局值；`_get_response_prefix()` 又在它之前增加一层显式前缀覆盖：

```python
def _get_enable_thinking(self, inputs=None):
    enable_thinking = (
        None
        if inputs is None
        else inputs.chat_template_kwargs.get('enable_thinking')
    )
    if enable_thinking is None:
        enable_thinking = self.enable_thinking
    return enable_thinking

def _get_response_prefix(self, inputs=None):
    response_prefix = (
        None
        if inputs is None
        else inputs.chat_template_kwargs.get('response_prefix')
    )
    if response_prefix is None:
        response_prefix = self.response_prefix
    if response_prefix is not None:
        return response_prefix

    enable_thinking = self._get_enable_thinking(inputs)
    if enable_thinking:
        return self.template_meta.thinking_prefix
    else:
        return self.template_meta.non_thinking_prefix
```

完整优先级为：

```text
单条样本 chat_template_kwargs.response_prefix
        ↓ 未设置
全局 response_prefix
        ↓ 未设置
单条样本 chat_template_kwargs.enable_thinking
        ↓ 未设置
全局 enable_thinking
        ↓
thinking_prefix / non_thinking_prefix
```

因此，`enable_thinking` 并不直接拼接字符串；它只在没有显式 `response_prefix` 时，帮助 `_get_response_prefix()` 从模板注册的两种前缀中选择一个

### 2. 推理路径：前缀如何进入 ChatML

Qwen3.5 从公开编码入口进入 ChatML 拼接的完整方法解析链是：

```text
Template.encode()
        ↓
Template._encode_truncated()
        ↓
Qwen3VLTemplate._encode()          # Qwen3_5Template 继承该实现
        ↓
Template._encode(self, inputs)     # Qwen3VLTemplate 显式调用基础实现
        ↓
Qwen3_5Template._swift_prepare_inputs()
        ↓
Template._swift_encode()
```

`Qwen3_5Template` 先规范消息内容，真正拼接 ChatML 的公共控制流位于基础类 `Template._swift_encode()`；thinking/non-thinking 的前缀也在这里进入 prompt。推理请求的最后一轮没有现成的 assistant 回答。在 `_swift_encode()` 中，只需关注最后一轮的两个分支：

```python
response_prefix = self._get_response_prefix(inputs)

if response is not None:
    context_list.append('{{RESPONSE}}')
elif response_prefix:
    context_list.append(response_prefix)
```

SFT 样本已经提供 assistant 标签，进入 `{{RESPONSE}}` 分支；推理请求的 `response is None`，因此把 `_get_response_prefix()` 选出的前缀直接写到 ChatML prompt 末尾。`enable_thinking` 的作用到这里才真正落到输入 token 上

`enable_thinking=True` 时：

```text
<|im_start|>assistant
<think>
```

`<think>` 尚未闭合，模型从思考区继续生成，随后自行产生 `</think>` 和最终答案

`enable_thinking=False` 时：

```text
<|im_start|>assistant
<think>

</think>

```

空思考块已经作为 prompt 写完，模型的生成起点位于 `</think>` 之后。non-thinking 因而是一种 **prefill protocol**，不是模型自己生成空 `<think></think>`，也不是解码器通过 token blacklist 强制关闭思考

#### 返回语义：为什么 ms-swift 会显示空 `<think></think>`

`enable_thinking=False` 时，空 `<think></think>` 已经作为 `response_prefix` 写入 prompt，模型只在它后面生成答案。因此，“ms-swift 输出空 think，而 Transformers 和 vLLM 不输出”的差异不在采样阶段，而在生成完成后的 response 构造阶段

**Transformers `model.generate()`**：`transformers/generation/utils.py` 中的标准处理方式是按输入长度截取新增 token：

```python
outputs = model.generate(**inputs, return_dict_in_generate=True)
input_length = inputs.input_ids.shape[1]
generated_tokens = outputs.sequences[:, input_length:]
response = tokenizer.decode(generated_tokens[0])
```

对 decoder-only 模型，`outputs.sequences` 包含 `prompt + completion`。常见推理代码按照输入长度切片，只解码新增 token；空思考块属于 prompt，因此不会出现在 `response` 中。严格来说，`model.generate()` 没有过滤 `<think></think>`：直接解码完整的 `outputs.sequences` 仍然可以看到它

**原生 vLLM**：真正决定返回文本的是 `vllm/v1/engine/detokenizer.py`。它将输出缓冲区初始化为空，并且只向其中追加引擎新产生的 token：

```python
class BaseIncrementalDetokenizer:
    def __init__(self, request):
        self.output_text = ''

    def update(self, new_token_ids, stop_terminated):
        for new_token_id in new_token_ids:
            self.token_ids.append(new_token_id)
            self.output_text += self.decode_next(new_token_id)
```

`vllm/v1/engine/output_processor.py` 随后直接用这个缓冲区构造 `CompletionOutput`：

```python
text = self.detokenizer.get_next_output_text(finished, delta)
if not delta:
    token_ids = self.detokenizer.output_token_ids

return CompletionOutput(
    text=text,
    token_ids=token_ids,
    ...,
)
```

在 detokenizer 内部，prompt token 只用于初始化前缀状态，保证第一个生成 token 能在正确的文本边界上解码；`output_text` 本身仍从空字符串开始，并且只接收 `new_token_ids`。因此，原生 vLLM 的 `CompletionOutput.text` 只包含模型新生成的答案。空 `<think></think>` 位于 prompt 中，从未写入这个输出缓冲区，所以不会出现在返回文本里

**ms-swift**：`swift/template/base.py` 先截取新增 token，再在解码阶段恢复模板前缀

```python
def get_generate_ids(self, generate_ids, num_prompt_tokens):
    if self.skip_prompt:
        generate_ids = generate_ids[..., num_prompt_tokens:]
    return generate_ids

def decode_generate_ids(
    self,
    generate_ids,
    *,
    first_token=True,
    template_inputs=None,
    **kwargs,
):
    generate_ids = self.skip_stop_tokens(generate_ids)
    response = self.tokenizer.decode(generate_ids, **kwargs)
    response_prefix = self._get_response_prefix(template_inputs)
    if first_token and response_prefix:
        response = response_prefix + response
    return response
```

Qwen3.5 继承基础模板的 `skip_prompt=True`。因此，ms-swift 的 Transformers backend 先通过 `get_generate_ids()` 切掉整个 prompt，再由 `decode_generate_ids()` 解码新增 token；随后 ms-swift 重新调用 `_get_response_prefix()`，显式执行 `response_prefix + response`。空 `<think></think>` 正是在这一步被拼回最终 response，而不是模型本轮新生成的 token

**ms-swift 的 vLLM backend**：`swift/infer_engine/vllm_engine.py` 不直接返回原生 vLLM 的 `output.text`，而是读取 `output.token_ids`，再次调用 Swift 模板解码：

```python
for output in result.outputs:
    output.token_ids = list(output.token_ids)
    response = self.template.decode_generate_ids(
        output.token_ids,
        template_inputs=inputs['template_inputs'],
    )
```

所以，原生 vLLM 返回的是 completion，而 ms-swift 的 vLLM backend 返回的是经过模板重建的 assistant response。底层采样没有改变，变化的只是返回文本是否重新包含 `response_prefix`

### 3. SFT 路径：标签规范化与 Loss Mask

SFT 样本已经包含最后一轮 assistant 回答，因此 `response is not None`，编码器把 `{{RESPONSE}}` 放入 ChatML，而不会把推理用的 `response_prefix` 再追加一次。此时模式来自 assistant 标签本身

进入 `_swift_encode()` 之前，`Qwen3_5Template._swift_prepare_inputs()` 先对消息执行与 Qwen3.5 Hugging Face Jinja 模板一致的规范化：user/system/tool 执行 `strip()`；assistant 若已经包含完整 `<think>...</think>`，则被整理成固定换行结构

```python
elif role == 'assistant':
    stripped = content.strip()
    if '</think>' in stripped and '<think>' in stripped:
        before, _, after = stripped.partition('</think>')
        reasoning = (
            before.rstrip('\n')
            .rsplit('<think>', 1)[-1]
            .lstrip('\n')
            .strip()
        )
        rest = after.lstrip('\n')
        message['content'] = (
            f'<think>\n{reasoning}\n</think>\n\n{rest}'
        )
    else:
        message['content'] = stripped
```

随后 `_add_non_thinking_prefix()` 扫描已有 assistant 消息。只要内容既不以 `<think>` 开头，也不以完整的 `non_thinking_prefix` 开头，就在原回答前补空思考块：

```python
def _add_non_thinking_prefix(self, inputs, thinking_prefix='<think>'):
    messages = inputs.messages
    non_thinking_prefix = self.template_meta.non_thinking_prefix

    if non_thinking_prefix:
        if ((not self.is_training
             or self.loss_scale.base_strategy == 'last_round')
                and not self.template_meta.preserve_thinking):
            start_idx = get_last_user_round(messages)
        else:
            start_idx = -1

        for i, message in enumerate(messages):
            if not self._is_add_non_thinking_round(
                messages, i, start_idx
            ):
                continue

            content = message['content']
            if isinstance(content, str):
                if not content.startswith(
                    (thinking_prefix, non_thinking_prefix)
                ):
                    message['content'] = (
                        non_thinking_prefix + content
                    )
```

`start_idx` 决定规范化覆盖哪些轮次：默认 SFT 的 `loss_scale=default` 从头处理所有 assistant 回答；`last_round` 策略只处理最后一个 user 之后的 assistant 回答。这与 loss 的轮次范围保持一致，但不改变“普通回答补空思考块、已有 thinking 回答保持不变”的判断。两类 SFT 数据经过这一阶段后共享统一外壳

```text
thinking 样本
assistant: <think>
真实推理过程
</think>

最终答案

non-thinking 样本
assistant: <think>

</think>

最终答案
```

这里没有从普通答案推导出真实思维链。`add_non_thinking_prefix=True` 只是把“没有推理过程”显式编码为空思考区；已有 `<think>真实推理</think>` 的样本保持 thinking，普通回答被归一化为 non-thinking

| assistant 原始标签 | `_add_non_thinking_prefix()` | 编码后的类型 |
| --- | --- | --- |
| `<think>真实推理</think>答案` | 不修改 | thinking |
| 普通答案 | 前置空 `<think></think>` | non-thinking |

这也解释为什么 `enable_thinking` 不是混合 SFT 的数据开关：最后一轮已有 `response` 时，`_swift_encode()` 已经进入 `{{RESPONSE}}` 分支，`_get_response_prefix()` 的返回值不会被追加到该标签前

空思考块虽然存在于 `input_ids`，却不需要模型学习生成。标准参数类 `swift/arguments/base_args/template_args.py` 会识别 `is_thinking=True` 且 `non_thinking_prefix` 非空的 hybrid-thinking 模板，并自动组合 `ignore_empty_think`：

```python
def _set_loss_scale(self):
    if not self.disable_ignore_empty_think \
            and self.template_meta is not None:
        template_meta = self.template_meta
        if template_meta.is_thinking \
                and template_meta.non_thinking_prefix:
            if self.loss_scale \
                    and 'ignore_empty_think' not in self.loss_scale:
                self.loss_scale += '+ignore_empty_think'
```

对应配置的核心正则是：

```json
{
  "^<think>\\s*</think>\\s*": [0.0]
}
```

它并不是在模型 forward 中搜索 token，而是在 ChatML context 仍保持字符串分段时，将匹配到的空思考块赋权为 `0`。随后 `_encode_context_list()` 才逐段 tokenize，并把权重为 `0` 的 token 转成 `labels=-100`：

```python
for context, loss_weight in zip(
    context_list,
    loss_scale_list,
):
    token_list = self._tokenize(context)
    input_ids += token_list

    if loss_weight > 0.0:
        labels += token_list
    else:
        labels += [-100] * len(token_list)
```

完整数据流是：

```text
普通 assistant 回答
        ↓ add_non_thinking_prefix
<think></think> + 最终答案
        ↓ ignore_empty_think
[空思考块, 最终答案]
[权重 0,    权重 1]
        ↓ tokenize / _encode_context_list
input_ids: [空思考 token | 答案 token]
labels:    [-100 ...      | 答案 token]
```

上图只展开 assistant response span；模板 suffix 的监督由后续 EOS 逻辑单独处理。因此 non-thinking SFT 保留与推理一致的 response 结构，但空思考块不参与监督，答案 token 以及模板要求的结束符仍参与训练。标准 CLI 会自动追加该策略；对应配置为：

```bash
--add_non_thinking_prefix true \
--loss_scale default+ignore_empty_think
```

### 4. OPD 路径：rollout 模式与训练序列对齐

普通 SFT 使用已有标签，所以 `enable_thinking` 不选择数据类型；OPD / on-policy GKD 在训练内部先让学生模型在线生成，因此其中包含一段真正的推理路径。`swift/rlhf_trainers/rollout_mixin.py` 的 `_generate_completions()` 先预处理样本，再进入 `template.generate_context()`：

```python
def _generate_completions(self, samples):
    samples = self._preprocess_inputs(samples)

    with unwrap_model_for_generation(...), \
            self.template.generate_context(), \
            self.multi_turn_completion_length_context():
        samples = self._infer_single_or_multi_turn(
            samples,
            self.request_config,
        )

    return samples
```

预处理的关键操作是删除已有 response：

```python
def _preprocess_inputs(self, samples):
    samples = self._set_inputs_system(samples)
    samples = self._add_prompt_id_to_inputs(samples)
    for sample in samples:
        remove_response(sample.messages)
    return samples
```

删除后，最后一轮重新变成等待生成的空 assistant，因此复用第 2 节的推理分支：

```text
离线样本
        ↓ remove_response
assistant(content=None)
        ↓ _get_response_prefix
enable_thinking=True  → thinking rollout
enable_thinking=False → non-thinking rollout
        ↓
学生生成 response_token_ids
```

“`enable_thinking` 只在推理阶段生效”与“OPD 训练中它会改变样本格式”并不矛盾：OPD 训练内部会执行一次学生推理。rollout 完成后还存在一个 token 对齐问题。response prefix 属于 prompt，学生返回的 `response_token_ids` 通常只包含新增 completion；重新编码训练序列时，必须把生成时使用的前缀补回，否则 rollout 与训练使用的 token 上下文不一致

`swift/rlhf_trainers/utils.py` 先按逐样本或全局 `enable_thinking` 取得相同的 prefix ids：

```python
def get_response_prefix_ids(
    template,
    sample_enable_thinking=None,
):
    effective = (
        sample_enable_thinking
        if sample_enable_thinking is not None
        else template.enable_thinking
    )
    if effective is True:
        prefix_str = template.template_meta.thinking_prefix
    elif effective is False:
        prefix_str = template.template_meta.non_thinking_prefix
    else:
        return None
    return template.tokenizer.encode(
        prefix_str,
        add_special_tokens=False,
    )
```

`encode_sample()` 从 `sample.extra['chat_template_kwargs']` 读取逐样本值，并把 prefix ids 与 rollout ids 一起交给 `replace_assistant_response_with_ids()`：

```python
ctk = sample.extra.get('chat_template_kwargs') or {}
sample_et = ctk.get('enable_thinking')
prefix_ids = get_response_prefix_ids(
    template,
    sample_enable_thinking=sample_et,
)

data['messages'] = replace_assistant_response_with_ids(
    messages,
    sample.response_token_ids,
    loss_mask,
    non_thinking_prefix_ids=prefix_ids,
)
```

虽然参数名仍叫 `non_thinking_prefix_ids`，当前调用传入的实际是本条样本所选择的 prefix：thinking 时为 `<think>\n`，non-thinking 时为空思考块。辅助函数在 completion 尚未包含该前缀时将其补到最后一轮，并把前缀对应的 loss mask 设为 `0`：

```python
if non_thinking_prefix_ids:
    n_prefix = len(non_thinking_prefix_ids)
    last_ids = list(completion_ids[-1])

    if last_ids[:n_prefix] != list(non_thinking_prefix_ids):
        if loss_mask is None:
            loss_mask = [
                [1] * len(ids)
                for ids in completion_ids
            ]
        completion_ids[-1] = (
            list(non_thinking_prefix_ids) + last_ids
        )
        loss_mask[-1] = (
            [0] * n_prefix + list(loss_mask[-1])
        )
```

这一步同时恢复两个 invariant：

- **上下文一致**：重新编码的 assistant 序列与 rollout 时模型实际看到的 response prefix 一致
- **监督一致**：属于 prompt 的 prefix token 被补回序列，但 `loss_mask=0`，训练只消费学生真正生成的 completion

逐样本混合 rollout 只需要设置：

```json
{"chat_template_kwargs": {"enable_thinking": true}}

{"chat_template_kwargs": {"enable_thinking": false}}
```

没有逐样本值时，再回退到全局 `--enable_thinking`

三类场景最终可以由“最后一轮 assistant 是否已经存在”统一解释：

| 场景 | 编码前的最后一轮 | 实际分支 | 模式来源 |
| --- | --- | --- | --- |
| 普通推理 | `assistant(content=None)` | 追加 `response_prefix` | `response_prefix / enable_thinking` |
| 普通 SFT | `assistant(content=label)` | 追加 `{{RESPONSE}}` | 标签内容与 `add_non_thinking_prefix` |
| OPD rollout | 先删除 response，再变成 `None` | 追加 `response_prefix` | 全局或逐样本 `enable_thinking` |

> [!IMPORTANT]
> Qwen3.5 推理模式的核心不是一个布尔变量，而是一套围绕 assistant 起点建立的 prefix protocol。`enable_thinking` 只负责在“没有 response、需要生成”时选择前缀；`add_non_thinking_prefix` 只负责在“已有 response、需要训练”时规范标签；OPD 先删除 response 再生成，所以从训练流程重新进入前一条推理分支

## Qwen3-Omni 多模态编码

### 1. Qwen3-Omni 的 ChatML 结构

Qwen3-Omni 沿用 Qwen 系列的 ChatML 角色边界。system、user 和 assistant 消息分别由 `<|im_start|>{role}` 与 `<|im_end|>` 包围；工具定义位于 system 内容中，工具调用与 thinking 位于 assistant 内容中，工具返回则由 `<tool_response>` 包装为下一条 user 消息

```text
<|im_start|>system
{SYSTEM}

# Tools
<tools>
{TOOL_DEFINITIONS}
</tools>
{TOOL_CALL_INSTRUCTION}<|im_end|>
<|im_start|>user
图像：<|vision_start|><|image_pad|><|vision_end|>
视频：<|vision_start|><|video_pad|><|vision_end|>
音频：<|audio_start|><|audio_pad|><|audio_end|>
{QUERY}<|im_end|>
<|im_start|>assistant
<think>
{REASONING}
</think>

<tool_call>
{"name":"{TOOL_NAME}","arguments":{TOOL_ARGUMENTS}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{TOOL_RESULT}
</tool_response><|im_end|>
<|im_start|>assistant
<think>
{REASONING_AFTER_TOOL}
</think>

{RESPONSE}<|im_end|>
```

tools、tool call、tool response 和 thinking 均为可选内容。image、video 和 audio 作为 user 内容中的行内占位区段：

```text
image → <|vision_start|><|image_pad|><|vision_end|>
video → <|vision_start|><|video_pad|><|vision_end|>
audio → <|audio_start|><|audio_pad|><|audio_end|>
```

`<|image_pad|>`、`<|video_pad|>` 和 `<|audio_pad|>` **只标识媒体插入位置**。tokenizer 初次编码时，每份媒体仍只对应一个 pad token；实际槽位数量由媒体 processor 的输出决定

> [!IMPORTANT]
> ChatML 中的单个媒体 pad token 只是语义占位符。模型接收的 `input_ids` 中，该 token 已被扩展为与对应 encoder 输出长度一致的连续 placeholder

`ChatmlTemplateMeta` 定义角色骨架，`Qwen3OmniTemplate` 定义模型使用的媒体占位 token：

```python
@dataclass
class ChatmlTemplateMeta(TemplateMeta):
    prompt: Prompt = field(default_factory=lambda: [
        '<|im_start|>user\n{{QUERY}}<|im_end|>\n'
        '<|im_start|>assistant\n'
    ])
    chat_sep: Optional[Prompt] = field(
        default_factory=lambda: ['<|im_end|>\n'])
    suffix: Prompt = field(
        default_factory=lambda: ['<|im_end|>\n'])
    system_prefix: Optional[Prompt] = field(default_factory=lambda: [
        '<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'
    ])


class Qwen3OmniTemplate(Qwen2_5OmniTemplate):
    version = 'omni_v3'
    placeholder_tokens = [
        '<|image_pad|>',
        '<|audio_pad|>',
        '<|video_pad|>',
    ]
```

### 2. 标准 `messages` 与内部输入

ms-swift 的标准多模态样本在 **`messages.content` 中保存媒体占位符**，在 **顶层数组中保存媒体数据**：

```json
{
  "messages": [
    {"role": "system", "content": "你是个有用无害的助手"},
    {
      "role": "user",
      "content": "<image>图片中是什么，<video>视频中是什么，<audio>音频中是什么"
    },
    {
      "role": "assistant",
      "content": "图片中是一头大象，视频中是一只小狗在草地上奔跑，音频中有鸟鸣声"
    }
  ],
  "images": ["/xxx/x.jpg"],
  "videos": ["/xxx/x.mp4"],
  "audios": ["/xxx/x.wav"]
}
```

`swift/template/template_inputs.py` 中的 `StdTemplateInputs.from_dict()` 将首条 system 消息移出普通对话轮次。媒体数组与 `content` 中的占位符保持原有顺序，不会在这一阶段加载或编码：

```text
StdTemplateInputs(
    system="你是个有用无害的助手",
    messages=[
        {
            "role": "user",
            "content": (
                "<image>图片中是什么，"
                "<video>视频中是什么，"
                "<audio>音频中是什么"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "图片中是一头大象，视频中是一只小狗在草地上奔跑，"
                "音频中有鸟鸣声"
            ),
        },
    ],
    images=["/xxx/x.jpg"],
    videos=["/xxx/x.mp4"],
    audios=["/xxx/x.wav"],
)
```

**第 `i` 个媒体占位符与对应数组的第 `i` 个元素配对**：`<image>` 对应 `images[i]`，`<video>` 对应 `videos[i]`，`<audio>` 对应 `audios[i]`。仅提供顶层媒体数组时，`_add_default_tags()` 会把缺少的标记补到第一条消息开头；显式标记则直接确定媒体与文本的相对位置

### 3. `replace_tag()` 将标准标记改写为 Omni 标记

`Template._swift_encode()` 先把 system、user 和 assistant 编译为 ChatML context list。此时 user 内容仍保留标准标记：

```text
<|im_start|>system
你是个有用无害的助手<|im_end|>
<|im_start|>user
<image>图片中是什么，<video>视频中是什么，<audio>音频中是什么<|im_end|>
<|im_start|>assistant
图片中是一头大象，视频中是一只小狗在草地上奔跑，音频中有鸟鸣声<|im_end|>
```

`Template._simplify_context_list()` 先通过 `_split_special_tokens()` 把 `<image>`、`<video>` 和 `<audio>` 从相邻文本中分离，再由 `_pre_tokenize()` 逐个调用模板的 `replace_tag()`：

```python
def _simplify_context_list(self, context_list, loss_scale_list, inputs):
    context_list, loss_scale_list = self._split_special_tokens(
        context_list, loss_scale_list)
    context_list, loss_scale_list = self._pre_tokenize(
        context_list, loss_scale_list, inputs)
    ...
```

Qwen3-Omni 继承 `swift/template/templates/qwen.py` 中的 `Qwen2_5OmniTemplate.replace_tag()`，并以 `version='omni_v3'` 选择相应分支：

```python
def replace_tag(self, media_type, index, inputs):
    from qwen_omni_utils import fetch_image, fetch_video

    if media_type == 'image':
        inputs.images[index] = fetch_image(
            {'image': inputs.images[index], **inputs.chat_template_kwargs},
            image_patch_size=self.processor.image_processor.patch_size,
        )
        return ['<|vision_start|><|image_pad|><|vision_end|>']

    elif media_type == 'audio':
        if self.mode != 'vllm':
            inputs.audios[index] = load_audio(
                inputs.audios[index], sampling_rate)
        return ['<|audio_start|><|audio_pad|><|audio_end|>']

    elif media_type == 'video':
        video, sample_fps = fetch_video(
            {'video': inputs.videos[index], **inputs.chat_template_kwargs},
            return_video_sample_fps=True,
            image_patch_size=self.processor.image_processor.patch_size,
        )
        inputs.videos[index] = video
        inputs.mm_processor_kwargs.setdefault('fps', []).append(sample_fps)
        return ['<|vision_start|><|video_pad|><|vision_end|>']
```

`replace_tag()` 同时完成两项变换：**加载媒体数据**，并将 **标准占位符替换为 Omni 专用标记**。示例中的 user 消息由此变为：

```text
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>图片中是什么，
<|vision_start|><|video_pad|><|vision_end|>视频中是什么，
<|audio_start|><|audio_pad|><|audio_end|>音频中是什么<|im_end|>
```

`_encode_context_list()` 对这组 context 逐段执行 tokenizer。此时**每种媒体仍只有一个 pad token**，尚未与真实媒体特征长度对齐

### 4. 媒体预处理与占位 token 扩展

`Qwen2_5OmniTemplate._encode()` 先取得包含单个媒体 pad token 的 `input_ids`，再调用 processor 生成媒体张量。`text=''` 使 processor 只处理媒体；临时产生的文本 `input_ids` 与 `attention_mask` 会被删除

```python
encoded = Template._encode(self, inputs)
inputs.audios = self._trim_omni_v3_audios(inputs.audios)

media_inputs = self.processor(
    text='',
    audio=inputs.audios or None,
    images=inputs.images or None,
    videos=inputs.videos or None,
    do_resize=False,
    return_tensors='pt',
)
media_inputs.pop('input_ids')
media_inputs.pop('attention_mask')

input_ids = encoded['input_ids']
labels = encoded['labels']
loss_scale = encoded.get('loss_scale')
config = self.config.thinker_config
```

image placeholder 的长度来自 `image_grid_thw`：

```python
image_token_id = [config.image_token_id]
image_idx_list = findall(input_ids, image_token_id)
image_grid_thw = media_inputs['image_grid_thw']
merge_size = self.processor.image_processor.merge_size

def _get_new_image_tokens(i):
    image_token_len = (
        image_grid_thw[i].prod()
        // (merge_size**2)
    )
    return image_token_id * image_token_len

input_ids, labels, loss_scale = self._extend_tokens(
    input_ids,
    labels,
    loss_scale,
    image_idx_list,
    _get_new_image_tokens,
)
```

video placeholder 的长度来自 `video_grid_thw`：

```python
video_token_id = [config.video_token_id]
video_idx_list = findall(input_ids, video_token_id)
video_grid_thw = media_inputs['video_grid_thw']
merge_size = self.processor.image_processor.merge_size

def _get_new_video_tokens(i):
    video_token_len = (
        video_grid_thw[i].prod()
        // (merge_size**2)
    )
    return video_token_id * video_token_len

input_ids, labels, loss_scale = self._extend_tokens(
    input_ids,
    labels,
    loss_scale,
    video_idx_list,
    _get_new_video_tokens,
)
```

audio placeholder 的长度来自 `feature_attention_mask`。`_get_feat_extract_output_lengths()` 将有效声学帧数转换为 audio encoder 的输出长度：

```python
def _get_feat_extract_output_lengths(self, input_lengths):
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    return (
        ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1
        + (input_lengths // 100) * 13
    )


audio_token_id = [config.audio_token_id]
audio_idx_list = findall(input_ids, audio_token_id)
feature_attention_mask = media_inputs['feature_attention_mask']
audio_feature_lengths = torch.sum(
    feature_attention_mask, dim=1)
audio_lengths = self._get_feat_extract_output_lengths(
    audio_feature_lengths)

def _get_new_audio_tokens(i):
    return audio_token_id * audio_lengths[i]

input_ids, labels, loss_scale = self._extend_tokens(
    input_ids,
    labels,
    loss_scale,
    audio_idx_list,
    _get_new_audio_tokens,
)
```

三类 placeholder 扩展完成后，媒体张量与更新后的 token 序列共同写入编码结果：

```python
encoded['input_ids'] = input_ids
encoded['labels'] = labels
encoded['loss_scale'] = loss_scale
encoded.update(media_inputs)
return encoded
```

processor 已经给出视觉网格和音频有效帧数时，三类 placeholder 的实际数量可以直接由同一段代码得到：

```python
merge_size = 2

image_grid_thw = torch.tensor([[1, 32, 48]])
video_grid_thw = torch.tensor([[8, 24, 40]])
feature_attention_mask = torch.ones(
    (1, 300), dtype=torch.long)

image_placeholder_len = int(
    image_grid_thw[0].prod() // (merge_size**2)
)
video_placeholder_len = int(
    video_grid_thw[0].prod() // (merge_size**2)
)
audio_placeholder_len = int(
    self._get_feat_extract_output_lengths(
        feature_attention_mask.sum(dim=1)
    )[0]
)

assert image_placeholder_len == 384
assert video_placeholder_len == 1920
assert audio_placeholder_len == 39
```

扩展前，每份媒体在 ChatML 中只有一个 pad token。**`_extend_tokens()` 只重复中间的媒体 token，两侧边界 token 保持不变**：

```text
<|vision_start|>
<|image_pad|> * image_placeholder_len
<|vision_end|>

<|vision_start|>
<|video_pad|> * video_placeholder_len
<|vision_end|>

<|audio_start|>
<|audio_pad|> * audio_placeholder_len
<|audio_end|>
```

代入该示例的实际长度：

```text
<|image_pad|> * 384
<|video_pad|> * 1920
<|audio_pad|> * 39
```

其中 `*` 表示重复相同 token，不是写入 ChatML 的文本字符。token id 层的扩展前结构为：

```python
vision_start_id = self._tokenize('<|vision_start|>')[0]
vision_end_id = self._tokenize('<|vision_end|>')[0]
audio_start_id = self._tokenize('<|audio_start|>')[0]
audio_end_id = self._tokenize('<|audio_end|>')[0]

image_segment = [
    vision_start_id,
    config.image_token_id,
    vision_end_id,
]
video_segment = [
    vision_start_id,
    config.video_token_id,
    vision_end_id,
]
audio_segment = [
    audio_start_id,
    config.audio_token_id,
    audio_end_id,
]
```

`_extend_tokens()` 只替换中间的媒体 token，起止标记不变：

```python
image_segment = (
    [vision_start_id]
    + [config.image_token_id] * image_placeholder_len
    + [vision_end_id]
)
video_segment = (
    [vision_start_id]
    + [config.video_token_id] * video_placeholder_len
    + [vision_end_id]
)
audio_segment = (
    [audio_start_id]
    + [config.audio_token_id] * audio_placeholder_len
    + [audio_end_id]
)

assert len(image_segment) == 386
assert len(video_segment) == 1922
assert len(audio_segment) == 41
```

> [!NOTE]
> 图像区段的 386 个 token 由 1 个 `vision_start`、384 个 `image_pad` 和 1 个 `vision_end` 构成。**placeholder 数只计算中间的媒体 token，不包含两侧边界 token**；video 和 audio 区段遵循相同规则

`feature_attention_mask` 决定 audio placeholder 的重复长度；`image_grid_thw` 与 `video_grid_thw` 决定 image/video placeholder 的重复长度。三类长度在同一个 `_encode()` 中计算，并统一交给 `_extend_tokens()` 改写 `input_ids`

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
```

processor 输出的 `pixel_values`、`pixel_values_videos`、`input_features` 保存在 `media_inputs`；`image_grid_thw`、`video_grid_thw` 与 `feature_attention_mask` 同时保留，供 collator、MRoPE 和模型 forward 使用。媒体 token 的 label 为 `-100`，不参与文本交叉熵损失

`use_audio_in_video=True` 时，`_get_new_tokens_use_audio_in_video()` 按 `video_second_per_grid` 将 video token 与视频音轨产生的 audio token 交错排列；普通 image、video 和独立 audio 仍走相同的 `_extend_tokens()` 路径

### 5. Batch、`attention_mask` 与多模态 RoPE

单样本编码完成后，`Template._data_collator()` 将不同长度的 `input_ids` 补齐到 batch 内最大长度，并据有效序列长度构造二维 `attention_mask`：

```python
seq_lens = [len(seq) for seq in res['input_ids']]
res['attention_mask'] = [
    torch.ones(seq_len, dtype=torch.int64)
    for seq_len in seq_lens
]

res['input_ids'] = self._pad_sequence(
    res['input_ids'], self.tokenizer.pad_token_id)
res['attention_mask'] = self._pad_sequence(
    res['attention_mask'], 0)
```

扩展后的 image、video 和 audio token 都属于有效序列，其 `attention_mask` 为 1；只有 batch padding 位置为 0。**`attention_mask` 描述语言序列，`feature_attention_mask` 描述有效声学帧**，二者不能混用

视觉数据由基础 collator 沿媒体维拼接：

```python
pixel_values = [
    b['pixel_values'] for b in batch
    if b.get('pixel_values') is not None
]
res['pixel_values'] = torch.concat(pixel_values)

pixel_values_videos = [
    b['pixel_values_videos'] for b in batch
    if b.get('pixel_values_videos') is not None
]
res['pixel_values_videos'] = torch.concat(pixel_values_videos)

res['image_grid_thw'] = self.concat_tensor(
    batch, 'image_grid_thw', 0)
res['video_grid_thw'] = self.concat_tensor(
    batch, 'video_grid_thw', 0)
```

音频特征先按 batch 内最大声学长度补齐，再沿音频样本维拼接：

```python
max_length = max(x.shape[-1] for x in input_features)
for i, input_feature in enumerate(input_features):
    mask = feature_attention_mask[i]
    input_features[i] = F.pad(
        input_feature,
        (0, max_length - input_feature.shape[-1]),
    )
    feature_attention_mask[i] = F.pad(
        mask,
        (0, max_length - mask.shape[-1]),
    )

res['input_features'] = torch.concat(input_features)
res['feature_attention_mask'] = torch.concat(
    feature_attention_mask)
```

典型 batch 的输入结构可表示为：

| 字段 | 形状 | 坐标含义 |
| --- | --- | --- |
| `input_ids` | `[B, L]` | ChatML 文本 token 与扩展后的媒体 token |
| `attention_mask` | `[B, L]` | 语言序列有效位置 |
| `pixel_values` | `[ΣP_image, patch_dim]` | 展平后的图像 patch |
| `image_grid_thw` | `[N_image, 3]` | 每幅图像的 `(T,H,W)` patch 网格 |
| `pixel_values_videos` | `[ΣP_video, patch_dim]` | 展平后的视频 patch |
| `video_grid_thw` | `[N_video, 3]` | 每段视频的 `(T,H,W)` patch 网格 |
| `input_features` | `[N_audio, 128, F_max]` | 补齐后的 mel 特征 |
| `feature_attention_mask` | `[N_audio, F_max]` | 有效声学特征帧 |

Qwen3-Omni 的 `get_rope_index()` 同时使用 token 序列、视觉网格、音频长度与视频时间间隔构造多模态位置。ms-swift 的模板调用为：

```python
feature_attention_mask = inputs.get('feature_attention_mask')
audio_feature_lengths = torch.sum(
    feature_attention_mask, dim=1)

position_ids, _ = self._get_get_rope_index()(
    input_ids,
    inputs.get('image_grid_thw'),
    inputs.get('video_grid_thw'),
    attention_mask,
    self.use_audio_in_video,
    audio_feature_lengths,
    video_second_per_grid,
)
```

模型返回的 `position_ids` 形状为 `[3, B, L]`，三个轴分别表示 temporal、height 和 width。普通文本与音频在三个轴上使用相同的一维递增位置；图像和视频 token 使用各自网格的三维坐标。视频 temporal 坐标还乘以 `video_second_per_grid × position_id_per_seconds`，从而保留抽帧后每个时间网格的实际间隔。ms-swift 在 collator 内临时增加线性的 text position 轴，形成 `[4, B, L]`：

```python
text_position_ids = torch.arange(
    seq_len, device=position_ids.device
).expand(1, *position_ids.shape[1:])

position_ids = torch.concat(
    [text_position_ids, position_ids], dim=0)
```

训练输入随后将三维多模态位置保存在 `position_ids`，并把第一维线性位置单独保存为 `text_position_ids`，供 causal mask、packing 和 sequence parallel 逻辑使用

### 6. `post_encode` 与 embedding scatter

`Qwen2_5OmniTemplate._post_encode()` 会在模板侧提前计算多模态 embedding。Qwen3-Omni 明确覆盖了该实现：

```python
class Qwen3OmniTemplate(Qwen2_5OmniTemplate):
    version = 'omni_v3'
    norm_bbox = 'norm1000'
    placeholder_tokens = [
        '<|image_pad|>',
        '<|audio_pad|>',
        '<|video_pad|>',
    ]

    def _post_encode(self, model, inputs):
        return inputs
```

> [!IMPORTANT]
> Qwen3-Omni 的 **`post_encode` 是直通操作**。`input_ids`、`pixel_values`、`pixel_values_videos`、`input_features` 及其长度元数据保持不变；**embedding 计算与替换发生在 `Qwen3OmniMoeThinkerForConditionalGeneration.forward()` 内部**

语言序列首先经过词嵌入层。此时媒体 pad token 与普通文本 token 一样，只对应词表中的占位向量：

```python
if inputs_embeds is None:
    inputs_embeds = self.get_input_embeddings()(input_ids)
```

设语言模型隐藏维度为 `D`，则：

```text
input_ids     : [B, L]
inputs_embeds : [B, L, D]
```

音频塔将 `input_features` 编码为 `[N_audio, D]`。模型在 `input_ids` 中定位全部 `audio_token_id`，将布尔 mask 扩展到 embedding 维，再按序写入音频特征：

```python
audio_features = self.get_audio_features(
    input_features,
    feature_attention_mask,
    audio_feature_lengths,
    return_dict=True,
).last_hidden_state

_, _, audio_mask = self.get_placeholder_mask(
    input_ids,
    inputs_embeds=inputs_embeds,
)

inputs_embeds = inputs_embeds.masked_scatter(
    audio_mask,
    audio_features,
)
```

图像与视频共享视觉塔。视觉编码器的 `pooler_output` 已完成 `merge_size²` 空间下采样，其第一维分别等于 `N_image` 与 `N_video`：

```python
image_outputs = self.get_image_features(
    pixel_values,
    image_grid_thw,
    return_dict=True,
)
image_embeds = image_outputs.pooler_output
image_mask, _, _ = self.get_placeholder_mask(
    input_ids,
    inputs_embeds=inputs_embeds,
    image_features=image_embeds,
)
inputs_embeds = inputs_embeds.masked_scatter(
    image_mask,
    image_embeds,
)

video_outputs = self.get_video_features(
    pixel_values_videos,
    video_grid_thw,
    return_dict=True,
)
video_embeds = video_outputs.pooler_output
_, video_mask, _ = self.get_placeholder_mask(
    input_ids,
    inputs_embeds=inputs_embeds,
    video_features=video_embeds,
)
inputs_embeds = inputs_embeds.masked_scatter(
    video_mask,
    video_embeds,
)
```

`get_placeholder_mask()` 直接由 token id 构造三类位置掩码：

```python
special_image_mask = input_ids == self.config.image_token_id
special_video_mask = input_ids == self.config.video_token_id
special_audio_mask = input_ids == self.config.audio_token_id

n_image_tokens = special_image_mask.sum()
n_video_tokens = special_video_mask.sum()

special_image_mask = special_image_mask.unsqueeze(-1)
special_video_mask = special_video_mask.unsqueeze(-1)
special_audio_mask = special_audio_mask.unsqueeze(-1)
```

多模态实现中常说的 **`scatter_mask`**，在当前 Qwen3-Omni 源码中对应 `special_image_mask`、`special_video_mask`、`special_audio_mask` 及其返回后的 `image_mask`、`video_mask`、`audio_mask`。这些 mask 共同定义 scatter 的目标位置

以图像为例，`image_mask` 的逻辑形状为 `[B,L,1]`，在 `masked_scatter()` 中广播为 `[B,L,D]`。若存在 `N_image` 个图像 token，则 mask 中参与替换的标量数为 `N_image × D`；`image_embeds` 的元素数同样为 `N_image × D`。Transformers 在 scatter 前显式检查这一不变量：

```python
torch_compilable_check(
    n_image_tokens * inputs_embeds.shape[-1]
    == image_features.numel(),
    "Image features and image tokens do not match",
)
```

video 使用相同检查。audio 的一致性由 ms-swift 与模型共用的卷积输出长度公式保证。`masked_scatter()` 按 mask 中 True 位置的线性顺序消费 source；processor 输出媒体特征的顺序与 `messages` 中占位符的顺序一致，因而第一个媒体占位区间接收第一份媒体的 embedding，后续媒体依次对应

```text
scatter 前
[text embedding]
[image_pad embedding × N_image]
[text embedding]
[video_pad embedding × N_video]
[text embedding]
[audio_pad embedding × N_audio]

scatter 后
[text embedding]
[image encoder output × N_image]
[text embedding]
[video encoder output × N_video]
[text embedding]
[audio encoder output × N_audio]
```

视觉编码器还返回 `deepstack_features`。`image_mask | video_mask` 构成 `visual_pos_masks`，文本解码器的前若干层在相同视觉位置继续叠加不同视觉层级的特征：

```python
visual_pos_masks = video_mask | image_mask

if (
    deepstack_visual_embeds is not None
    and layer_idx in range(len(deepstack_visual_embeds))
):
    hidden_states = self._deepstack_process(
        hidden_states,
        visual_pos_masks,
        deepstack_visual_embeds[layer_idx],
    )
```

ms-swift 在 DeepSpeed 路径中通过 `_compat_qwen3_omni_mixed_data()` 接管部分 forward 逻辑，但保持相同语义：视觉与音频特征仍由各自 token mask 写入 `inputs_embeds`，DeepStack 特征仍由 `visual_pos_masks` 定位

### 7. 输入链路的张量闭合

```text
messages.content 中的 <image>/<video>/<audio>
        ↓ StdTemplateInputs.from_dict
messages + images/videos/audios
        ↓ Template._swift_encode
包含标准媒体标记的 ChatML context list
        ↓ _split_special_tokens + replace_tag
<|image_pad|> / <|video_pad|> / <|audio_pad|>
        ↓ tokenizer
每种媒体一个初始 placeholder token
        ↓ Qwen3-Omni processor
pixel_values / pixel_values_videos / input_features
image_grid_thw / video_grid_thw / feature_attention_mask
        ↓ _get_feat_extract_output_lengths + _extend_tokens
与下采样特征等长的多模态 input_ids
        ↓ data_collator
input_ids + attention_mask + position_ids + media tensors
        ↓ Qwen3OmniMoeThinker.forward
text embedding + image/video/audio encoder output
        ↓ get_placeholder_mask + masked_scatter
统一的 inputs_embeds[B, L, D]
```

> [!IMPORTANT]
> 三种模态共享同一项对齐条件：**语言序列中的媒体 token 数必须等于对应 encoder 输出的 embedding 数**。`replace_tag()` 确定媒体边界，`image_grid_thw`、`video_grid_thw` 与音频长度函数确定槽位数量，`_extend_tokens()` 在 `input_ids` 中建立槽位，`get_placeholder_mask()` 与 `masked_scatter()` 完成最终的 embedding 写入
