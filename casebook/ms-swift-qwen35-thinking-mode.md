# ms-swift Qwen3.5 推理模式

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

## 1. 从 `TemplateMeta` 到实际 `response prefix`

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

`is_thinking=True` 不表示每次推理都必须思考。Qwen3.5 同时注册非空的 `non_thinking_prefix`，因此它是 hybrid-thinking，而不是只能进入 thinking 的纯思考模板。模板实例化时，`swift/template/base.py` 才把参数层的 `enable_thinking=None` 解析成实际布尔值：

```python
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

## 2. 推理路径：前缀如何进入 ChatML

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

`Qwen3_5Template` 先规范消息内容，真正拼接 ChatML 的公共控制流位于基础类 `Template._swift_encode()`；thinking/non-thinking 的前缀也在这里进入 prompt

推理请求的最后一轮没有现成的 assistant 回答。在 `_swift_encode()` 中，只需关注最后一轮的两个分支：

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

### 返回语义：为什么 ms-swift 会显示空 `<think></think>`

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

**ms-swift 的 vLLM backend**

`swift/infer_engine/vllm_engine.py` 不直接返回原生 vLLM 的 `output.text`，而是读取 `output.token_ids`，再次调用 Swift 模板解码：

```python
for output in result.outputs:
    output.token_ids = list(output.token_ids)
    response = self.template.decode_generate_ids(
        output.token_ids,
        template_inputs=inputs['template_inputs'],
    )
```

所以，原生 vLLM 返回的是 completion，而 ms-swift 的 vLLM backend 返回的是经过模板重建的 assistant response。底层采样没有改变，变化的只是返回文本是否重新包含 `response_prefix`

## 3. SFT 路径：标签规范化与 Loss Mask

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

上图只展开 assistant response span；模板 suffix 的监督由后续 EOS 逻辑单独处理。因此 non-thinking SFT 保留与推理一致的 response 结构，但空思考块不参与监督，答案 token 以及模板要求的结束符仍参与训练。标准 CLI 会自动追加这一策略；显式记录下面的配置，可以直接表达预期语义：

```bash
--add_non_thinking_prefix true \
--loss_scale default+ignore_empty_think
```

## 4. OPD 路径：rollout 模式与训练序列对齐

普通 SFT 使用已有标签，所以 `enable_thinking` 不选择数据类型；OPD / on-policy GKD 在训练内部先让学生模型在线生成，因此其中包含一段真正的推理路径

`swift/rlhf_trainers/rollout_mixin.py` 的 `_generate_completions()` 先预处理样本，再进入 `template.generate_context()`：

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

这就是“`enable_thinking` 只在推理阶段生效”与“OPD 训练中它会改变样本格式”并不矛盾的原因：OPD 训练内部确实执行一次学生推理rollout 完成后还存在一个 token 对齐问题。response prefix 属于 prompt，学生返回的 `response_token_ids` 通常只包含新增 completion；重新编码训练序列时，必须把生成时使用的前缀补回，否则 rollout 时的 token 上下文与训练时的 token 上下文不一致

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
```

或：

```json
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
