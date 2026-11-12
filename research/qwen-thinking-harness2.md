# Qwen thinking 与 Harness² 一手资料核查

核查日期：2026-11-12。范围仅为官方模型资料、官方推理框架文档/源码、论文原文和作者仓库。本次只编写研究记录，没有修改运行配置、启动模型或运行实验。

## 结论

1. 目录名 `Qwen3.8-27B` 不能证明 checkpoint 的官方型号。如果本地配置显示 `Qwen3_5ForConditionalGeneration` / `qwen3_5`，应按 **Qwen3.5 架构**核对模板、tokenizer、推理实现；但这仍不能证明权重与官方 Qwen3.5-27B 完全一致。官方 Qwen3.5-27B 的配置确实使用上述架构标识。[Q1][Q2]
2. 官方 Qwen3.5-27B **默认 thinking**。`chat_template_kwargs.enable_thinking` 控制模板的生成前缀；`--reasoning-parser qwen3` 则把输出里的 thinking 与普通回答分离。两者职责不同。工具调用另用 `--enable-auto-tool-choice --tool-call-parser qwen3_coder`。[Q1][Q3][V1]
3. 官方原生上下文为 **262,144 tokens**，模型卡建议复杂任务尽量保留至少 **128K** 上下文以保存 thinking 能力。这个建议不是“低于128K就不能 thinking”的硬门槛，也不保证当前4卡配置一定能承担128K。输入、思考、答案、工具参数共同消耗序列/生成预算；不能只打开开关而忽略输出空间。[Q1][V3]
4. **Harness² / Harness^2** 可明确对应作者论文 *Harness²: Recursive Agent Harnessing for an Open World*（arXiv:2610.14224，v1提交于2026-10-09）及 `google-research/harness2`。它是固定模型权重、双层递归优化 agent 周边系统的方法，**不是特指上下文摘要/超长截断算法**。[H1][H2]
5. 我们现有“单个失败轨迹 → 一条过程提示 → 两任务重跑”的先导，不能称作 Harness² 复现。Harness² 还有当前任务的候选提出、试跑对比、组合，以及跨任务保存已采用 harness 和有证据经验的改进库。[H1][H2][H3]

## 1. Qwen3.5 架构、官方 thinking 模板与 vLLM

### 1.1 先确定什么，什么仍未知

官方 Qwen3.5-27B 是带视觉编码器的27B模型，文本配置包含64层、`hidden_size=5120`、交替线性/全注意力、`max_position_embeddings=262144`，架构标识为 `Qwen3_5ForConditionalGeneration`。[Q1][Q2] 不能因系列介绍涉及 MoE 就把这个27B检查点自动描述为 MoE；具体以该检查点的配置为准。

对用户目录的可靠身份核查应包括：本地配置、tokenizer、chat template 与 generation config 的实际内容，以及可追溯下载revision或文件摘要。只有目录名、API served-model-name 或架构类名都不足以证明权重来源。本次未远程读取或重新认证这些权重。以下配置结论对“与官方 Qwen3.5-27B 模板兼容的检查点”成立，而不是对目录中的任意改名权重无条件成立。[Q1][Q2][Q3]

### 1.2 模板开关的实际行为

官方 Jinja 模板在 `add_generation_prompt` 时输出：

```text
# enable_thinking=false
<|im_start|>assistant\n<think>\n\n</think>\n\n

# enable_thinking=true，或未定义
<|im_start|>assistant\n<think>\n
```

因此关闭 thinking 是预先给出空 thinking 段，开启则让模型继续生成该段。模型卡说明 thinking 内容由 `<think> ... </think>` 标识。[Q1][Q3] 不应把“没有 reasoning parser”当成“没有生成 thinking”：parser 负责抽取，模板负责模式前缀。[V1]

vLLM/SGLang 的 OpenAI兼容请求示意如下，仅作资料说明，**本次没有执行**：

```python
extra_body={"chat_template_kwargs": {"enable_thinking": True}}
# 关闭时改为 False
```

官方模型卡还明确区分阿里云 Model Studio：其 API 使用顶层 `enable_thinking`，不是上述 vLLM 的嵌套写法。不能混用服务商接口。[Q1]

### 1.3 vLLM 服务参数各自负责什么

官方 Qwen3.5-27B 模型卡的工具调用示例为：

```bash
# 官方例子，非本机容量承诺，也未执行
vllm serve Qwen/Qwen3.5-27B \
  --tensor-parallel-size 8 --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

- `--reasoning-parser qwen3`：解析 reasoning 与回答内容；不是增加推理能力或上下文的开关。[V1]
- `--enable-auto-tool-choice --tool-call-parser qwen3_coder`：支持工具选择与工具调用协议解析；它不会替 agent 实际执行工具。[Q1][V3]
- `--default-chat-template-kwargs '{"enable_thinking": false}'`：服务端提供默认的模板参数。若客户端也传模式参数，不能只检查启动命令就断言一次具体请求的有效模式；应检查实际请求和模板。[V3]
- `--language-model-only`：官方提供的纯文本部署选项，跳过视觉编码器；此时不能把原生看图能力视作当前 agent 已具备的工具能力。[Q1][V3]

Qwen模型卡发布时要求新版本/main/nightly推理框架；vLLM资料持续更新，因此应把实际vLLM版本和模板revision一起固定。当前研究以 **vLLM v0.25.1** 的源码/文档核对接口，不能把其他版本的字段习惯直接照搬。[Q1][V1][V2]

### 1.4 工具循环中的 reasoning 历史：不能简单全部丢弃

Qwen模型卡建议多轮历史保留最终回答、移除过往thinking；但模板源码有一个重要细节：它寻找最近一次真实user请求（排除伪装成user的 `<tool_response>`），较早user轮的assistant thinking不再渲染，而**当前user请求之后的多步工具链**会继续渲染 `reasoning_content`。[Q1][Q3]

因此推荐理解是：让匹配版本的官方模板区分“旧对话轮”和“当前工具链”，而不是每次工具返回后手动抹掉所有thinking。反过来，也不应无限累计所有旧轮thinking。[Q3]

vLLM v0.25.1 官方文档注明响应字段已由 `reasoning_content` 改名为 **`reasoning`**。该版本 `chat_utils.py` 从输入assistant消息读取 `reasoning`，随后同时设置 `reasoning` 与 `reasoning_content` 供模板兼容使用。[V1][V2] 这意味着后续适配器必须核对真实响应、SDK/LiteLLM字段映射和回传消息，不能只读取一个旧字段而静默丢失当前工具链的thinking。这是源码审查结论，本次未运行接口测试。

### 1.5 上下文、生成预算与采样注意事项

官方模型卡给出的原生上下文为262,144，建议至少128K以保持复杂任务的thinking能力；超过原生长度另有YaRN/RoPE scaling配置示例，上限示例为1,010,000。这些是模型/推理配置能力，不是自动摘要、自动换会话或无限上下文。[Q1][V3]

开启thinking后，思考也占生成token；若生成上限过小，可能在形成有效工具调用或最终答案前就截断。官方thinking请求示例使用 `max_tokens=81920`，non-thinking示例用32768。它们是官方示例预算，不能直接当成当前每个工具步都必须配置的值，也不能在没测容量时直接复制。[Q1]

官方thinking采样建议区分任务：[Q1]

- 通用任务：`temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0`。
- 精确编码任务（例如WebDev）：`temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0`。

这些推荐不等于已证明最适合本项目。当前先导的thinking关闭、24K上下文、每步3072生成tokens、temperature0是一套局部执行条件；它的负结果不能推广成官方thinking设置下的能力上限。**也没有一手证据证明仅开启thinking就必然提高本项目成绩。** 若将来改变模式或预算，应当作为新的完整对照条件，保存旧结果。

## 2. Harness² 到底指什么

### 2.1 第一方出处与命名

作者论文标题为 *Harness²: Recursive Agent Harnessing for an Open World*，Ruiyao Xu等，arXiv:2610.14224v1。官方作者仓库位于Google Research组织，README链接该论文与项目页，并声明不是Google官方支持产品。研究记录固定代码revision为 `6868817518876de8fd170c6a8354daca273b6191`（2026-11-02）；论文固定v1。[H1][H2]

README对harness的定义包括模型周围的prompts、tools、memory、skills、orchestration等；harnessing就是优化这个系统。模型权重保持不变。可编辑表面有八类：system prompt、guardrails、memory、tool guidance、subagents、skills、scripts、plugins。[H2]

### 2.2 两层递归

**任务层：Propose–Probe–Compose。** 对当前任务提出harness修改，实际试跑候选，通过执行行为与产物的对比来组合改进harness；改进过程不依赖隐藏真值verifier。可采用并行候选扩展，也可串行多步改进。[H1][H2][H3]

**Harnessing层：改进过程本身积累经验。** 在采用harness后反思，把采用的完整harness及有执行证据支持的lesson写入同域经验库，供后续任务改进使用。官方实现中同域任务顺序执行、只读取早先条目，不同域可并行；运行库按benchmark/executor/mode/budget/seed隔离。[H2][H3][H5]

这里“无ground-truth反馈”约束的是在线改进阶段，不表示论文没有离线评分。官方命令把refinement、judging、reporting分开；基线评分也有独立选项。[H3]

### 2.3 与上下文管理的真实联系

官方 `core/context.py` 的做法是把证据放到磁盘树里：当前任务、完整轨迹、产物、当前harness、库索引、候选池；给改进agent的prompt是短MISSION和文件指针，让它按需读取。源码注明不为塞进prompt而截断完整轨迹，但个别observation仍可能有明确renderer截断标记。**“证据文件完整”不等于“每次模型上下文无限或无截断”。**[H4]

官方 `core/library.py` 采用同域库，存完整采用harness与经验记录，索引包含检索线索；经验至少包含 `cue`、`lesson`、`evidence`。它强调携带完整harness树而不是只有摘要。改进agent通过索引浏览相关条目。[H5]

所以可借鉴的上下文设计是“外部证据/经验文件 + 索引 + 按需读取 + 有证据的反思”，不能把Harness²概括成“超长时自动压缩历史”。底层执行器的上下文维护仍是另一层机制。

### 2.4 对当前项目的含义与边界

现有先导只生成了一条过程经验并重放；它没有完整的候选编辑池、行为对比组合、可持续更新的改进harness，因此结果不构成对Harness²的复现或反证。[H1][H2]

官方报告的收益来自其论文中的模型/执行器/预算/任务设置，不能直接预测Qwen27B加mini-swe-agent的收益。更合适的借鉴是将改动绑定具体执行证据、比较多个行为结果、保存采用理由、通过同域索引限制经验适用范围，并在完整成本预算下做独立评估。[H1][H3][H4][H5]

官方实验指南另外承认公开benchmark的rubric可被联网executor下载，默认release不限制网络，并建议setup后在网络层限制相关源站。这说明引用它的方法并不自动保证评测隔离；本项目已有隔离策略仍需独立保留。[H3]

## 一手来源索引

- **[Q1]** Qwen官方Qwen3.5-27B模型卡，核查时revision `fc05daec18b0a78c049392ed2e771dde82bdf654`：https://huggingface.co/Qwen/Qwen3.5-27B/blob/fc05daec18b0a78c049392ed2e771dde82bdf654/README.md
- **[Q2]** 同revision官方模型配置：https://huggingface.co/Qwen/Qwen3.5-27B/blob/fc05daec18b0a78c049392ed2e771dde82bdf654/config.json
- **[Q3]** 同revision官方chat template：https://huggingface.co/Qwen/Qwen3.5-27B/blob/fc05daec18b0a78c049392ed2e771dde82bdf654/chat_template.jinja
- **[V1]** vLLM v0.25.1官方Reasoning Outputs文档源码：https://github.com/vllm-project/vllm/blob/v0.25.1/docs/features/reasoning_outputs.md
- **[V2]** vLLM v0.25.1官方消息转换源码（`_parse_chat_message_content`）：https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/entrypoints/chat_utils.py
- **[V3]** vLLM官方Qwen3.5/3.6部署recipe（动态文档；大模型硬件示例不能直接套用27B）：https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html
- **[H1]** Harness²论文原文v1：https://arxiv.org/abs/2610.14224v1 ，全文：https://arxiv.org/html/2610.14224v1
- **[H2]** 作者官方仓库README，固定revision：https://github.com/google-research/harness2/blob/6868817518876de8fd170c6a8354daca273b6191/README.md ；项目页：https://harness2.github.io
- **[H3]** 官方实验指南：https://github.com/google-research/harness2/blob/6868817518876de8fd170c6a8354daca273b6191/docs/EXPERIMENTS.md
- **[H4]** 官方证据上下文树实现：https://github.com/google-research/harness2/blob/6868817518876de8fd170c6a8354daca273b6191/harness2/core/context.py
- **[H5]** 官方同域经验库实现：https://github.com/google-research/harness2/blob/6868817518876de8fd170c6a8354daca273b6191/harness2/core/library.py
