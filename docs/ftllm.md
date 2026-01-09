
# fastllm 核心调用链（源码级）

本文目标：把 fastllm 的“从输入字符串到流式 token 输出”的链路讲清楚，做到你能顺着源码定位到每个关键函数、知道性能/行为由哪些开关决定。

适用范围：当前仓库 fastllm-windows 的 C++ 推理栈（`basellm`/`ComputeGraph`/`Executor`/`Tokenizer`/`JinjaTemplate`）与常用入口（benchmark/webui/apiserver/pytools）。

---

## 分析：你在追的到底是什么

fastllm 的“推理一次”通常包含 5 层：

1) **入口层**：示例程序/服务端把请求（messages 或 prompt）转换成 tokens（或字符串 prompt）。
2) **模板层**：chat_template（Jinja）或老式 `MakeInput/MakeHistory` 把 messages 变 prompt。
3) **分词层**：`Tokenizer::Encode` 把 prompt 变 token ids。
4) **推理层**：`basellm::Forward/ForwardBatch` 生成 logits 并更新 KV cache；graph 模型走 `RunComputeGraph`。
5) **采样/流式层**：`LLMSampling(...)` 从 logits 采样出下一 token，并通过同步 callback 或异步队列输出。

其中最关键的“核心环”是：

$$\text{tokens} \rightarrow \text{FillLLMInputs} \rightarrow \text{Forward/ForwardBatch} \rightarrow \text{logits} \rightarrow \text{sampling} \rightarrow \text{next token}$$

---

## 伪代码：端到端（同步与异步）

### A) 同步（阻塞）生成：`basellm::Response/ResponseBatch`

```text
prompt/messages
	-> ApplyChatTemplate(...)  [可选]
	-> Tokenizer.Encode(prompt) -> inputTokens
	-> init pastKeyValues (KV cache per layer)
	-> loop until eos/limit:
			 FillLLMInputs(...): inputIds, attentionMask, positionIds
			 Forward/ForwardBatch(..., pastKeyValues, generationConfig, lastTokens)
			 if greedy: argmax else: LLMSampling(logits, ...)
			 decode(token) -> append output
			 callback(stream chunk)
```

### B) 异步（非阻塞）生成：`basellm::LaunchResponseTokens + FetchResponseTokens`

```text
handle = LaunchResponseTokens(inputTokens, generationConfig)

while true:
	if CanFetchResponse(handle):
		 tk = FetchResponseTokens(handle)
		 if tk == -1: break
		 yield tk
	else:
		 sleep(0)
```

异步模式的“核心机制”是：后台 `mainLoop` 线程批量收集多个 handle 的待推理 token（prefill / decode），调用 `Forward/ForwardBatch` 后把采样结果 push 到每个 handle 的 `resultTokenQueue`。

---

## 关键结构体与职责

### 1) `fastllm::basellm`：推理与生成的公共基类

定义见 [include/models/basellm.h](../include/models/basellm.h)。

- `Forward(...)`：**核心虚函数**，单步推理：输入 `inputIds/attentionMask/positionIds`，更新 `pastKeyValues` 并产生 logits。
- `ForwardBatch(...)`：批量推理（有的模型会走图执行或 device batch 快路径）。
- `Response/ResponseBatch(...)`：同步生成（循环 Forward + 采样 + decode）。实现见 [src/models/basellm.cpp](../src/models/basellm.cpp)。
- `LaunchResponseTokens/FetchResponseTokens/AbortResponse`：异步生成的 handle/队列协议。

### 2) `ResponseContext`：每个异步请求的状态机

同样在 [include/models/basellm.h](../include/models/basellm.h)。关键字段：

- `pastKeyValues`：每层 KV cache（每层一对 `(k,v)`），并通过 `Data::SetKVCache()` 标记。
- `currentTokens/allTokens`：当前待 prefill/decode 的 tokens 与累计 tokens。
- `resultTokenQueue`：后台推理线程产出的 token 队列（`FetchResponseTokens` 读取）。
- `generationConfig`：采样/停止条件/输出长度等。
- `isEnding/isAbort/error`：生命周期控制。

### 3) `Tokenizer`：Encode/Decode 与 chat_template 挂载

实现见 [src/tokenizer.cpp](../src/tokenizer.cpp)。

- `Tokenizer::Encode(const std::string&) -> Data`：输出 `FLOAT32` 的 token id 序列（形状 `{1, n}`）。
- `Tokenizer::Decode(const Data&) -> std::string`：token ids 回文本。
- `Tokenizer::SetTokenizerConfig(json)`：会把 `chat_template` 写入 `Tokenizer::chatTemplate`（若 config 提供）。

### 4) `JinjaTemplate`：chat_template 的执行器

实现见 [src/template.cpp](../src/template.cpp)。

- `basellm::ApplyChatTemplate(...)` 会构建 `JinjaVar`，再 `JinjaTemplate(chatTemplate).Apply(var)` 生成 prompt。

### 5) `ComputeGraph + RunComputeGraph`：图模型执行

实现/入口见 [src/graph.cpp](../src/graph.cpp) 与 [src/models/graphllm.cpp](../src/models/graphllm.cpp)。

- `OptimizeComputeGraph(graph, weight)`：图级优化（例如连续 Linear 合并 + Split 回填、Linear+Silu+MulTo 合并 Swiglu）。
- `RunComputeGraph(...)`：把图 ops 逐个翻译成 `Executor::Run(opType, datas, params)`。

### 6) `Executor`：设备选择与 op profiler

实现见 [src/executor.cpp](../src/executor.cpp)。

- `Executor::Run(...)`：
	- 遍历可用 device（cuda/multicuda/tfacc/numa/cpu），找到 `CanRun` 的设备。
	- 将输入/权重 Data 迁移到目标 device（`Data::ToDevice`）。
	- 调用 device 的 `Reshape` 与 `Run`。
	- 记录 `profiler[opType] += spend`。
- `ClearProfiler/PrintProfiler`：示例程序中已用 CLI 开关包裹一次推理。

---

## 入口→模板→分词：prompt 是怎么来的

### 1) messages → prompt（优先 chat_template）

实现见 [src/models/basellm.cpp](../src/models/basellm.cpp) 的 `basellm::ApplyChatTemplate`。

- 若 `weight.tokenizer.chatTemplate` **为空**：走老式拼接协议：
	- 遍历 messages：`assistant` 消息会调用 `MakeHistory(history, round, user, assistant)`
	- 最后调用 `MakeInput(history, round, last_user)` 生成 prompt
- 若 chat_template **存在**：
	- `ChatMessagesToJinjaVar(messages)` 把消息变成 `JinjaVar`（含 `messages/add_generation_prompt/tools`）
	- 将 tokenizer config 中的额外字段注入 JinjaVar（如各种 `*_token`）
	- `JinjaTemplate(chatTemplate).Apply(var)` 得到 prompt

### 2) prompt → tokens

`Tokenizer::Encode` 的行为取决于 `TokenizerType`：

- BPE/GLM：优先 sentencepiece（若编译启用并加载 `tokenizer_serialized`），否则走内部 BPE。
- QWEN：内部 BPE/合并逻辑，并支持识别/注入一些特殊 token（例如 `<|im_start|>` 等）。

分词输出是 `Data(FLOAT32, {1, n}, ids)`，后续用 `((float*)data.cpuData)[i]` 读取。

---

## 生成主循环：同步 ResponseBatch 是怎么跑的

实现见 [src/models/basellm.cpp](../src/models/basellm.cpp) 的 `basellm::ResponseBatch`（单条 `Response` 类似）。核心点：

1) `Tokenizer.Encode(prompt)` 得到每条输入的 `inputTokens[i]`。
2) 初始化 `pastKeyValues`（每层 `(k,v)` 都 `SetKVCache()`）。
3) 初始化 `LastTokensManager`（用于重复惩罚/last_n 策略）。
4) 每轮：
	 - `FillLLMInputsBatch(...)` 构造 `inputIds/attentionMask/positionIds`。
	 - 调用 `ForwardBatch`（模型实现决定是图执行还是传统实现）。
	 - 得到 `ret[i]`（下一 token）。
	 - 终止条件：`eos_token_id` / `eos_token_ids` / `stop_token_ids` / `output_token_limit`。
	 - `Tokenizer.Decode(...)` 把本轮 token 拼到 `outputs[i]`。
	 - retCb 流式回调输出 chunk。

注意：`FillLLMInputs` 会在 prefill（seqLen>1）时生成 positionIds（按 promptLen 对齐），并按 `NeedAttentionMask(qlen,klen)` 决定是否构造 attentionMask；Graph 模型里也对 mask 有优化路径（例如 `NeedAttentionMask` 在部分条件下返回 false）。

---

## 异步 mainLoop：多请求如何拼 batch、如何流式输出

实现见 [src/models/basellm.cpp](../src/models/basellm.cpp) 的 `basellm::LaunchResponseTokens`。

### 1) handle 创建与请求入队

- `handleId = responseContextDict.CreateHandle()` 创建请求。
- `ResponseContext::Init(block_cnt, dataType)` 初始化 KV cache。
- 把 `currentTokens/allTokens/generationConfig/multimodalInput` 写入 context。
- 触发 `dictCV.notify_one()` 唤醒 mainLoop。

### 2) mainLoop 批处理策略（核心）

mainLoop 会在每次循环里：

- 清理 abort 的请求：`isAbort == true` 的 handle 会被移除。
- 估算 KV cache 预算（尤其 CUDA 会根据 free mem 推一个 kvCacheLimit），计算 `maxTotalLens/maxBatch/promptLimit`。
- 对请求排序（当前实现按 `-currentTokens.size()` 排序），分两阶段收集：
	- **prefill 阶段**（`preTokens == 0`）
	- **decode 阶段**（`preTokens > 0`）

对每个被选中的请求：

- `FillLLMInputs(tokens, intParams, ...)` 生成该请求本轮的 `inputIds/attentionMask/positionIds`。
- 收集到 `ids/attentionMasks/positionIds/seqLens/pastKeyValues/generationConfigs/logits`。

然后：

- 若 `seqLens.size() > 1`：
	- `canDoBatchForward` 为真则走 `ForwardBatch(batch, inputIds, attentionMasks, positionIds, seqLens, pastKeyValues, generationConfigs, ...)`
	- 否则逐条 `Forward(...)`
- 若 `seqLens.size() == 1`：
	- 对超长 prefill 有切片策略（`first/part`），避免一次性 prefill 太长。
	- multimodal 请求会走 `ForwardMultimodal(...)`。

推理结束后，针对每个 handle 的采样结果 `curRet`：

- 触发停止条件则 `isEnding=true`；否则：
	- `currentTokens = {curRet}`
	- `resultTokenQueue.push(curRet)`（这就是流式 fetch 的来源）
	- `allTokens.push_back(curRet)`
	- `LastTokensUnit::Push(curRet)`

### 3) Fetch/Abort 协议

- `FetchResponseTokens(handle)`：
	- 队列有 token 就 pop 返回。
	- 若 `isEnding` 则删除 handle 并返回：正常结束 `-1`；prompt 过长错误 `-2`。
- `AbortResponse(handle)`：标记 `isAbort=true`，mainLoop 会清理并允许释放资源。

---

## Graph 模型：ForwardBatch 如何落到 Executor

典型路径（GraphLLM）：

1) `GraphLLMModel::ForwardBatch(...)`：构造 inputs/weights/pastKeys/pastValues/masks。
2) `RunComputeGraph(graph, deviceMap, inputs, weightDicts, outputs, pastKeys, pastValues, masks)`。
3) `RunComputeGraph`：遍历 `graph.ops`，把每个 op 翻译为若干次 `Executor::Run(opType, datas, params)`。
4) `Executor::Run`：选择设备、迁移 Data、执行 device kernel，并记 profiler。
5) `GraphLLMModel::ForwardBatch` 在拿到 logits 后：
	 - greedy：`TopK(...,1)`
	 - sampling：`LLMSampling(logits, base, generationConfig, lastTokens.units[b])`

你想做“算子级优化/定位瓶颈”，基本就从 `Executor::Run` 的 opType 维度入手（需要时可在你自己的入口代码里手动调用 `ClearProfiler/PrintProfiler`）。

---

## C++/Python 入口对照（最短路径）

### C++ 侧

- benchmark/webui/apiserver：最终都会调用到 `basellm` 的 `Response/ResponseBatch` 或异步 handle 协议（取决于入口实现）。
- 你要定位“请求参数→GenerationConfig”的映射，重点看各入口解析参数后如何填 `GenerationConfig`（例如 top_k、top_p、temperature、repeat_penalty、stop_token_ids、output_token_limit）。

### Python 侧（pytools/ftllm）

- `tools/src/pytools.cpp` 暴露了 `apply_chat_template(...)` 等桥接函数，底层仍调用 `basellm::ApplyChatTemplate`。
- Python server/CLI 属于“入口层差异”，核心推理栈仍以 C++ 实现为准。

---

## 入口参数 → GenerationConfig 映射表

这一节解决两个问题：

1) 入口（HTTP/CLI/Python）给的参数，最终落到 `GenerationConfig` 哪个字段？
2) 没传参数时，默认行为是什么（尤其是“是否采样/是否贪心”）？

### 0) GenerationConfig 字段默认值（C++）

定义见 [include/fastllm.h](../include/fastllm.h)。

| GenerationConfig 字段 | 默认值 | 作用/备注 |
| --- | --- | --- |
| `output_token_limit` | `-1` | `<=0` 代表不限制（入口通常会覆盖成一个安全上限） |
| `output_token_least` | `0` | 最少输出 token 数（配合 EOS reset） |
| `last_n` | `64` | 重复惩罚统计窗口 |
| `repeat_penalty` | `1.0` | `1.0` 代表不惩罚；>1 通常更合理 |
| `top_k` | `1` | `1` 近似贪心；>1 才会触发 top-k 采样 |
| `top_p` | `1.0` | nucleus 采样阈值（需要采样时才有意义） |
| `temperature` | `1.0` | 温度参数 |
| `output_logits` | `false` | 是否回传 logits（会增加开销/内存） |
| `add_special_tokens` | `true` | 主要对 chatglm prompt 生效 |
| `stop_token_ids` | 空 | token 级 stop（注意：不是 stop string） |

补充：`GenerationConfig::IsSimpleGreedy()` 的判定逻辑是 `repeat_penalty≈1 && top_k<=1` 才认为是“简单贪心”。

### 1) C++ apiserver：`/v1/chat/completions`（OpenAI-like）

实现见 [example/apiserver/apiserver.cpp](../example/apiserver/apiserver.cpp)。

| 请求 JSON 字段 | 代码落点 | GenerationConfig 字段 | 默认值（入口侧） | 说明 |
| --- | --- | --- | --- | --- |
| `max_tokens` | `int_value()` | `output_token_limit` | `256` | 不传时用 256（与 C++ 默认 `-1` 不同） |
| `temperature` | `number_value()` | `temperature` | 不覆盖 | 仅在传入且为 number 时覆盖 |
| `top_p` | `number_value()` | `top_p` | 不覆盖 | 仅在传入且为 number 时覆盖 |
| `top_k` | `number_value()` | `top_k` | 不覆盖 | 这里取的是 number，会隐式转 int（建议传整数） |
| `frequency_penalty` | `number_value()` | `repeat_penalty` | 不覆盖 | 语义上这是“重复惩罚”，但 OpenAI 的 `frequency_penalty` 与这里并非严格等价；建议传 >= 1.0 |
| `stream` | bool | 无 | `false` | 只影响 HTTP chunked 输出，不影响采样策略 |
| `messages`/`prompt` | 构造 chatMessages | 无 | - | 最终走 `ApplyChatTemplate(messages)` 再 `Tokenizer::Encode` |
| `stop`/`presence_penalty` | - | `stop_token_ids`（部分支持） | - | `stop` 仅在 **能编码为单个 token** 时生效；多 token stop string 当前会被忽略。`presence_penalty` 仍未映射 |

另外：同文件里还有一个更简单的路由（把 `prompt` 当 user 消息），它只设置了 `max_tokens`（默认 200），其它字段不处理。

### 2) C++ webui：`POST /chat`

实现见 [example/webui/webui.cpp](../example/webui/webui.cpp)。

- 入口只负责维护 `messages` 并走 `ApplyChatTemplate(messages)`。
- 推理调用：`model->LaunchResponseTokens(tokens)`，**未传 GenerationConfig**，即完全使用 C++ 默认值（偏贪心，`top_k=1`）。
- 如果你想让 webui 支持 `top_p/top_k/temperature/repeat_penalty`，最直接的改法是：在 `/chat` 的请求体里加 JSON，再构造一个 `GenerationConfig` 传给 `LaunchResponseTokens(tokens, config)`。

### 3) C++ benchmark（吞吐测量）

实现见 [example/benchmark/benchmark.cpp](../example/benchmark/benchmark.cpp)。

| CLI 参数/配置 | GenerationConfig 字段 | 说明 |
| --- | --- | --- |
| `--limit`（或 config.limit） | `output_token_limit` | benchmark 只设置了输出长度，其它采样参数用默认（偏贪心） |

### 4) Python openai_server：ChatCompletionRequest → 启动流式 handle

实现见 [tools/fastllm_pytools/openai_server/fastllm_completion.py](../tools/fastllm_pytools/openai_server/fastllm_completion.py) 与协议定义 [tools/fastllm_pytools/openai_server/protocal/openai_protocol.py](../tools/fastllm_pytools/openai_server/protocal/openai_protocol.py)。

| OpenAI 请求字段 | Python 侧处理 | 下层映射（概念） | 说明 |
| --- | --- | --- | --- |
| `max_tokens` | `max_length = request.max_tokens or 32768` | `output_token_limit` | Python server 默认给了一个很大的上限（更像“尽量不停”） |
| `min_tokens` | `min_length = request.min_tokens or 0` | `output_token_least` | 透传到下层最少输出 |
| `temperature` | 直接传入 | `temperature` | |
| `top_p` | 直接传入 | `top_p` | |
| `top_k` | 直接传入 | `top_k` | 协议默认是 `-1`，如果你没传，可能会把 -1 透下去（建议显式传 `1..50`） |
| `frequency_penalty` | 若非 0 用它，否则 1.0 | `repeat_penalty` | 这里把 `frequency_penalty` 当作重复惩罚系数用（并非 OpenAI 原语义） |
| `stop`/`presence_penalty` | 当前未见传入下层 | - | openai_server 侧目前主要靠 token 级停止（EOS/limit），stop string 未在此处落到 `stop_token_ids` |
| `stream` | 决定返回 generator | - | 流式断开会 `abort_handle(handle)` |

### 5) Python 直调 LLM：`llm.py` 的 ctypes 映射（补充）

实现见 [tools/fastllm_pytools/llm.py](../tools/fastllm_pytools/llm.py)。

- `generate(**kwargs)` 会把 kwargs 组装成 `GenerationConfig`（Python 侧类），再通过 ctypes 调用底层 `launch_response_llm_model(...)`。
- stop token 走的是 **token id 列表**（`stop_token_ids`），最终会填到 C++ `GenerationConfig.stop_token_ids`（对应 pytools.cpp 里把数组插入 multiset 的逻辑）。

---

## 性能与风险点（读源码时最常踩的坑）

1) **KV cache 内存预算**：异步 mainLoop 会根据设备 free mem 推一个 `kvCacheLimit`，影响 `maxTotalLens/maxBatch/promptLimit`；出现“突然不接新请求/强制 ending”的情况先看这里。
2) **mask 构造成本**：`FillLLMInputs` 在 qlen>1 且 `NeedAttentionMask` 返回 true 时会构造较大的 attentionMask；对长 prefill 是热点。
3) **长 prefill 切片**：`seqLens[0] > first` 会被切片 Forward，多次调用 `Forward`；这会改变 profiler 分布（更偏向 prefill op）。
4) **并发与 profiler**：`Executor` 的 profiler 是进程内 map 累加；如果你在服务端并发请求里启用 profiler，需要自行做“单请求隔离/串行化”以免统计被污染。

---

## 下一步建议（如果你要继续下钻）

- 想看具体模型（llama/qwen3/deepseek_v2）在 `Forward` 里怎么拼图/怎么调用 `RunComputeGraph`：从 [src/model.cpp](../src/model.cpp) 里 `CreateModelWithType` 找到具体 class，再看对应的 `src/models/*.cpp`。
- 想把“采样参数映射”写成一张表：从入口层（apiserver/webui/python server）把 `GenerationConfig` 的字段列出来，对照 `LLMSampling` 调用点。

