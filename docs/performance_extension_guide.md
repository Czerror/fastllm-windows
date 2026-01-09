# FastLLM（Windows fork）性能与扩展指南

> 目标：给出一条“可度量 → 可复现 → 可优化 → 可扩展”的最短路径。
> 
> 本文基于仓库当前实现：C++ 推理核（`basellm`/`Executor`/`ComputeGraph`）+ Python 工具链（`tools/fastllm_pytools`）+ 示例程序（`benchmark/webui/apiserver`）。

---

## 1. 性能度量：先建立基线

### 1.1 用 `benchmark` 分离 prefill 与 decode

C++ 基准入口：`benchmark.exe`（源码：`example/benchmark/benchmark.cpp`）。

建议两组基线：
- **prefill 吞吐（tokens/s）**：用相同 prompt、固定 batch，主要看提示词编码+prefill 速度。
- **decode 吞吐（tokens/s）**：固定 `output_token_limit`，并确保相同采样参数（top_p/top_k/temperature/repeat_penalty）。

Windows 示例（按你的实际模型路径替换）：
- `ftllm bench -p <模型或HF目录> -t 8 -b 8 -l 256 -f <prompts.txt>`

注意：`benchmark.cpp` 会对每条输入先 `ApplyChatTemplate` 并 `Encode`，`prompt token number` 是对这些 prompt 的总 token 数。

### 1.2 用 server 观测“并发 + 延迟”

你有两套 server：
- C++ 原生：`apiserver.exe`（源码：`example/apiserver/apiserver.cpp`）
- Python（FastAPI/uvicorn）：`python -m ftllm serve ...`（源码：`tools/fastllm_pytools/server.py`）

建议至少测三类指标：
- **TTFT（time to first token）**：prefill 主导
- **TPS（tokens per second）**：decode 主导
- **并发稳定性**：错误率、是否出现请求堆积、是否 OOM/显存暴涨

### 1.3 （可选）算子级耗时统计

`Executor` 内部维护了按 opType 聚合的耗时统计接口，但本仓库的示例程序已移除“自动打印 profiler”的开关与输出逻辑。
如果你后续需要做算子级定位，建议在你自己的驱动/压测程序里按需启用，并自行处理并发隔离口径。

---

## 2. 入口选型与 `ftllm` 路由规则（必读）

这一节解决三个问题：
- 我该选 C++ 还是 Python？
- `ftllm <command>` 最终会调用哪个产物？
- 为什么有时会“自动切 Python”？

### 2.1 C++ vs Python：能力/部署/参数面速查

| 场景 / 需求 | 推荐入口 | `ftllm` 命令示例 | 关键理由 |
| --- | --- | --- | --- |
| 只需要 OpenAI 风格 `chat.completions`（尽量轻、尽量快） | C++ `apiserver.exe` | `ftllm serve -p <模型> --port 8080` | 原生二进制依赖少；通常更接近“最低开销路径” |
| 需要 host 绑定 / API key / dev_mode / 以及更多服务端能力 | Python `python -m ftllm serve` | `ftllm -py serve -p <模型> --host 127.0.0.1 --api_key xxx` | Python 侧参数面更全；便于扩展 |
| 需要 HuggingFace Repo ID 下载 | Python | `ftllm download <repo_id>` | C++ 不负责下载；该命令强制走 Python |
| 需要 LoRA（`--lora` 或模型目录自带 LoRA） | Python（自动） | `ftllm run -p <模型> --lora <lora_path>` | `ftllm` 会自动检测并切换 Python |
| 本地交互式聊天（CLI） | C++ `FastllmStudio_cli.exe` | `ftllm run -p <模型>` 或 `ftllm run <模型路径>` | 原生交互 CLI；支持把 `-p/--path` 自动转换为位置参数 |
| Web 聊天界面 | C++ `webui.exe`（默认） | `ftllm webui -p <模型> --port 1616` | 一键启动；适合快速验证 |
| 性能基准测试 | C++ `benchmark.exe` | `ftllm bench -p <模型> -t 8 -b 8 -l 256 -f <prompts.txt>` | 最适合做基线与回归 |
| 模型量化 | C++ `quant.exe` | `ftllm quant -p <模型文件> -o <输出>` | 纯工具链任务，适合原生 |

### 2.2 `ftllm` 命令到二进制/后端的映射

`ftllm` 默认优先走 C++ 原生程序，映射关系如下：
- `serve/server/api` → `apiserver.exe`
- `webui/web` → `webui.exe`
- `chat/run` → `FastllmStudio_cli.exe`
- `bench/benchmark` → `benchmark.exe`
- `quant/quantize` → `quant.exe`

特别说明（Windows 常踩坑）：
- `FastllmStudio_cli.exe` 只接受位置参数 `model_path`，不支持 `-p/--path`。
- 你仍然可以用统一写法：`ftllm run -p <模型>`，`ftllm` 会把 `-p/--path` 转换为位置参数再调用 CLI。

### 2.3 触发“自动切 Python”的条件（你看到它突然用 python 的原因）

满足任一条件，`ftllm` 会走 `python -m ftllm ...`：
- 显式指定 `-py`
- 命令属于 Python-only：`download/ui/config/export`
- 参数包含 Python-only 参数集合（例如 `--host/--api_key/--custom/--dtype_config/--cache_history/--tool_call_parser/...`）
- 显式使用 `--lora` 或者模型目录自动检测到 LoRA（如存在 `lora/adapter_config.json` 或 `adapter_config.json`）

建议：
- 要“稳定跑 C++ 原生”：避免传入 Python-only 参数；LoRA 场景直接用 `-py`。
- 要“稳定跑 Python”：显式加 `-py`，不要依赖自动切换。

---

## 3. 关键性能热点（按调用链定位）

### 3.1 Executor 的“数据搬运”可能成为隐性大头

`Executor::Run()` 会对 `datas` 里的每个 `Data*` 执行 `ToDevice(device)`（含 batch 形式的 `Data**`）。

风险点：
- 小 op 密集时，频繁 `ToDevice` + `Reshape` 的调度开销会显著。
- `GetKVCacheInCPU()` / `GetHistoryCacheInCPU()` 打开时会出现 `lockInCPU`，导致即使 first device 是 GPU 也会强制走 CPU。

可度量方式：
- 建议优先用 `benchmark` 的吞吐/延迟指标做基线与回归；需要进一步下钻时，再考虑做算子级耗时统计。

### 3.2 ComputeGraph 的“图级优化”当前只做了两类关键融合

图优化入口：`OptimizeComputeGraph()`（源码：`src/graph.cpp`）。

已实现：
- **连续 Linear 合并**：多个共享同一 `input` 的 `Linear` 合并成一个大 `Linear`，再 `Split` 回原输出。
- **Linear + Silu + MulTo → Swiglu**：典型 MLP fused pattern。

优化建议（可扩展方向）：
- 把更多“常见 pattern”加入 `OptimizeComputeGraph`（例如 RMSNorm+Linear、RoPE+Attention 前后固定 permute 等），优先挑 profiler 占比最高的组合。

### 3.3 FusedAttention 的真实开销点：KV 扩容 + CatDirect(Batch)

`RunComputeGraph()` 对 `FusedAttention` 有专门路径（源码：`src/graph.cpp`）。

关键行为：
- 通过 `unitLen`（默认 128）做 KV cache 扩容对齐（`cache->Expansion(newDims)`），避免频繁 realloc。
- 对 batch=1：直接 `CatDirect` 把 `curk/curv` append 到 `pastKeys/pastValues`。
- 对 batch>1：在 `seqLens` 全为 1 时走 `CatDirectBatch` + `AttentionBatch`，是 decode 最重要快路径。

可优化点（按收益排序）：
1) **减少不必要的 permute/reshape**：当前路径会构造 `axisData` 并多次 `PermuteSelf`；如果你的模型结构固定，可以在图构建阶段预排布减少 permute。
2) **复用 axisData / 避免重复分配**：`axisData.Allocate()` 在热循环中出现多次（尤其 batch>1 分支），可考虑做静态缓存或把 axis 作为常量参数传入。
3) **KV 扩容策略**：unitLen=128 是折中；长上下文/高并发时可以通过 `kv_cache_limit` / `tokensLimit` 限制上界，避免显存/内存抖动。

---

## 4. 可回归优化清单（精简版）

建议优先把“测量口径”固定下来，再做任何优化：
- 固定 prompt 文件（同一份 `prompts.txt`）
- 固定输出 token 上限与采样参数（`top_p/top_k/temperature/repeat_penalty`）
- 固定并发/批次（C++ `--batch` 或 benchmark 的 `-b`）

最小回归组合（推荐先跑 3 次取均值）：
- `ftllm bench -p <模型> -t 8 -b 1 -l 256 -f <prompts.txt>`（prefill/decode 吞吐基线）
2) 长上下文/高并发先控上界：设置 C++ `tokensLimit`（`apiserver --tokens`）与 Python `kv_cache_limit`，避免扩容抖动。
3) 若 `CatDirectBatch`/`CatDirect` 占比很高：说明 KV append 成本显著，优先减少不必要的 permute/reshape。

回归指标：
- TTFT 变化不大但 TPS 提升（decode 优化常见特征）
- `CatDirect(Batch)` 与 `PermuteSelf` 占比是否下降

#### 4.2.3 `PermuteSelf`/`CatDirect(Batch)` 占比异常（布局与小 op 调度）

优先动作：
1) 优先做图级融合：把“固定前后处理”合成更少的 op（减少调度 + 减少中间张量）。
2) 检查是否存在“每步都重新构造 axis/shape”的路径（热循环重复分配会放大开销）。

回归指标：
- profiler 中 `PermuteSelf`/`CatDirect(Batch)` 的绝对时间和占比都应下降

#### 4.2.4 `ToDevice`/CPU 拷贝类 op 异常（隐性搬运）

优先动作：
1) 检查是否触发了 `lockInCPU`（某些 cache 行为可能导致强制驻留 CPU）。
2) 确认 device 选择与权重/缓存所在 device 一致，避免每步来回搬。
3) 小 op 密集时优先减少跨 device 数据流：要么全放 GPU，要么明确分层并让边界少发生。

回归指标：
- profiler 的数据搬运类 op 占比应显著下降
- 吞吐提升通常比 TTFT 更明显

---

## 5. 调参建议（能直接落地）

### 5.1 动态量化（dtype_config）

文档：`docs/dtype_config.md`。

核心策略：
- 用正则匹配权重名 → 为不同子模块指定 dtype（如 MLA fp8、MoE int4、gate float16）。
- **后面的规则优先级更高**。

推荐流程：
1) 先全局 `int4g`（baseline）
2) 给 attention/MLA 关键层切 `fp8` 或 `float16`
3) gate/ln 保持 float16/float32
4) 用 `benchmark` 回归速度与质量

### 5.2 混合推理（mixforward / moe_device）

文档：`docs/mixforward.md`。

适用：MoE 模型，常见目标是“让 decode 更快、显存更稳定”。
- 默认：`--device cuda --moe_device cpu`
- 进阶：把部分 moe 层放到 `cuda` 或 `multicuda`（注意上下文长度可能下降）。

### 5.3 缓存策略（cache_history / kv_cache_limit / tokensLimit）

- Python：
  - `--cache_history` / `--cache_fast`（`tools/fastllm_pytools/util.py`）
  - `--kv_cache_limit`（字符串，支持 `auto`）
- C++：
  - `apiserver` 支持 `--tokens`（映射到 `model->tokensLimit`）

推荐：
- 先把吞吐拉起来：开启 cache_history（适合长对话/重复前缀）。
- 再做稳定性：设置 kv_cache_limit/tokensLimit 防止长上下文把显存吃爆。

---

## 6. 扩展最短路径（按“改动最小、收益最大”排序）

### 6.1 新模型：优先用 Python ComputeGraph 描述

文档：`docs/custom.md` + `docs/custom_op.md`。

最短路径：
1) 写一个 Python 文件，继承 `ftllm.llm.ComputeGraph`
2) 在 `build()` 里用 `self.Embedding/Linear/FusedAttention/...` 描述计算
3) 文件末尾声明 `__model__ = YourModelClass`
4) 启动时加 `--custom your_model.py`

适用场景：
- 新模型结构还在迭代
- 你希望快速验证图结构是否正确、是否能跑通

### 6.2 新算子：先从“设备侧 op”加起，再考虑图层语法糖

核心接口：
- `BaseDevice::CanRun/Reshape/Run`（`include/device.h`）
- `Executor::Run` 会自动挑能跑的 device（`src/executor.cpp`）

建议落地顺序：
1) 先在 CPU device 上实现（便于调试正确性）
2) 再加 CUDA/TOPS/TFACC 等对应实现
3) 需要让 Python 自定义模型能调用时，再把 op 以 `ComputeGraph` 形式暴露（对应 `include/graph.h` + `src/graph.cpp` 的解析/执行分发）

### 6.3 新设备：实现 BaseDevice + 注册到 Executor

最短路径：
1) 新建 device 目录：`include/devices/<name>/` + `src/devices/<name>/`
2) 实现 `BaseDevice` 的内存分配/拷贝/ops 表
3) 在 `Executor::Executor()` 里按宏开关注册（参考 `CudaDevice/TopsDevice/TfaccDevice/NumaDevice`）

风险点：
- `ToDevice()` 语义要和现有 `Data` 的 `dataDevice`/`lockInCPU` 机制一致，否则会出现隐性拷贝或错误。

---

## 7. 推荐的性能报告模板（可直接粘贴到 issue/PR）

| 项目 | 优化前(ms) | 优化后(ms) | 改善% | 说明 |
| ---- | ---------- | ---------- | ----- | ---- |
| TTFT |  |  |  |  |
| Decode TPS |  |  |  |  |
| Attention 占比 |  |  |  | （可选）算子级耗时统计 |
| PermuteSelf 占比 |  |  |  | （可选）算子级耗时统计 |

---

## 8. 下一步建议（按你现在的仓库状态）

1) 先用 `benchmark` 固定一组 prompt 文件，记录 `prompt speed` + `output speed` 作为主基线。
2) 如需进一步定位热点，可做算子级耗时统计，确认 top-3 op（常见是 Linear/Attention/CatDirect/PermuteSelf）。
3) 再决定是走：
   - dtype_config 动态量化（更偏“省显存/提吞吐”）
   - mixforward（更偏“MoE decode 提速”）
   - 图融合（更偏“减少 permute/小 op 调度开销”）

