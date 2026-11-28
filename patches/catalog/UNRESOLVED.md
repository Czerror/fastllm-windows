# 无法以“新增文件”方式解决的上游修改目录（待决策）

架构原则：dev = 纯上游 + `patches/` 新增文件，不修改上游核心源码。
以下条目是旧补丁体系中**必须修改上游源码**才能实现的功能，逐一列出供决策。
旧补丁全文保留在 `patches/catalog/source-patches/` 供参考，**不会被自动应用**。

## 已批准保留（唯一源码修改）

| 编号 | 内容 | 说明 |
|---|---|---|
| 02 | NCCL 最小守卫 | Windows 无 NCCL。`fastllm-multicuda.cu` 用 `#ifndef FASTLLM_NO_NCCL` 包裹整个 NCCL 段并附带降级桩实现；Linux 行为不变。补丁位于 `patches/approved/02-cuda-nccl-guard.patch`，由 apply-patches.sh 自动应用，需 CUDA 构建验证。 |
| 04 | alivethreadpool chrono include（候选） | 上游 `alivethreadpool.h` 使用 `std::chrono` 但未包含 `<chrono>`（依赖 GCC 传递包含）。MSVC 无法用 /FI 强制包含（nvcc+STL 会触发 UCRT 双重定义），故以 1 行 include 修复（`patches/approved/04-alivethreadpool-chrono.patch`）。 |
| 06 | avx2 MSVC 对齐（候选） | 上游 `avx2.cpp:635` 的后置 `__attribute__((aligned(32)))` 在 MSVC 无法通过宏映射（`__declspec(align)` 必须前置）。当前以 1 行 `alignas(32)` 修复（`patches/approved/06-avx2-msvc-align.patch`）随构建应用。若你希望零源码修改，可改用 clang-cl 工具集（本机暂未安装）或舍弃 AVX2 调试日志路径。 |
| 11 | cpudevice 对齐 placement new（候选） | 上游 `cpudevice.cpp:2652` 的 `new (std::align_val_t{64})` 在 MSVC C++20 下报 C2956（对齐 placement delete 选择错误）。当前以 2 行标准写法修复（`patches/approved/11-cpudevice-align-new.patch`）随构建应用。 |
| 12 | FlashInfer MSVC 兼容（新增） | vendored `third_party/flashinfer` 4 处 shim：`FLASHINFER_INLINE` 改 `__forceinline__`、`math.cuh` 补 `ushort`、`topk.cuh` 补 `__builtin_expect`、`decode.cuh` 的 `__attribute__((shared))` 改 `__shared__`。由 `inject_minimal_patches.py` 内容寻址注入（幂等），dev 树保持纯上游；仅 Windows/MSVC 生效。 |
| 13 | Windows 禁用 FlashInfer attention（新增） | 上游 `fastllm-attention.cu` 的 `FASTLLM_ENABLE_FLASHINFER` 定义增加 `!defined(_WIN32)` 条件：Windows 下 FlashInfer mla.cuh 与 nvcc 模板解析不兼容，走内置原生分页注意力回退（`#ifndef` 分支）；采样路径（`sampling.cuh`）不受影响。由 `inject_minimal_patches.py` 注入，Linux 行为不变。 |
| 14 | Windows 宿主(cl.exe)编译兼容（新增） | ① `basellm.cpp` 的 `__int128`（MSVC x64 不支持）改 `long double`；② `multicudadevice.cpp` 的 `strcasecmp` 映射 `_stricmp`；③ `cudadevice.cpp` 上游将 CUTLASS/Triton FP8 实现整体 `#if !defined(_WIN32)` 排除但调用点未同步守卫，在 `#else` 分支补同名 Windows 桩（返回 false 走 native FP8）；④ 修正 `TryCudaTritonChunkGdnPrefill` 桩签名（8 参 → 10 参，与调用点一致）。均由 `inject_minimal_patches.py` 注入，Linux 行为不变。 |

## 待决策：舍弃（推荐）或改上游

| 编号 | 功能 | 修改内容 | 无法新增的原因 | 舍弃的影响 |
|---|---|---|---|---|
| 12/14/04 | 日志回调系统 | basellm.cpp `SetLogCallback` 实现；basellm.h 注入 `LogEvent/LogData`；pybinding 日志绑定；log_handler.py | 上游**没有任何日志钩子 API**，回调必须插入底层日志调用点；新增文件无法拦截上游内部日志 | **已决策：舍弃**。`log_handler.py` 保留为新增文件但无数据源 |
| 16 | 推理统计 | basellm `GetStats` + context 统计计数；`inference_stats.h` | 上游无 `GetStats`/统计字段，计数点在上游 decode 循环内 | **已决策：舍弃底层统计**，`inference_stats.h`（`InferenceStatsHelper`）已恢复为启用特性，由新增服务层自行计时 |
| 17/18 | 自动设备映射 | model.cpp `ApplyAutoDeviceMap` 实现；basellm.h 声明 | 上游无此 API，且需要在模型加载流程中插入调用 | **已决策：舍弃**。多卡需手动 `SetDeviceMap` |
| 03 | template tojson | template.cpp 注入 `tojson` Jinja 过滤器/函数 | 上游无此功能（仅 Python 端有无关的 `to_json`） | **已决策：舍弃**。影响分析：仅影响使用 `{{ ... | tojson }}` 的第三方 Jinja chat 模板渲染；上游自带模板/模型不受影响（上游本身无此过滤器也能正常工作） |
| 07 | pytools 大改 | pytools.cpp：删除 CUDA 导出、改 warmup、多模态签名、统计缓存 | 上游 pytools.cpp 已更新（warmup 返回 `const char*`，Python 端已匹配）；原改动多为删除上游功能/行为变更 | **已决策：直接使用上游完整版本**（现状即如此），无影响 |
| 13 | diskdevice Windows | diskdevice.cpp 的 `_open/_read/mmap` 兼容层 | 上游用 POSIX `open/pread/mmap`，Windows 无法编译 | **已决策：功能完整保留**。`include/sys/mman.h` + `include/unistd.h` 提供 Win32 实现；`posix_memalign` 已升级为 `_aligned_malloc` + 文件内 `free→_aligned_free` 配对（真 4096 对齐） |
| 15 | WarmUp/isComplete | basellm 注入 `WarmUp` 枚举 | 上游已有 `WarmUp()/AutoWarmup()` | **已决策：舍弃**（已过时，无影响） |
| 10 | 示例 Windows 适配 | apiserver/webui/benchmark 修改 | 上游示例已自带 `_WIN32` 分支 | 无需处理（本地/CI 编译通过） |
| 08 | Python 工具改动 | pyfastllm `__init__` 日志导入；chat.py/download.py UI；tool_parsers 注册 Glm47 | 上游无 `glm47_moe_tool_parser.py`，该注册在纯上游树会导入失败 | **已决策：舍弃**。影响分析：① pyfastllm 日志导入——随日志系统舍弃，无影响；② chat.py/download.py 的 console 进度/标题——纯 UI 增强，丢失仅影响展示；③ Glm47 parser 注册——上游无对应文件，本就不该应用 |
| 05 | 测试修复 | 修改测试文件 | 上游测试随源码演进 | 无需处理（非打包功能） |
| 09 | .gitmodules | 子模块条目 | 已改由 apply-patches.sh 构建时添加子模块 | 无需处理 |

## 已通过“新增文件/构建参数”解决（无需决策）

| 原问题 | 解决方式 |
|---|---|
| `__attribute__`（avx2/avx512vnni） | 新增 `include/utils/msvc_attr_shim.h`，CMake 仅对这两个文件 `/FI` 强制包含 |
| M_PI（fastllm.cpp） | 构建参数 `/D_USE_MATH_DEFINES` |
| chatglm u8 字面量 | 构建参数 `/Zc:char8_t-` |
| C++17 默认（gguf 需 C++20） | 构建参数 `CMAKE_CXX_STANDARD=20` |
| USE_MMAP（Windows 无 sys/mman.h） | CMake 在 Windows 自动关闭 |
| sentencepiece（上游无此子模块，可选依赖） | dev 默认 `USE_SENTENCEPIECE=OFF` 不添加子模块；手动需要时通过 build-dev.yml 的 `build_sentencepiece` 输入临时添加子模块（固定 11051e3b）并以 ON 构建 |
| ftllm 客户端 + console/help | 纯新增文件 |
| chat_handler/minja 模板 | 文件保留在 `patches/catalog/features-archive/`；**接线需改上游 apiserver** → 可选：舍弃，或新增独立服务可执行文件（工作量较大） |

## 建议

1. 日志回调/统计/自动设备映射：推荐舍弃（保持零源码修改）；如需保留，请指定保留哪一项，我们再评估最小上游改动。
2. diskdevice：可在“Windows 舍弃（当前）”与“新增 `diskdevice_win.cpp` 兼容实现（保留功能）”之间选择。
3. chat_handler：若需要聊天模板渲染，建议以新增独立 server 程序实现。

## CUDA 构建探针结论（2026-08-05，本机 CUDA 13.1 + MSVC 19.44）

- 配置成功：全部上游 CUDA 特性开启（CUTLASS FP8、DeepSeek-V4 SM120 稀疏 MLA/DeepGEMM、TurboMind、120f 归一化）。
- 本项目核心 CUDA 文件（含 NCCL 守卫后的 multicuda）已编译通过；构建在 **上游 vendored 第三方库** 处失败：
  - `third_party/flashinfer/math.cuh`：`ushort` 未定义（FlashInfer 头文件与 CUDA 13.1 类型暴露顺序不兼容），连带 `vec_dtypes.cuh` 报错。
- 后续预计还会遇到 deep_gemm / turbomind / cutlass 等第三方库的 Windows 工具链兼容问题（未继续深挖）。
- 选项：
  A. 接受 Windows CUDA 暂不可用（推荐，保持零源码修改）；
  B. 逐个给上游 vendored 第三方库打最小兼容补丁（工作量中等，需逐库验证）；
  C. 裁剪 FlashInfer/DeepGemm 等 SM120 专用路径（回到旧版“裁剪”方案，违背本次目标）。

## CUDA 构建修复进展（2026-08-07，迁移到 windows-latest + CUDA latest）

- 已确认并修复：`windows-latest` 已迁移 VS 2026，CUDA 13.1 的 `cudafe++` 在 MSVC 19.51 下崩溃；
  曾以 `windows-2022` + CUDA 13.1 作为官方支持组合验证通过（commit `445805bb`）。
- 2026-08-07 完成迁移：**CUDA job 改为 `windows-latest` + CUDA 13.3.1（latest）**，
  迁移测试（SP ON）run #31209273043 成功，CPU/CUDA 双包产出；默认（SP OFF）与手动（SP ON）构建最终验证中。
- 仍需 `/Zc:preprocessor`（经 workflow 的 `-DCMAKE_CUDA_FLAGS` 透传）：turbomind `TM_PP_*` 宏在
  MSVC 传统预处理器下展开错误（各 CUDA 版本均存在）。
- 已批准注入：FlashInfer MSVC 兼容（本表 12 号）；Windows 禁用 FlashInfer attention（本表 13 号）；
  DeepGemm SM120 特化内核在 Windows 构建中禁用（DeepGemm 仅支持 Linux，`patches/cmake/CMakeLists.windows.txt` 门控）；
  宿主 cl.exe 编译兼容（本表 14 号）。
- 子模块管理遵循上游：pybind11 对齐上游记录提交（不再自动跟随远程最新）；sentencepiece 上游未定义，
  dev 默认 `USE_SENTENCEPIECE=OFF`，手动需要时通过 `build_sentencepiece` 输入临时添加子模块（11051e3b）。
