# fastllm-windows 上游合并计划

> **目标**: 将上游 [ztxz16/fastllm](https://github.com/ztxz16/fastllm) 领先的 4000+ 提交合并到 [Czerror/fastllm-windows](https://github.com/Czerror/fastllm-windows)，同时完整保留所有 Windows 特定适配。

---

## 零、前置信息收集

### 0.1 当前状态

| 项目 | 仓库 | 最新提交 |
|------|------|----------|
| 本项目 | `Czerror/fastllm-windows` (master) | `9cfff9f` — "chore: Auto-update submodules" |
| 上游 | `ztxz16/fastllm` (master) | `ed79ba5` — "fix" (2026-06-09) |

### 0.2 已确认的 Windows 独有文件（必须保留）

| 类别 | 文件 | 说明 |
|------|------|------|
| **构建入口** | `build.ps1` | PowerShell 交互式构建脚本 |
| | `点我启动编译.bat` | 批处理双击入口 |
| | `setup-env.ps1` | 一键环境安装 |
| **平台入口** | `ftllm.cpp` | Windows 专属统一命令行入口 (1772行) |
| **运行时** | `VC++/*.dll` (10个) | VC++ Redist 便携分发 |
| **VS 项目** | `example/Win32Demo/*` | VS .sln/.vcxproj 原生项目 |
| **CI/CD** | `.github/workflows/build-windows.yml` | Windows 自动构建工作流 |
| **脚本** | `simple_install.sh` | 简化安装 (可能新增) |

### 0.3 上游新增目录（本项缺失，需直接引入）

```
src/blocks/                    ← 新增：块操作抽象
src/devices/disk/              ← 新增：磁盘设备
src/devices/cuda/              ← 大幅扩展：CUDA 完整实现
src/devices/multicuda/         ← 新增：多 GPU 支持
src/devices/numas/             ← 新增：NUMA 支持
src/devices/tops/              ← 新增：燧原 Tops 支持
include/blocks/                ← 新增：块操作头文件
include/devices/cuda/          ← 新增：CUDA 设备头文件
include/devices/disk/          ← 新增：磁盘设备头文件
include/devices/numas/         ← 新增：NUMA 头文件
include/devices/tops/          ← 新增：Tops 头文件
third_party/turbomind/         ← 新增：TurboMind AWQ 内核
third_party/flashinfer/        ← 新增：FlashInfer 集成
```

### 0.4 上游新增模型文件（本项缺失）

| 模型 | 源文件 | 头文件 | 大小 |
|------|--------|--------|------|
| DeepSeek V4 | `src/models/deepseekv4.cpp` | `include/models/deepseekv4.h` | 198KB |
| Gemma 4 | `src/models/gemma4.cpp` | `include/models/gemma4.h` | 98KB |
| MiniMax M2 | `src/models/minimax_m2.cpp` | `include/models/minimax_m2.h` | 97KB |
| Qwen2 | `src/models/qwen2.cpp` | `include/models/qwen2.h` | 15KB |
| Qwen3.5 | `src/models/qwen3_5.cpp` | `include/models/qwen3_5.h` | 756KB |
| Step3-3.5 | `src/models/step3p5.cpp` | `include/models/step3p5.h` | 292KB |

### 0.5 上游新增 CPU 设备文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `src/devices/cpu/deepseekv4ops.cpp` | 47KB | DeepSeek V4 专用 CPU 算子 |
| `src/devices/cpu/linear.cpp` | 145KB | CPU 线性运算 |
| `src/devices/cpu/cpudevicebatch.cpp` | 13KB | CPU 批量推理 |

### 0.6 上游新增核心文件

| 文件 | 说明 |
|------|------|
| `src/device.cpp` | 设备抽象层 |
| `include/device.h` | 设备抽象头文件 |
| `include/models/qwen3_cuda_common.h` (44KB) | Qwen3 CUDA 公共头文件 |

---

## 一、推荐合并策略：分阶段渐进式合并

### 为什么不用 `git merge --squash`？

因为上游 4000+ 提交涉及大量文件变更，直接 squash merge 会丢失上游历史且极难解决冲突。

### 推荐策略：三轮渐进式 Cherry-Pick + 批量覆盖

```
阶段 A: 结构对齐 — 引入上游新增目录和文件（低冲突）
阶段 B: CMakeLists.txt 合并 — 手工融合两个版本的构建系统（高冲突，关键步骤）
阶段 C: 源文件覆盖 — 用上游版本替换共享文件，重新应用 Windows #ifdef（中高冲突）
阶段 D: 验证编译 — Windows 本地编译验证和修复
```

---

## 二、详细步骤清单

### 阶段 A: 结构对齐（低风险）

> **目标**: 让项目目录结构与上游一致，引入所有新增文件和目录。

#### 步骤 A1: 备份当前状态

```bash
# 在 GitHub 上创建备份分支
git checkout -b backup/$(date +%Y%m%d)
git push origin backup/$(date +%Y%m%d)
```

**⚠️ 需检查**: 确认当前工作区干净 (`git status` 无未提交变更)。

#### 步骤 A2: 创建合并工作分支

```bash
git checkout master
git checkout -b merge/upstream-latest
```

使用 GitHub MCP [`create_branch`](owner=Czerror, repo=fastllm-windows, branch=merge/upstream-latest)。

#### 步骤 A3: 添加上游远程（如尚未配置）

在本地仓库添加 upstream 远程：
```bash
git remote add upstream https://github.com/ztxz16/fastllm.git
git fetch upstream master
```

#### 步骤 A4: 引入新增目录和文件（无冲突）

以下文件和目录在 Windows 分支中**不存在**，可以直接从上游复制：

```bash
# 新增目录 — 直接检出
git checkout upstream/master -- \
    src/blocks/ \
    src/devices/disk/ \
    src/devices/cuda/ \
    src/devices/multicuda/ \
    src/devices/numas/ \
    src/devices/tops/ \
    include/blocks/ \
    include/devices/cuda/ \
    include/devices/disk/ \
    include/devices/numas/ \
    include/devices/tops/ \
    include/device.h \
    src/device.cpp

# 新增模型文件
git checkout upstream/master -- \
    src/models/deepseekv4.cpp \
    src/models/gemma4.cpp \
    src/models/minimax_m2.cpp \
    src/models/qwen2.cpp \
    src/models/qwen3_5.cpp \
    src/models/step3p5.cpp \
    include/models/deepseekv4.h \
    include/models/gemma4.h \
    include/models/minimax_m2.h \
    include/models/qwen2.h \
    include/models/qwen3_5.h \
    include/models/qwen3_cuda_common.h \
    include/models/step3p5.h

# 新增 CPU 设备文件
git checkout upstream/master -- \
    src/devices/cpu/deepseekv4ops.cpp \
    src/devices/cpu/linear.cpp \
    src/devices/cpu/cpudevicebatch.cpp

# 新增第三方库
git checkout upstream/master -- \
    third_party/turbomind/ \
    third_party/flashinfer/
```

**预期结果**: 无冲突，纯新增。

#### 步骤 A5: 检查上游新增的 graph 模型文件

上游 `src/models/graph/` 目录可能包含不同于本项目的文件。需要对比并只添加缺失的。

**⚠️ 需检查**: 使用 `diff` 对比两个仓库 `src/models/graph/` 目录。

---

### 阶段 B: CMakeLists.txt 合并（高风险 ⚠️）

> **目标**: 将上游 CMakeLists.txt 的新功能合并到 Windows 版本中。

#### 步骤 B1: 获取上游 CMakeLists.txt 全文

已从 GitHub MCP 获取：[上游 CMakeLists.txt (SHA: 7a532e5)](https://github.com/ztxz16/fastllm/blob/master/CMakeLists.txt)

#### 步骤 B2: 逐段对比与合并决策表

| 上游区块 | 本项目处理 | 冲突级别 |
|----------|-----------|----------|
| `project(fastllm)` → 无 `-windows` 后缀 | **保留本项目**: `project(fastllm-windows)` | 低 |
| `option(USE_NUMAS ... ON)` | **保留本项目**: 保持为 option，Windows 下游设为 OFF | 低 |
| `option(USE_TOPS ... OFF)` | **新增**: 直接引入，无人维护时不启用即可 | 无 |
| `option(USE_IVCOREX ... OFF)` | **新增**: 直接引入 | 无 |
| `CUDA_ARCH` 改为 `native` 默认 | **保留本项目**: `75;80;86;89;90;120` 显式指定 | 低 |
| ROCm 设备检测函数 `detect_rocm_devices()` | **新增**: 直接引入（Windows ROCm 实验性） | 无 |
| `MAKE_WHL_X86` 支持 | **新增**: 直接引入 | 无 |
| MSVC 编译器标志 | **关键冲突**: 上游版本 vs 本项目版本 | **高** |
| AVX512 → `/arch:AVX10.1`（上游）vs `/arch:AVX512`（本项目） | **需决策**: 评估哪个对 Windows 更合适 | 中 |
| `/Ob2` → `/Ob1 /Gy` 修复 | **上游已包含**: 本项目的 int4 乱码修复已被上游采纳 | 无 |
| `src/blocks/` 和 `src/devices/` 新源文件 | **直接引入**: 所有新增的 GLOB 和显式列表 | 低 |
| `include/devices/` 新目录 | **直接引入** | 低 |
| `USE_CUDA` 大幅扩展（多GPU、TurboMind、FlashInfer）| **直接引入**: Windows CUDA 构建也需要这些 | 中 |
| `USE_ROCM` hipify 流程 | **保留**: Windows CI 不启用 ROCm | 低 |
| `USE_NUMAS` 条件 | **保留本项目**: Windows 上禁用 | 低 |
| `USE_TOPS` 条件 | **新增** | 无 |
| `USE_IVCOREX` 条件 | **新增** | 无 |
| `PY_API` Python 绑定路径 | **保留本项目**: Python 路径不同 | 中 |
| `fastllm_tools` 构建后复制 | **保留本项目**: `.dll` vs `.so` 差异 | **高** |
| `build_info.json` 生成 | **新增**: 直接引入 | 无 |
| `webui`/`benchmark`/`apiserver` 等可执行文件 | **保留本项目**: 路径可能不同 | 中 |

#### 步骤 B3: MSVC 编译器标志合并（关键冲突）

**上游版本**:
```cmake
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    string(REPLACE "/Ob2" "/Ob1 /Gy" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX /std:c++17 /arch:AVX2 /source-charset:utf-8")
```

**本项目版本**:
```cmake
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DNOMINMAX /std:c++20 /arch:AVX2 /source-charset:utf-8 /execution-charset:utf-8 /bigobj /MP2 /D_CRT_SECURE_NO_WARNINGS /wd4244 /wd4267 /wd4190 /wd4996")
```

**合并决策**:

| 标志 | 上游 | 本项目 | 建议采用 | 原因 |
|------|------|--------|----------|------|
| C++ 标准 | `/std:c++17` | `/std:c++20` | **`/std:c++20`** | gguf.cpp 需要 C++20 指定初始化器 |
| `/execution-charset:utf-8` | 无 | 有 | **保留** | 中文模型路径支持 |
| `/bigobj` | 无 | 有 | **保留** | 大 .obj 文件需要 |
| `/MP2` | 无 | 有 | **保留** | 防 CL.exe 崩溃 |
| `/D_CRT_SECURE_NO_WARNINGS` | 无 | 有 | **保留** | MSVC CRT 安全检查警告 |
| `/wdXXXX` 警告抑制 | 无 | 有 | **保留** | 干净编译输出 |
| `/Ob1 /Gy` | 有 | 无 (通过 `/Ob2` 替换) | **采用上游** | 修复 int4 乱码 |
| AVX512 标志 | `/arch:AVX10.1` | `/arch:AVX512` | **需验证** | ⚠️ 见下方分析 |

#### 步骤 B4: AVX512 vs AVX10.1 决策

- `/arch:AVX512` 是 MSVC 传统的 AVX-512 启用方式
- `/arch:AVX10.1` 是 MSVC 2022 17.10+ 新增的，支持 AVX10.1 指令集
- **建议**: 如果 MSVC 版本 >= 19.40 (VS 2022 17.10)，使用 `/arch:AVX10.1`；否则回退到 `/arch:AVX512`

#### 步骤 B5: CMakeLists.txt 最终合并方案

**采用"以上游为基础，叠加本项目 Windows 适配"的策略**:

1. 以**上游 CMakeLists.txt** 为底本
2. 应用以下本项目独有修改：
   - `project(fastllm-windows)` 
   - C++ 标准改为 `/std:c++20`
   - 添加 `/execution-charset:utf-8 /bigobj /MP2 /D_CRT_SECURE_NO_WARNINGS /wd4244 /wd4267 /wd4190 /wd4996`
   - CUDA_ARCH 覆盖为 `75;80;86;89;90;120`
   - `USE_MMAP` 在 WIN32 下强制 OFF
   - `fastllm_tools` 构建后复制逻辑保留 `.dll` 路径
   - ROCm Windows 实验性警告保留
   - AVX512 编译选项使用条件判断

---

### 阶段 C: 源文件覆盖（中高风险）

> **目标**: 用上游最新版本更新所有共享的 `.cpp`/`.h` 文件，同时保留 Windows 条件编译块。

#### 步骤 C1: 识别需要条件编译保护的文件

根据之前分析，以下文件包含 Windows 特定的 `#ifdef _WIN32`/`#ifdef _MSC_VER` 代码：

| 文件 | Windows 适配内容 | 合并策略 |
|------|-----------------|----------|
| `include/fastllm.h:8` | `#define NOMINMAX` | 上游已包含 — **直接覆盖** |
| `include/utils/utils.h:30-237` | `<Windows.h>`, `<intrin.h>`, CPUID | **需对比** — 上游可能有新 CPU 指令检测 |
| `include/utils/console.h:15-164` | `SetConsoleOutputCP`, ANSI 启用 | **需对比** |
| `include/utils/avxMath.h:35-177` | `__declspec(align(32))`, `__forceinline` | **需对比** |
| `src/model.cpp:435` | `_fseeki64` | **需对比** — 上游 182KB vs 本地 |
| `src/fastllm.cpp:2059` | `_ftelli64` | **需对比** — 上游 210KB vs 本地 |
| `src/devices/cpu/cpudevice.cpp:1359` | `_aligned_malloc` | **需对比** — 上游 512KB vs 本地 |
| `src/devices/cpu/avx2.cpp:665` | `__forceinline` | **需对比** |
| `src/devices/cpu/amx.cpp:49` | `#if !defined(_WIN32)` 排除 | **需对比** |
| `src/models/chatglm.cpp:969` | `u8` 字面量转换 | **需对比** |

#### 步骤 C2: 文件覆盖决策矩阵

| 文件 | 建议策略 | 理由 |
|------|----------|------|
| `include/fastllm.h` | **上游覆盖** | NOMINMAX 保护上游已包含 |
| `include/utils/utils.h` | **手工合并** | 上游可能有新的 CPUID 检测，需保留 `<intrin.h>` 路径 |
| `include/utils/console.h` | **手工合并** | Windows 控制台 UTF-8 设置上游可能没有 |
| `include/utils/avxMath.h` | **上游覆盖 + 补丁** | 上游可能新增了数学函数，覆盖后再检查对齐宏 |
| `include/utils/armMath.h` | **上游覆盖** | ARM 无关 Windows |
| `include/executor.h` | **上游覆盖** | 可能新增 API |
| `include/graph.h` | **上游覆盖** | 可能新增图操作 |
| `include/model.h` | **上游覆盖** | 可能新增模型接口 |
| `include/template.h` | **上游覆盖** | 可能新增模板功能 |
| `include/models/*.h` | **上游覆盖全部** | 共享模型头文件以获取新模型支持 |
| `src/fastllm.cpp` | **手工合并** ⚠️ | 210KB 大文件，需逐段检查 Windows 适配 |
| `src/model.cpp` | **手工合并** ⚠️ | 182KB，文件 I/O 的 `_fseeki64`/`_ftelli64` 需保留 |
| `src/executor.cpp` | **上游覆盖** | 可能新增执行器功能 |
| `src/graph.cpp` | **上游覆盖** | 可能新增图计算 |
| `src/template.cpp` | **上游覆盖** | 可能新增模板 |
| `src/tokenizer.cpp` | **上游覆盖** | 可能新增分词器 |
| `src/pybinding.cpp` | **手工合并** | Python 绑定可能有路径差异 |
| `src/devices/cpu/*.cpp` | **手工合并** ⚠️ | 512KB cpudevice.cpp 是重中之重 |
| `src/models/*.cpp` | **上游覆盖全部** | 共享模型实现以获取 bug 修复和新功能 |

#### 步骤 C3: 高风险文件重点合并计划

**`src/devices/cpu/cpudevice.cpp`** (上游 512KB vs 本地未知大小):
- 先对整个函数做 diff
- 重点关注 `_aligned_malloc`/`_aligned_free` 出现的位置
- 重点关注 `_fseeki64`/`_ftelli64` 出现的位置
- 可能需要逐函数对比

**`src/fastllm.cpp`** (上游 210KB vs 本地):
- 重点关注文件 I/O 的 `_ftelli64` 替换
- 上游可能重构了某些函数

**`src/model.cpp`** (上游 182KB vs 本地):
- 重点关注 `_fseeki64` 替换
- 上游可能新增了模型格式支持

#### 步骤 C4: Windows `#ifdef` 补丁清单

合并过程中可能丢失的 Windows 适配代码需要重新添加：

```cpp
// 1. include/fastllm.h — NOMINMAX（上游已含，验证即可）
#if defined(_WIN32) || defined(_WIN64)
#define NOMINMAX
#endif

// 2. include/utils/utils.h — <intrin.h> CPUID
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

// 3. include/utils/utils.h — filesystem 回退
#if _MSC_VER <= 1900
namespace fs = std::experimental::filesystem;
#else
namespace fs = std::filesystem;
#endif

// 4. include/utils/avxMath.h — 对齐声明
#if defined(_MSC_VER)
#define ALIGN32 __declspec(align(32))
#else
#define ALIGN32 __attribute__((aligned(32)))
#endif

// 5. src/devices/cpu/amx.cpp — AMX 排除
#if !defined(_WIN32)
    // Linux AMX 初始化
#else
    printf("AMX is not supported yet on Windows.\n");
#endif

// 6. 文件 I/O — 大文件支持
#ifdef _WIN32
    _fseeki64(f, offset, SEEK_SET);
#else
    fseek(f, offset, SEEK_SET);
#endif

// 7. 内存对齐
#ifdef _MSC_VER
    ptr = _aligned_malloc(size, alignment);
#else
    ptr = aligned_alloc(alignment, size);
#endif
```

---

### 阶段 D: 验证编译

#### 步骤 D1: 清理并重新配置 CMake

```bash
# Windows 本地
./build.ps1 -CleanCache -Target cpu -Auto
```

#### 步骤 D2: 检查编译错误类型

| 预期错误类型 | 原因 | 修复方法 |
|-------------|------|----------|
| `NOMINMAX` 未定义 | include 顺序变化 | 确保 `fastllm.h` 第一个包含 |
| `std::filesystem` 命名空间错误 | MSVC 版本 | 添加 `#if _MSC_VER <= 1900` 回退 |
| `__cpuid` 未定义 | `<intrin.h>` 缺失 | 添加 `#include <intrin.h>` |
| `aligned_alloc` 未找到 | MSVC 不兼容 | 替换为 `_aligned_malloc` |
| `fseek`/`ftell` 2GB 截断 | 32位偏移 | 替换为 `_fseeki64`/`_ftelli64` |
| 中文注释乱码 | 编码问题 | 确认 `/source-charset:utf-8 /execution-charset:utf-8` |
| CUDA 编译错误 | NVCC + MSVC 兼容 | 检查 `-Xcompiler` 标志 |
| 新增模型文件编译错误 | 依赖新的头文件 | 检查 include 路径和新 API |

#### 步骤 D3: CI 验证

推送 `merge/upstream-latest` 到 GitHub，触发 `.github/workflows/build-windows.yml`。

---

## 三、风险矩阵

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| CMakeLists.txt 合并冲突导致构建失败 | 高 | 严重 | 手工逐段合并，保留所有 MSVC 标志 |
| 上游重构了核心 API 导致模型文件不兼容 | 中 | 严重 | 先对比头文件 API 变化 |
| `cpudevice.cpp` (512KB) 合并丢失 Windows 适配 | 高 | 严重 | diff 逐函数检查 |
| CUDA 路径差异导致 NVCC 编译失败 | 中 | 中 | 检查所有 CUDA 源文件路径 |
| Python 绑定 (.pyd) 链接失败 | 中 | 中 | 检查 pybinding.cpp 的 Python 路径 |
| `third_party/turbomind` 在 MSVC 下编译失败 | 中 | 中 | TurboMind 主要为 Linux 设计 |
| 上游 AVX10.1 标志不兼容旧 MSVC | 低 | 中 | 添加 MSVC 版本检测 |
| `src/devices/cuda/` 太新导致 Windows CUDA 不兼容 | 中 | 高 | 可能需要保持旧版 CUDA 实现 |

---

## 四、执行顺序总览

```
┌─────────────────────────────────────────────────────┐
│ 前置检查: 确认工作区干净，创建备份分支               │
├─────────────────────────────────────────────────────┤
│ Phase A: 结构对齐                                    │
│  A1. git checkout -b merge/upstream-latest           │
│  A2. 添加 upstream remote + fetch                    │
│  A3. git checkout upstream/master -- <新目录/文件>    │
│  A4. 提交: "引入上游新增目录和文件"                   │
├─────────────────────────────────────────────────────┤
│ Phase B: CMakeLists.txt 合并                         │
│  B1. 以上游 CMakeLists.txt 为底本                    │
│  B2. 手工叠加本项目 Windows 适配                     │
│  B3. 提交: "合并上游 CMakeLists.txt + Windows 适配"   │
├─────────────────────────────────────────────────────┤
│ Phase C: 源文件覆盖                                  │
│  C1. 低风险文件批量覆盖（上游覆盖）                   │
│  C2. 高风险文件手工合并（diff + 补丁）                │
│  C3. 重新应用所有 Windows #ifdef 补丁                 │
│  C4. 提交: "合并上游源文件 + 保留 Windows 适配"       │
├─────────────────────────────────────────────────────┤
│ Phase D: 验证                                        │
│  D1. Windows 本地编译验证 (CPU)                      │
│  D2. Windows 本地编译验证 (CUDA，如有 GPU)            │
│  D3. CI 自动构建验证                                 │
│  D4. 修复编译错误                                    │
├─────────────────────────────────────────────────────┤
│ 完成: 合并到 master + 打 tag 发布                     │
└─────────────────────────────────────────────────────┘
```

---

## 五、关键注意事项

### ⚠️ 绝对不能做的事

1. **不要删除任何 Windows 独有文件**: `ftllm.cpp`, `build.ps1`, `setup-env.ps1`, `VC++/`, `example/Win32Demo/`, `点我启动编译.bat`, `.github/workflows/build-windows.yml`
2. **不要修改 `project()` 名称**: 保持 `fastllm-windows`
3. **不要在 Windows 上启用 `USE_MMAP`**: 保持 OFF
4. **不要丢弃 `/std:c++20`**: gguf.cpp 必须 C++20
5. **不要丢弃 `/bigobj` 和 `/MP2`**: MSVC 编译稳定性必需

### ✅ 必须做的事

1. **保留所有 `#ifdef _WIN32`/`#ifdef _MSC_VER` 代码块**
2. **保留 `_aligned_malloc`/`_aligned_free` 替代 `aligned_alloc`**
3. **保留 `_fseeki64`/`_ftelli64` 替代 `fseek`/`ftell`**
4. **保留 `/DNOMINMAX` 宏定义**
5. **保留 `/source-charset:utf-8 /execution-charset:utf-8` 编码标志**
6. **保留 CUDA NVCC 的 MSVC 兼容标志**

---

## 六、后续维护建议

合并完成后，建议：

1. **设置 GitHub Actions 自动同步**: 创建一个定期运行的 workflow，检查上游新提交并自动创建 PR
2. **最小化 Windows 补丁集**: 将 Windows 适配代码尽可能封装为宏或独立文件，而非散落在各处
3. **向上游提交 PR**: 将通用的 MSVC 兼容性改进（如 `/Ob1 /Gy`、部分 `#ifdef _WIN32` 补丁）贡献回上游
4. **版本号对齐**: 合并后版本号应与上游 `v0.1.7.0` 保持一致，附加 `-windows` 后缀
