# fastllm-windows patches

## 架构

- **dev 分支 = 纯上游 + 本目录**：除 `patches/` 与 `.github/workflows/`（CI 基础设施）外，dev 与上游 `ztxz16/fastllm` 逐文件一致。
- 所有 Windows 适配均为**新增文件**或**构建系统参数**，不修改上游核心源码。
- 已批准的源码修改（构建时注入，dev 树保持纯上游）：
  - NCCL 最小守卫（`approved/02-cuda-nccl-guard.patch`，Windows CUDA 需要）；
  - FlashInfer MSVC 兼容（`inject_minimal_patches.py` 注入 `__forceinline__`/`ushort`/`__builtin_expect`/`__shared__` 四处 shim）。
- Windows CUDA 构建禁用 DeepGemm SM120 特化内核（DeepGemm 仅支持 Linux，见 `cmake/CMakeLists.windows.txt`）。

## 目录说明

- `features/` — 纯新增文件：`ftllm.cpp` 客户端、`console.h`/`help_text.*`、`msvc_attr_shim.h`、Python 新增工具。
- `cmake/CMakeLists.windows.txt` — 最小构建适配：上游全部源文件与目标 + ftllm 客户端；构建时复制为 `CMakeLists.txt`。
- `approved/` — 已批准的最小源码修改（NCCL 守卫）。
- `catalog/` — 无法新增解决的旧修改清单（`UNRESOLVED.md`）与旧补丁存档，待决策后清理。

## 应用

```bash
bash patches/apply-patches.sh
cmake -B build -A x64 -DPY_API=ON -DUSE_SENTENCEPIECE=ON -DUNIT_TEST=OFF
cmake --build build --config Release --parallel
```
