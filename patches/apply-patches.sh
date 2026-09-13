#!/usr/bin/env bash
# ============================================================
# fastllm-windows 补丁应用脚本(最小化)
# 架构: dev = 纯上游 + 本目录新增文件,不修改上游核心源码
#   1) 复制 ftllm 客户端与纯新增头文件/脚本
#   2) 复制最小适配 CMakeLists(保留上游全部源文件与目标)
#   3) 应用唯一已批准的源码修改: NCCL 守卫(Windows CUDA)
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "  fastllm-windows Patch System (minimal)"
echo "=========================================="
echo "Project root: $PROJECT_ROOT"
echo ""

echo "[1/5] Installing ftllm client + additive headers..."
mkdir -p include/utils
cp patches/features/ftllm.cpp ftllm.cpp
cp patches/features/include/utils/console.h include/utils/console.h
cp patches/features/include/utils/help_text.h include/utils/help_text.h
cp patches/features/include/utils/help_text.json include/utils/help_text.json
cp patches/features/include/utils/inference_stats.h include/utils/inference_stats.h
cp patches/features/include/utils/msvc_attr_shim.h include/utils/msvc_attr_shim.h
cp patches/features/include/utils/msvc_std_shim.h include/utils/msvc_std_shim.h
mkdir -p include/sys
cp patches/features/include/sys/syscall.h include/sys/syscall.h
cp patches/features/include/sys/mman.h include/sys/mman.h
cp patches/features/include/unistd.h include/unistd.h
mkdir -p runtime_libs/vcrt
cp patches/runtime-libs/vcrt/*.dll runtime_libs/vcrt/
echo "  ✓ ftllm.cpp / console.h / help_text.* / msvc 垫片 / VC++ 运行库"

echo "[2/5] Installing Python additive files..."
mkdir -p tools/fastllm_pytools
cp patches/features/tools/fastllm_pytools/console.py tools/fastllm_pytools/console.py
cp patches/features/tools/fastllm_pytools/help_text.py tools/fastllm_pytools/help_text.py
cp patches/features/tools/fastllm_pytools/__main__.py tools/fastllm_pytools/__main__.py
cp patches/features/tools/fastllm_pytools/requirements.txt tools/fastllm_pytools/requirements.txt
mkdir -p pyfastllm/fastllm/utils
cp patches/features/pyfastllm/fastllm/utils/log_handler.py pyfastllm/fastllm/utils/log_handler.py
mkdir -p docs
cp patches/docs/使用说明.md docs/使用说明.md
echo "  ✓ python 新增文件"

echo "[3/5] Installing minimal CMakeLists..."
cp patches/cmake/CMakeLists.windows.txt CMakeLists.txt
echo "  ✓ CMakeLists.txt (最小构建适配,保留上游全部源文件)"

echo "[4/4] Injecting approved minimal source guards (内容寻址,不依赖行号/上下文)..."
python patches/tools/inject_minimal_patches.py

echo ""
echo "=========================================="
echo "  All additive files installed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  Build: cmake -B build -A x64 -DPY_API=ON -DUSE_SENTENCEPIECE=OFF -DUNIT_TEST=OFF"
echo "  Or on Windows: .\\build.ps1"
echo ""
