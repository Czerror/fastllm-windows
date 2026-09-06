#!/usr/bin/env python3
# ============================================================
# 已批准最小修改的寻址注入脚本
# 通过内容定位(而非 git apply 的行号/上下文)对上游文件注入修改,
# 对上游更新更健壮; 幂等(已注入则跳过)。
# 覆盖: 02 NCCL 守卫 / 04 alivethreadpool chrono / 06 avx2 alignas / 11 cpudevice align
# 用法: python patches/tools/inject_minimal_patches.py
# ============================================================
import pathlib
import sys

# CI Windows runner 控制台默认 cp1252,强制 UTF-8 输出避免中文报错
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPO = pathlib.Path(__file__).resolve().parent.parent.parent


def load(path):
    p = REPO / path
    return p, p.read_text(encoding="utf-8")


def save(p, text):
    p.write_text(text, encoding="utf-8", newline="\n")


def inject_nccl_guard():
    p, text = load("src/devices/multicuda/fastllm-multicuda.cu")
    orig = text

    # 1) 守卫 <nccl.h> 包含
    guarded_include = "#ifndef FASTLLM_NO_NCCL\n#include <nccl.h>\n#endif"
    if guarded_include not in text:
        if "#include <nccl.h>" not in text:
            print("  ⚠ 02: 未找到 #include <nccl.h>,跳过")
            return False
        text = text.replace("#include <nccl.h>", guarded_include, 1)

    # 2) 在 NCCL 区间起始处插入 #ifndef
    marker = "// 全局变量存储通信器"
    guard_line = "#ifndef FASTLLM_NO_NCCL\n" + marker
    if marker in text and guard_line not in text:
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if line.startswith(marker):
                if i > 0 and lines[i - 1].strip() == "#ifndef FASTLLM_NO_NCCL":
                    break
                lines.insert(i, "#ifndef FASTLLM_NO_NCCL")
                break
        text = "\n".join(lines)

    # 3) 文件末尾追加降级桩实现 + #endif
    stubs_marker = "// NCCL 不可用时的桩实现"
    if stubs_marker not in text:
        stubs = '''

#else
// NCCL 不可用时的桩实现 (Windows 等平台, 与上游行为一致的降级路径)
namespace {
    size_t FastllmNcclStubDataTypeBytes(int dataType) {
        if (dataType == fastllm::DataType::FLOAT32) {
            return sizeof(float);
        } else if (dataType == fastllm::DataType::FLOAT16 || dataType == fastllm::DataType::BFLOAT16) {
            return sizeof(uint16_t);
        }
        return 0;
    }
}

bool FastllmCudaPeerAccessInit(const std::vector<int>& devices) {
    return false;
}

bool FastllmInitNccl(const std::vector<int>& devices) {
    return false;
}

bool FastllmInitNcclGraphPeer(int srcDevice, int dstDevice) {
    return false;
}

uint64_t FastllmGetNcclGeneration() {
    return 0;
}

bool FastllmNcclGraphPeerCopy(int dstDevice, void *dst, int srcDevice, const void *src, size_t bytes) {
    return false;
}

bool FastllmCanUseTP2P2PAllReduceAdd(int count, int dataType, int deviceId) {
    return false;
}

bool FastllmTryTP2P2PAllReduceAdd(void* data, void* dest, int count, int dataType, int deviceId) {
    return false;
}

void FastllmNcclBroadcast(void* data, int count, int dataType, int root, int deviceId) {
}

void FastllmNcclBroadcastFrom(void* send, void* recv, int count, int dataType, int root, int deviceId) {
    if (send == nullptr || recv == nullptr || count <= 0 || send == recv) {
        return;
    }
    size_t bytes = (size_t)count * FastllmNcclStubDataTypeBytes(dataType);
    if (bytes > 0) {
        FastllmCudaCopyFromDeviceToDevice(recv, send, bytes);
    }
}

void FastllmNcclAllReduce(void* data, void* dest, int count, int dataType, int deviceId) {
    if (data == nullptr || dest == nullptr || count <= 0 || data == dest) {
        return;
    }
    size_t bytes = (size_t)count * FastllmNcclStubDataTypeBytes(dataType);
    if (bytes > 0) {
        FastllmCudaCopyFromDeviceToDevice(dest, data, bytes);
    }
}

void FastllmNcclAllReduceNoCustom(void* data, void* dest, int count, int dataType, int deviceId) {
    FastllmNcclAllReduce(data, dest, count, dataType, deviceId);
}

void FastllmNcclReduce(void* data, void* dest, int count, int dataType, int root, int deviceId) {
    FastllmNcclAllReduce(data, dest, count, dataType, deviceId);
}
#endif
'''
        text = text.rstrip("\n") + stubs

    if text != orig:
        save(p, text)
        print("  ✓ 02: NCCL 守卫已注入")
    else:
        print("  ✓ 02: NCCL 守卫已存在")
    return True


def inject_alivethreadpool_chrono():
    p, text = load("include/devices/cpu/alivethreadpool.h")
    if "#include <chrono>" in text:
        print("  ✓ 04: chrono include 已存在")
        return True
    if "#include <vector>" not in text:
        print("  ⚠ 04: 未找到 #include <vector>,跳过")
        return False
    text = text.replace("#include <vector>", "#include <vector>\n#include <chrono>", 1)
    save(p, text)
    print("  ✓ 04: chrono include 已注入")
    return True


def inject_avx2_alignas():
    p, text = load("src/devices/cpu/avx2.cpp")
    old = "        int16_t values[16] __attribute__((aligned(32)));"
    new = "        alignas(32) int16_t values[16];"
    if new in text:
        print("  ✓ 06: avx2 alignas 已存在")
        return True
    if old not in text:
        print("  ⚠ 06: 未找到 __attribute__((aligned(32))) 目标行,跳过")
        return False
    text = text.replace(old, new, 1)
    save(p, text)
    print("  ✓ 06: avx2 alignas 已注入")
    return True


def inject_cpudevice_align():
    p, text = load("src/devices/cpu/cpudevice.cpp")
    old = "            taskStates[i] = new (std::align_val_t{64}) TaskState();"
    new = (
        "            void *taskStateMem = ::operator new(sizeof(TaskState), std::align_val_t{64});\n"
        "            taskStates[i] = new (taskStateMem) TaskState();"
    )
    if "taskStateMem = ::operator new" in text:
        print("  ✓ 11: cpudevice 对齐注入已存在")
        return True
    if old not in text:
        print("  ⚠ 11: 未找到 aligned placement-new 目标行,跳过")
        return False
    text = text.replace(old, new, 1)
    save(p, text)
    print("  ✓ 11: cpudevice 对齐注入已完成")
    return True


def inject_flashinfer_msvc_compat():
    # FlashInfer 头文件在 MSVC/nvcc 下的最小兼容注入(仅 Windows 生效, Linux 行为不变):
    #   1) FLASHINFER_INLINE 的 __attribute__((always_inline)) -> __forceinline__
    #   2) math.cuh 的 ushort 类型
    #   3) topk.cuh 的 __builtin_expect
    #   4) attention/decode.cuh 的 __attribute__((shared)) -> __shared__
    # 内容寻址、幂等; dev 树保持纯上游, 构建时才注入。
    ok = True

    p, text = load("third_party/flashinfer/vec_dtypes.cuh")
    old = "#define FLASHINFER_INLINE inline __attribute__((always_inline)) __device__"
    new = (
        "#if defined(_MSC_VER)\n"
        "#define FLASHINFER_INLINE inline __forceinline__ __device__\n"
        "#else\n"
        "#define FLASHINFER_INLINE inline __attribute__((always_inline)) __device__\n"
        "#endif"
    )
    if "#define FLASHINFER_INLINE inline __forceinline__ __device__" not in text:
        if old not in text:
            print("  ⚠ FI1: 未找到 FLASHINFER_INLINE 定义,跳过")
            ok = False
        else:
            save(p, text.replace(old, new, 1))
            print("  ✓ FI1: FLASHINFER_INLINE MSVC 兼容已注入")
    else:
        print("  ✓ FI1: FLASHINFER_INLINE 兼容已存在")

    p, text = load("third_party/flashinfer/math.cuh")
    if "FLASHINFER_MSVC_USHORT" not in text:
        marker = "#include <cstdint>"
        if marker not in text:
            print("  ⚠ FI2: 未找到 #include <cstdint>,跳过")
            ok = False
        else:
            insert = (
                "\n#if defined(_MSC_VER) && !defined(FLASHINFER_MSVC_USHORT)\n"
                "typedef unsigned short ushort;\n"
                "#define FLASHINFER_MSVC_USHORT\n"
                "#endif\n"
            )
            save(p, text.replace(marker, marker + insert, 1))
            print("  ✓ FI2: ushort typedef 已注入")
    else:
        print("  ✓ FI2: ushort typedef 已存在")

    p, text = load("third_party/flashinfer/topk.cuh")
    if "FLASHINFER_MSVC_BUILTIN_EXPECT" not in text:
        marker = '#include "vec_dtypes.cuh"'
        if marker not in text:
            print('  ⚠ FI3: 未找到 #include "vec_dtypes.cuh",跳过')
            ok = False
        else:
            insert = (
                "\n#if defined(_MSC_VER) && !defined(FLASHINFER_MSVC_BUILTIN_EXPECT)\n"
                "#define __builtin_expect(x, expected) (x)\n"
                "#define FLASHINFER_MSVC_BUILTIN_EXPECT\n"
                "#endif\n"
            )
            save(p, text.replace(marker, marker + insert, 1))
            print("  ✓ FI3: __builtin_expect 兼容已注入")
    else:
        print("  ✓ FI3: __builtin_expect 兼容已存在")

    p, text = load("third_party/flashinfer/attention/decode.cuh")
    old = "  extern __attribute__((shared)) uint8_t smem[];"
    new = (
        "#if defined(_MSC_VER)\n"
        "  extern __shared__ uint8_t smem[];\n"
        "#else\n"
        "  extern __attribute__((shared)) uint8_t smem[];\n"
        "#endif"
    )
    injected_block = "#if defined(_MSC_VER)\n  extern __shared__ uint8_t smem[];\n#else"
    if injected_block not in text:
        if old not in text:
            print("  ⚠ FI4: 未找到 __attribute__((shared)) 声明,跳过")
            ok = False
        else:
            save(p, text.replace(old, new, 1))
            print("  ✓ FI4: __shared__ 兼容已注入")
    else:
        print("  ✓ FI4: __shared__ 兼容已存在")

    return ok


def inject_attention_no_flashinfer_windows():
    # Windows/MSVC 下禁用 FlashInfer attention 路径:
    # vendored FlashInfer 的 mla.cuh 等头文件与 Windows nvcc 存在模板解析不兼容,
    # 且上游 fastllm-attention.cu 已内置原生分页注意力回退(#ifndef 分支)。
    # 仅 Windows 生效, Linux 行为不变; 采样路径(sampling.cuh)仍启用。
    p, text = load("src/devices/cuda/attention/fastllm-attention.cu")
    old = (
        "#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 700)\n"
        "#define FASTLLM_ENABLE_FLASHINFER\n"
        "#endif"
    )
    new = (
        "#if !defined(_WIN32) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 700))\n"
        "#define FASTLLM_ENABLE_FLASHINFER\n"
        "#endif"
    )
    if "#if !defined(_WIN32) && (!defined(__CUDA_ARCH__)" not in text:
        if old not in text:
            print("  ⚠ FI5: 未找到 FASTLLM_ENABLE_FLASHINFER 定义,跳过")
            return False
        save(p, text.replace(old, new, 1))
        print("  ✓ FI5: Windows 禁用 FlashInfer attention 已注入")
    else:
        print("  ✓ FI5: Windows 禁用 FlashInfer attention 已存在")
    return True


def inject_host_msvc_compat():
    # cl.exe 宿主编译兼容(仅 Windows/MSVC 生效):
    #   1) basellm.cpp: MSVC x64 不支持 __int128 -> long double(80位, 精度足够)
    #   2) multicudadevice.cpp: strcasecmp -> _stricmp
    #   3) cudadevice.cpp: 上游把 CUTLASS/Triton FP8 实现整体 #if !defined(_WIN32)
    #      排除, 但调用点未同步守卫 -> 在 #else 分支补同名 Windows 桩(返回 false,
    #      走 native FP8 路径); 同时修正 TryCudaTritonChunkGdnPrefill 桩签名
    #      (上游桩仍是 8 参, 调用点已改为 10 参)。
    ok = True

    p, text = load("src/models/basellm.cpp")
    if "__int128" in text:
        save(p, text.replace("__int128", "long double"))
        print("  ✓ H1: basellm.cpp __int128 -> long double")
    else:
        print("  ✓ H1: basellm.cpp __int128 已替换")

    p, text = load("src/devices/multicuda/multicudadevice.cpp")
    marker = "#include <cstring>"
    if "#define strcasecmp _stricmp" not in text:
        if marker not in text:
            print("  ⚠ H2: 未找到 #include <cstring>,跳过")
            ok = False
        else:
            insert = (
                "\n#if defined(_MSC_VER) && !defined(strcasecmp)\n"
                "#define strcasecmp _stricmp\n"
                "#endif\n"
            )
            save(p, text.replace(marker, marker + insert, 1))
            print("  ✓ H2: multicudadevice.cpp strcasecmp -> _stricmp")
    else:
        print("  ✓ H2: multicudadevice.cpp strcasecmp 已替换")

    p, text = load("src/devices/cuda/cudadevice.cpp")
    anchor = (
        "    static bool TryCudaTritonLinearFp8Block128(\n"
        "        Data &, Data &, const Data &, Data &, int, int, int) {\n"
        "        return false;\n"
        "    }\n"
    )
    stubs = (
        "#if defined(_WIN32)\n"
        "    static bool TryCudaCutlassLinearFp8Block128(\n"
        "        Data &, Data &, const Data &, Data &, int, int, int) {\n"
        "        return false;\n"
        "    }\n"
        "\n"
        "    static bool TryCudaCutlassLinearFp8PerChannel(\n"
        "        Data &, Data &, const Data &, Data &, int, int, int) {\n"
        "        return false;\n"
        "    }\n"
        "#endif // defined(_WIN32)\n"
        "\n"
    )
    if "// Windows 桩: CUTLASS FP8" not in text:
        if anchor not in text:
            print("  ⚠ H3: 未找到 Triton FP8 桩锚点,跳过")
            ok = False
        else:
            text = text.replace(anchor, "// Windows 桩: CUTLASS FP8\n" + stubs + anchor, 1)
            print("  ✓ H3: cudadevice.cpp Windows CUTLASS FP8 桩已注入")
    else:
        print("  ✓ H3: cudadevice.cpp Windows CUTLASS FP8 桩已存在")

    old_triton = (
        "    static bool TryCudaTritonChunkGdnPrefill(\n"
        "        Data &, Data &, Data &, Data &, Data &, Data &, Data &, Data &) {\n"
        "        return false;\n"
        "    }\n"
    )
    new_triton = (
        "    static bool TryCudaTritonChunkGdnPrefill(\n"
        "        Data &, Data &, Data &, Data &, Data &, Data *, Data &, Data &, Data &, bool) {\n"
        "        return false;\n"
        "    }\n"
    )
    if new_triton not in text:
        if old_triton not in text:
            print("  ⚠ H4: 未找到 8 参 TryCudaTritonChunkGdnPrefill 桩,跳过")
            ok = False
        else:
            save(p, text.replace(old_triton, new_triton, 1))
            print("  ✓ H4: TryCudaTritonChunkGdnPrefill 桩签名已修正")
    else:
        print("  ✓ H4: TryCudaTritonChunkGdnPrefill 桩签名已存在")

    return ok


def main():
    print("[inject] 寻址注入已批准的最小源码修改...")
    results = [
        inject_nccl_guard(),
        inject_alivethreadpool_chrono(),
        inject_avx2_alignas(),
        inject_cpudevice_align(),
        inject_flashinfer_msvc_compat(),
        inject_attention_no_flashinfer_windows(),
        inject_host_msvc_compat(),
    ]
    if not all(results):
        print("[inject] 存在未注入项,请检查上游文件变化")
        return 1
    print("[inject] 全部完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
