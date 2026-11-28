// ============================================================
// MSVC __attribute__ 兼容垫片(纯新增文件,不修改上游源码)
// 由 CMakeLists 仅对 avx2.cpp / avx512vnni.cpp 强制包含(/FI)。
// 这两个文件仅剩 __attribute__((always_inline)) 用法(对齐用法已由
// approved/06-avx2-msvc-align.patch 修复),MSVC 无对应 __declspec,
// 直接丢弃该提示(等价于普通 inline)。
// ============================================================
#pragma once

#ifdef _MSC_VER
#ifndef FASTLLM_MSVC_ATTR_SHIM_INCLUDED
#define FASTLLM_MSVC_ATTR_SHIM_INCLUDED

#define __attribute__(x)

#endif // FASTLLM_MSVC_ATTR_SHIM_INCLUDED
#endif // _MSC_VER
