// ============================================================
// MSVC 平台垫片(纯新增文件,不修改上游源码)
// 上游 amx.cpp 无条件 #include <sys/syscall.h>;MSVC 无此系统头,
// 但实际 syscall 调用位于 #if defined(__AMX_TILE__) 内(MSVC 不定义),
// 因此提供一个空头即可。GCC/Clang 下透传系统头保持 Linux 行为。
// ============================================================
#pragma once

#if defined(_MSC_VER)
// MSVC: 空实现,AMX 相关代码已被 __AMX_TILE__ 排除
#else
#include_next <sys/syscall.h>
#endif
