// ============================================================
// MSVC 标准库传递包含垫片(纯新增文件,不修改上游源码)
// 上游部分头文件依赖 GCC 的传递包含(如 <thread> 间接引入 <chrono>),
// MSVC 下不成立。此垫片由 CMake 全局 /FI 强制包含,补齐缺失的标准头。
// ============================================================
#pragma once

#ifdef _MSC_VER
#include <chrono>

// 全局预定义 64 位 off_t(优先于 UCRT <sys/types.h> 的 32 位 long 版本),
// 保证 diskdevice 等文件在 Windows 上支持 >2GB 偏移
#ifndef _OFF_T_DEFINED
typedef long long off_t;
#define _OFF_T_DEFINED
#endif
#endif
