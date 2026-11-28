// ============================================================
// MSVC unistd.h 兼容垫片(纯新增文件,不修改上游源码)
// 提供 open/close/pread 的 Windows 等价实现。
// Linux/其他平台透传系统头(include_next)。
// ============================================================
#pragma once

#if defined(_MSC_VER)

#include <io.h>
#include <fcntl.h>
#include <cstddef>

#ifndef _SSIZE_T_DEFINED
typedef long long ssize_t;
#define _SSIZE_T_DEFINED
#endif

#ifndef open
#define open _open
#endif
#ifndef close
#define close _close
#endif

inline ssize_t pread(int fd, void *buf, size_t count, long long offset) {
    if (_lseeki64(fd, offset, SEEK_SET) < 0) {
        return -1;
    }
    return _read(fd, buf, (unsigned int)count);
}

#else
#include_next <unistd.h>
#endif
