// ============================================================
// MSVC sys/mman.h 兼容垫片(纯新增文件,不修改上游源码)
// 为 diskdevice.cpp 提供 Windows 的 mmap/munmap/posix_fadvise 实现。
// Linux/其他平台透传系统头(include_next)。
// ============================================================
#pragma once

#if defined(_MSC_VER)

#include <windows.h>
#include <io.h>
#include <cstddef>
#include <errno.h>
#include <stdlib.h>
#include <malloc.h>

#ifndef PROT_READ
#define PROT_READ 0x1
#endif
#ifndef MAP_PRIVATE
#define MAP_PRIVATE 0x2
#endif
#ifndef MAP_FAILED
#define MAP_FAILED ((void *)-1)
#endif
#ifndef POSIX_FADV_WILLNEED
#define POSIX_FADV_WILLNEED 3
#endif

inline void *mmap(void *, size_t length, int, int, int fd, long long offset) {
    HANDLE hFile = (HANDLE)_get_osfhandle(fd);
    if (hFile == INVALID_HANDLE_VALUE) {
        return MAP_FAILED;
    }
    HANDLE hMap = CreateFileMappingW(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (hMap == nullptr) {
        return MAP_FAILED;
    }
    void *p = MapViewOfFile(hMap, FILE_MAP_READ,
                            (DWORD)((unsigned long long)offset >> 32),
                            (DWORD)offset, length);
    CloseHandle(hMap);
    return p;
}

inline int munmap(void *addr, size_t) {
    return UnmapViewOfFile(addr) ? 0 : -1;
}

inline int madvise(void *, size_t, int) {
    return 0;
}

inline int posix_fadvise(int, long long, long long, int) {
    return 0;
}

#ifndef _SC_PAGESIZE
#define _SC_PAGESIZE 1
#endif

inline long sysconf(int name) {
    if (name == _SC_PAGESIZE) {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return (long)si.dwPageSize;
    }
    return -1;
}

// 真对齐实现: 使用 _aligned_malloc,并把本文件作用域内的 free 映射为 _aligned_free
// (本垫片仅被 diskdevice.cpp 包含,该文件内所有 free() 均释放 posix_memalign 分配的缓冲区)
#define free _aligned_free
inline int posix_memalign(void **memptr, size_t alignment, size_t size) {
    void *p = _aligned_malloc(size ? size : 1, alignment);
    if (p == nullptr) {
        return ENOMEM;
    }
    *memptr = p;
    return 0;
}

#else
#include_next <sys/mman.h>
#endif
