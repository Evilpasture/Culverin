#pragma once

#include <stddef.h>
#include <stdlib.h>

#ifdef _WIN32
#    include <malloc.h>
#endif

// Native heap primitives for state owned by the physics core. Keep these
// allocations paired with the matching native free instead of CPython's heap.
static inline void *CPH_Alloc(size_t size) {
    return malloc(size);
}

static inline void *CPH_Calloc(size_t count, size_t size) {
    return calloc(count, size);
}

static inline void *CPH_Realloc(void *ptr, size_t size) {
    return realloc(ptr, size);
}

static inline void CPH_Free(void *ptr) {
    free(ptr);
}

static inline void *CPH_AlignedAlloc(size_t size, size_t alignment) {
    if ((alignment & (alignment - 1)) != 0 || alignment == 0) {
        return nullptr;
    }
    if (alignment < sizeof(void *)) {
        alignment = sizeof(void *);
    }
#if defined(_WIN32)
    return _aligned_malloc(size, alignment);
#else
    void *ptr = nullptr;
    return posix_memalign(&ptr, alignment, size) == 0 ? ptr : nullptr;
#endif
}

static inline void CPH_AlignedFree(void *ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

static inline void CPH_SafeFree(void **ptr) {
    if (ptr && *ptr) {
        CPH_Free(*ptr);
        *ptr = nullptr;
    }
}

static inline void CPH_SafeAlignedFree(void **ptr) {
    if (ptr && *ptr) {
        CPH_AlignedFree(*ptr);
        *ptr = nullptr;
    }
}

#define CPH_SAFE_FREE(ptr) CPH_SafeFree((void **)&(ptr))
#define CPH_SAFE_FREE_ALIGNED(ptr) CPH_SafeAlignedFree((void **)&(ptr))
