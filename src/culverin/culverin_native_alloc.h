#pragma once

#include <stddef.h>
#include <stdlib.h>

#ifdef _WIN32
#    include <malloc.h>
#endif

// Native heap primitives for state owned by the physics core. Keep these
// allocations paired with the matching native free instead of CPython's heap.
static inline void *CulvNative_Malloc(size_t size) {
    return malloc(size);
}

static inline void *CulvNative_Calloc(size_t count, size_t size) {
    return calloc(count, size);
}

static inline void *CulvNative_Realloc(void *ptr, size_t size) {
    return realloc(ptr, size);
}

static inline void CulvNative_Free(void *ptr) {
    free(ptr);
}

static inline void *CulvNative_AlignedMalloc(size_t size, size_t alignment) {
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

static inline void CulvNative_AlignedFree(void *ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

static inline void CulvNative_SafeFree(void **ptr) {
    if (ptr && *ptr) {
        CulvNative_Free(*ptr);
        *ptr = nullptr;
    }
}

static inline void CulvNative_SafeAlignedFree(void **ptr) {
    if (ptr && *ptr) {
        CulvNative_AlignedFree(*ptr);
        *ptr = nullptr;
    }
}

#define CULVERIN_SAFE_FREE(ptr) CulvNative_SafeFree((void **)&(ptr))
#define CULVERIN_SAFE_FREE_ALIGNED(ptr) CulvNative_SafeAlignedFree((void **)&(ptr))
