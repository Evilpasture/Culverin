#pragma once
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// Comment this line out to disable all debug prints
#define CULVERIN_DEBUG

// --- Compiler Hints ---
#if defined(__GNUC__) || defined(__clang__)
#    define LIKELY(x) __builtin_expect(!!(x), 1)
#    define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
// Fallback for MSVC or other compilers that don't support built-in expect
#    define LIKELY(x) (x)
#    define UNLIKELY(x) (x)
#endif

// Use restrict keyword to tell the compiler these buffers do not overlap.
// This is the single best way to enable SIMD auto-vectorization.
#ifdef _MSC_VER
#    define CULV_RESTRICT __restrict
// MSVC doesn't have a direct equivalent to assume_aligned,
// though __assume( ((intptr_t)x & 31) == 0 ) is sometimes used.
#    define CULV_ASSUME_ALIGNED(x, alignment) (x)
#    define CULV_FORCE_INLINE __forceinline
#else
#    define CULV_RESTRICT __restrict__
#    define CULV_ASSUME_ALIGNED(x, alignment) __builtin_assume_aligned((x), (alignment))
#    define CULV_FORCE_INLINE inline __attribute__((always_inline))
#endif

// Use a prefixed function to avoid collision
[[noreturn]]
static inline void culv_unreachable(void) {
#if defined(CULVERIN_DEBUG)
    fprintf(stderr, "Unreachable hit at %s:%d\n", __FILE__, __LINE__);
    abort();
#elif defined(_MSC_VER)
    __assume(0);
#else
    __builtin_unreachable();
#endif
}

// #define CULVERIN_PROFILE_SYNC

#ifdef CULVERIN_PROFILE_SYNC
#    include <stdio.h>
static inline uint64_t rdtsc() {
#    ifdef _MSC_VER
    return __rdtsc();
#    elif defined(__aarch64__) || defined(__arm64__)
    // ARM64 / Apple Silicon equivalent
    uint64_t val;
    __asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(val));
    return val;
#    else
    // Original x86 block
    uint32_t lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
#    endif
}
#endif

#if defined(__clang__) || defined(__GNUC__)
// __builtin_prefetch(addr, rw, locality)
// rw: 0 = read, 1 = write
// locality: 0 = none, 3 = high (keep in L1)
#    define CULV_PREFETCH(addr) __builtin_prefetch((const void *)(addr), 0, 3)
#    define CULV_PREFETCH_WRITE(addr) __builtin_prefetch((const void *)(addr), 1, 3)
#elif defined(_MSC_VER)
#    include <mmintrin.h>
// _mm_prefetch(addr, hint)
// _MM_HINT_T0 = Prefetch into all cache levels (L1, L2, L3)
#    define CULV_PREFETCH(addr) _mm_prefetch((const char *)(addr), _MM_HINT_T0)
#    define CULV_PREFETCH_WRITE(addr) _mm_prefetch((const char *)(addr), _MM_HINT_T0)
#else
// Fallback for compilers that don't support prefetching
#    define CULV_PREFETCH(addr) ((void)0)
#    define CULV_PREFETCH_WRITE(addr) ((void)0)
#endif

// Use a nested check to avoid the "macro not defined" evaluation error
#if defined(__has_c_attribute)
#    if __has_c_attribute(nodiscard)
#        define CULV_NODISCARD [[nodiscard]]
#        define CULV_MAYBE_UNUSED [[maybe_unused]]
#    else
#        define CULV_NODISCARD
#        define CULV_MAYBE_UNUSED
#    endif
#elif defined(_MSC_VER)
#    define CULV_NODISCARD _Check_return_
#    define CULV_MAYBE_UNUSED
#elif defined(__GNUC__) || defined(__clang__)
#    define CULV_NODISCARD __attribute__((warn_unused_result))
#    define CULV_MAYBE_UNUSED __attribute__((unused))
#else
#    define CULV_NODISCARD
#    define CULV_MAYBE_UNUSED
#endif

// --- Compiler Assume Hint ---
// Tells the compiler an expression is guaranteed to be true, allowing it to
// optimize away range checks and improve loop unrolling.
#if defined(__clang__) || defined(__GNUC__)
#   define CULV_ASSUME(x) do { if (!(x)) __builtin_unreachable(); } while (0)
#elif defined(_MSC_VER)
#   define CULV_ASSUME(x) __assume(x)
#else
#   define CULV_ASSUME(x) ((void)0)
#endif

// Force TSan to ignore a specific function
#if defined(__has_feature)
#  if __has_feature(thread_sanitizer)
#    define CULV_NO_TSAN __attribute__((no_sanitize("thread")))
#  else
#    define CULV_NO_TSAN
#  endif
#elif defined(__SANITIZE_THREAD__)
#  define CULV_NO_TSAN __attribute__((no_sanitize("thread")))
#else
#  define CULV_NO_TSAN
#endif

// NOLINTNEXTLINE(readability-identifier-naming)
#define PyCFunction_DeclareMethod CULV_NODISCARD static PyObject *
// NOLINTNEXTLINE(readability-identifier-naming)
#define PyCFunction_DeclareMethodFromModule CULV_NODISCARD extern PyObject *
// NOLINTNEXTLINE(readability-identifier-naming)
#define PyGetSet_DeclareGetter CULV_NODISCARD extern PyObject *
// NOLINTNEXTLINE(readability-identifier-naming)
#define PyGetSet_DeclareSetter CULV_NODISCARD extern PyObject *

// For tp_alloc and factory-style slots
// NOLINTNEXTLINE(readability-identifier-naming)
#define PyType_DeclareSlot_Object CULV_NODISCARD static PyObject *
// NOLINTNEXTLINE(readability-identifier-naming)
#define PyType_DeclareSlot_ObjectFromModule CULV_NODISCARD extern PyObject *

// For tp_init and status-returning slots
// NOLINTNEXTLINE(readability-identifier-naming)
#define PyType_DeclareSlot_Status CULV_NODISCARD static int
// NOLINTNEXTLINE(readability-identifier-naming)
#define PyType_DeclareSlot_StatusFromModule CULV_NODISCARD extern int

// For tp_dealloc (Cannot be nodiscard because it returns void)
// NOLINTNEXTLINE(readability-identifier-naming)
#define PyType_DeclareSlot_Void static void
// NOLINTNEXTLINE(readability-identifier-naming)
#define PyType_DeclareSlot_VoidFromModule extern void

CULV_MAYBE_UNUSED static constexpr size_t MEMORY_ALIGNMENT_SIZE = 64;

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L
#    define CULV_REPRODUCIBLE [[reproducible]]
#    define CULV_UNSEQUENCED [[unsequenced]]
#else
#    define CULV_REPRODUCIBLE
#    define CULV_UNSEQUENCED
#endif
