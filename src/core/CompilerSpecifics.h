#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CPH_GLUE(a, b) a##b
#define CPH_CONCAT(a, b) CPH_GLUE(a, b)

#ifdef __cplusplus
#    include <atomic>
#    define CPH_ATOMIC(t) std::atomic<t>
#else
#    include <stdatomic.h>
#    define CPH_ATOMIC(t) _Atomic(t)
#endif

#if defined(__cplusplus)
using CPH_u23 = uint32_t;
using CPH_u1  = uint8_t;
using CPH_u5  = uint8_t;
#elif (defined(__GNUC__) && __GNUC__ < 14)
typedef uint32_t CPH_u23;
typedef uint8_t CPH_u1;
typedef uint8_t CPH_u5;
#else
// Native C23
typedef unsigned _BitInt(23) CPH_u23;
typedef unsigned _BitInt(1) CPH_u1;
typedef unsigned _BitInt(5) CPH_u5;
#endif

// #define CULVERIN_DEBUG

// --- Compiler Hints ---
#if defined(__GNUC__) || defined(__clang__)
#    define CPH_LIKELY(x) __builtin_expect(!!(x), 1)
#    define CPH_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
// Fallback for MSVC or other compilers that don't support built-in expect
#    define CPH_LIKELY(x) (x)
#    define CPH_UNLIKELY(x) (x)
#endif

// Use restrict keyword to tell the compiler these buffers do not overlap.
// This is the single best way to enable SIMD auto-vectorization.
#ifdef _MSC_VER
// MSVC: __restrict is standard here.
#    define CPH_RESTRICT __restrict

// MSVC doesn't have a direct equivalent to builtin_assume_aligned.
// We use __assume to hint the optimizer about the pointer's alignment bits.
#    define CPH_ASSUME_ALIGNED(x, alignment)                                                      \
        (__assume(((uintptr_t)(x) & ((alignment) - 1)) == 0), (x))

#    define CPH_FORCE_INLINE __forceinline
#else
// Clang/GCC: Use the double-underscore version for maximum compatibility.
#    define CPH_RESTRICT __restrict__

// The Builtin is an expression, not an attribute.
// It "cleanses" the pointer and returns it with alignment metadata attached.
#    define CPH_ASSUME_ALIGNED(x, alignment) (__builtin_assume_aligned((x), (alignment)))

#    define CPH_FORCE_INLINE inline __attribute__((always_inline))
#endif

// Use a prefixed function to avoid collision
[[noreturn]]
static inline void CPH_Unreachable(void) {
#if defined(CULVERIN_DEBUG)
    fprintf(stderr, "Unreachable hit at %s:%d\n", __FILE__, __LINE__);
    abort();
#elif defined(_MSC_VER)
    __assume(false);
#else
    __builtin_unreachable();
#endif
}

#if defined(__clang__) || defined(__GNUC__)
// rw: 0 = read, 1 = write | locality: 3 = high (keep in L1)
#    define CPH_PREFETCH_READ(addr) __builtin_prefetch((const void *)(addr), 0, 3)
#    define CPH_PREFETCH_WRITE(addr) __builtin_prefetch((const void *)(addr), 1, 3)
#elif defined(_MSC_VER)
#    include <mmintrin.h>
// MSVC uses hints: _MM_HINT_T0 = all cache levels
#    define CPH_PREFETCH_READ(addr) _mm_prefetch((const char *)(addr), _MM_HINT_T0)
#    define CPH_PREFETCH_WRITE(addr) _mm_prefetch((const char *)(addr), _MM_HINT_T0)
#else
#    define CPH_PREFETCH_READ(addr) ((void)0)
#    define CPH_PREFETCH_WRITE(addr) ((void)0)
#endif

// --- Compiler Assume Hint ---
// Tells the compiler an expression is guaranteed to be true, allowing it to
// optimize away range checks and improve loop unrolling.
#if defined(CULVERIN_DEBUG)
#    define CPH_ASSUME(x)                                                                         \
        do {                                                                                       \
            if (!(x)) {                                                                            \
                fprintf(stderr, "Assumption failed: %s at %s:%d\n", #x, __FILE__, __LINE__);       \
                abort();                                                                           \
            }                                                                                      \
        } while (0)
#elif defined(__clang__) || defined(__GNUC__)
#    define CPH_ASSUME(x)                                                                         \
        do {                                                                                       \
            if (!(x))                                                                              \
                __builtin_unreachable();                                                           \
        } while (0)
#elif defined(_MSC_VER)
#    define CPH_ASSUME(x) __assume(x)
#else
#    define CPH_ASSUME(x) ((void)0)
#endif

// Force TSan to ignore a specific function
#if defined(__has_feature)
#    if __has_feature(thread_sanitizer)
#        define CPH_NO_TSAN __attribute__((no_sanitize("thread")))
#    else
#        define CPH_NO_TSAN
#    endif
#elif defined(__SANITIZE_THREAD__)
#    define CPH_NO_TSAN __attribute__((no_sanitize("thread")))
#else
#    define CPH_NO_TSAN
#endif

[[maybe_unused]] static constexpr size_t kMemoryAlignmentSize = 64;

#define CPH_STR_HELPER(x) #x
#define CPH_STR(x) CPH_STR_HELPER(x)

#define CPH_JOIN_HELPER(x, y) x##y
#define CPH_JOIN(x, y) CPH_JOIN_HELPER(x, y)

#if defined(__clang__)
#    define CPH_UNROLL_LOOP(n) _Pragma(CPH_STR(clang loop unroll_count(n)))
#elif defined(__GNUC__)
#    define CPH_UNROLL_LOOP(x) _Pragma(CPH_STR(GCC unroll(x)))
#else
#    define CPH_UNROLL_LOOP(x)
#endif


#ifdef __cplusplus
// In C++, use the standard decltype
#    define CPH_TYPE_OF(x) decltype(x)
#else
// In C, use C23 typeof or the GCC/Clang extension
#    if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L
#        define CPH_TYPE_OF(x) typeof(x)
#    else
// Fallback for older C or compilers with extensions
#        define CPH_TYPE_OF(x) __typeof__(x)
#    endif
#endif