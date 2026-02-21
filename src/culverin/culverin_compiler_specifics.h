#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// Comment this line out to disable all debug prints
#define CULVERIN_DEBUG

// --- Compiler Hints ---
#if defined(__GNUC__) || defined(__clang__)
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
// Fallback for MSVC or other compilers that don't support built-in expect
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#endif

// Use restrict keyword to tell the compiler these buffers do not overlap.
// This is the single best way to enable SIMD auto-vectorization.
#ifdef _MSC_VER
  #define CULV_RESTRICT __restrict
  // MSVC doesn't have a direct equivalent to assume_aligned, 
  // though __assume( ((intptr_t)x & 31) == 0 ) is sometimes used.
  #define CULV_ASSUME_ALIGNED(x, alignment) (x) 
  #define CULV_FORCE_INLINE __forceinline
#else
  #define CULV_RESTRICT __restrict__
  #define CULV_ASSUME_ALIGNED(x, alignment) __builtin_assume_aligned((x), (alignment))
  #define CULV_FORCE_INLINE inline __attribute__((always_inline))
#endif

#if defined(CULVERIN_DEBUG)
    // Debug/Development: Deterministic Panic
    [[noreturn]] static inline void culv_panic(const char* msg, const char* file, int line) {
        fprintf(stderr, "PANIC: %s at %s:%d\n", msg, file, line);
        abort(); // Or __builtin_trap() for a debugger break
    }
    #undef unreachable
    // NOLINTNEXTLINE(readability-identifier-naming)
    #define unreachable() do { \
        printf("Unreachable hit at %s:%d\n", __FILE__, __LINE__); \
        abort(); \
    } while(0)
#else
    // Release/Production: Pure Optimization Hint
    #if defined(_MSC_VER)
        #define unreachable() __assume(0)
    #else
        #define unreachable() __builtin_unreachable()
    #endif
#endif

#define CULVERIN_PROFILE_SYNC

#ifdef CULVERIN_PROFILE_SYNC
#include <stdio.h>
static inline uint64_t rdtsc() {
#ifdef _MSC_VER
  return __rdtsc();
#elif defined(__aarch64__) || defined(__arm64__)
  // ARM64 / Apple Silicon equivalent
  uint64_t val;
  __asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(val));
  return val;
#else
  // Original x86 block
  uint32_t lo, hi;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  return ((uint64_t)hi << 32) | lo;
#endif
}
#endif

#if defined(__clang__) || defined(__GNUC__)
    // __builtin_prefetch(addr, rw, locality)
    // rw: 0 = read, 1 = write
    // locality: 0 = none, 3 = high (keep in L1)
    #define CULV_PREFETCH(addr) __builtin_prefetch((const void*)(addr), 0, 3)
    #define CULV_PREFETCH_WRITE(addr) __builtin_prefetch((const void*)(addr), 1, 3)
#elif defined(_MSC_VER)
    #include <mmintrin.h>
    // _mm_prefetch(addr, hint)
    // _MM_HINT_T0 = Prefetch into all cache levels (L1, L2, L3)
    #define CULV_PREFETCH(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T0)
    #define CULV_PREFETCH_WRITE(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T0)
#else
    // Fallback for compilers that don't support prefetching
    #define CULV_PREFETCH(addr) ((void)0)
    #define CULV_PREFETCH_WRITE(addr) ((void)0)
#endif

#if defined(__has_c_attribute) && __has_c_attribute(nodiscard)
#  define CULV_NODISCARD [[nodiscard]]
#elif defined(_MSC_VER)
#  define CULV_NODISCARD _Check_return_
#elif defined(__GNUC__) || defined(__clang__)
#  define CULV_NODISCARD __attribute__((warn_unused_result))
#else
#  define CULV_NODISCARD
#endif