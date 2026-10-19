#pragma once

// Compiler intrinsics that have no portable C++23 spelling yet. Everything
// that the language can now express (atomics, alignment, unreachable,
// assume, byte-width integer types) lives in the language, not here.

#include <cstddef>
#include <cstdlib>
#include <utility>


namespace CPH {

#define CPH_GLUE(a, b) a##b
#define CPH_CONCAT(a, b) CPH_GLUE(a, b)

// Branch-prediction hints on *expressions*. [[likely]]/[[unlikely]] are
// statement attributes, so the builtin still needs a macro.
#if defined(__GNUC__) || defined(__clang__)
#    define CPH_LIKELY(x) __builtin_expect(!!(x), 1)
#    define CPH_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#    define CPH_LIKELY(x) (x)
#    define CPH_UNLIKELY(x) (x)
#endif

// `restrict` is the single best lever for SIMD auto-vectorization of the SoA
// shadow-buffer loops: it promises the buffers do not overlap. C++ has no
// standard spelling; this is the compiler's.
#ifdef _MSC_VER
#    define CPH_RESTRICT __restrict
#else
#    define CPH_RESTRICT __restrict__
#endif

[[noreturn]] inline void Unreachable() {
#if defined(CULVERIN_DEBUG)
    std::abort();
#else
    std::unreachable();
#endif
}

// Disable ThreadSanitizer instrumentation for one function. Used at Jolt
// callback boundaries where Jolt's own synchronization is invisible to TSan.
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

inline constexpr std::size_t kMemoryAlignmentSize = 64;
inline constexpr std::size_t kAvxAlignment        = 32;

#define CPH_STR_HELPER(x) #x
#define CPH_STR(x) CPH_STR_HELPER(x)

#if defined(__clang__)
#    define CPH_UNROLL_LOOP(n) _Pragma(CPH_STR(clang loop unroll_count(n)))
#elif defined(__GNUC__)
#    define CPH_UNROLL_LOOP(n) _Pragma(CPH_STR(GCC unroll(n)))
#else
#    define CPH_UNROLL_LOOP(n)
#endif

} // namespace CPH
