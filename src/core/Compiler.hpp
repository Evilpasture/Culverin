#pragma once

// Compiler intrinsics that have no portable C++23 spelling yet. Everything
// that the language can now express (atomics, alignment, unreachable,
// assume, byte-width integer types, branch hints) lives in the language,
// not here.

#include <cstddef>
#include <cstdlib>
#include <type_traits>
#include <utility>


namespace CPH {

#define CPH_GLUE(a, b) a##b
#define CPH_CONCAT(a, b) CPH_GLUE(a, b)

// `restrict` is the single best lever for SIMD auto-vectorization of the SoA
// shadow-buffer loops: it promises the buffers do not overlap. C++ has no
// standard spelling; this is the compiler's.
#ifdef _MSC_VER
#    define CPH_RESTRICT __restrict
#else
#    define CPH_RESTRICT __restrict__
#endif

// Disable ThreadSanitizer instrumentation for one function. Used at Jolt
// callback boundaries where Jolt's own synchronization is invisible to TSan.
// Spelled as a C++ attribute rather than __attribute__(()). The vendor
// qualifier matters: GCC 14 silently IGNORES clang::no_sanitize (with a
// -Wattributes warning) and only honours gnu::no_sanitize, so picking the
// wrong one here would quietly re-instrument the annotated function.
#if defined(__has_cpp_attribute)
#    if __has_cpp_attribute(clang::no_sanitize)
#        define CPH_NO_TSAN [[clang::no_sanitize("thread")]]
#    elif __has_cpp_attribute(gnu::no_sanitize)
#        define CPH_NO_TSAN [[gnu::no_sanitize("thread")]]
#    else
#        define CPH_NO_TSAN
#    endif
#else
#    define CPH_NO_TSAN
#endif

inline constexpr std::size_t kMemoryAlignmentSize = 64;
inline constexpr std::size_t kAvxAlignment        = 32;

// --- Compile-time loop unrolling -------------------------------------------------------------
// Replaces the CPH_UNROLL_LOOP pragma macro. A pragma asks the compiler to
// unroll and hopes; these expand the iterations in the language, so the
// unrolling is guaranteed and visible to the optimiser as ordinary code.

/// Call `f` once per index in [0, N), passing std::integral_constant<size_t, I>
/// so the index is a compile-time constant inside the body.
template <size_t N, typename F>
constexpr void Unroll(F &&f) {
    [&f]<size_t... Is>(std::index_sequence<Is...>) -> auto {
        (std::forward<F>(f)(std::integral_constant<size_t, Is> {}), ...);
    }(std::make_index_sequence<N> {});
}

/// As above, but caps the unroll factor by the element size: wider elements
/// get a smaller factor so the unrolled body does not blow out the I-cache.
template <typename T, size_t N, typename F>
constexpr void Unroll(F &&f) {
    constexpr size_t max_unroll = (sizeof(T) > 32) ? 4 : 8;
    constexpr size_t actual_n   = (N > max_unroll) ? max_unroll : N;

    [&f]<size_t... Is>(std::index_sequence<Is...>) -> auto {
        (std::forward<F>(f)(std::integral_constant<size_t, Is> {}), ...);
    }(std::make_index_sequence<actual_n> {});
}

/// Run `f(i)` for i in [0, total), unrolled by Factor with a scalar tail.
template <size_t Factor, typename F>
constexpr void UnrollLoop(size_t total, F &&f) {
    size_t i = 0;
    for (; i + Factor <= total; i += Factor) {
        Unroll<Factor>([&f, i](auto index) { std::forward<F>(f)(i + index); });
    }
    for (; i < total; ++i) {
        f(i);
    }
}

/// Call `f()` exactly N times.
template <size_t N, typename F>
constexpr void Repeat(F &&f) {
    [&f]<size_t... Is>(std::index_sequence<Is...>) -> auto {
        ((static_cast<void>(Is), std::forward<F>(f)()), ...);
    }(std::make_index_sequence<N> {});
}

} // namespace CPH
