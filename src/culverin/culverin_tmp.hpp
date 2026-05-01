#pragma once
#include <utility>

// Locality hints for clarity
namespace CPH {
enum class CacheLevel : uint8_t { L1 = 3, L2 = 2, L3 = 1, stream = 0 };
enum class AccessType : uint8_t { Read = 0, Write = 1 };

template <AccessType Access = AccessType::Read, CacheLevel Level = CacheLevel::L1>
[[gnu::always_inline]] inline void Prefetch(const void *addr) noexcept {
#if defined(__clang__) || defined(__GNUC__)
    __builtin_prefetch(addr, static_cast<int>(Access), static_cast<int>(Level));
#elif defined(_MSC_VER)
    // MSVC doesn't have a direct 1:1 for __builtin_prefetch's rw param
    if constexpr (Access == AccessType::Write) {
        // PREFETCHW support is CPU-specific; T0 is the standard fallback
        _mm_prefetch(static_cast<const char *>(addr), _MM_HINT_T0);
    } else {
        if constexpr (Level == CacheLevel::L1)
            _mm_prefetch(static_cast<const char *>(addr), _MM_HINT_T0);
        else if constexpr (Level == CacheLevel::L2)
            _mm_prefetch(static_cast<const char *>(addr), _MM_HINT_T1);
        else
            _mm_prefetch(static_cast<const char *>(addr), _MM_HINT_NTA);
    }
#endif
}

template <size_t N, typename F> constexpr void Unroll(F &&f) {
    [&f]<size_t... Is>(std::index_sequence<Is...>) -> auto {
        (f(std::integral_constant<size_t, Is>{}), ...);
    }(std::make_index_sequence<N>{});
}

template <typename T, size_t N, typename F> constexpr void Unroll(F &&f) {
    constexpr size_t MAX_UNROLL = (sizeof(T) > 32) ? 4 : 8;
    constexpr size_t ActualN    = (N > MAX_UNROLL) ? MAX_UNROLL : N;

    [&f]<size_t... Is>(std::index_sequence<Is...>) -> auto {
        (f(std::integral_constant<size_t, Is>{}), ...);
    }(std::make_index_sequence<ActualN>{});
}

template <size_t N, typename F> constexpr void Repeat(F &&f) {
    [&f]<size_t... Is>(std::index_sequence<Is...>) -> auto {
        ((static_cast<void>(Is), f()), ...);
    }(std::make_index_sequence<N>{});
}

} // namespace CPH