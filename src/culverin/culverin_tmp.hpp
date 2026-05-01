#pragma once
#include <type_traits>
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

template <size_t Factor, typename F> constexpr void UnrollLoop(size_t total, F &&f) {
    size_t i = 0;
    if (total >= Factor) {
        for (; i <= total - Factor; i += Factor) {
            Unroll<Factor>([&](auto index) -> auto { f(i + index); });
        }
    }
    // Clean up the remainder
    for (; i < total; ++i) {
        f(i);
    }
}

template <size_t N, typename F> constexpr void Repeat(F &&f) {
    [&f]<size_t... Is>(std::index_sequence<Is...>) -> auto {
        ((static_cast<void>(Is), f()), ...);
    }(std::make_index_sequence<N>{});
}

template <typename T> struct RestrictSpan {
    using element_type = T;
    using value_type   = std::remove_cv_t<T>;
    using size_type    = std::size_t;
    using pointer      = T *;
    using reference    = T &;

    pointer __restrict__ _ptr;
    size_type _len;

    // 1. Manual constructor (like std::span(ptr, len))
    [[gnu::always_inline]]
    constexpr RestrictSpan(pointer p, size_type l) noexcept
        : _ptr(p), _len(l) {}

    // Allows RestrictSpan<T> to convert to RestrictSpan<const T>
    template <typename U>
        requires std::is_convertible_v<U *, T *>
    [[gnu::always_inline]]
    constexpr RestrictSpan(const RestrictSpan<U> &other) noexcept
        : _ptr(other.data()), _len(other.size()) {}

    // 2. C-style array constructor (like std::span(T(&arr)[N]))
    template <size_type N>
    [[gnu::always_inline]]
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    constexpr RestrictSpan(T (&arr)[N]) noexcept
        : _ptr(arr), _len(N) {}

    // 3. std::array constructor (handles const/non-const element types)
    template <size_type N>
    [[gnu::always_inline]]
    constexpr RestrictSpan(std::array<std::remove_const_t<T>, N> &arr) noexcept
        : _ptr(arr.data()), _len(N) {}

    // 4. Subspan helper (Crucial for your BATCH_SIZE logic)
    [[nodiscard, gnu::always_inline]]
    constexpr auto first(size_type count) const noexcept -> RestrictSpan<T> {
        return RestrictSpan<T>(_ptr, count);
    }

    // Standard Accessors
    [[nodiscard, gnu::always_inline]] constexpr auto operator[](size_type i) const noexcept
        -> reference {
        return _ptr[i];
    }
    [[nodiscard, gnu::always_inline]] constexpr auto data() const noexcept -> pointer {
        return _ptr;
    }
    [[nodiscard, gnu::always_inline]] constexpr auto size() const noexcept -> size_type {
        return _len;
    }
    [[nodiscard, gnu::always_inline]] auto begin() const noexcept -> pointer { return _ptr; }
    [[nodiscard, gnu::always_inline]] auto end() const noexcept -> pointer { return _ptr + _len; }
};
// NOLINTNEXTLINE(modernize-avoid-c-arrays)
template <typename T, std::size_t N> RestrictSpan(T (&)[N]) -> RestrictSpan<T>;
template <typename T, std::size_t N> RestrictSpan(std::array<T, N> &) -> RestrictSpan<T>;

} // namespace CPH
