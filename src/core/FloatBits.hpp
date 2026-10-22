#pragma once

#include <bit>
#include <cstdint>
#include <type_traits>

namespace CPH {

/// Bit-level finiteness test.
///
/// `std::isfinite` cannot be used in this tree. The build enables -ffast-math,
/// which implies -ffinite-math-only, and clang reports every call site with
///
///     warning: use of infinity is undefined behavior due to the currently
///     enabled floating-point options [-Wnan-infinity-disabled]
///
/// The same flag lets the optimizer assume the argument is finite, so the check
/// would also be folded to `true` and silently stop doing anything. Testing the
/// exponent bits is well-defined under every floating-point mode and survives
/// LTO, which is why this is one implementation rather than one per call site.
///
/// Callers that need a user-facing rejection use ValidateFinite() in the
/// bindings; callers that need to sanitize engine input use IsFinite() here.
[[nodiscard]] inline bool IsFinite(float value) noexcept {
    constexpr std::uint32_t kExponentMask = 0x7F800000U;
    // volatile: keep fast-math and LTO from constant-folding the bit pattern of
    // a value they believe to be finite.
    volatile std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    return (bits & kExponentMask) != kExponentMask;
}

[[nodiscard]] inline bool IsFinite(double value) noexcept {
    constexpr std::uint64_t kExponentMask = 0x7FF0000000000000ULL;
    volatile std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
    return (bits & kExponentMask) != kExponentMask;
}

template <typename T> [[nodiscard]] inline bool IsFinite(T value) noexcept {
    if constexpr (std::is_same_v<std::remove_cv_t<T>, float>) {
        return IsFinite(static_cast<float>(value));
    } else if constexpr (std::is_same_v<std::remove_cv_t<T>, double>) {
        return IsFinite(static_cast<double>(value));
    } else {
        static_assert(std::is_arithmetic_v<T>, "finite checks require an arithmetic type");
        return true;
    }
}

} // namespace CPH
