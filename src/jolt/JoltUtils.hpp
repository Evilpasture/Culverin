#pragma once

#include "jolt/JoltConfig.hpp"
#include <joltc.h>

#include <cstdint>
#include <memory>


namespace CPH {

inline constexpr std::uint32_t kJphInvalidBodyId = 0xFFFFFFFF;

// Jolt BodyID layout: [8-bit sequence | 24-bit index]; the index is masked
// to 23 bits in practice (the top bit of the low 24 is a flag).
[[nodiscard]] constexpr std::uint32_t JPH_ID_TO_INDEX(std::uint32_t id) noexcept {
    return id & 0x007FFFFF;
}

// RAII owner for JoltC objects with a `*_Destroy` function. Usage:
//   JoltPtr<JPH_Shape, JPH_Shape_Destroy> shape{...};
template <class T, auto Destroy>
struct JoltDeleter {
    void operator()(T *ptr) const noexcept {
        if (ptr != nullptr) {
            Destroy(ptr);
        }
    }
};

template <class T, auto Destroy>
using JoltPtr = std::unique_ptr<T, JoltDeleter<T, Destroy>>;

} // namespace CPH
