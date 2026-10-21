#pragma once

#include "jolt/Jolt.hpp"
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

// JoltC's value types are NOT layout-compatible with Jolt's: JPH_Vec3 is 12
// bytes against JPH::Vec3's 16, and JPH_RVec3 is 24 against JPH::RVec3's 32.
// Convert field by field; a reinterpret_cast would read past the end.
[[nodiscard]] inline JPH::Vec3 ToJolt(const JPH_Vec3 &v) noexcept {
    return JPH::Vec3(v.x, v.y, v.z);
}
[[nodiscard]] inline JPH::RVec3 ToJolt(const JPH_RVec3 &v) noexcept {
    return JPH::RVec3(v.x, v.y, v.z);
}
[[nodiscard]] inline JPH::Quat ToJolt(const JPH_Quat &q) noexcept {
    return JPH::Quat(q.x, q.y, q.z, q.w);
}
[[nodiscard]] inline JPH::BodyID ToJoltBodyId(std::uint32_t packed) noexcept {
    return JPH::BodyID(packed);
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
