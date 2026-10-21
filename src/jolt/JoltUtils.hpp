#pragma once

#include "jolt/Jolt.hpp"
#include "jolt/JoltConfig.hpp"
#include <joltc.h>

#include <cstdint>
#include <memory>


namespace CPH {

// Body IDs use Jolt's own constants (JPH::BodyID::cInvalidBodyID /
// cMaxBodyIndex); there is no C-binding equivalent to mirror, so the C
// interface must not re-declare them.

// The JoltC value types (JPH_Vec3/JPH_RVec3/JPH_Quat) are gone from the C++
// paths: every payload is a native JPH::Vec3/JPH::RVec3/JPH::Quat now, so the
// field-by-field ToJolt() shim that bridged the two had no callers left. They
// were never layout-compatible (12 bytes vs 16, 24 vs 32), which is exactly why
// the bridge had to copy component-wise rather than reinterpret_cast.
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
