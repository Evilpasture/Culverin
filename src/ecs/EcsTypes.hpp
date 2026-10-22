#pragma once

// Vocabulary shared by the native ECS core and anything that embeds it.
// Deliberately free of <Python.h>: the ECS is a standalone C++ data
// structure, and the Python layer is only one possible front end for it.

#include "core/Error.hpp"
#include "core/Handle.hpp"
#include <cstdint>
#include <string_view>

namespace CPH {

// A 64-bit entity handle: [32-bit generation | 32-bit index]. Same class
// template as BodyHandle/ConstraintHandle (core/Handle.hpp), so entity
// handles pack, validate and compare exactly the way body handles do.
using Entity = GenerationalHandle<EntityTag>;

// Errors raised by the ECS data structures. No zero enumerator, as
// CPH::Error reserves 0 for "no error".
enum class EcsError : std::uint8_t {
    OutOfMemory = 1,
    ComponentResizeWhileViewed,
    RegisterWhileViewed,
    ClearWhileViewed,
    InvalidEntity,
    InvalidComponent,
    ComponentSizeMismatch,
    HandleComponentSize,
    PositionComponentSize,
    RotationComponentSize,
    InvalidHandleComponent,
};

// These strings are observable: the Python binding raises them verbatim, and
// they were the wording of the pre-migration binding. Keep them byte-identical
// (they are found by ADL from CPH::Error, so they also show up in any native
// diagnostic that prints a failure).
inline std::string_view ErrorMessage(EcsError error) noexcept {
    switch (error) {
    case EcsError::OutOfMemory:
        return "out of memory";
    case EcsError::ComponentResizeWhileViewed:
        return "Cannot resize ECS component while a memoryview is held. "
               "Delete the array or view before adding more entities.";
    case EcsError::RegisterWhileViewed:
        return "Cannot register new components while a view is held.";
    case EcsError::ClearWhileViewed:
        return "Cannot clear registry while memoryviews are exported.";
    case EcsError::InvalidEntity:
        return "Invalid or stale entity handle";
    case EcsError::InvalidComponent:
        return "Invalid component ID";
    // The binding formats the full "expected N, got M" wording, because the
    // sizes are known only at the Python buffer boundary.
    case EcsError::ComponentSizeMismatch:
        return "Data size mismatch";
    case EcsError::HandleComponentSize:
        return "Handle component must be 8 bytes (uint64)";
    case EcsError::PositionComponentSize:
        return "Position component must be 12 bytes (3x float32)";
    case EcsError::RotationComponentSize:
        return "Rotation component must be 16 bytes (4x float32)";
    case EcsError::InvalidHandleComponent:
        return "Invalid handle component ID";
    }
    return "unknown ECS error";
}

} // namespace CPH
