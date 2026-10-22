#pragma once

#include "core/FloatBits.hpp"

#include <Jolt/Jolt.h>

#include <Jolt/Math/Quat.h>

#include <cmath>

namespace CPH {

// Jolt consumes every rotation as a unit quaternion. `Quat::operator*(Vec3Arg)`
// opens with `JPH_ASSERT(IsNormalized())` (Quat.inl:398), and that overload is
// reached from `Body::SetPositionAndRotationInternal` through
// `inRotation * mShape->GetCenterOfMass()` (Body.cpp:106). The Python API takes
// raw `(x, y, z, w)` tuples, so the canonical hand-written "90 degrees around Y"
// -- (0.0, 0.7071, 0.0, 0.7071) -- is not unit: its length squared is 0.9999808,
// i.e. off by 1.9e-5, outside Jolt's 1e-5 tolerance in Vec4::IsNormalized.
//
// With Jolt's asserts compiled in (USE_ASSERTS=ON, or any build without NDEBUG:
// Core.h:562 defines JPH_DEBUG, IssueReporting.h:14 turns that into
// JPH_ENABLE_ASSERTS) that input is a hard trap inside PhysicsWorld::StepNative,
// between Body::ApplyBodyCreationSettings and SetPositionAndRotationInternal.
// With asserts compiled out it is still wrong, just silently: the centre-of-mass
// offset is scaled by |q|^2 and every rotated body lands off its requested pose.
//
// The fix belongs at the boundary, not in Jolt: sanitize the value once, where
// an API-supplied quaternion becomes engine state.
//
// Already-unit values (within the fast-path tolerance, ten times tighter than
// Jolt's) are returned untouched, so bit patterns -- and therefore simulation
// results and the oracle hashes -- are unchanged for every well-formed input.
[[nodiscard]] inline auto SanitizeRotation(const JPH::Quat &rotation) noexcept -> JPH::Quat {
    const float length_sq = rotation.LengthSq();

    if (!IsFinite(length_sq) || length_sq <= 1.0e-12F) {
        // Degenerate input (all-zero or non-finite): fall back to a well-formed
        // rotation rather than propagate NaNs through every later frame. Same
        // policy as the plane-normal sanitizer in InternalQuery.cpp.
        return JPH::Quat::sIdentity();
    }

    if (std::abs(length_sq - 1.0F) <= 1.0e-6F) {
        return rotation;
    }

    return rotation.Normalized();
}

} // namespace CPH
