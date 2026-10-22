#pragma once

// Shared native value types. The stride layouts in this header are public
// contracts: Python exposes them zero-copy through the buffer protocol.

#include "core/Compiler.hpp"
#include "core/Handle.hpp"
#include "jolt/Jolt.hpp"
#include "jolt/JoltConfig.hpp"
#include <Jolt/Physics/Body/MotionType.h>

#include <cstdint>
#include <utility>

namespace CPH {

// --- Jolt precision ---

// Jolt's own precision alias is JPH::Real (double under JPH_DOUBLE_PRECISION,
// float otherwise). This header used to declare a project-local `JPH::Real`
// typedef in the global scope plus the `JPH_REAL_CHAR` / `CPH::kRealFormatString`
// macros, duplicating a type Jolt already defines and planting `JPH_`-prefixed
// macros outside Jolt's control. The typedef is gone in favour of JPH::Real;
// `JPH_REAL_CHAR` had no users at all and is deleted; the string form becomes
// the typed constant below, which is the only remaining caller. No `JPH_*`
// macro is defined by Culverin any more.
inline constexpr const char *kRealFormatString =
    sizeof(JPH::Real) == sizeof(double) ? "d" : "f";

// --- Shadow-buffer record layouts (stride 4, SIMD-aligned) ---

// Positions: float64 in double-precision builds, w is padding (0.0).
struct PosStride {
    alignas(32) JPH::Real x;
    JPH::Real y;
    JPH::Real z;
    JPH::Real w;
};

// Rotations (x, y, z, w) and velocities (x, y, z, pad).
struct AuxStride {
    alignas(16) float x;
    float y, z, w;
};

static_assert(sizeof(PosStride) == sizeof(JPH::Real) * 4);
static_assert(sizeof(AuxStride) == sizeof(float) * 4);

// BodyHandle and ConstraintHandle are strongly-typed 64-bit generational
// handles; see core/Handle.hpp. Layout, and therefore the Python-visible
// integer, is unchanged.

// --- Public enumerations (values are part of the Python API) ---

enum class ConstraintType : std::uint8_t {
    Fixed    = 0,
    Point    = 1,
    Hinge    = 2,
    Slider   = 3,
    Distance = 4,
    Cone     = 5
};

enum class ContactEventType : std::uint8_t {
    Added     = 0,
    Persisted = 1,
    Removed   = 2
};

// Values are Jolt's own EMotionType ordinals (Static=0, Kinematic=1,
// Dynamic=2), cast to the wire-width underlying type. The cast is explicit
// because EMotionType is a scoped enum and this one is 8-bit.
enum class MotionType : std::uint8_t {
    Static    = static_cast<std::uint8_t>(JPH::EMotionType::Static),
    Kinematic = static_cast<std::uint8_t>(JPH::EMotionType::Kinematic),
    Dynamic   = static_cast<std::uint8_t>(JPH::EMotionType::Dynamic),
};

enum class ShapeType : std::uint8_t {
    Box         = 0,
    Sphere      = 1,
    Capsule     = 2,
    Cylinder    = 3,
    Plane       = 4,
    Mesh        = 5,
    Heightfield = 6,
    ConvexHull  = 7
};

enum class ObjectLayer : std::uint32_t {
    Static = 0,
    Dynamic,
    Vehicle, // For raycasts/sensors that shouldn't hit the vehicle itself
    Count
};

struct BodyCreationProps {
    float mass;
    float friction;
    float restitution;
    int isSensor;
    int useCcd;
};

struct BodyConfig {
    float mass                   = 0.0f;
    float friction               = 0.0f;
    float restitution            = 0.0f;
    int isSensor                = 0;
    int useCcd                  = 0;
    int motionType              = 0;
    std::uint32_t allowedDofs   = 0;
};

} // namespace CPH
