#pragma once

// Shared native value types. The stride layouts in this header are public
// contracts: Python exposes them zero-copy through the buffer protocol.

#include "core/Compiler.hpp"
#include "jolt/JoltConfig.hpp"
#include <joltc.h>

#include <cstdint>
#include <utility>

// --- Jolt precision compatibility ---
#ifndef JPH_Real
#    ifdef JPH_DOUBLE_PRECISION
using JPH_Real = double;
#        define JPH_REAL_CHAR 'd'
#        define JPH_REAL_STRING "d"
#    else
using JPH_Real = float;
#        define JPH_REAL_CHAR 'f'
#        define JPH_REAL_STRING "f"
#    endif
#endif


namespace CPH {

// --- Shadow-buffer record layouts (stride 4, SIMD-aligned) ---

// Positions: float64 in double-precision builds, w is padding (0.0).
struct PosStride {
    alignas(32) JPH_Real x;
    JPH_Real y;
    JPH_Real z;
    JPH_Real w;
};

// Rotations (x, y, z, w) and velocities (x, y, z, pad).
struct AuxStride {
    alignas(16) float x;
    float y, z, w;
};

static_assert(sizeof(PosStride) == sizeof(JPH_Real) * 4);
static_assert(sizeof(AuxStride) == sizeof(float) * 4);

// --- Generational handles: (generation << 32) | slot index ---
using BodyHandle       = std::uint64_t;
using ConstraintHandle = std::uint64_t;

inline constexpr std::uint32_t kHandleIndexBits = 32;
inline constexpr std::uint64_t kHandleIndexMask = 0xFFFFFFFF;

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

enum class MotionType : std::uint8_t {
    Static    = JPH_MotionType_Static,
    Kinematic = JPH_MotionType_Kinematic,
    Dynamic   = JPH_MotionType_Dynamic,
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
    int is_sensor;
    int use_ccd;
};

struct BodyConfig {
    float mass                   = 0.0f;
    float friction               = 0.0f;
    float restitution            = 0.0f;
    int is_sensor                = 0;
    int use_ccd                  = 0;
    int motion_type              = 0;
    std::uint32_t allowed_dofs   = 0;
};

} // namespace CPH
