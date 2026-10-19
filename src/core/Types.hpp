#pragma once

// Shared native value types. The stride layouts in this header are public
// contracts: Python exposes them zero-copy through the buffer protocol.

#include "core/Compiler.hpp"
#include "jolt/JoltConfig.hpp"
#include <joltc.h>

#include <cstdint>

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

struct Vec3f {
    float x;
    float y;
    float z;
};

static_assert(sizeof(PosStride) == sizeof(JPH_Real) * 4);
static_assert(sizeof(AuxStride) == sizeof(float) * 4);

// --- Generational handles: (generation << 32) | slot index ---
using BodyHandle       = std::uint64_t;
using ConstraintHandle = std::uint64_t;

inline constexpr std::uint32_t kHandleIndexBits = 32;
inline constexpr std::uint64_t kHandleIndexMask = 0xFFFFFFFF;

// --- Public enumerations (values are part of the Python API) ---

enum ConstraintType : std::uint8_t {
    CONSTRAINT_FIXED    = 0,
    CONSTRAINT_POINT    = 1,
    CONSTRAINT_HINGE    = 2,
    CONSTRAINT_SLIDER   = 3,
    CONSTRAINT_DISTANCE = 4,
    CONSTRAINT_CONE     = 5
};

enum ContactEventType : std::uint8_t {
    EVENT_ADDED     = 0,
    EVENT_PERSISTED = 1,
    EVENT_REMOVED   = 2
};

enum MotionType : std::uint8_t {
    MOTION_STATIC    = JPH_MotionType_Static,
    MOTION_KINEMATIC = JPH_MotionType_Kinematic,
    MOTION_DYNAMIC   = JPH_MotionType_Dynamic,
};

enum ShapeType : std::uint8_t {
    SHAPE_BOX         = 0,
    SHAPE_SPHERE      = 1,
    SHAPE_CAPSULE     = 2,
    SHAPE_CYLINDER    = 3,
    SHAPE_PLANE       = 4,
    SHAPE_MESH        = 5,
    SHAPE_HEIGHTFIELD = 6,
    SHAPE_CONVEX_HULL = 7
};

enum ObjectLayer : std::uint32_t {
    OBJECT_LAYER_STATIC = 0,
    OBJECT_LAYER_DYNAMIC,
    OBJECT_LAYER_VEHICLE, // For raycasts/sensors that shouldn't hit the vehicle itself
    OBJECT_LAYER_COUNT
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
