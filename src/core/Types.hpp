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

// --- Shadow-buffer record layouts (stride 4, SIMD-aligned) ---

// Positions: float64 in double-precision builds, w is padding (0.0).
struct CPH_PosStride {
    alignas(32) JPH_Real x;
    JPH_Real y;
    JPH_Real z;
    JPH_Real w;
};

// Rotations (x, y, z, w) and velocities (x, y, z, pad).
struct CPH_AuxStride {
    alignas(16) float x;
    float y, z, w;
};

struct CPH_Vec3f {
    float x;
    float y;
    float z;
};

static_assert(sizeof(CPH_PosStride) == sizeof(JPH_Real) * 4);
static_assert(sizeof(CPH_AuxStride) == sizeof(float) * 4);

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

enum CPH_ShapeType : std::uint8_t {
    CPH_SHAPE_BOX         = 0,
    CPH_SHAPE_SPHERE      = 1,
    CPH_SHAPE_CAPSULE     = 2,
    CPH_SHAPE_CYLINDER    = 3,
    CPH_SHAPE_PLANE       = 4,
    CPH_SHAPE_MESH        = 5,
    CPH_SHAPE_HEIGHTFIELD = 6,
    CPH_SHAPE_CONVEX_HULL = 7
};

enum ObjectLayer : std::uint32_t {
    OBJECT_LAYER_STATIC = 0,
    OBJECT_LAYER_DYNAMIC,
    OBJECT_LAYER_VEHICLE, // For raycasts/sensors that shouldn't hit the vehicle itself
    OBJECT_LAYER_COUNT
};

struct CPH_BodyCreationProps {
    float mass;
    float friction;
    float restitution;
    int is_sensor;
    int use_ccd;
};

struct CPH_BodyConfig {
    float mass                   = 0.0f;
    float friction               = 0.0f;
    float restitution            = 0.0f;
    int is_sensor                = 0;
    int use_ccd                  = 0;
    int motion_type              = 0;
    std::uint32_t allowed_dofs   = 0;
};
