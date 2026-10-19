#pragma once
#include "jolt/JoltConfig.h"
#include "core/CompilerSpecifics.h"
#include <joltc.h>
#include <stdint.h>

// --- Jolt Precision Compatibility ---
#ifndef JPH_Real
#    ifdef JPH_DOUBLE_PRECISION
typedef double JPH_Real;
#        define JPH_REAL_CHAR 'd'
#        define JPH_REAL_STRING "d"
#    else
typedef float JPH_Real;
#        define JPH_REAL_CHAR 'f'
#        define JPH_REAL_STRING "f"
#    endif
#endif

// --- Memory Stride Helpers ---

typedef struct {
    alignas(32) JPH_Real x;
    JPH_Real y;
    JPH_Real z;
    JPH_Real w;
} CPH_PosStride;

// Maps to self->physics.rotations, velocities (Packed X, Y, Z, W)
typedef struct {
    alignas(16) float x;
    float y, z, w;
} CPH_AuxStride;

typedef struct {
    float x;
    float y;
    float z;
} CPH_Vec3f; // General Vec3f

// Sanity check sizes
static_assert(sizeof(CPH_PosStride) == sizeof(JPH_Real) * 4);
static_assert(sizeof(CPH_AuxStride) == sizeof(float) * 4);

// Minimal Handle Helper
// Python handles will be 64-bit integers: (Generation << 32) | SlotIndex
typedef uint64_t BodyHandle;

// Constraint Types
typedef enum ConstraintType : uint8_t {
    CONSTRAINT_FIXED    = 0,
    CONSTRAINT_POINT    = 1,
    CONSTRAINT_HINGE    = 2,
    CONSTRAINT_SLIDER   = 3,
    CONSTRAINT_DISTANCE = 4,
    CONSTRAINT_CONE     = 5
} ConstraintType;

// Minimal Handle for Constraints (Distinct from BodyHandle)
typedef uint64_t ConstraintHandle;

// --- Contact Lifecycle Types ---
typedef enum ContactEventType : uint8_t {
    EVENT_ADDED     = 0,
    EVENT_PERSISTED = 1,
    EVENT_REMOVED   = 2
} ContactEventType;

typedef enum MotionType : uint8_t {
    MOTION_STATIC    = JPH_MotionType_Static,
    MOTION_KINEMATIC = JPH_MotionType_Kinematic,
    MOTION_DYNAMIC   = JPH_MotionType_Dynamic,
} MotionType;

typedef enum CPH_ShapeType : uint8_t {
    CPH_SHAPE_BOX         = 0,
    CPH_SHAPE_SPHERE      = 1,
    CPH_SHAPE_CAPSULE     = 2,
    CPH_SHAPE_CYLINDER    = 3,
    CPH_SHAPE_PLANE       = 4,
    CPH_SHAPE_MESH        = 5,
    CPH_SHAPE_HEIGHTFIELD = 6,
    CPH_SHAPE_CONVEX_HULL = 7
} CPH_ShapeType;

typedef struct {
    float mass;
    float friction;
    float restitution;
    int is_sensor;
    int use_ccd;
} CPH_BodyCreationProps;

typedef struct {
    float mass;
    float friction;
    float restitution;
    int is_sensor;
    int use_ccd;
    int motion_type;
    uint32_t allowed_dofs;
} CPH_BodyConfig;

typedef enum ObjectLayer : uint32_t {
    OBJECT_LAYER_STATIC = 0,
    OBJECT_LAYER_DYNAMIC,
    OBJECT_LAYER_VEHICLE, // For raycasts/sensors that shouldn't hit the vehicle itself
    OBJECT_LAYER_COUNT
} ObjectLayer;

[[maybe_unused]] static constexpr uint32_t kHandleIndexBits = 32;
[[maybe_unused]] static constexpr uint64_t kHandleIndexMask = 0xFFFFFFFF;
