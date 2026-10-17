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
} PosStride;

// Maps to self->physics.rotations, velocities (Packed X, Y, Z, W)
typedef struct {
    alignas(16) float x;
    float y, z, w;
} AuxStride;

typedef struct {
    float x;
    float y;
    float z;
} Vec3f; // General Vec3f

// Sanity check sizes
static_assert(sizeof(PosStride) == sizeof(JPH_Real) * 4);
static_assert(sizeof(AuxStride) == sizeof(float) * 4);

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

typedef enum CulvShapeType : uint8_t {
    CULV_SHAPE_BOX         = 0,
    CULV_SHAPE_SPHERE      = 1,
    CULV_SHAPE_CAPSULE     = 2,
    CULV_SHAPE_CYLINDER    = 3,
    CULV_SHAPE_PLANE       = 4,
    CULV_SHAPE_MESH        = 5,
    CULV_SHAPE_HEIGHTFIELD = 6,
    CULV_SHAPE_CONVEX_HULL = 7
} CulvShapeType;

typedef struct {
    float mass;
    float friction;
    float restitution;
    int is_sensor;
    int use_ccd;
} BodyCreationProps;

typedef struct {
    float mass;
    float friction;
    float restitution;
    int is_sensor;
    int use_ccd;
    int motion_type;
    uint32_t allowed_dofs;
} BodyConfig;

typedef enum ObjectLayer : uint32_t {
    OBJECT_LAYER_STATIC = 0,
    OBJECT_LAYER_DYNAMIC,
    OBJECT_LAYER_VEHICLE, // For raycasts/sensors that shouldn't hit the vehicle itself
    OBJECT_LAYER_COUNT
} ObjectLayer;

[[maybe_unused]] static constexpr uint32_t HANDLE_INDEX_BITS = 32;
[[maybe_unused]] static constexpr uint64_t HANDLE_INDEX_MASK = 0xFFFFFFFF;
