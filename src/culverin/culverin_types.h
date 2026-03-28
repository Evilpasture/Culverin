#pragma once
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
    JPH_Real x;
    JPH_Real y;
    JPH_Real z;
    JPH_Real w;
} PosStride;

// Maps to self->rotations, velocities (Packed X, Y, Z, W)
typedef struct {
    float x, y, z, w;
} AuxStride;

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

static constexpr uint32_t HANDLE_INDEX_BITS = 32;
static constexpr uint64_t HANDLE_INDEX_MASK = 0xFFFFFFFF;
