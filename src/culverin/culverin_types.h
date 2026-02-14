#pragma once
#include "joltc.h"

// --- Jolt Precision Compatibility ---
#ifndef JPH_Real
  #ifdef JPH_DOUBLE_PRECISION
    typedef double JPH_Real;
  #else
    typedef float JPH_Real;
  #endif
#endif


// --- Memory Stride Helpers ---
// Maps to self->positions (Packed X, Y, Z)
typedef struct { JPH_Real x, y, z, _pad; } PosStride;

// Maps to self->rotations, velocities (Packed X, Y, Z, W)
typedef struct { float x, y, z, w; } AuxStride; 

// Sanity check sizes
_Static_assert(sizeof(PosStride) == sizeof(JPH_Real) * 4, "PosStride padding error");
_Static_assert(sizeof(AuxStride) == sizeof(float) * 4,    "AuxStride padding error");

// Minimal Handle Helper
// Python handles will be 64-bit integers: (Generation << 32) | SlotIndex
typedef uint64_t BodyHandle;

// Constraint Types
typedef enum ConstraintType : uint8_t {
  CONSTRAINT_FIXED = 0,
  CONSTRAINT_POINT = 1,
  CONSTRAINT_HINGE = 2,
  CONSTRAINT_SLIDER = 3,
  CONSTRAINT_DISTANCE = 4,
  CONSTRAINT_CONE = 5
} ConstraintType;

// Minimal Handle for Constraints (Distinct from BodyHandle)
typedef uint64_t ConstraintHandle;

// --- Contact Lifecycle Types ---
typedef enum ContactEventType : uint8_t {
  EVENT_ADDED = 0,
  EVENT_PERSISTED = 1,
  EVENT_REMOVED = 2
} ContactEventType;