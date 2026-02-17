#pragma once
#include <stdint.h>

// --- Jolt Precision Compatibility ---
#ifndef JPH_Real
  #ifdef JPH_DOUBLE_PRECISION
    typedef double JPH_Real;
    #define JPH_REAL_CHAR 'd'
    #define JPH_REAL_STRING "d"
  #else
    typedef float JPH_Real;
    #define JPH_REAL_CHAR 'f'
    #define JPH_REAL_STRING "f"
  #endif
#endif

// --- Memory Stride Helpers ---
// Use a clear name to avoid collisions with Jolt's internal defines
constexpr int CULV_STRIDE_ALIGN = 32;

typedef struct {
    // Align the first member to force the whole struct to 32-byte alignment
    alignas(CULV_STRIDE_ALIGN) JPH_Real x;
    JPH_Real y;
    JPH_Real z;
    // Explicitly calculate padding based on the target width
    JPH_Real _pad[(CULV_STRIDE_ALIGN / sizeof(JPH_Real)) - 3];
} PosStride;

// Maps to self->rotations, velocities (Packed X, Y, Z, W)
typedef struct { alignas(16) float x; float y, z, w; } AuxStride; 

// Sanity check sizes
static_assert(sizeof(PosStride) == 32);
static_assert(sizeof(AuxStride) == 16);

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