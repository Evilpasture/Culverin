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