#pragma once

#include "core/Types.h"
#include <stdint.h>

// Native shadow storage for a Jolt soft body. No Python references or APIs.
typedef struct {
    JPH_Real *vertices;
    float *normals;
    float *velocities;
    uint32_t num_vertices;
} SoftBodyShadow;
