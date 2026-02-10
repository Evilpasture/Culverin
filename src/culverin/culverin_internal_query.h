#pragma once
#include "joltc.h"

// --- Shape Caching ---
typedef struct {
  uint32_t type; // 0=Box, 1=Sphere, 2=Capsule, 3=Cylinder, 4=Plane
  float p1, p2, p3, p4;
} ShapeKey;

typedef struct {
  ShapeKey key;
  JPH_Shape *shape;
} ShapeEntry;

#include "culverin.h"

JPH_Shape *find_or_create_shape(struct PhysicsWorldObject *self, int type,
                                       const float *params);