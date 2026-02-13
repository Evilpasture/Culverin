#pragma once
#include "joltc.h"

struct PhysicsWorldObject;
typedef struct PhysicsWorldObject PhysicsWorldObject;

// --- Shape Caching ---
typedef struct {
  uint32_t type; // 0=Box, 1=Sphere, 2=Capsule, 3=Cylinder, 4=Plane
  float p1, p2, p3, p4;
} ShapeKey;

typedef struct {
  ShapeKey key;
  JPH_Shape *shape;
} ShapeEntry;

typedef struct {
  JPH_ShapeCastResult hit;
  bool has_hit;
} CastShapeContext;

// Optional filter to ignore a specific body
typedef struct {
  JPH_BodyID ignore_id;
} CastShapeFilter;

#include "culverin.h"

JPH_Shape *find_or_create_shape_locked(struct PhysicsWorldObject *self, int type,
                                const float *params);

void free_shape_cache(PhysicsWorldObject *self);

bool execute_raycast_query(PhysicsWorldObject *self, JPH_BodyID ignore_bid,
                           const JPH_RVec3 *origin, const JPH_Vec3 *direction,
                           JPH_RayCastResult *hit);

void extract_hit_normal(PhysicsWorldObject *self, JPH_BodyID bodyID,
                        JPH_SubShapeID subShapeID2, const JPH_RVec3 *origin,
                        const JPH_Vec3 *ray_dir, JPH_Real fraction,
                        JPH_Vec3 *normal_out);

void shapecast_execute_internal(PhysicsWorldObject *self,
                                const JPH_Shape *shape,
                                const JPH_RMat4 *transform,
                                const JPH_Vec3 *sweep_dir,
                                JPH_BodyID ignore_bid, CastShapeContext *ctx);

// Collector for Shapecast
static inline float CastShape_ClosestCollector(void *context, const JPH_ShapeCastResult *result) {
  CastShapeContext *ctx = (CastShapeContext *)context;
  if (result->fraction < ctx->hit.fraction) {
    ctx->hit = *result;
    ctx->has_hit = true;
  }
  return result->fraction;
}

static inline bool JPH_API_CALL CastShape_BodyFilter(void *userData,
                                                     JPH_BodyID bodyID) {
  CastShapeFilter *ctx = (CastShapeFilter *)userData;
  return (ctx->ignore_id == 0 || bodyID != ctx->ignore_id);
}
