#pragma once
#include "core/Types.h"
#include "joltc.h"

struct CPH_PhysicsWorld;

// --- Shape Caching ---
typedef struct {
    uint32_t type; // 0=Box, 1=Sphere, 2=Capsule, 3=Cylinder, 4=Plane
    float p1, p2, p3, p4;
} CPH_ShapeKey;

typedef struct {
    CPH_ShapeKey key;
    JPH_Shape *shape;
} CPH_ShapeEntry;

typedef struct {
    JPH_ShapeCastResult hit;
    bool has_hit;
} CPH_CastShapeContext;

// Optional filter to ignore a specific body
typedef struct {
    JPH_BodyID ignore_id;
} CPH_CastShapeFilter;

JPH_Shape *CPH_FindOrCreateShapeLocked(struct CPH_PhysicsWorld *world, int type, const float *params);

void CPH_FreeShapeCache(struct CPH_PhysicsWorld *world);

bool CPH_ShapecastExecuteInternal(struct CPH_PhysicsWorld *world, const JPH_Shape *shape,
                                const JPH_RMat4 *transform, const JPH_Vec3 *sweep_dir,
                                JPH_BodyID ignore_bid, CPH_CastShapeContext *ctx);

// Collector for Shapecast
static inline float CastShape_ClosestCollector(void *context, const JPH_ShapeCastResult *result) {
    CPH_CastShapeContext *ctx = (CPH_CastShapeContext *)context;
    if (result->fraction < ctx->hit.fraction) {
        ctx->hit     = *result;
        ctx->has_hit = true;
    }
    return result->fraction;
}

static inline bool JPH_API_CALL UnifiedBodyFilter(void *userData, JPH_BodyID bodyID) {
    // If no context was provided (e.g. overlap_sphere), allow everything
    if (userData == nullptr) {
        return true;
    }

    // Otherwise, check the ignore_id
    CPH_CastShapeFilter *ctx = (CPH_CastShapeFilter *)userData;
    return (bool)(ctx->ignore_id == 0 || bodyID != ctx->ignore_id);
}
