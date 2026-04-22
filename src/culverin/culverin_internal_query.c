#include "culverin_internal_query.h"
#include "culverin.h"
#include "culverin_physics_world.h"

/**
 * REQUIRES: SHADOW_LOCK held AND g_jph_trampoline_lock held
 */
JPH_Shape *find_or_create_shape_locked(PhysicsWorldObject *self, int type, const float *params) {
    // 1. KEY NORMALIZATION & SANITIZATION
    // p1-p3 clamped to 1mm minimum to prevent solver issues.
    // unused parameters are zeroed to maximize cache hits.
    const float p1 = (params[0] < 1e-3f) ? 1e-3f : params[0];
    const float p2 = (params[1] < 1e-3f) ? 1e-3f : params[1];
    const float p3 = (params[2] < 1e-3f) ? 1e-3f : params[2];
    const float p4 = (type == CULV_SHAPE_PLANE) ? params[3] : 0.0f;

    // 2. CACHE LOOKUP
    for (size_t i = 0; i < self->shape_cache_count; ++i) {
        const ShapeKey *cached = &self->shape_cache[i].key;
        if (cached->type == (uint32_t)type && cached->p1 == p1 && cached->p2 == p2 &&
            cached->p3 == p3 && cached->p4 == p4) {
            return self->shape_cache[i].shape;
        }
    }

    // 3. PRE-ALLOCATION CHECK (Fail early before expensive Jolt work)
    if (self->shape_cache_count >= self->shape_cache_capacity) {
        size_t new_cap = (self->shape_cache_capacity == 0) ? 16 : self->shape_cache_capacity * 2;
        ShapeEntry *new_ptr =
            (ShapeEntry *)CULV_RAW_REALLOC(self->shape_cache, new_cap * sizeof(ShapeEntry));
        if (UNLIKELY(!new_ptr)) {
            return nullptr;
        }
        self->shape_cache          = new_ptr;
        self->shape_cache_capacity = new_cap;
    }

    // 4. JOLT CREATION (Type-Specific Dispatch)
    JPH_Shape *shape = nullptr;
    switch (type) {
    case CULV_SHAPE_BOX: {
        JPH_Vec3 half_extents = {p1, p2, p3};
        auto s                = JPH_BoxShapeSettings_Create(&half_extents, 0.05f);
        if (s) {
            shape = (JPH_Shape *)JPH_BoxShapeSettings_CreateShape(s);
            JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)s);
        }
        break;
    }
    case CULV_SHAPE_SPHERE: {
        auto s = JPH_SphereShapeSettings_Create(p1);
        if (s) {
            shape = (JPH_Shape *)JPH_SphereShapeSettings_CreateShape(s);
            JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)s);
        }
        break;
    }
    case CULV_SHAPE_CAPSULE: {
        auto s = JPH_CapsuleShapeSettings_Create(p1, p2);
        if (s) {
            shape = (JPH_Shape *)JPH_CapsuleShapeSettings_CreateShape(s);
            JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)s);
        }
        break;
    }
    case CULV_SHAPE_CYLINDER: {
        auto s = JPH_CylinderShapeSettings_Create(p1, p2, 0.05f);
        if (s) {
            shape = (JPH_Shape *)JPH_CylinderShapeSettings_CreateShape(s);
            JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)s);
        }
        break;
    }
    case CULV_SHAPE_PLANE: {
        JPH_Plane plane = {{p1, p2, p3}, p4};
        auto s          = JPH_PlaneShapeSettings_Create(&plane, nullptr, 1000.0f);
        if (s) {
            shape = (JPH_Shape *)JPH_PlaneShapeSettings_CreateShape(s);
            JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)s);
        }
        break;
    }
    default:
        break;
    }

    if (UNLIKELY(!shape)) {
        return nullptr;
    }

    // 5. CACHE COMMIT
    ShapeEntry *entry = &self->shape_cache[self->shape_cache_count++];
    entry->key        = (ShapeKey){.type = (uint32_t)type, .p1 = p1, .p2 = p2, .p3 = p3, .p4 = p4};
    entry->shape      = shape;

    return shape;
}

void free_shape_cache(PhysicsWorldObject *self) {
    if (!self->shape_cache) {
        return;
    }

    for (size_t i = 0; i < self->shape_cache_count; i++) {
        if (self->shape_cache[i].shape) {
            JPH_Shape_Destroy(self->shape_cache[i].shape);
        }
    }
    CULV_RAW_FREE(self->shape_cache);
    self->shape_cache       = nullptr;
    self->shape_cache_count = 0;
}

// Helper 1: Run the Raycast
// ASSUMPTION: Caller has already acquired g_jph_trampoline_lock and released GIL.
bool execute_raycast_query(PhysicsWorldObject *self, JPH_BodyID ignore_bid, const JPH_RVec3 *origin,
                           const JPH_Vec3 *direction, JPH_RayCastResult *hit) {
    // 1. Filter Setup (Safe, doesn't touch shared Jolt memory yet)
    JPH_BroadPhaseLayerFilter *bp_f = JPH_BroadPhaseLayerFilter_Create(nullptr);
    JPH_ObjectLayerFilter *obj_f    = JPH_ObjectLayerFilter_Create(nullptr);

    CastShapeFilter filter_ctx = {.ignore_id = ignore_bid};
    JPH_BodyFilter *bf         = JPH_BodyFilter_Create(&filter_ctx);

    // 2. Execution
    const JPH_NarrowPhaseQuery *query = JPH_PhysicsSystem_GetNarrowPhaseQuery(self->system);
    bool has_hit = JPH_NarrowPhaseQuery_CastRay(query, origin, direction, hit, bp_f, obj_f, bf);

    // 3. Cleanup
    JPH_BodyFilter_Destroy(bf);
    JPH_BroadPhaseLayerFilter_Destroy(bp_f);
    JPH_ObjectLayerFilter_Destroy(obj_f);

    return has_hit;
}

// Helper 2: Extract World Space Normal after hit
// ASSUMPTION: Caller holds the Jolt lock.
void extract_hit_normal(PhysicsWorldObject *self, JPH_BodyID bodyID, JPH_SubShapeID subShapeID2,
                        const JPH_RVec3 *origin, const JPH_Vec3 *ray_dir, JPH_Real fraction,
                        JPH_Vec3 *normal_out) {
    const JPH_BodyLockInterface *lock_iface = JPH_PhysicsSystem_GetBodyLockInterface(self->system);
    JPH_BodyLockRead lock;
    JPH_BodyLockInterface_LockRead(lock_iface, bodyID, &lock);

    if (lock.body) {
        // Perform hit point calculation in high precision
        JPH_RVec3 hit_p = {origin->x + (JPH_Real)ray_dir->x * fraction,
                           origin->y + (JPH_Real)ray_dir->y * fraction,
                           origin->z + (JPH_Real)ray_dir->z * fraction};
        JPH_Body_GetWorldSpaceSurfaceNormal(lock.body, subShapeID2, &hit_p, normal_out);
    } else {
        *normal_out = (JPH_Vec3){0, 1, 0};
    }
    JPH_BodyLockInterface_UnlockRead(lock_iface, &lock);
}

// Helper 3: Internal logic to run the actual query
// ASSUMPTION: Caller has already acquired g_jph_trampoline_lock and released GIL.
void shapecast_execute_internal(PhysicsWorldObject *self, const JPH_Shape *shape,
                                const JPH_RMat4 *transform, const JPH_Vec3 *sweep_dir,
                                JPH_BodyID ignore_bid, CastShapeContext *ctx) {
    JPH_BroadPhaseLayerFilter *bp_f = JPH_BroadPhaseLayerFilter_Create(nullptr);

    JPH_ObjectLayerFilter *obj_f = JPH_ObjectLayerFilter_Create(nullptr);

    CastShapeFilter filter_ctx = {.ignore_id = ignore_bid};
    JPH_BodyFilter *bf         = JPH_BodyFilter_Create(&filter_ctx);

    JPH_STACK_ALLOC(JPH_ShapeCastSettings, settings);
    JPH_ShapeCastSettings_Init(settings);
    settings->backFaceModeTriangles = JPH_BackFaceMode_IgnoreBackFaces;
    settings->backFaceModeConvex    = JPH_BackFaceMode_IgnoreBackFaces;

    JPH_RVec3 base_offset          = {0, 0, 0};
    const JPH_NarrowPhaseQuery *nq = JPH_PhysicsSystem_GetNarrowPhaseQuery(self->system);

    JPH_NarrowPhaseQuery_CastShape(nq, shape, transform, sweep_dir, settings, &base_offset,
                                   CastShape_ClosestCollector, ctx, bp_f, obj_f, bf, nullptr);

    JPH_BodyFilter_Destroy(bf);
    JPH_BroadPhaseLayerFilter_Destroy(bp_f);
    JPH_ObjectLayerFilter_Destroy(obj_f);
}
