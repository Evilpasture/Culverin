#include "physics/InternalQuery.hpp"
#include "core/Buffer.hpp"
#include "physics/PhysicsWorldState.hpp"
#include <cmath>

/**
 * REQUIRES: SHADOW_LOCK held. It serializes both shape-cache access and shape
 * creation. This helper does not access the physics system and does not require
 * jph_trampoline_lock.
 */
JPH_Shape *CPH_FindOrCreateShapeLocked(CPH_PhysicsWorld *world, int type, const float *params) {
    // 1. KEY NORMALIZATION & SANITIZATION
    // Defensive clamping
    float p1;
    float p2;
    float p3;
    float p4;

    if (type == CPH_SHAPE_PLANE) {
        // Plane normal components must represent a unit vector.
        // We defensively normalize them and avoid clamping zero-components.
        float nx     = params[0];
        float ny     = params[1];
        float nz     = params[2];
        float len_sq = (nx * nx) + (ny * ny) + (nz * nz);

        if (len_sq < 1e-6f) {
            // Safeguard against degenerate (0,0,0) vectors by defaulting to straight up
            p1 = 0.0f;
            p2 = 1.0f;
            p3 = 0.0f;
        } else {
            float len = std::sqrt(len_sq);
            p1        = nx / len;
            p2        = ny / len;
            p3        = nz / len;
        }
        p4 = params[3]; // Keep plane offset
    } else {
        // Standard geometric shapes are clamped to a 1mm minimum to protect solver stability.
        p1 = (params[0] < 1e-3f) ? 1e-3f : params[0];
        p2 = (params[1] < 1e-3f) ? 1e-3f : params[1];
        p3 = (params[2] < 1e-3f) ? 1e-3f : params[2];
        p4 = 0.0f;
    }
    // 2. CACHE LOOKUP
    for (size_t i = 0; i < world->shape_cache_count; ++i) {
        const CPH_ShapeKey *cached = &world->shape_cache[i].key;
        if (cached->type == (uint32_t)type && cached->p1 == p1 && cached->p2 == p2 &&
            cached->p3 == p3 && cached->p4 == p4) {
            return world->shape_cache[i].shape;
        }
    }

    // 3. PRE-ALLOCATION CHECK (Fail early before expensive Jolt work)
    if (world->shape_cache_count >= world->shape_cache.size()) {
        const std::size_t new_cap = world->shape_cache.size() == 0 ? 16 : world->shape_cache.size() * 2;
        if (CPH_UNLIKELY(!world->shape_cache.reallocate(new_cap))) {
            return nullptr;
        }
    }

    // 4. JOLT CREATION (Type-Specific Dispatch)
    JPH_Shape *shape = nullptr;
    switch (type) {
    case CPH_SHAPE_BOX: {
        JPH_Vec3 half_extents = {p1, p2, p3};
        auto s                = JPH_BoxShapeSettings_Create(&half_extents, 0.05f);
        if (s) {
            shape = (JPH_Shape *)JPH_BoxShapeSettings_CreateShape(s);
            JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)s);
        }
        break;
    }
    case CPH_SHAPE_SPHERE: {
        auto s = JPH_SphereShapeSettings_Create(p1);
        if (s) {
            shape = (JPH_Shape *)JPH_SphereShapeSettings_CreateShape(s);
            JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)s);
        }
        break;
    }
    case CPH_SHAPE_CAPSULE: {
        auto s = JPH_CapsuleShapeSettings_Create(p1, p2);
        if (s) {
            shape = (JPH_Shape *)JPH_CapsuleShapeSettings_CreateShape(s);
            JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)s);
        }
        break;
    }
    case CPH_SHAPE_CYLINDER: {
        auto s = JPH_CylinderShapeSettings_Create(p1, p2, 0.05f);
        if (s) {
            shape = (JPH_Shape *)JPH_CylinderShapeSettings_CreateShape(s);
            JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)s);
        }
        break;
    }
    case CPH_SHAPE_PLANE: {
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

    if (CPH_UNLIKELY(!shape)) {
        return nullptr;
    }

    // 5. CACHE COMMIT
    CPH_ShapeEntry &entry = world->shape_cache[world->shape_cache_count++];
    entry.key   = CPH_ShapeKey{.type = (uint32_t)type, .p1 = p1, .p2 = p2, .p3 = p3, .p4 = p4};
    entry.shape = shape;

    return shape;
}


// Execute one native shapecast. The caller owns query registration and holds
// the shape alive for the duration of this call.
bool CPH_ShapecastExecuteInternal(CPH_PhysicsWorld *world, const JPH_Shape *shape,
                                const JPH_RMat4 *transform, const JPH_Vec3 *sweep_dir,
                                JPH_BodyID ignore_bid, CPH_CastShapeContext *ctx) {
    JPH_BroadPhaseLayerFilter *bp_f = JPH_BroadPhaseLayerFilter_Create(nullptr);
    JPH_ObjectLayerFilter *obj_f    = JPH_ObjectLayerFilter_Create(nullptr);

    CPH_CastShapeFilter filter_ctx = {ignore_bid};
    JPH_BodyFilter *bf         = JPH_BodyFilter_Create(&filter_ctx);

    if (!bp_f || !obj_f || !bf) {
        if (bf) {
            JPH_BodyFilter_Destroy(bf);
        }
        if (bp_f) {
            JPH_BroadPhaseLayerFilter_Destroy(bp_f);
        }
        if (obj_f) {
            JPH_ObjectLayerFilter_Destroy(obj_f);
        }
        return false;
    }

    JPH_ShapeCastSettings shape_cast_settings{};
    JPH_ShapeCastSettings *settings = &shape_cast_settings;
    JPH_ShapeCastSettings_Init(settings);
    settings->backFaceModeTriangles = JPH_BackFaceMode_IgnoreBackFaces;
    settings->backFaceModeConvex    = JPH_BackFaceMode_IgnoreBackFaces;

    JPH_RVec3 base_offset          = {0, 0, 0};
    const JPH_NarrowPhaseQuery *nq = JPH_PhysicsSystem_GetNarrowPhaseQuery(world->system);

    JPH_NarrowPhaseQuery_CastShape(nq, shape, transform, sweep_dir, settings, &base_offset,
                                   CastShape_ClosestCollector, ctx, bp_f, obj_f, bf, nullptr);

    JPH_BodyFilter_Destroy(bf);
    JPH_BroadPhaseLayerFilter_Destroy(bp_f);
    JPH_ObjectLayerFilter_Destroy(obj_f);
    return true;
}
