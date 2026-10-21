#include "physics/InternalQuery.hpp"
#include "core/Buffer.hpp"
#include "physics/PhysicsWorldState.hpp"
#include <Jolt/Physics/Body/BodyFilter.h>
#include <Jolt/Physics/Collision/NarrowPhaseQuery.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/CylinderShape.h>
#include <Jolt/Physics/Collision/Shape/PlaneShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <cmath>

/**
 * REQUIRES: SHADOW_LOCK held. It serializes both shape-cache access and shape
 * creation. This helper does not access the physics system and does not require
 * jph_trampoline_lock.
 */

namespace CPH {

namespace {

[[nodiscard]] JPH::ShapeRefC create_shape(const JPH::ShapeSettings &settings) {
    const auto result = settings.Create();
    return result.IsValid() ? JPH::ShapeRefC(result.Get()) : JPH::ShapeRefC{};
}

} // namespace

JPH::ShapeRefC PhysicsWorld::FindOrCreateShapeLocked(int type, std::span<const float, 4> params) {
    const ShapeType shape_type = static_cast<ShapeType>(type);

    // 1. KEY NORMALIZATION & SANITIZATION
    // Defensive clamping
    float p1;
    float p2;
    float p3;
    float p4;

    if (shape_type == ShapeType::Plane) {
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
    const ShapeKey key{.type = (uint32_t)type, .p1 = p1, .p2 = p2, .p3 = p3, .p4 = p4};
    for (size_t i = 0; i < shape_cache_count; ++i) {
        if (shape_cache[i].key == key) {
            return shape_cache[i].shape;
        }
    }

    // 3. PRE-ALLOCATION CHECK (Fail early before expensive Jolt work)
    if (shape_cache_count >= shape_cache.size()) {
        const std::size_t new_cap = shape_cache.size() == 0 ? 16 : shape_cache.size() * 2;
        if (!shape_cache.reallocate(new_cap)) [[unlikely]] {
            return {};
        }
    }

    // 4. JOLT CREATION (Type-Specific Dispatch)
    JPH::ShapeRefC shape;
    switch (shape_type) {
    case ShapeType::Box:
        shape = create_shape(JPH::BoxShapeSettings(JPH::Vec3(p1, p2, p3), 0.05f));
        break;
    case ShapeType::Sphere:
        shape = create_shape(JPH::SphereShapeSettings(p1));
        break;
    case ShapeType::Capsule:
        shape = create_shape(JPH::CapsuleShapeSettings(p1, p2));
        break;
    case ShapeType::Cylinder:
        shape = create_shape(JPH::CylinderShapeSettings(p1, p2, 0.05f));
        break;
    case ShapeType::Plane:
        shape = create_shape(
            JPH::PlaneShapeSettings(JPH::Plane(JPH::Vec3(p1, p2, p3), p4), nullptr, 1000.0f));
        break;
    default:
        break;
    }

    if (shape == nullptr) [[unlikely]] {
        return {};
    }

    // 5. CACHE COMMIT
    ShapeEntry &entry = shape_cache[shape_cache_count++];
    entry.key         = key;
    entry.shape       = shape;

    return shape;
}


JPH::ShapeRefC PhysicsWorld::FindOrCreateShape(int type, std::span<const float, 4> params) {
    const std::scoped_lock guard(shadow_lock);
    return FindOrCreateShapeLocked(type, params);
}

// Execute one native shapecast. The caller owns query registration and holds
// the shape alive for the duration of this call.
void ShapecastExecuteInternal(PhysicsWorld &world, const JPH::Shape *shape,
                              JPH::RMat44Arg transform, JPH::Vec3Arg sweep_dir,
                              JPH::BodyID ignore_bid, JPH::CastShapeCollector &collector) {
    // Scale rides in the world transform, so the cast shape is unscaled here --
    // the same construction the JoltC wrapper performed.
    const JPH::RShapeCast shape_cast = JPH::RShapeCast::sFromWorldTransform(
        shape, JPH::Vec3::sReplicate(1.0f), transform, sweep_dir);

    JPH::ShapeCastSettings settings;
    // IgnoreBackFaces is Jolt's default for both modes; stated so the intent
    // survives an upstream default change.
    settings.SetBackFaceMode(JPH::EBackFaceMode::IgnoreBackFaces);

    // An invalid BodyID matches no real body, so this is "ignore nothing".
    const JPH::IgnoreSingleBodyFilter body_filter(ignore_bid);
    world.physics_system.GetNarrowPhaseQuery().CastShape(shape_cast, settings, JPH::RVec3::sZero(),
                                                         collector, {}, {}, body_filter);
}

} // namespace CPH
