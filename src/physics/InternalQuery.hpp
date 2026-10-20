#pragma once

#include "core/Types.hpp"
#include "jolt/Jolt.hpp"
#include <Jolt/Physics/Collision/Shape/Shape.h>
#include <joltc.h>

#include <cstdint>


namespace CPH {

struct PhysicsWorld;

// --- Shape caching ---

struct ShapeKey {
    std::uint32_t type; // ShapeType
    float p1, p2, p3, p4;

    [[nodiscard]] friend bool operator==(const ShapeKey &, const ShapeKey &) = default;
};

// Owning cache entry: the refcounted shape lives with the entry.
struct ShapeEntry {
    ShapeKey key{};
    JPH::ShapeRefC shape;
};


// --- Shapecast execution shared with the query layer ---

struct CastShapeContext {
    JPH_ShapeCastResult hit;
    bool has_hit;
};

struct CastShapeFilter {
    JPH_BodyID ignore_id;
};

bool ShapecastExecuteInternal(PhysicsWorld &world, const JPH_Shape *shape,
                                  const JPH_RMat4 *transform, const JPH_Vec3 *sweep_dir,
                                  JPH_BodyID ignore_bid, CastShapeContext *ctx);

// Jolt collector: keeps the closest hit. Signature fixed by joltc.h.
inline float CastShape_ClosestCollector(void *context, const JPH_ShapeCastResult *result) {
    auto *ctx = static_cast<CastShapeContext *>(context);
    if (result->fraction < ctx->hit.fraction) {
        ctx->hit     = *result;
        ctx->has_hit = true;
    }
    return result->fraction;
}

// Jolt body filter: allows everything except an optional ignored body.
inline bool JPH_API_CALL UnifiedBodyFilter(void *userData, JPH_BodyID bodyID) {
    if (userData == nullptr) {
        return true;
    }
    const auto *ctx = static_cast<CastShapeFilter *>(userData);
    return ctx->ignore_id == 0 || bodyID != ctx->ignore_id;
}

} // namespace CPH
