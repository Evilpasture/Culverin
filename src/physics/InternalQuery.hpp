#pragma once

#include "core/Types.hpp"
#include <joltc.h>

#include <cstdint>

struct CPH_PhysicsWorld;

// --- Shape caching ---

struct CPH_ShapeKey {
    std::uint32_t type; // CPH_ShapeType
    float p1, p2, p3, p4;

    [[nodiscard]] friend bool operator==(const CPH_ShapeKey &, const CPH_ShapeKey &) = default;
};

// Owning cache entry: the shape is destroyed with the entry.
struct CPH_ShapeEntry {
    CPH_ShapeKey key{};
    JPH_Shape *shape = nullptr;

    CPH_ShapeEntry() = default;
    ~CPH_ShapeEntry() {
        if (shape != nullptr) {
            JPH_Shape_Destroy(shape);
        }
    }

    CPH_ShapeEntry(const CPH_ShapeEntry &)            = delete;
    CPH_ShapeEntry &operator=(const CPH_ShapeEntry &) = delete;

    CPH_ShapeEntry(CPH_ShapeEntry &&other) noexcept
        : key(other.key), shape(other.shape) {
        other.shape = nullptr;
    }
    CPH_ShapeEntry &operator=(CPH_ShapeEntry &&other) noexcept {
        if (this != &other) {
            if (shape != nullptr) {
                JPH_Shape_Destroy(shape);
            }
            key         = other.key;
            shape       = other.shape;
            other.shape = nullptr;
        }
        return *this;
    }
};

// REQUIRES: shadow_lock held (documented at the definition).
JPH_Shape *CPH_FindOrCreateShapeLocked(CPH_PhysicsWorld *world, int type,
                                       const float *params);

// --- Shapecast execution shared with the query layer ---

struct CPH_CastShapeContext {
    JPH_ShapeCastResult hit;
    bool has_hit;
};

struct CPH_CastShapeFilter {
    JPH_BodyID ignore_id;
};

bool CPH_ShapecastExecuteInternal(CPH_PhysicsWorld *world, const JPH_Shape *shape,
                                  const JPH_RMat4 *transform, const JPH_Vec3 *sweep_dir,
                                  JPH_BodyID ignore_bid, CPH_CastShapeContext *ctx);

// Jolt collector: keeps the closest hit. Signature fixed by joltc.h.
inline float CastShape_ClosestCollector(void *context, const JPH_ShapeCastResult *result) {
    auto *ctx = static_cast<CPH_CastShapeContext *>(context);
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
    const auto *ctx = static_cast<CPH_CastShapeFilter *>(userData);
    return ctx->ignore_id == 0 || bodyID != ctx->ignore_id;
}
