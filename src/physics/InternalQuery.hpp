#pragma once

#include "core/Types.hpp"
#include "jolt/Jolt.hpp"
#include <Jolt/Physics/Collision/Shape/Shape.h>
#include <Jolt/Physics/Collision/ShapeCast.h>

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

// Closest-hit collector for shape casts.
//
// Jolt ships ClosestHitCollisionCollector, but it ranks hits by
// ShapeCastResult::GetEarlyOutFraction(), which switches to -penetrationDepth
// for a cast that starts inside a shape. The pre-A4 C collector compared the
// raw cast fraction instead, so this reproduces that ordering -- including the
// fact that a hit at exactly fraction 1.0 was never recorded, because the
// comparison started from 1.0f.
class ClosestFractionCastShapeCollector final : public JPH::CastShapeCollector {
  public:
    ClosestFractionCastShapeCollector() noexcept {
        // ShapeCastResult leaves mFraction uninitialized; seed it the way the
        // C version did so the first comparison is against 1.0f.
        mHit.mFraction = 1.0f;
    }

    void AddHit(const JPH::ShapeCastResult &inResult) override {
        if (inResult.mFraction < mHit.mFraction) {
            mHit     = inResult;
            mHasHit = true;
        }
        UpdateEarlyOutFraction(inResult.mFraction);
    }

    [[nodiscard]] bool has_hit() const noexcept { return mHasHit; }
    [[nodiscard]] const JPH::ShapeCastResult &hit() const noexcept { return mHit; }

  private:
    JPH::ShapeCastResult mHit{};
    bool mHasHit = false;
};

// Executes one shape cast against the world's narrow-phase query.
//
// The filters are stack objects: Jolt's filter base classes allow everything
// by default, so the only filter with behaviour is the single-body ignore.
// `ignore_bid` may be invalid (0xFFFFFFFF), which ignores nothing. The caller
// owns query registration and keeps `shape` alive for the duration.
void ShapecastExecuteInternal(PhysicsWorld &world, const JPH::Shape *shape,
                              JPH::RMat44Arg transform, JPH::Vec3Arg sweep_dir,
                              JPH::BodyID ignore_bid, JPH::CastShapeCollector &collector);

} // namespace CPH
