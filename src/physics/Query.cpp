#include "physics/Query.hpp"
#include "core/Buffer.hpp"
#include "math/MathCore.hpp"
#include "physics/InternalQuery.hpp"
#include "physics/PhysicsWorldState.hpp"
#include "physics/SlotState.hpp"
#include <Jolt/Geometry/AABox.h>
#include <Jolt/Physics/Body/BodyLock.h>
#include <Jolt/Physics/Collision/BroadPhase/BroadPhaseQuery.h>
#include <Jolt/Physics/Collision/CastResult.h>
#include <Jolt/Physics/Collision/CollideShape.h>
#include <Jolt/Physics/Collision/NarrowPhaseQuery.h>
#include <Jolt/Physics/Collision/RayCast.h>
#include <cmath>
#include <cstring>


namespace CPH {

static constexpr float kQueryDirectionEpsilonSquared = 1e-12f;
static constexpr float kQueryDirectionUnitTolerance  = 1e-4f;

static void WaitUntilCanQuery(PhysicsWorld *world, std::unique_lock<std::mutex> &shadow) {
    const auto blocked = [&] {
        return world->isStepping.load(std::memory_order::relaxed) ||
               world->stepRequested.load(std::memory_order::relaxed);
    };
    if (blocked()) {
        world->waitingThreads.fetch_add(1, std::memory_order::relaxed);
        while (blocked()) {
            // Never hold shadow_lock while parked on the condition.
            shadow.unlock();
            {
                std::unique_lock sync(world->stepSync.mutex);
                world->stepSync.cond.wait(sync, [&] { return !blocked(); });
            }
            shadow.lock();
        }
        world->waitingThreads.fetch_sub(1, std::memory_order::relaxed);
    }
}

static void QueryBegin(PhysicsWorld *world) {
    std::unique_lock shadow(world->shadowLock);
    WaitUntilCanQuery(world, shadow);
    // The gate check and this increment are one critical section, so a
    // step cannot start between them.
    world->activeQueries.fetch_add(1, std::memory_order::acquire);
}

void PhysicsWorld::QueryEnd() {
    const std::scoped_lock sync(stepSync.mutex);
    if (activeQueries.fetch_sub(1, std::memory_order::acq_rel) == 1) {
        stepSync.cond.notify_all();
    }
}


static std::optional<uint32_t>
QueryHandleIsValidLocked(const PhysicsWorld &world, BodyHandle handle) {
    if (!handle) {
        return std::nullopt;
    }
    const std::optional<std::uint32_t> slot = world.Resolve(handle);
    if (!slot) {
        return std::nullopt;
    }

    const uint8_t state = (world.bodies.slotStates[*slot]).load(std::memory_order::acquire);
    if (!GetSlotPredicate(state, kMaskImmStandard).isImmediate) {
        return std::nullopt;
    }
    return slot;
}

static void FilterQueryHits(PhysicsWorld *world, PhysicsQueryHitBuffer &hits) {
    const std::scoped_lock shadow(world->shadowLock);
    size_t write_index = 0;
    for (size_t i = 0; i < hits.count; ++i) {
        if (QueryHandleIsValidLocked(*world, hits.hits[i])) {
            hits.hits[write_index++] = hits.hits[i];
        }
    }
    hits.count = write_index;
}

// Maps an ignore handle to the body it names. A missing or stale handle yields
// an invalid BodyID (0xFFFFFFFF), which is how a query spells "ignore nothing":
// no real body ever carries that value.
static JPH::BodyID ResolveIgnoreBodyLocked(const PhysicsWorld &world, BodyHandle ignore_handle) {
    if (const std::optional<uint32_t> ignore_slot =
            QueryHandleIsValidLocked(world, ignore_handle)) {
        const size_t body_count = (world.count).load(std::memory_order::relaxed);
        const uint32_t dense    = world.bodies.slotToDense[*ignore_slot];
        if (dense < body_count) {
            return JPH::BodyID(world.bodies.bodyIds[dense]);
        }
    }
    return JPH::BodyID();
}

static JPH::ShapeRefC QueryBeginWithShape(PhysicsWorld *world, int shape_type,
                                             std::span<const float, 4> params,
                                             JPH::BodyID *ignore_body, BodyHandle ignore_handle) {
    std::unique_lock shadow(world->shadowLock);
    WaitUntilCanQuery(world, shadow);

    // The shadow lock serializes cache access. Avoid taking jph_trampoline_lock
    // here: other native paths may hold it while waiting for shadow_lock.
    // The returned ref keeps the cached shape alive for the whole query.
    const JPH::ShapeRefC shape = world->FindOrCreateShapeLocked(shape_type, params);

    if (shape && ignore_body) {
        *ignore_body = ResolveIgnoreBodyLocked(*world, ignore_handle);
    }

    // Match the binding's former lock scope: shape/ignore resolution and query
    // registration are atomic with respect to step flags and shadow mutations.
    if (shape) {
        (world->activeQueries).fetch_add(1, std::memory_order::acquire);
    }
    return shape;
}

static JPH::BodyID QueryBeginWithIgnore(PhysicsWorld *world, BodyHandle ignore_handle) {
    std::unique_lock shadow(world->shadowLock);
    WaitUntilCanQuery(world, shadow);

    const JPH::BodyID ignore_body = ResolveIgnoreBodyLocked(*world, ignore_handle);

    (world->activeQueries).fetch_add(1, std::memory_order::acquire);
    return ignore_body;
}

struct OverlapContext {
    PhysicsWorld &world;
    PhysicsQueryHitBuffer &hits;
    Error error;
};

static void OverlapRecordHit(OverlapContext &context, JPH::BodyID body_id) {
    if (context.error) {
        return;
    }

    const BodyHandle handle = static_cast<BodyHandle>(
        context.world.physicsSystem.GetBodyInterface().GetUserData(body_id));
    if (handle && !context.hits.append(handle)) {
        context.error = QueryError::OutOfMemory;
    }
}

// Streams every narrow-phase overlap result into the caller's buffer. The
// pre-A4 C callback returned a constant early-out fraction, which never pruned,
// so this collector never forces an early out either: one handle per reported
// result, in query order.
class OverlapShapeCollector final : public JPH::CollideShapeCollector {
  public:
    explicit OverlapShapeCollector(OverlapContext &context) : mContext(context) {}

    void AddHit(const JPH::CollideShapeResult &inResult) override {
        OverlapRecordHit(mContext, inResult.mBodyID2);
    }

  private:
    OverlapContext &mContext;
};

// Same, for the broad-phase query, which reports bare body IDs.
class OverlapBodyCollector final : public JPH::CollideShapeBodyCollector {
  public:
    explicit OverlapBodyCollector(OverlapContext &context) : mContext(context) {}

    void AddHit(const JPH::BodyID &inBodyID) override { OverlapRecordHit(mContext, inBodyID); }

  private:
    OverlapContext &mContext;
};

std::expected<void, Error>
PhysicsWorld::OverlapSphere(JPH::RVec3 center, float radius, PhysicsQueryHitBuffer &hits) {
    if (!hits.hits) {
        return std::unexpected(QueryError::InvalidArgument);
    }

    const float shape_params[4] = {radius, 0.0f, 0.0f, 0.0f};
    const JPH::ShapeRefC shape = QueryBeginWithShape(
        this, std::to_underlying(ShapeType::Sphere), std::span<const float, 4>(shape_params, 4),
        nullptr, BodyHandle{});
    if (!shape) {
        return std::unexpected(QueryError::ShapeFailure);
    }

    const JPH::RMat44 transform =
        JPH::RMat44::sRotationTranslation(JPH::Quat::sIdentity(), center);
    const JPH::CollideShapeSettings settings;

    // No filters: every layer is queried and an overlap has no body to ignore.
    OverlapContext context = {.world = *this, .hits = hits, .error = {}};
    OverlapShapeCollector collector(context);
    physicsSystem.GetNarrowPhaseQuery().CollideShape(
        shape, JPH::Vec3::sReplicate(1.0f), transform, settings, JPH::RVec3::sZero(), collector);
    if (!context.error) {
        FilterQueryHits(this, hits);
    }
    QueryEnd();
    if (context.error) {
        return std::unexpected(context.error);
    }
    return {};
}

std::expected<void, Error>
PhysicsWorld::OverlapAabb(JPH::RVec3 min, JPH::RVec3 max, PhysicsQueryHitBuffer &hits) {
    if (!hits.hits) {
        return std::unexpected(QueryError::InvalidArgument);
    }

    QueryBegin(this);

    // Jolt rounds the double-precision corners outward to float, which keeps the
    // box a superset of the requested one.
    const JPH::AABox box(min, max);

    OverlapContext context = {.world = *this, .hits = hits, .error = {}};
    OverlapBodyCollector collector(context);
    physicsSystem.GetBroadPhaseQuery().CollideAABox(box, collector);
    if (!context.error) {
        FilterQueryHits(this, hits);
    }
    QueryEnd();
    if (context.error) {
        return std::unexpected(context.error);
    }
    return {};
}

static float RayDirectionScale(JPH::Vec3Arg direction, float max_distance, float *magnitude_sq) {
    const float length_sq = direction.GetX() * direction.GetX() +
                            direction.GetY() * direction.GetY() +
                            direction.GetZ() * direction.GetZ();
    *magnitude_sq = length_sq;
    if (length_sq < kQueryDirectionEpsilonSquared) {
        return 0.0f;
    }
    if (std::fabs(length_sq - 1.0f) < kQueryDirectionUnitTolerance) {
        return max_distance;
    }
    return max_distance * FastRsqrt(length_sq);
}

std::expected<PhysicsRaycastHit, Error> PhysicsWorld::Raycast(
    JPH::RVec3 origin, JPH::Vec3 direction, float max_distance, BodyHandle ignore_handle) {
    PhysicsRaycastHit hit{};

    float magnitude_sq = 0.0f;
    const float scale  = RayDirectionScale(direction, max_distance, &magnitude_sq);
    if (magnitude_sq < kQueryDirectionEpsilonSquared) {
        return hit;
    }

    // A missing/stale ignored handle intentionally means "ignore nothing".
    const JPH::BodyID ignore_body      = QueryBeginWithIgnore(this, ignore_handle);
    const JPH::IgnoreSingleBodyFilter body_filter(ignore_body);

    const JPH::Vec3 ray_direction(direction.GetX() * scale, direction.GetY() * scale,
                                  direction.GetZ() * scale);
    const JPH::RRayCast ray(origin, ray_direction);
    JPH::RayCastResult ray_hit;
    const bool has_hit = physicsSystem.GetNarrowPhaseQuery().CastRay(ray, ray_hit, {}, {},
                                                                      body_filter);

    if (has_hit) {
        const BodyHandle handle = static_cast<BodyHandle>(
            physicsSystem.GetBodyInterface().GetUserData(ray_hit.mBodyID));
        bool valid_handle = false;
        if (handle) {
            const std::scoped_lock shadow(shadowLock);
            valid_handle = QueryHandleIsValidLocked(*this, handle).has_value();
        }

        if (valid_handle) {
            JPH::Vec3 normal;
            const JPH::BodyLockRead body_lock(physicsSystem.GetBodyLockInterfaceNoLock(),
                                              ray_hit.mBodyID);
            if (body_lock.Succeeded()) {
                const JPH::RVec3 hit_position = ray.GetPointOnRay(ray_hit.mFraction);
                normal = body_lock.GetBody().GetWorldSpaceSurfaceNormal(ray_hit.mSubShapeID2,
                                                                       hit_position);
            }

            hit.handle   = handle;
            hit.fraction = ray_hit.mFraction;
            hit.normal   = normal;
            hit.hasHit  = true;
        }
    }
    QueryEnd();
    return hit;
}

std::expected<void, Error> PhysicsWorld::RaycastBatch(
    std::span<const float> starts, std::span<const float> directions, float max_distance,
    std::span<RayCastBatchResult> results) {
    const size_t count = results.size();
    if (starts.size() != 3 * count || directions.size() != 3 * count) {
        return std::unexpected(QueryError::InvalidArgument);
    }
    if (count == 0) {
        return {};
    }

    std::memset(results.data(), 0, count * sizeof(RayCastBatchResult));
    QueryBegin(this);

    const JPH::NarrowPhaseQuery &query      = physicsSystem.GetNarrowPhaseQuery();
    const JPH::BodyLockInterface &lock_interface = physicsSystem.GetBodyLockInterfaceNoLock();
    JPH::BodyInterface &body_interface       = physicsSystem.GetBodyInterface();

    for (size_t i = 0; i < count; ++i) {
        const size_t offset = i * 3;
        const JPH::Vec3 direction(directions[offset], directions[offset + 1],
                                  directions[offset + 2]);
        float magnitude_sq = 0.0f;
        const float scale  = RayDirectionScale(direction, max_distance, &magnitude_sq);
        if (magnitude_sq < kQueryDirectionEpsilonSquared) {
            continue;
        }

        const JPH::Vec3 ray_direction(direction.GetX() * scale, direction.GetY() * scale,
                                      direction.GetZ() * scale);
        const JPH::RVec3 origin((double)starts[offset], (double)starts[offset + 1],
                                (double)starts[offset + 2]);
        const JPH::RRayCast ray(origin, ray_direction);
        JPH::RayCastResult ray_hit;

        // A batched ray takes no filters, as before: it ignores nothing.
        if (!query.CastRay(ray, ray_hit)) {
            continue;
        }

        const BodyHandle handle =
            static_cast<BodyHandle>(body_interface.GetUserData(ray_hit.mBodyID));
        if (!handle) {
            continue;
        }

        RayCastBatchResult *result = &results[i];
        result->handle             = handle.Raw();
        result->fraction           = ray_hit.mFraction;
        result->subShapeID         = ray_hit.mSubShapeID2.GetValue();

        const JPH::BodyLockRead body_lock(lock_interface, ray_hit.mBodyID);
        if (body_lock.Succeeded()) {
            const JPH::RVec3 hit_position = ray.GetPointOnRay(ray_hit.mFraction);
            const JPH::Vec3 normal =
                body_lock.GetBody().GetWorldSpaceSurfaceNormal(ray_hit.mSubShapeID2, hit_position);
            result->nx = normal.GetX();
            result->ny = normal.GetY();
            result->nz = normal.GetZ();
            result->px = (float)hit_position.GetX();
            result->py = (float)hit_position.GetY();
            result->pz = (float)hit_position.GetZ();
        }
    }

    // Validate all collected body handles and resolve their material IDs while
    // the slot map is protected. active_queries prevents step/resize from
    // invalidating this batch while the Jolt queries are in flight.
    const std::scoped_lock shadow(shadowLock);
    const size_t body_count = this->count.load(std::memory_order::relaxed);
    for (size_t i = 0; i < results.size(); ++i) {
        RayCastBatchResult *result = &results[i];
        const std::optional<uint32_t> slot = QueryHandleIsValidLocked(*this, BodyHandle(result->handle));
        if (!slot) {
            std::memset(result, 0, sizeof(*result));
            continue;
        }

        const uint32_t dense = bodies.slotToDense[*slot];
        if (dense < body_count && bodies.materialIds) {
            result->materialId = bodies.materialIds[dense];
        }
    }

    QueryEnd();
    return {};
}

std::expected<PhysicsShapecastHit, Error> PhysicsWorld::Shapecast(
    int shape_type, JPH::RVec3 position, JPH::Quat rotation, JPH::Vec3 sweep_direction,
    std::span<const float, 4> size, BodyHandle ignore_handle) {
    PhysicsShapecastHit hit{};

    const float magnitude_sq = sweep_direction.GetX() * sweep_direction.GetX() +
                               sweep_direction.GetY() * sweep_direction.GetY() +
                               sweep_direction.GetZ() * sweep_direction.GetZ();
    if (magnitude_sq < kQueryDirectionEpsilonSquared) {
        return hit;
    }

    JPH::BodyID ignore_body; // invalid until resolved: "ignore nothing"
    const JPH::ShapeRefC shape =
        QueryBeginWithShape(this, shape_type, size, &ignore_body, ignore_handle);
    if (!shape) {
        return std::unexpected(QueryError::ShapeFailure);
    }

    const JPH::RMat44 transform = JPH::RMat44::sRotationTranslation(rotation, position);
    ClosestFractionCastShapeCollector collector;
    ShapecastExecuteInternal(*this, shape, transform, sweep_direction, ignore_body, collector);

    if (collector.has_hit()) {
        const JPH::ShapeCastResult &cast_hit = collector.hit();
        const BodyHandle handle = static_cast<BodyHandle>(
            physicsSystem.GetBodyInterface().GetUserData(cast_hit.mBodyID2));
        bool valid_handle = false;
        if (handle) {
            const std::scoped_lock shadow(shadowLock);
            valid_handle = QueryHandleIsValidLocked(*this, handle).has_value();
        }

        if (valid_handle) {
            float nx                     = -cast_hit.mPenetrationAxis.GetX();
            float ny                     = -cast_hit.mPenetrationAxis.GetY();
            float nz                     = -cast_hit.mPenetrationAxis.GetZ();
            const float normal_length_sq = nx * nx + ny * ny + nz * nz;
            if (normal_length_sq > kQueryDirectionEpsilonSquared) {
                const float inverse_length = FastRsqrt(normal_length_sq);
                nx *= inverse_length;
                ny *= inverse_length;
                nz *= inverse_length;
            }

            hit.handle   = handle;
            hit.fraction = cast_hit.mFraction;
            hit.point    = cast_hit.mContactPointOn2;
            hit.normal   = JPH::Vec3(nx, ny, nz);
            hit.hasHit  = true;
        }
    }

    QueryEnd();
    return hit;
}

} // namespace CPH
