#pragma once

// Native contact listener (Phase 2, step A3 — see docs/CPP_INTERFACE_PLAN.md).
// One instance lives inside PhysicsWorld for the world's lifetime; it replaces
// the JoltC proc table that used to be registered process-wide at module init.
//
// Jolt calls these callbacks from the simulation threads with all bodies
// locked, so they must not allocate unboundedly or take locks. The event
// records they fill are 128-byte POD slots in PhysicsWorld::contact_buffer,
// claimed atomically and published with a release fence.

#include "jolt/Jolt.hpp"
#include <Jolt/Physics/Collision/ContactListener.h>


namespace CPH {

struct PhysicsWorld;

class WorldContactListener final : public JPH::ContactListener {
public:
    explicit WorldContactListener(PhysicsWorld *world) noexcept : world_(world) {}

    [[nodiscard]] JPH::ValidateResult
    OnContactValidate(const JPH::Body &inBody1, const JPH::Body &inBody2,
                      JPH::RVec3Arg inBaseOffset,
                      const JPH::CollideShapeResult &inCollisionResult) noexcept override;
    void OnContactAdded(const JPH::Body &inBody1, const JPH::Body &inBody2,
                        const JPH::ContactManifold &inManifold,
                        JPH::ContactSettings &ioSettings) noexcept override;
    void OnContactPersisted(const JPH::Body &inBody1, const JPH::Body &inBody2,
                            const JPH::ContactManifold &inManifold,
                            JPH::ContactSettings &ioSettings) noexcept override;
    void OnContactRemoved(const JPH::SubShapeIDPair &inSubShapePair) noexcept override;

private:
    PhysicsWorld *world_;
};

} // namespace CPH
