#include "physics/ContactListener.hpp"

#include "physics/ContactEventDefinitions.hpp"
#include "physics/PhysicsWorldState.hpp"

// Native Jolt headers for zero-cost abstractions and SIMD math.
#include <Jolt/Physics/Body/Body.h>
#include <Jolt/Physics/Collision/Shape/SubShapeIDPair.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>


namespace CPH {

namespace {

// =================================================================================================
// TMP EVENT GENERATOR
// Isolates thread synchronization, bounds checking, and canonicalization into a single template.
// =================================================================================================
template <auto EventType, typename InfoExtractor>
CPH_NO_TSAN inline void RecordContact(PhysicsWorld *world, BodyHandle r1, BodyHandle r2,
                                      InfoExtractor &&extract_info) noexcept {
    // 1. Thread-safe Index Allocation
    const size_t idx =
        world->contactAtomicIdx.fetch_add(1, std::memory_order::relaxed);
    if (idx >= world->contactMaxCapacity) [[unlikely]] {
        return;
    }

    // 2. Fetch the newly structured 128-byte event. The buffer is
    // allocate_for_overwrite, so zero the claimed slot first: fields this
    // writer does not set are then defined zeros (legacy exposed
    // uninitialized bytes here).
    ContactEvent &event = world->contactBuffer[idx];
    std::memset(&event, 0, sizeof(event));
    ContactEventSlim *slim  = &event.slim;
    ContactEventFatExt *fat = &event.fat;

    // 3. Set Event Type
    slim->flags = static_cast<uint32_t>(EventType);

    // 4. Canonicalize handles (Always sort lowest-to-highest for determinism)
    const bool swapped = (r1 > r2);
    slim->body1.store(swapped ? r2 : r1, std::memory_order::relaxed);
    slim->body2.store(swapped ? r1 : r2, std::memory_order::relaxed);

    // 5. Inject event-specific logic (Inlined lambda)
    extract_info(slim, fat, swapped);

    // 6. Memory Fence to publish the event to the Python extraction thread
    std::atomic_thread_fence(std::memory_order::release);
}

// =================================================================================================
// UNIFIED MANIFOLD PROCESSOR (Added / Persisted)
// Leverages native JPH::Vec3 math for vectorized AVX/SSE acceleration.
// =================================================================================================
template <auto EventType>
inline void ProcessManifoldEvent(PhysicsWorld *world, const JPH::Body &body1,
                                 const JPH::Body &body2,
                                 const JPH::ContactManifold &manifold) noexcept {
    const BodyHandle r1(body1.GetUserData());
    const BodyHandle r2(body2.GetUserData());

    const auto slot1 = r1.Index();
    const auto slot2 = r2.Index();

    if (slot1 >= world->slotCapacity || slot2 >= world->slotCapacity) [[unlikely]] {
        return;
    }

    const uint32_t idx1 = world->bodies.slotToDense[slot1];
    const uint32_t idx2 = world->bodies.slotToDense[slot2];

    // Collision Bitmask Validation
    if (!(world->bodies.categories[idx1] & world->bodies.masks[idx2]) ||
        !(world->bodies.categories[idx2] & world->bodies.masks[idx1])) {
        return;
    }

    // Defer memory allocation and math to the unified generator
    RecordContact<EventType>(
        world, r1, r2,
        [&](ContactEventSlim *slim, ContactEventFatExt *fat, bool swapped) -> auto {
            // Native SIMD Vector Loading
            JPH::Vec3 n = manifold.mWorldSpaceNormal;
            if (swapped) {
                n = -n;
            }

            slim->nx = n.GetX();
            slim->ny = n.GetY();
            slim->nz = n.GetZ();

            const JPH::RVec3 p = manifold.GetWorldSpaceContactPointOn1(0);

            slim->px = static_cast<JPH::Real>(p.GetX());
            slim->py = static_cast<JPH::Real>(p.GetY());
            slim->pz = static_cast<JPH::Real>(p.GetZ());

            if (body1.IsSensor() || body2.IsSensor()) {
                slim->impulse       = 0.0f;
                slim->slidingSpeed = 0.0f;
            } else {
                // Native Branchless Velocity Math
                const JPH::Vec3 v1 =
                    body1.IsStatic() ? JPH::Vec3::sZero() : body1.GetLinearVelocity();
                const JPH::Vec3 v2 =
                    body2.IsStatic() ? JPH::Vec3::sZero() : body2.GetLinearVelocity();

                const JPH::Vec3 dv = swapped ? (v2 - v1) : (v1 - v2);

                // SIMD Dot Product & Length
                const float dot     = dv.Dot(n);
                slim->impulse       = std::abs(dot);
                slim->slidingSpeed = dv.LengthSq() - (dot * dot);
            }

            fat->mat1 = world->bodies.materialIds[idx1];
            fat->mat2 = world->bodies.materialIds[idx2];
        });
}

} // namespace

// =================================================================================================
// LISTENER ENTRY POINTS
// =================================================================================================

JPH::ValidateResult
WorldContactListener::OnContactValidate(const JPH::Body &inBody1, const JPH::Body &inBody2,
                                        [[maybe_unused]] JPH::RVec3Arg inBaseOffset,
                                        [[maybe_unused]] const JPH::CollideShapeResult
                                            &inCollisionResult) noexcept {
    const BodyHandle r1(inBody1.GetUserData());
    const BodyHandle r2(inBody2.GetUserData());

    const auto slot1 = r1.Index();
    const auto slot2 = r2.Index();

    if (slot1 >= world_->slotCapacity || slot2 >= world_->slotCapacity) [[unlikely]] {
        return JPH::ValidateResult::RejectContact;
    }

    const uint32_t idx1 = world_->bodies.slotToDense[slot1];
    const uint32_t idx2 = world_->bodies.slotToDense[slot2];

    const uint32_t cat1  = world_->bodies.categories[idx1];
    const uint32_t mask1 = world_->bodies.masks[idx1];
    const uint32_t cat2  = world_->bodies.categories[idx2];
    const uint32_t mask2 = world_->bodies.masks[idx2];

    if (((cat1 & mask2) == 0u) || ((cat2 & mask1) == 0u)) {
        return JPH::ValidateResult::RejectContact;
    }

    return JPH::ValidateResult::AcceptContact;
}

void WorldContactListener::OnContactAdded(const JPH::Body &inBody1, const JPH::Body &inBody2,
                                          const JPH::ContactManifold &inManifold,
                                          [[maybe_unused]] JPH::ContactSettings &ioSettings) noexcept {
    ProcessManifoldEvent<ContactEventType::Added>(world_, inBody1, inBody2, inManifold);
}

void WorldContactListener::OnContactPersisted(const JPH::Body &inBody1, const JPH::Body &inBody2,
                                              const JPH::ContactManifold &inManifold,
                                              [[maybe_unused]] JPH::ContactSettings &ioSettings) noexcept {
    ProcessManifoldEvent<ContactEventType::Persisted>(world_, inBody1, inBody2, inManifold);
}

CPH_NO_TSAN
void WorldContactListener::OnContactRemoved(const JPH::SubShapeIDPair &inSubShapePair) noexcept {
    const uint32_t i1 = inSubShapePair.GetBody1ID().GetIndex();
    const uint32_t i2 = inSubShapePair.GetBody2ID().GetIndex();

    BodyHandle r1;
    BodyHandle r2;

    if (world_->idToHandleMap) [[likely]] {
        if (i1 <= world_->maxJoltBodies) {
            r1 = world_->idToHandleMap[i1].load(std::memory_order::acquire);
        }
        if (i2 <= world_->maxJoltBodies) {
            r2 = world_->idToHandleMap[i2].load(std::memory_order::acquire);
        }
    }

    if (!r1.IsValid() || !r2.IsValid()) [[unlikely]] {
        return;
    }

    // Removed bodies only receive a subset of the data
    RecordContact<ContactEventType::Removed>(
        world_, r1, r2,
        [](ContactEventSlim *slim, ContactEventFatExt * /*fat*/, bool /*swapped*/) -> void {
            slim->px            = 0.0f;
            slim->py            = 0.0f;
            slim->pz            = 0.0f;
            slim->nx            = 0.0f;
            slim->ny            = 0.0f;
            slim->nz            = 0.0f;
            slim->impulse       = 0.0f;
            slim->slidingSpeed = 0.0f;
        });
}

} // namespace CPH
