#include "physics/ContactListener.h"
#include "physics/ContactEventDefinitions.h"
#include "physics/PhysicsWorldState.h"

// Include native Jolt headers for zero-cost abstractions and SIMD math
#include <Jolt/Jolt.h>
#include <Jolt/Physics/Body/Body.h>
#include <Jolt/Physics/Collision/ContactListener.h>

#include <atomic>
#include <cmath>
#include <cstdint>

namespace {

// =================================================================================================
// TMP EVENT GENERATOR
// Isolates thread synchronization, bounds checking, and canonicalization into a single template.
// =================================================================================================
template <auto EventType, typename InfoExtractor>
[[gnu::no_sanitize("thread")]]
inline void RecordContact(CPH_PhysicsWorld *world, uint64_t r1, uint64_t r2,
                          InfoExtractor &&extract_info) noexcept {
    // 1. Thread-safe Index Allocation
    const size_t idx =
        std::atomic_fetch_add_explicit(&world->contact_atomic_idx, 1, std::memory_order_relaxed);
    if (idx >= world->contact_max_capacity) [[unlikely]] {
        return;
    }

    // 2. Fetch the newly structured 128-byte event
    CPH_ContactEvent *ev_base   = GetEventAt(world->contact_buffer, idx);
    CPH_ContactEventSlim *slim  = GetSlimHeader(ev_base);
    CPH_ContactEventFatExt *fat = GetFatExtension(ev_base);

    // 3. Set Event Type
    slim->flags = static_cast<uint32_t>(EventType);

    // 4. Canonicalize handles (Always sort lowest-to-highest for determinism)
    const bool swapped = (r1 > r2);
    std::atomic_store_explicit(&slim->body1, swapped ? r2 : r1, std::memory_order_relaxed);
    std::atomic_store_explicit(&slim->body2, swapped ? r1 : r2, std::memory_order_relaxed);

    // 5. Inject event-specific logic (Inlined lambda)
    extract_info(slim, fat, swapped);

    // 6. Memory Fence to publish the event to the Python extraction thread
    std::atomic_thread_fence(std::memory_order_release);
}

// =================================================================================================
// UNIFIED MANIFOLD PROCESSOR (Added / Persisted)
// Leverages native JPH::Vec3 math for vectorized AVX/SSE acceleration.
// =================================================================================================
template <auto EventType>
inline void ProcessManifoldEvent(CPH_PhysicsWorld *world, const JPH_Body *c_body1,
                                 const JPH_Body *c_body2,
                                 const JPH_ContactManifold *c_manifold) noexcept {

    // Zero-cost cast to native Jolt objects
    const auto *body1    = reinterpret_cast<const JPH::Body *>(c_body1);
    const auto *body2    = reinterpret_cast<const JPH::Body *>(c_body2);
    const auto *manifold = reinterpret_cast<const JPH::ContactManifold *>(c_manifold);

    const uint64_t r1 = body1->GetUserData();
    const uint64_t r2 = body2->GetUserData();

    const auto slot1 = static_cast<uint32_t>(r1 & kHandleIndexMask);
    const auto slot2 = static_cast<uint32_t>(r2 & kHandleIndexMask);

    if (slot1 >= world->slot_capacity || slot2 >= world->slot_capacity) [[unlikely]] {
        return;
    }

    const uint32_t idx1 = world->slot_to_dense[slot1];
    const uint32_t idx2 = world->slot_to_dense[slot2];

    // Collision Bitmask Validation
    if (!(world->categories[idx1] & world->masks[idx2]) ||
        !(world->categories[idx2] & world->masks[idx1])) {
        return;
    }

    // Defer memory allocation and math to the unified generator
    RecordContact<EventType>(
        world, r1, r2, [&](CPH_ContactEventSlim *slim, CPH_ContactEventFatExt *fat, bool swapped) -> auto {
            // Native SIMD Vector Loading
            JPH::Vec3 n = manifold->mWorldSpaceNormal;
            if (swapped) {
                n = -n;
            }

            slim->nx = n.GetX();
            slim->ny = n.GetY();
            slim->nz = n.GetZ();

            const JPH::RVec3 p = manifold->GetWorldSpaceContactPointOn1(0);

            slim->px = static_cast<JPH::Real>(p.GetX());
            slim->py = static_cast<JPH::Real>(p.GetY());
            slim->pz = static_cast<JPH::Real>(p.GetZ());

            if (body1->IsSensor() || body2->IsSensor()) {
                slim->impulse       = 0.0f;
                slim->sliding_speed = 0.0f;
            } else {
                // Native Branchless Velocity Math
                const JPH::Vec3 v1 =
                    body1->IsStatic() ? JPH::Vec3::sZero() : body1->GetLinearVelocity();
                const JPH::Vec3 v2 =
                    body2->IsStatic() ? JPH::Vec3::sZero() : body2->GetLinearVelocity();

                const JPH::Vec3 dv = swapped ? (v2 - v1) : (v1 - v2);

                // SIMD Dot Product & Length
                const float dot     = dv.Dot(n);
                slim->impulse       = std::abs(dot);
                slim->sliding_speed = dv.LengthSq() - (dot * dot);
            }

            fat->mat1 = world->material_ids[idx1];
            fat->mat2 = world->material_ids[idx2];
        });
}

// =================================================================================================
// JOLT CALLBACK TARGETS
// =================================================================================================

void JPH_API_CALL OnContactAdded(void *userData, const JPH_Body *body1, const JPH_Body *body2,
                                 const JPH_ContactManifold *manifold,
                                 JPH_ContactSettings * /*unused*/) noexcept {
    ProcessManifoldEvent<EVENT_ADDED>(static_cast<CPH_PhysicsWorld *>(userData), body1, body2,
                                      manifold);
}

void JPH_API_CALL OnContactPersisted(void *userData, const JPH_Body *body1, const JPH_Body *body2,
                                     const JPH_ContactManifold *manifold,
                                     JPH_ContactSettings * /*unused*/) noexcept {
    ProcessManifoldEvent<EVENT_PERSISTED>(static_cast<CPH_PhysicsWorld *>(userData), body1, body2,
                                          manifold);
}

[[gnu::no_sanitize("thread")]]
void JPH_API_CALL OnContactRemoved(void *userData, const JPH_SubShapeIDPair *pair) noexcept {
    auto *world = static_cast<CPH_PhysicsWorld *>(userData);

    const uint32_t i1 = pair->Body1ID & JPH::BodyID::cMaxBodyIndex;
    const uint32_t i2 = pair->Body2ID & JPH::BodyID::cMaxBodyIndex;

    uint64_t r1 = 0;
    uint64_t r2 = 0;

    if (world->id_to_handle_map != nullptr) [[likely]] {
        if (i1 <= world->max_jolt_bodies) {
            r1 = std::atomic_load_explicit(&world->id_to_handle_map[i1], std::memory_order_acquire);
        }
        if (i2 <= world->max_jolt_bodies) {
            r2 = std::atomic_load_explicit(&world->id_to_handle_map[i2], std::memory_order_acquire);
        }
    }

    if (r1 == 0 || r2 == 0) [[unlikely]] {
        return;
    }

    // Removed bodies only receive a subset of the data
    RecordContact<EVENT_REMOVED>(
        world, r1, r2,
        [](CPH_ContactEventSlim *slim, CPH_ContactEventFatExt * /*fat*/, bool /*swapped*/) -> void {
            slim->px            = 0.0f;
            slim->py            = 0.0f;
            slim->pz            = 0.0f;
            slim->nx            = 0.0f;
            slim->ny            = 0.0f;
            slim->nz            = 0.0f;
            slim->impulse       = 0.0f;
            slim->sliding_speed = 0.0f;
        });
}

auto JPH_API_CALL OnContactValidate(void *userData, const JPH_Body *c_body1,
                                    const JPH_Body *c_body2, const JPH_RVec3 * /*unused*/,
                                    const JPH_CollideShapeResult * /*unused*/) noexcept
    -> JPH_ValidateResult {
    auto *world = static_cast<CPH_PhysicsWorld *>(userData);

    const auto *body1 = reinterpret_cast<const JPH::Body *>(c_body1);
    const auto *body2 = reinterpret_cast<const JPH::Body *>(c_body2);

    const uint64_t r1 = body1->GetUserData();
    const uint64_t r2 = body2->GetUserData();

    const auto slot1 = static_cast<uint32_t>(r1 & kHandleIndexMask);
    const auto slot2 = static_cast<uint32_t>(r2 & kHandleIndexMask);

    if (slot1 >= world->slot_capacity || slot2 >= world->slot_capacity) [[unlikely]] {
        return JPH_ValidateResult_RejectContact;
    }

    const uint32_t idx1 = world->slot_to_dense[slot1];
    const uint32_t idx2 = world->slot_to_dense[slot2];

    const uint32_t cat1  = world->categories[idx1];
    const uint32_t mask1 = world->masks[idx1];
    const uint32_t cat2  = world->categories[idx2];
    const uint32_t mask2 = world->masks[idx2];

    if (((cat1 & mask2) == 0u) || ((cat2 & mask1) == 0u)) {
        return JPH_ValidateResult_RejectContact;
    }

    return JPH_ValidateResult_AcceptContact;
}

} // namespace

// =================================================================================================
// EXPORT STRUCTURE
// =================================================================================================

extern "C" const JPH_ContactListener_Procs contact_procs = {.OnContactValidate = OnContactValidate,
                                                            .OnContactAdded    = OnContactAdded,
                                                            .OnContactPersisted =
                                                                OnContactPersisted,
                                                            .OnContactRemoved = OnContactRemoved};