#pragma once

// Core world lifecycle and stepping. Native operations never call the
// Python API; the CPython binding translates errors to exceptions.
// Construction/destruction of PhysicsWorld itself (member state, Jolt
// handle teardown) is the struct's constructor/destructor.

#include "core/Error.hpp"
#include "physics/CommandBuffer.hpp"
#include "physics/PhysicsParams.hpp"
#include "physics/PhysicsWorldState.hpp"

#include <cstddef>
#include <expected>
#include <span>

// Allocates body/constraint/contact storage for `max_bodies`, with room
// for `initial_count` bodies available immediately. Returns 0 / -1 (OOM).

namespace CPH {

int AllocateBuffers(PhysicsWorld *world, int max_bodies, std::size_t initial_count);
void InitializeFreeSlots(PhysicsWorld *world, std::size_t start_idx);

int PhysicsWorldInitJolt(PhysicsWorld *world, WorldSettings settings,
                             GravityVector gravity);
[[nodiscard]] std::expected<void, Error> PhysicsWorldLoadBakedScene(PhysicsWorld *world,
                                                  const BakedSceneData *scene);

// Slot and resize mutations require shadow_lock and a non-stepping world.
[[nodiscard]] std::expected<void, Error> PhysicsWorldResize(PhysicsWorld *world,
                                                            std::size_t new_capacity);
[[nodiscard]] std::expected<void, Error> PhysicsWorldCommitCreate(
    PhysicsWorld *world, JPH::BodyCreationSettings &settings, std::uint32_t slot_state,
    BodyHandle *out_handle);
[[nodiscard]] std::expected<void, Error> PhysicsWorldCommitCreateSoft(
    PhysicsWorld *world, JPH::SoftBodyCreationSettings &settings, std::uint32_t slot_state,
    BodyHandle *out_handle);
void ConfigureBodySettings(JPH::BodyCreationSettings &settings, const JPH::Shape *shape,
                               BodyConfig config);
void ApplyBodyCreationProps(JPH::BodyCreationSettings &settings, const JPH::Shape *shape,
                                BodyCreationProps props);

// Caller holds shadow_lock. Swaps the active command queue with the spare
// and returns the captured commands (still owned by the world's spare
// buffer; flush them before the next detach).
std::span<PhysicsCommand> PhysicsWorldDetachCommands(PhysicsWorld *world);

// Called with the GIL released after the binding marked the world stepping
// and detached the commands; flushes commands, steps Jolt, syncs shadows.
void PhysicsWorldStepNative(PhysicsWorld *world, float dt,
                                std::span<PhysicsCommand> commands);

// Caller holds shadow_lock; finalizes step counters/flags and wakes waiters.
std::size_t PhysicsWorldFinalizeStep(PhysicsWorld *world, float dt);

int VerifyAbiAlignment(JPH_BodyInterface *body_interface);

} // namespace CPH
