#pragma once

// Core world lifecycle and stepping. Native operations never call the
// Python API; the CPython binding translates status codes to exceptions.
// Construction/destruction of CPH_PhysicsWorld itself (member state, Jolt
// handle teardown) is the struct's constructor/destructor.

#include "physics/CommandBuffer.hpp"
#include "physics/PhysicsParams.hpp"
#include "physics/PhysicsWorldState.hpp"

#include <cstddef>
#include <span>

// Allocates body/constraint/contact storage for `max_bodies`, with room
// for `initial_count` bodies available immediately. Returns 0 / -1 (OOM).
int CPH_AllocateBuffers(CPH_PhysicsWorld *world, int max_bodies, std::size_t initial_count);
void CPH_InitializeFreeSlots(CPH_PhysicsWorld *world, std::size_t start_idx);

int CPH_PhysicsWorldInitJolt(CPH_PhysicsWorld *world, CPH_WorldSettings settings,
                             CPH_GravityVector gravity);
PhysicsWorldStatus CPH_PhysicsWorldLoadBakedScene(CPH_PhysicsWorld *world,
                                                  const CPH_BakedSceneData *scene);

// Slot and resize mutations require shadow_lock and a non-stepping world.
PhysicsWorldStatus CPH_PhysicsWorldResize(CPH_PhysicsWorld *world, std::size_t new_capacity);
PhysicsWorldStatus CPH_PhysicsWorldCommitCreate(CPH_PhysicsWorld *world,
                                                 JPH_BodyCreationSettings *settings,
                                                 std::uint32_t slot_state, BodyHandle *out_handle);
PhysicsWorldStatus CPH_PhysicsWorldCommitCreateSoft(CPH_PhysicsWorld *world,
                                                     JPH_SoftBodyCreationSettings *settings,
                                                     std::uint32_t slot_state,
                                                     BodyHandle *out_handle);
void CPH_ConfigureBodySettings(JPH_BodyCreationSettings *settings, JPH_Shape *shape,
                               CPH_BodyConfig config);
void CPH_ApplyBodyCreationProps(JPH_BodyCreationSettings *settings, JPH_Shape *shape,
                                CPH_BodyCreationProps props);

// Caller holds shadow_lock. Swaps the active command queue with the spare
// and returns the captured commands (still owned by the world's spare
// buffer; flush them before the next detach).
std::span<PhysicsCommand> CPH_PhysicsWorldDetachCommands(CPH_PhysicsWorld *world);

// Called with the GIL released after the binding marked the world stepping
// and detached the commands; flushes commands, steps Jolt, syncs shadows.
void CPH_PhysicsWorldStepNative(CPH_PhysicsWorld *world, float dt,
                                std::span<PhysicsCommand> commands);

// Caller holds shadow_lock; finalizes step counters/flags and wakes waiters.
std::size_t CPH_PhysicsWorldFinalizeStep(CPH_PhysicsWorld *world, float dt);

int CPH_VerifyAbiAlignment(JPH_BodyInterface *body_interface);
