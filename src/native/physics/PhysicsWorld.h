#pragma once

#include "physics/PhysicsParams.h"
#include "physics/PhysicsWorldState.h"

// Native operations never call the Python API. The CPython binding translates
// native status codes into exceptions where appropriate.
#ifdef __cplusplus
extern "C" {
#endif

void CPH_PhysicsWorldInitState(CPH_PhysicsWorld *world);
int CPH_AllocateBuffers(CPH_PhysicsWorld *world, int max_bodies, size_t initial_count);
int CPH_AllocateConstraints(CPH_PhysicsWorld *world);
void CPH_InitializeFreeSlots(CPH_PhysicsWorld *world, size_t start_idx);
int CPH_PhysicsWorldInitJolt(CPH_PhysicsWorld *world, CPH_WorldSettings settings, CPH_GravityVector gravity);
PhysicsWorldStatus CPH_PhysicsWorldLoadBakedScene(CPH_PhysicsWorld *world, const CPH_BakedSceneData *scene);

// Slot and resize mutations require shadow_lock and a non-stepping world.
PhysicsWorldStatus CPH_PhysicsWorldResize(CPH_PhysicsWorld *world, size_t new_capacity);
PhysicsWorldStatus CPH_PhysicsWorldCommitCreate(CPH_PhysicsWorld *world,
                                              JPH_BodyCreationSettings *settings,
                                              uint32_t slot_state, BodyHandle *out_handle);
PhysicsWorldStatus CPH_PhysicsWorldCommitCreateSoft(CPH_PhysicsWorld *world,
                                                    JPH_SoftBodyCreationSettings *settings,
                                                    uint32_t slot_state, BodyHandle *out_handle);
void CPH_ConfigureBodySettings(JPH_BodyCreationSettings *settings, JPH_Shape *shape, CPH_BodyConfig config);
void CPH_ApplyBodyCreationProps(JPH_BodyCreationSettings *settings, JPH_Shape *shape,
                               CPH_BodyCreationProps props);

// Caller holds shadow_lock while transferring ownership of the active queue.
PhysicsCommand *CPH_PhysicsWorldDetachCommandQueue(CPH_PhysicsWorld *world, size_t *command_count);

// Called with the GIL released after the binding has marked the world stepping
// and detached the command queue; performs the Jolt update and shadow sync.
void CPH_PhysicsWorldStepNative(CPH_PhysicsWorld *world, float dt, PhysicsCommand *queue,
                               size_t command_count);
// Caller holds shadow_lock; finalizes native counters/flags and wakes waiters.
size_t CPH_PhysicsWorldFinalizeStep(CPH_PhysicsWorld *world, float dt);
int CPH_VerifyAbiAlignment(JPH_BodyInterface *body_interface);

void CPH_FreeNewBuffers(CPH_NewBuffers *buffers);
void CPH_FreeConstraints(CPH_PhysicsWorld *world);
void CPH_FreeShadowBuffers(CPH_PhysicsWorld *world);
void CPH_PhysicsWorldFreeNative(CPH_PhysicsWorld *world);

#ifdef __cplusplus
}
#endif
