#pragma once

#include "physics/PhysicsParams.h"
#include "physics/PhysicsWorldState.h"

// Native operations never call the Python API. The CPython binding translates
// native status codes into exceptions where appropriate.
void physics_world_init_state(PhysicsWorld *world);
int allocate_buffers(PhysicsWorld *world, int max_bodies, size_t initial_count);
int allocate_constraints(PhysicsWorld *world);
void initialize_free_slots(PhysicsWorld *world, size_t start_idx);
int physics_world_init_jolt(PhysicsWorld *world, WorldSettings settings, GravityVector gravity);
PhysicsWorldStatus physics_world_load_baked_scene(PhysicsWorld *world, const BakedSceneData *scene);

// Slot and resize mutations require shadow_lock and a non-stepping world.
PhysicsWorldStatus physics_world_resize(PhysicsWorld *world, size_t new_capacity);
PhysicsWorldStatus physics_world_commit_create(PhysicsWorld *world,
                                              JPH_BodyCreationSettings *settings,
                                              uint32_t slot_state, BodyHandle *out_handle);
PhysicsWorldStatus physics_world_commit_create_soft(PhysicsWorld *world,
                                                    JPH_SoftBodyCreationSettings *settings,
                                                    uint32_t slot_state, BodyHandle *out_handle);
void configure_body_settings(JPH_BodyCreationSettings *settings, JPH_Shape *shape, BodyConfig config);
void apply_body_creation_props(JPH_BodyCreationSettings *settings, JPH_Shape *shape,
                               BodyCreationProps props);

// Caller holds shadow_lock while transferring ownership of the active queue.
PhysicsCommand *physics_world_detach_command_queue(PhysicsWorld *world, size_t *command_count);

// Called with the GIL released after the binding has marked the world stepping
// and detached the command queue; performs the Jolt update and shadow sync.
void physics_world_step_native(PhysicsWorld *world, float dt, PhysicsCommand *queue,
                               size_t command_count);
// Caller holds shadow_lock; finalizes native counters/flags and wakes waiters.
size_t physics_world_finalize_step(PhysicsWorld *world, float dt);
int verify_abi_alignment(JPH_BodyInterface *body_interface);

void free_new_buffers(NewBuffers *buffers);
void free_constraints(PhysicsWorld *world);
void free_shadow_buffers(PhysicsWorld *world);
void physics_world_free_native(PhysicsWorld *world);
