#pragma once

#include "physics/PhysicsWorldState.h"

typedef enum PhysicsWorldRuntimeStatus : int {
    PHYSICS_WORLD_RUNTIME_STATUS_OK               = 0,
    PHYSICS_WORLD_RUNTIME_STATUS_OUT_OF_MEMORY    = -1,
    PHYSICS_WORLD_RUNTIME_STATUS_STALE_HANDLE     = -2,
    PHYSICS_WORLD_RUNTIME_STATUS_PENDING_CREATE   = -3,
    PHYSICS_WORLD_RUNTIME_STATUS_OVERFLOW         = -4,
    PHYSICS_WORLD_RUNTIME_STATUS_INVALID_ARGUMENT = -5,
    PHYSICS_WORLD_RUNTIME_STATUS_WRONG_BODY_TYPE  = -6,
    PHYSICS_WORLD_RUNTIME_STATUS_OUT_OF_RANGE     = -7,
    PHYSICS_WORLD_RUNTIME_STATUS_MISSING_BUFFER   = -8,
} PhysicsWorldRuntimeStatus;

typedef enum PhysicsWorldSnapshotStatus : int {
    PHYSICS_WORLD_SNAPSHOT_STATUS_OK                = 0,
    PHYSICS_WORLD_SNAPSHOT_STATUS_OUT_OF_MEMORY     = -1,
    PHYSICS_WORLD_SNAPSHOT_STATUS_INVALID_ARGUMENT  = -2,
    PHYSICS_WORLD_SNAPSHOT_STATUS_TRUNCATED         = -3,
    PHYSICS_WORLD_SNAPSHOT_STATUS_CAPACITY_MISMATCH = -4,
    PHYSICS_WORLD_SNAPSHOT_STATUS_OVERFLOW          = -5,
} PhysicsWorldSnapshotStatus;

typedef struct PhysicsBodyStats {
    PosStride position;
    AuxStride rotation;
    AuxStride linear_velocity;
} PhysicsBodyStats;

// Engine and native world-state operations. CPython bindings own argument
// parsing, result conversion, and translation of these status codes.
bool physics_world_attach_contact_listener(PhysicsWorld *world);
void physics_world_sync_loaded_state(PhysicsWorld *world, size_t saved_count);
PhysicsWorldRuntimeStatus physics_world_take_contact_events(PhysicsWorld *world,
                                                            struct ContactEvent **out_events,
                                                            size_t *out_count);
PhysicsWorldSnapshotStatus physics_world_save_snapshot(PhysicsWorld *world, uint8_t **out_data,
                                                       size_t *out_size);
PhysicsWorldSnapshotStatus physics_world_load_snapshot(PhysicsWorld *world, const uint8_t *data,
                                                       size_t data_size, size_t *out_saved_count,
                                                       size_t *out_world_capacity,
                                                       size_t *out_saved_capacity);
PhysicsWorldRuntimeStatus physics_world_set_gravity(PhysicsWorld *world, JPH_Vec3 gravity);
PhysicsWorldRuntimeStatus physics_world_get_gravity(PhysicsWorld *world, JPH_Vec3 *out_gravity);
PhysicsWorldRuntimeStatus physics_world_get_body_stats(PhysicsWorld *world, BodyHandle handle,
                                                       PhysicsBodyStats *out_stats);
PhysicsWorldRuntimeStatus physics_world_get_soft_body_vertex_count(PhysicsWorld *world,
                                                                   BodyHandle handle,
                                                                   uint32_t *out_count);
PhysicsWorldRuntimeStatus physics_world_get_soft_body_vertex_position(PhysicsWorld *world,
                                                                      BodyHandle handle,
                                                                      uint32_t vertex_index,
                                                                      JPH_Vec3 *out_position,
                                                                      uint32_t *out_count);
PhysicsWorldRuntimeStatus physics_world_get_soft_body_local_vertices(PhysicsWorld *world,
                                                                     BodyHandle handle,
                                                                     JPH_Vec3 **out_positions,
                                                                     uint32_t *out_count);
PhysicsWorldRuntimeStatus physics_world_apply_buoyancy(PhysicsWorld *world, BodyHandle handle,
                                                       JPH_Real surface_y, float buoyancy,
                                                       float linear_drag, float angular_drag,
                                                       JPH_Vec3 fluid_velocity, float dt,
                                                       bool *out_submerged);
PhysicsWorldRuntimeStatus physics_world_apply_buoyancy_batch(
    PhysicsWorld *world, const BodyHandle *handles, size_t handle_count, JPH_Real surface_y,
    float buoyancy, float linear_drag, float angular_drag, JPH_Vec3 fluid_velocity, float dt);
PhysicsWorldRuntimeStatus physics_world_get_motion_type(PhysicsWorld *world, BodyHandle handle,
                                                        JPH_MotionType *out_motion_type);
PhysicsWorldRuntimeStatus physics_world_get_user_data(PhysicsWorld *world, BodyHandle handle,
                                                      uint64_t *out_user_data);
PhysicsWorldRuntimeStatus physics_world_get_index(PhysicsWorld *world, BodyHandle handle,
                                                  uint32_t *out_index);
bool physics_world_is_alive(PhysicsWorld *world, BodyHandle handle);
PhysicsWorldRuntimeStatus physics_world_is_active(PhysicsWorld *world, BodyHandle handle,
                                                  bool *out_active);
PhysicsWorldRuntimeStatus
physics_world_get_active_indices(PhysicsWorld *world, uint32_t **out_indices, size_t *out_count);
PhysicsWorldRuntimeStatus physics_world_get_render_state(PhysicsWorld *world, float alpha,
                                                         float **out_state, size_t *out_count);
PhysicsWorldRuntimeStatus physics_world_set_collision_filter(PhysicsWorld *world, BodyHandle handle,
                                                             uint32_t category, uint32_t mask);
PhysicsWorldRuntimeStatus physics_world_register_material(PhysicsWorld *world, uint32_t id,
                                                          float friction, float restitution);
PhysicsWorldRuntimeStatus
physics_world_build_debug_data(PhysicsWorld *world, bool draw_shapes, bool draw_constraints,
                               bool draw_bounding_box, bool draw_centers, bool wireframe,
                               DebugVertex **out_lines, size_t *out_line_count,
                               DebugVertex **out_triangles, size_t *out_triangle_count);
