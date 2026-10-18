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

typedef struct CPH_PhysicsBodyStats {
    CPH_PosStride position;
    CPH_AuxStride rotation;
    CPH_AuxStride linear_velocity;
} CPH_PhysicsBodyStats;

// Engine and native world-state operations. CPython bindings own argument
// parsing, result conversion, and translation of these status codes.
bool CPH_PhysicsWorldAttachContactListener(CPH_PhysicsWorld *world);
void CPH_PhysicsWorldSyncLoadedState(CPH_PhysicsWorld *world, size_t saved_count);
PhysicsWorldRuntimeStatus CPH_PhysicsWorldTakeContactEvents(CPH_PhysicsWorld *world,
                                                            struct CPH_ContactEvent **out_events,
                                                            size_t *out_count);
PhysicsWorldSnapshotStatus CPH_PhysicsWorldSaveSnapshot(CPH_PhysicsWorld *world, uint8_t **out_data,
                                                       size_t *out_size);
PhysicsWorldSnapshotStatus CPH_PhysicsWorldLoadSnapshot(CPH_PhysicsWorld *world, const uint8_t *data,
                                                       size_t data_size, size_t *out_saved_count,
                                                       size_t *out_world_capacity,
                                                       size_t *out_saved_capacity);
PhysicsWorldRuntimeStatus CPH_PhysicsWorldSetGravity(CPH_PhysicsWorld *world, JPH_Vec3 gravity);
PhysicsWorldRuntimeStatus CPH_PhysicsWorldGetGravity(CPH_PhysicsWorld *world, JPH_Vec3 *out_gravity);
PhysicsWorldRuntimeStatus CPH_PhysicsWorldGetBodyStats(CPH_PhysicsWorld *world, BodyHandle handle,
                                                       CPH_PhysicsBodyStats *out_stats);
PhysicsWorldRuntimeStatus CPH_PhysicsWorldGetSoftBodyVertexCount(CPH_PhysicsWorld *world,
                                                                   BodyHandle handle,
                                                                   uint32_t *out_count);
PhysicsWorldRuntimeStatus CPH_PhysicsWorldGetSoftBodyVertexPosition(CPH_PhysicsWorld *world,
                                                                      BodyHandle handle,
                                                                      uint32_t vertex_index,
                                                                      JPH_Vec3 *out_position,
                                                                      uint32_t *out_count);
PhysicsWorldRuntimeStatus CPH_PhysicsWorldGetSoftBodyLocalVertices(CPH_PhysicsWorld *world,
                                                                     BodyHandle handle,
                                                                     JPH_Vec3 **out_positions,
                                                                     uint32_t *out_count);
PhysicsWorldRuntimeStatus CPH_PhysicsWorldApplyBuoyancy(CPH_PhysicsWorld *world, BodyHandle handle,
                                                       JPH_Real surface_y, float buoyancy,
                                                       float linear_drag, float angular_drag,
                                                       JPH_Vec3 fluid_velocity, float dt,
                                                       bool *out_submerged);
PhysicsWorldRuntimeStatus CPH_PhysicsWorldApplyBuoyancyBatch(
    CPH_PhysicsWorld *world, const BodyHandle *handles, size_t handle_count, JPH_Real surface_y,
    float buoyancy, float linear_drag, float angular_drag, JPH_Vec3 fluid_velocity, float dt);
PhysicsWorldRuntimeStatus CPH_PhysicsWorldGetMotionType(CPH_PhysicsWorld *world, BodyHandle handle,
                                                        JPH_MotionType *out_motion_type);
PhysicsWorldRuntimeStatus CPH_PhysicsWorldGetUserData(CPH_PhysicsWorld *world, BodyHandle handle,
                                                      uint64_t *out_user_data);
PhysicsWorldRuntimeStatus CPH_PhysicsWorldGetIndex(CPH_PhysicsWorld *world, BodyHandle handle,
                                                  uint32_t *out_index);
bool CPH_PhysicsWorldIsAlive(CPH_PhysicsWorld *world, BodyHandle handle);
PhysicsWorldRuntimeStatus CPH_PhysicsWorldIsActive(CPH_PhysicsWorld *world, BodyHandle handle,
                                                  bool *out_active);
PhysicsWorldRuntimeStatus
CPH_PhysicsWorldGetActiveIndices(CPH_PhysicsWorld *world, uint32_t **out_indices, size_t *out_count);
PhysicsWorldRuntimeStatus CPH_PhysicsWorldGetRenderState(CPH_PhysicsWorld *world, float alpha,
                                                         float **out_state, size_t *out_count);
PhysicsWorldRuntimeStatus CPH_PhysicsWorldSetCollisionFilter(CPH_PhysicsWorld *world, BodyHandle handle,
                                                             uint32_t category, uint32_t mask);
PhysicsWorldRuntimeStatus CPH_PhysicsWorldRegisterMaterial(CPH_PhysicsWorld *world, uint32_t id,
                                                          float friction, float restitution);
PhysicsWorldRuntimeStatus
CPH_PhysicsWorldBuildDebugData(CPH_PhysicsWorld *world, bool draw_shapes, bool draw_constraints,
                               bool draw_bounding_box, bool draw_centers, bool wireframe,
                               CPH_DebugVertex **out_lines, size_t *out_line_count,
                               CPH_DebugVertex **out_triangles, size_t *out_triangle_count);
