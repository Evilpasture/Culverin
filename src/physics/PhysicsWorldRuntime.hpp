#pragma once

// Engine and native world-state operations. CPython bindings own argument
// parsing, result conversion, and translation of these status codes.
// Functions that produce arrays fill an owning Buffer (nothrow); the
// caller never frees native memory by hand.

#include "core/Buffer.hpp"
#include "physics/ContactEventDefinitions.hpp"
#include "physics/PhysicsWorldState.hpp"

#include <cstddef>
#include <cstdint>

enum PhysicsWorldRuntimeStatus : int {
    PHYSICS_WORLD_RUNTIME_STATUS_OK               = 0,
    PHYSICS_WORLD_RUNTIME_STATUS_OUT_OF_MEMORY    = -1,
    PHYSICS_WORLD_RUNTIME_STATUS_STALE_HANDLE     = -2,
    PHYSICS_WORLD_RUNTIME_STATUS_PENDING_CREATE   = -3,
    PHYSICS_WORLD_RUNTIME_STATUS_OVERFLOW         = -4,
    PHYSICS_WORLD_RUNTIME_STATUS_INVALID_ARGUMENT = -5,
    PHYSICS_WORLD_RUNTIME_STATUS_WRONG_BODY_TYPE  = -6,
    PHYSICS_WORLD_RUNTIME_STATUS_OUT_OF_RANGE     = -7,
    PHYSICS_WORLD_RUNTIME_STATUS_MISSING_BUFFER   = -8,
};

enum PhysicsWorldSnapshotStatus : int {
    PHYSICS_WORLD_SNAPSHOT_STATUS_OK                = 0,
    PHYSICS_WORLD_SNAPSHOT_STATUS_OUT_OF_MEMORY     = -1,
    PHYSICS_WORLD_SNAPSHOT_STATUS_INVALID_ARGUMENT  = -2,
    PHYSICS_WORLD_SNAPSHOT_STATUS_TRUNCATED         = -3,
    PHYSICS_WORLD_SNAPSHOT_STATUS_CAPACITY_MISMATCH = -4,
    PHYSICS_WORLD_SNAPSHOT_STATUS_OVERFLOW          = -5,
};

struct CPH_PhysicsBodyStats {
    CPH_PosStride position;
    CPH_AuxStride rotation;
    CPH_AuxStride linear_velocity;
};

bool CPH_PhysicsWorldAttachContactListener(CPH_PhysicsWorld *world);
void CPH_PhysicsWorldSyncLoadedState(CPH_PhysicsWorld *world, std::size_t saved_count);

PhysicsWorldRuntimeStatus
CPH_PhysicsWorldTakeContactEvents(CPH_PhysicsWorld *world, Buffer<CPH_ContactEvent> &out_events);
PhysicsWorldSnapshotStatus CPH_PhysicsWorldSaveSnapshot(CPH_PhysicsWorld *world,
                                                         Buffer<std::uint8_t> &out_data);
PhysicsWorldSnapshotStatus CPH_PhysicsWorldLoadSnapshot(CPH_PhysicsWorld *world,
                                                         const std::uint8_t *data,
                                                         std::size_t data_size,
                                                         std::size_t *out_saved_count,
                                                         std::size_t *out_world_capacity,
                                                         std::size_t *out_saved_capacity);
PhysicsWorldRuntimeStatus CPH_PhysicsWorldSetGravity(CPH_PhysicsWorld *world, JPH_Vec3 gravity);
PhysicsWorldRuntimeStatus CPH_PhysicsWorldGetGravity(CPH_PhysicsWorld *world,
                                                      JPH_Vec3 *out_gravity);
PhysicsWorldRuntimeStatus CPH_PhysicsWorldGetBodyStats(CPH_PhysicsWorld *world, BodyHandle handle,
                                                        CPH_PhysicsBodyStats *out_stats);
PhysicsWorldRuntimeStatus
CPH_PhysicsWorldGetSoftBodyVertexCount(CPH_PhysicsWorld *world, BodyHandle handle,
                                        std::uint32_t *out_count);
PhysicsWorldRuntimeStatus
CPH_PhysicsWorldGetSoftBodyVertexPosition(CPH_PhysicsWorld *world, BodyHandle handle,
                                           std::uint32_t vertex_index, JPH_Vec3 *out_position,
                                           std::uint32_t *out_count);
PhysicsWorldRuntimeStatus
CPH_PhysicsWorldGetSoftBodyLocalVertices(CPH_PhysicsWorld *world, BodyHandle handle,
                                          Buffer<JPH_Vec3> &out_positions,
                                          std::uint32_t &out_count);
PhysicsWorldRuntimeStatus CPH_PhysicsWorldApplyBuoyancy(CPH_PhysicsWorld *world, BodyHandle handle,
                                                         JPH_Real surface_y, float buoyancy,
                                                         float linear_drag, float angular_drag,
                                                         JPH_Vec3 fluid_velocity, float dt,
                                                         bool *out_submerged);
PhysicsWorldRuntimeStatus CPH_PhysicsWorldApplyBuoyancyBatch(
    CPH_PhysicsWorld *world, const BodyHandle *handles, std::size_t handle_count,
    JPH_Real surface_y, float buoyancy, float linear_drag, float angular_drag,
    JPH_Vec3 fluid_velocity, float dt);
PhysicsWorldRuntimeStatus CPH_PhysicsWorldGetMotionType(CPH_PhysicsWorld *world, BodyHandle handle,
                                                         JPH_MotionType *out_motion_type);
PhysicsWorldRuntimeStatus CPH_PhysicsWorldGetUserData(CPH_PhysicsWorld *world, BodyHandle handle,
                                                       std::uint64_t *out_user_data);
PhysicsWorldRuntimeStatus CPH_PhysicsWorldGetIndex(CPH_PhysicsWorld *world, BodyHandle handle,
                                                    std::uint32_t *out_index);
bool CPH_PhysicsWorldIsAlive(CPH_PhysicsWorld *world, BodyHandle handle);
PhysicsWorldRuntimeStatus CPH_PhysicsWorldIsActive(CPH_PhysicsWorld *world, BodyHandle handle,
                                                    bool *out_active);
PhysicsWorldRuntimeStatus
CPH_PhysicsWorldGetActiveIndices(CPH_PhysicsWorld *world, Buffer<std::uint32_t> &out_indices,
                                  std::size_t &out_count);
PhysicsWorldRuntimeStatus CPH_PhysicsWorldGetRenderState(CPH_PhysicsWorld *world, float alpha,
                                                          Buffer<float> &out_state,
                                                          std::size_t &out_count);
PhysicsWorldRuntimeStatus
CPH_PhysicsWorldSetCollisionFilter(CPH_PhysicsWorld *world, BodyHandle handle,
                                    std::uint32_t category, std::uint32_t mask);
PhysicsWorldRuntimeStatus CPH_PhysicsWorldRegisterMaterial(CPH_PhysicsWorld *world,
                                                            std::uint32_t id, float friction,
                                                            float restitution);
PhysicsWorldRuntimeStatus
CPH_PhysicsWorldBuildDebugData(CPH_PhysicsWorld *world, bool draw_shapes, bool draw_constraints,
                                bool draw_bounding_box, bool draw_centers, bool wireframe,
                                Buffer<CPH_DebugVertex> &out_lines,
                                Buffer<CPH_DebugVertex> &out_triangles);
