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


namespace CPH {

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

struct PhysicsBodyStats {
    PosStride position;
    AuxStride rotation;
    AuxStride linear_velocity;
};

bool PhysicsWorldAttachContactListener(PhysicsWorld *world);
void PhysicsWorldSyncLoadedState(PhysicsWorld *world, std::size_t saved_count);

PhysicsWorldRuntimeStatus
PhysicsWorldTakeContactEvents(PhysicsWorld *world, Buffer<ContactEvent> &out_events);
PhysicsWorldSnapshotStatus PhysicsWorldSaveSnapshot(PhysicsWorld *world,
                                                         Buffer<std::uint8_t> &out_data);
PhysicsWorldSnapshotStatus PhysicsWorldLoadSnapshot(PhysicsWorld *world,
                                                         const std::uint8_t *data,
                                                         std::size_t data_size,
                                                         std::size_t *out_saved_count,
                                                         std::size_t *out_world_capacity,
                                                         std::size_t *out_saved_capacity);
PhysicsWorldRuntimeStatus PhysicsWorldSetGravity(PhysicsWorld *world, JPH_Vec3 gravity);
PhysicsWorldRuntimeStatus PhysicsWorldGetGravity(PhysicsWorld *world,
                                                      JPH_Vec3 *out_gravity);
PhysicsWorldRuntimeStatus PhysicsWorldGetBodyStats(PhysicsWorld *world, BodyHandle handle,
                                                        PhysicsBodyStats *out_stats);
PhysicsWorldRuntimeStatus
PhysicsWorldGetSoftBodyVertexCount(PhysicsWorld *world, BodyHandle handle,
                                        std::uint32_t *out_count);
PhysicsWorldRuntimeStatus
PhysicsWorldGetSoftBodyVertexPosition(PhysicsWorld *world, BodyHandle handle,
                                           std::uint32_t vertex_index, JPH_Vec3 *out_position,
                                           std::uint32_t *out_count);
PhysicsWorldRuntimeStatus
PhysicsWorldGetSoftBodyLocalVertices(PhysicsWorld *world, BodyHandle handle,
                                          Buffer<JPH_Vec3> &out_positions,
                                          std::uint32_t &out_count);
PhysicsWorldRuntimeStatus PhysicsWorldApplyBuoyancy(PhysicsWorld *world, BodyHandle handle,
                                                         JPH_Real surface_y, float buoyancy,
                                                         float linear_drag, float angular_drag,
                                                         JPH_Vec3 fluid_velocity, float dt,
                                                         bool *out_submerged);
PhysicsWorldRuntimeStatus PhysicsWorldApplyBuoyancyBatch(
    PhysicsWorld *world, const BodyHandle *handles, std::size_t handle_count,
    JPH_Real surface_y, float buoyancy, float linear_drag, float angular_drag,
    JPH_Vec3 fluid_velocity, float dt);
PhysicsWorldRuntimeStatus PhysicsWorldGetMotionType(PhysicsWorld *world, BodyHandle handle,
                                                         JPH_MotionType *out_motion_type);
PhysicsWorldRuntimeStatus PhysicsWorldGetUserData(PhysicsWorld *world, BodyHandle handle,
                                                       std::uint64_t *out_user_data);
PhysicsWorldRuntimeStatus PhysicsWorldGetIndex(PhysicsWorld *world, BodyHandle handle,
                                                    std::uint32_t *out_index);
bool PhysicsWorldIsAlive(PhysicsWorld *world, BodyHandle handle);
PhysicsWorldRuntimeStatus PhysicsWorldIsActive(PhysicsWorld *world, BodyHandle handle,
                                                    bool *out_active);
PhysicsWorldRuntimeStatus
PhysicsWorldGetActiveIndices(PhysicsWorld *world, Buffer<std::uint32_t> &out_indices,
                                  std::size_t &out_count);
PhysicsWorldRuntimeStatus PhysicsWorldGetRenderState(PhysicsWorld *world, float alpha,
                                                          Buffer<float> &out_state,
                                                          std::size_t &out_count);
PhysicsWorldRuntimeStatus
PhysicsWorldSetCollisionFilter(PhysicsWorld *world, BodyHandle handle,
                                    std::uint32_t category, std::uint32_t mask);
PhysicsWorldRuntimeStatus PhysicsWorldRegisterMaterial(PhysicsWorld *world,
                                                            std::uint32_t id, float friction,
                                                            float restitution);
PhysicsWorldRuntimeStatus
PhysicsWorldBuildDebugData(PhysicsWorld *world, bool draw_shapes, bool draw_constraints,
                                bool draw_bounding_box, bool draw_centers, bool wireframe,
                                Buffer<DebugVertex> &out_lines,
                                Buffer<DebugVertex> &out_triangles);

} // namespace CPH
