#pragma once

// Engine and native world-state operations. CPython bindings own argument
// parsing, result conversion, and translation of these errors.
// Functions that produce arrays fill an owning Buffer (nothrow); the
// caller never frees native memory by hand.

#include "core/Buffer.hpp"
#include "core/Error.hpp"
#include "physics/ContactEventDefinitions.hpp"
#include "physics/PhysicsWorldState.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>


namespace CPH {

enum class WorldRuntimeError : std::uint32_t {
    OutOfMemory     = 1,
    StaleHandle     = 2,
    PendingCreate   = 3,
    Overflow        = 4,
    InvalidArgument = 5,
    WrongBodyType   = 6,
    OutOfRange      = 7,
    MissingBuffer   = 8,
};

enum class WorldSnapshotError : std::uint32_t {
    OutOfMemory      = 1,
    InvalidArgument  = 2,
    Truncated        = 3,
    CapacityMismatch = 4,
    Overflow         = 5,
};

struct PhysicsBodyStats {
    PosStride position;
    AuxStride rotation;
    AuxStride linear_velocity;
};

bool PhysicsWorldAttachContactListener(PhysicsWorld *world);
void PhysicsWorldSyncLoadedState(PhysicsWorld *world, std::size_t saved_count);

[[nodiscard]] std::expected<void, Error>
PhysicsWorldTakeContactEvents(PhysicsWorld *world, Buffer<ContactEvent> &out_events);
[[nodiscard]] std::expected<void, Error> PhysicsWorldSaveSnapshot(PhysicsWorld *world,
                                                         Buffer<std::uint8_t> &out_data);
[[nodiscard]] std::expected<void, Error> PhysicsWorldLoadSnapshot(PhysicsWorld *world,
                                                         const std::uint8_t *data,
                                                         std::size_t data_size,
                                                         std::size_t *out_saved_count,
                                                         std::size_t *out_world_capacity,
                                                         std::size_t *out_saved_capacity);
[[nodiscard]] std::expected<void, Error> PhysicsWorldSetGravity(PhysicsWorld *world,
                                                                JPH_Vec3 gravity);
[[nodiscard]] std::expected<void, Error> PhysicsWorldGetGravity(PhysicsWorld *world,
                                                      JPH_Vec3 *out_gravity);
[[nodiscard]] std::expected<void, Error>
PhysicsWorldGetBodyStats(PhysicsWorld *world, BodyHandle handle, PhysicsBodyStats *out_stats);
[[nodiscard]] std::expected<void, Error>
PhysicsWorldGetSoftBodyVertexCount(PhysicsWorld *world, BodyHandle handle,
                                        std::uint32_t *out_count);
[[nodiscard]] std::expected<void, Error>
PhysicsWorldGetSoftBodyVertexPosition(PhysicsWorld *world, BodyHandle handle,
                                           std::uint32_t vertex_index, JPH_Vec3 *out_position,
                                           std::uint32_t *out_count);
[[nodiscard]] std::expected<void, Error>
PhysicsWorldGetSoftBodyLocalVertices(PhysicsWorld *world, BodyHandle handle,
                                          Buffer<JPH_Vec3> &out_positions,
                                          std::uint32_t &out_count);
[[nodiscard]] std::expected<void, Error>
PhysicsWorldApplyBuoyancy(PhysicsWorld *world, BodyHandle handle, JPH_Real surface_y,
                          float buoyancy, float linear_drag, float angular_drag,
                          JPH_Vec3 fluid_velocity, float dt, bool *out_submerged);
[[nodiscard]] std::expected<void, Error> PhysicsWorldApplyBuoyancyBatch(
    PhysicsWorld *world, const BodyHandle *handles, std::size_t handle_count,
    JPH_Real surface_y, float buoyancy, float linear_drag, float angular_drag,
    JPH_Vec3 fluid_velocity, float dt);
[[nodiscard]] std::expected<void, Error>
PhysicsWorldGetMotionType(PhysicsWorld *world, BodyHandle handle, JPH_MotionType *out_motion_type);
[[nodiscard]] std::expected<void, Error>
PhysicsWorldGetUserData(PhysicsWorld *world, BodyHandle handle, std::uint64_t *out_user_data);
[[nodiscard]] std::expected<void, Error>
PhysicsWorldGetIndex(PhysicsWorld *world, BodyHandle handle, std::uint32_t *out_index);
bool PhysicsWorldIsAlive(PhysicsWorld *world, BodyHandle handle);
[[nodiscard]] std::expected<void, Error>
PhysicsWorldIsActive(PhysicsWorld *world, BodyHandle handle, bool *out_active);
[[nodiscard]] std::expected<void, Error>
PhysicsWorldGetActiveIndices(PhysicsWorld *world, Buffer<std::uint32_t> &out_indices,
                                  std::size_t &out_count);
[[nodiscard]] std::expected<void, Error>
PhysicsWorldGetRenderState(PhysicsWorld *world, float alpha, Buffer<float> &out_state,
                           std::size_t &out_count);
[[nodiscard]] std::expected<void, Error>
PhysicsWorldSetCollisionFilter(PhysicsWorld *world, BodyHandle handle,
                                    std::uint32_t category, std::uint32_t mask);
[[nodiscard]] std::expected<void, Error> PhysicsWorldRegisterMaterial(PhysicsWorld *world,
                                                            std::uint32_t id, float friction,
                                                            float restitution);
[[nodiscard]] std::expected<void, Error>
PhysicsWorldBuildDebugData(PhysicsWorld *world, bool draw_shapes, bool draw_constraints,
                                bool draw_bounding_box, bool draw_centers, bool wireframe,
                                Buffer<DebugVertex> &out_lines,
                                Buffer<DebugVertex> &out_triangles);

} // namespace CPH
