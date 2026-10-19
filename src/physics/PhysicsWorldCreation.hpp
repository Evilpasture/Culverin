#pragma once

#include "core/Error.hpp"
#include "physics/PhysicsParams.hpp"
#include "physics/PhysicsWorldState.hpp"

#include <expected>


namespace CPH {

enum class WorldCreationError : std::uint32_t {
    BodySettingsFailed = 1,
    InvalidArgument    = 2,
    ShapeFailed        = 3,
    OutOfMemory        = 4,
    InvalidMeshIndex   = 5,
    RagdollFailed      = 6,
};

enum class PhysicsWorldBodySetup : uint8_t {
    Configured,
    CreationProps,
    MaterialOnly,
    None,
};

struct PhysicsWorldBodyCreateRequest {
    // Shape-cache key; unused (0) when the caller creates the shape itself
    // (mesh/heightfield paths).
    int shape_type          = 0;
    float shape_params[4]   = {};
    JPH_RVec3 position      = {};
    JPH_Quat rotation       = {0.0f, 0.0f, 0.0f, 1.0f};
    BodyConfig config;
    uint32_t category       = kAllLayerBits;
    uint32_t mask           = kAllLayerBits;
    uint32_t material_id    = 0;
    uint64_t user_data      = 0;
};

struct PhysicsWorldCompoundPart {
    int shape_type          = 0;
    float shape_params[4]   = {};
    JPH_Vec3 local_position = {};
    JPH_Quat local_rotation = {0.0f, 0.0f, 0.0f, 1.0f};
};

struct PhysicsWorldSoftBodyCreateRequest {
    JPH_SoftBodySharedSettings *shared_settings = nullptr;
    uint32_t vertex_count                       = 0;
    JPH_RVec3 position                          = {};
    JPH_Quat rotation                           = {0.0f, 0.0f, 0.0f, 1.0f};
    uint64_t user_data                          = 0;
    uint32_t category                           = kAllLayerBits;
    uint32_t mask                               = kAllLayerBits;
    float pressure                              = 0.0f;
    float vertex_radius                         = 0.0f;
    float linear_damping                        = 0.0f;
    uint32_t num_iterations                     = 0;
    float max_linear_velocity                   = 0.0f;
    float gravity_factor                        = 0.0f;
    float friction                              = 0.0f;
    float restitution                           = 0.0f;
    bool make_rotation_identity                 = false;
    bool update_position                        = false;
    bool faces_double_sided                     = false;
};


void PhysicsWorldResolveMaterial(PhysicsWorld *world, uint32_t material_id, float input_friction,
                                    float input_restitution, float *out_friction,
                                    float *out_restitution);
[[nodiscard]] std::expected<void, Error> PhysicsWorldCreateBody(PhysicsWorld *world,
                                                     const PhysicsWorldBodyCreateRequest *request,
                                                     BodyHandle *out_handle, size_t *out_count);
[[nodiscard]] std::expected<void, Error>
PhysicsWorldCreateBodiesBatch(PhysicsWorld *world, const PosStride *positions,
                                  const ShapeParams *sizes, size_t batch_count, int shape_type,
                                  int motion_type, BodyHandle *out_handles, size_t *out_count);
[[nodiscard]] std::expected<void, Error> PhysicsWorldCreateSoftBody(
    PhysicsWorld *world, const PhysicsWorldSoftBodyCreateRequest *request, BodyHandle *out_handle,
    size_t *out_count);
JPH_RagdollSettings *PhysicsWorldCreateRagdollSettings(JPH_Skeleton *skeleton);
[[nodiscard]] std::expected<void, Error>
PhysicsWorldCreateRagdollInstance(PhysicsWorld *world, JPH_RagdollSettings *settings,
                                      JPH_RVec3 position, JPH_Quat rotation, uint64_t user_data,
                                      JPH_Ragdoll **out_ragdoll, size_t *out_body_count);
[[nodiscard]] std::expected<void, Error> PhysicsWorldRegisterRagdollBodies(
    PhysicsWorld *world, JPH_Ragdoll *ragdoll, size_t body_count, uint64_t user_data,
    uint32_t category, uint32_t mask, uint32_t material_id, uint32_t *body_slots,
    size_t *out_world_count);
void PhysicsWorldDestroyRagdoll(PhysicsWorld *world, JPH_Ragdoll *ragdoll);
[[nodiscard]] std::expected<void, Error>
PhysicsWorldCreateConvexHull(PhysicsWorld *world, const float *points, size_t point_count,
                                 const PhysicsWorldBodyCreateRequest *request,
                                 BodyHandle *out_handle, size_t *out_count);
[[nodiscard]] std::expected<void, Error>
PhysicsWorldCreateCompoundBody(PhysicsWorld *world, const PhysicsWorldCompoundPart *parts,
                                   size_t part_count, const PhysicsWorldBodyCreateRequest *request,
                                   BodyHandle *out_handle, size_t *out_count);
[[nodiscard]] std::expected<void, Error>
PhysicsWorldCreateMeshBody(PhysicsWorld *world, const float *vertices, size_t vertex_count,
                               const uint32_t *indices, size_t triangle_count, JPH_RVec3 position,
                               JPH_Quat rotation, uint32_t category, uint32_t mask,
                               uint64_t user_data, BodyHandle *out_handle, size_t *out_count,
                               uint32_t out_bad_indices[3]);
[[nodiscard]] std::expected<void, Error> PhysicsWorldCreateHeightfieldBody(
    PhysicsWorld *world, const float *heights, size_t grid_size, JPH_Vec3 scale, JPH_RVec3 position,
    JPH_Quat rotation, float friction, float restitution, uint32_t category, uint32_t mask,
    uint32_t material_id, uint64_t user_data, BodyHandle *out_handle, size_t *out_count);

} // namespace CPH
