#pragma once

#include "physics/PhysicsParams.hpp"
#include "physics/PhysicsWorldState.hpp"

enum PhysicsWorldCreationStatus : int {
    PHYSICS_WORLD_CREATION_STATUS_OK                   = 0,
    PHYSICS_WORLD_CREATION_STATUS_BODY_SETTINGS_FAILED = -1,
    PHYSICS_WORLD_CREATION_STATUS_NATIVE_ERROR         = -2,
    PHYSICS_WORLD_CREATION_STATUS_INVALID_ARGUMENT     = -3,
    PHYSICS_WORLD_CREATION_STATUS_SHAPE_FAILED         = -4,
    PHYSICS_WORLD_CREATION_STATUS_OUT_OF_MEMORY        = -5,
    PHYSICS_WORLD_CREATION_STATUS_INVALID_MESH_INDEX   = -6,
    PHYSICS_WORLD_CREATION_STATUS_RAGDOLL_FAILED       = -7,
};

enum PhysicsWorldBodySetup : uint8_t {
    PHYSICS_WORLD_BODY_SETUP_CONFIGURED,
    PHYSICS_WORLD_BODY_SETUP_CREATION_PROPS,
    PHYSICS_WORLD_BODY_SETUP_MATERIAL_ONLY,
    PHYSICS_WORLD_BODY_SETUP_NONE,
};

struct CPH_PhysicsWorldBodyCreateRequest {
    // Shape-cache key; unused (0) when the caller creates the shape itself
    // (mesh/heightfield paths).
    int shape_type          = 0;
    float shape_params[4]   = {};
    JPH_RVec3 position;
    JPH_Quat rotation;
    CPH_BodyConfig config;
    uint32_t category;
    uint32_t mask;
    uint32_t material_id    = 0;
    uint64_t user_data;
};

struct CPH_PhysicsWorldCompoundPart {
    int shape_type;
    float shape_params[4];
    JPH_Vec3 local_position;
    JPH_Quat local_rotation;
};

struct CPH_PhysicsWorldSoftBodyCreateRequest {
    JPH_SoftBodySharedSettings *shared_settings;
    uint32_t vertex_count;
    JPH_RVec3 position;
    JPH_Quat rotation;
    uint64_t user_data;
    uint32_t category;
    uint32_t mask;
    float pressure;
    float vertex_radius;
    float linear_damping;
    uint32_t num_iterations;
    float max_linear_velocity;
    float gravity_factor;
    float friction;
    float restitution;
    bool make_rotation_identity;
    bool update_position;
    bool faces_double_sided;
};


void CPH_PhysicsWorldResolveMaterial(CPH_PhysicsWorld *world, uint32_t material_id, float input_friction,
                                    float input_restitution, float *out_friction,
                                    float *out_restitution);
PhysicsWorldCreationStatus CPH_PhysicsWorldCreateBody(CPH_PhysicsWorld *world,
                                                     const CPH_PhysicsWorldBodyCreateRequest *request,
                                                     BodyHandle *out_handle, size_t *out_count,
                                                     PhysicsWorldStatus *out_native_status);
PhysicsWorldCreationStatus
CPH_PhysicsWorldCreateBodiesBatch(CPH_PhysicsWorld *world, const CPH_PosStride *positions,
                                  const CPH_ShapeParams *sizes, size_t batch_count, int shape_type,
                                  int motion_type, BodyHandle *out_handles, size_t *out_count,
                                  PhysicsWorldStatus *out_native_status);
PhysicsWorldCreationStatus CPH_PhysicsWorldCreateSoftBody(
    CPH_PhysicsWorld *world, const CPH_PhysicsWorldSoftBodyCreateRequest *request, BodyHandle *out_handle,
    size_t *out_count, PhysicsWorldStatus *out_native_status);
JPH_RagdollSettings *CPH_PhysicsWorldCreateRagdollSettings(JPH_Skeleton *skeleton);
PhysicsWorldCreationStatus
CPH_PhysicsWorldCreateRagdollInstance(CPH_PhysicsWorld *world, JPH_RagdollSettings *settings,
                                      JPH_RVec3 position, JPH_Quat rotation, uint64_t user_data,
                                      JPH_Ragdoll **out_ragdoll, size_t *out_body_count);
PhysicsWorldCreationStatus CPH_PhysicsWorldRegisterRagdollBodies(
    CPH_PhysicsWorld *world, JPH_Ragdoll *ragdoll, size_t body_count, uint64_t user_data,
    uint32_t category, uint32_t mask, uint32_t material_id, uint32_t *body_slots,
    size_t *out_world_count, PhysicsWorldStatus *out_native_status);
void CPH_PhysicsWorldDestroyRagdoll(CPH_PhysicsWorld *world, JPH_Ragdoll *ragdoll);
PhysicsWorldCreationStatus
CPH_PhysicsWorldCreateConvexHull(CPH_PhysicsWorld *world, const float *points, size_t point_count,
                                 const CPH_PhysicsWorldBodyCreateRequest *request,
                                 BodyHandle *out_handle, size_t *out_count,
                                 PhysicsWorldStatus *out_native_status);
PhysicsWorldCreationStatus
CPH_PhysicsWorldCreateCompoundBody(CPH_PhysicsWorld *world, const CPH_PhysicsWorldCompoundPart *parts,
                                   size_t part_count, const CPH_PhysicsWorldBodyCreateRequest *request,
                                   BodyHandle *out_handle, size_t *out_count,
                                   PhysicsWorldStatus *out_native_status);
PhysicsWorldCreationStatus
CPH_PhysicsWorldCreateMeshBody(CPH_PhysicsWorld *world, const float *vertices, size_t vertex_count,
                               const uint32_t *indices, size_t triangle_count, JPH_RVec3 position,
                               JPH_Quat rotation, uint32_t category, uint32_t mask,
                               uint64_t user_data, BodyHandle *out_handle, size_t *out_count,
                               PhysicsWorldStatus *out_native_status, uint32_t out_bad_indices[3]);
PhysicsWorldCreationStatus CPH_PhysicsWorldCreateHeightfieldBody(
    CPH_PhysicsWorld *world, const float *heights, size_t grid_size, JPH_Vec3 scale, JPH_RVec3 position,
    JPH_Quat rotation, float friction, float restitution, uint32_t category, uint32_t mask,
    uint32_t material_id, uint64_t user_data, BodyHandle *out_handle, size_t *out_count,
    PhysicsWorldStatus *out_native_status);
