#pragma once

#include "physics/PhysicsWorldState.h"

typedef enum PhysicsRagdollStatus : int {
    PHYSICS_RAGDOLL_STATUS_OK                 = 0,
    PHYSICS_RAGDOLL_STATUS_INVALID_ARGUMENT   = -1,
    PHYSICS_RAGDOLL_STATUS_INVALID_SHAPE      = -2,
    PHYSICS_RAGDOLL_STATUS_JOINT_OUT_OF_RANGE = -3,
} PhysicsRagdollStatus;

typedef struct PhysicsRagdollPartRequest {
    int joint_index;
    int parent_index;
    float mass;
    float twist_min;
    float twist_max;
    float cone_angle;
    JPH_Vec3 axis;
    JPH_Vec3 normal;
    bool has_position;
    JPH_RVec3 position;
} PhysicsRagdollPartRequest;

typedef struct PhysicsRagdollBodyHandleInfo {
    BodyHandle handle;
    bool alive;
} PhysicsRagdollBodyHandleInfo;

typedef struct PhysicsRagdollBodyDebugInfo {
    JPH_RVec3 position;
    JPH_Quat rotation;
    JPH_Vec3 linear_velocity;
} PhysicsRagdollBodyDebugInfo;

JPH_Skeleton *physics_ragdoll_skeleton_create(void);
void physics_ragdoll_skeleton_destroy(JPH_Skeleton *skeleton);
int physics_ragdoll_skeleton_add_joint(JPH_Skeleton *skeleton, const char *name, int parent_index);
int physics_ragdoll_skeleton_get_joint_index(JPH_Skeleton *skeleton, const char *name);
bool physics_ragdoll_skeleton_finalize(JPH_Skeleton *skeleton);

void physics_ragdoll_settings_destroy(JPH_RagdollSettings *settings);
bool physics_ragdoll_settings_stabilize(JPH_RagdollSettings *settings);
JPH_Shape *physics_ragdoll_find_or_create_shape(PhysicsWorld *world, int shape_type,
                                                const float shape_params[4]);
PhysicsRagdollStatus physics_ragdoll_settings_add_part(JPH_RagdollSettings *settings,
                                                           JPH_Shape *shape,
                                                           const PhysicsRagdollPartRequest *request,
                                                           int *out_joint_count);

int physics_ragdoll_get_joint_count(JPH_Ragdoll *ragdoll);
bool physics_ragdoll_drive_to_pose(JPH_Ragdoll *ragdoll, JPH_RVec3 root_position,
                                   const JPH_Mat4 *matrices, int joint_count);
int physics_ragdoll_get_body_count(JPH_Ragdoll *ragdoll);
void physics_ragdoll_copy_body_handles(PhysicsWorld *world, const uint32_t *body_slots,
                                       size_t body_count,
                                       PhysicsRagdollBodyHandleInfo *out_handles);
bool physics_ragdoll_copy_debug_info(PhysicsWorld *world, JPH_Ragdoll *ragdoll,
                                     PhysicsRagdollBodyDebugInfo *out_info, size_t body_count);
void physics_ragdoll_deallocate(PhysicsWorld *world, JPH_Ragdoll *ragdoll,
                                const uint32_t *body_slots, size_t body_count);
