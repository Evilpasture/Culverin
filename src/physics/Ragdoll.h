#pragma once

#include "physics/PhysicsWorldState.h"

typedef enum PhysicsRagdollStatus : int {
    PHYSICS_RAGDOLL_STATUS_OK                 = 0,
    PHYSICS_RAGDOLL_STATUS_INVALID_ARGUMENT   = -1,
    PHYSICS_RAGDOLL_STATUS_INVALID_SHAPE      = -2,
    PHYSICS_RAGDOLL_STATUS_JOINT_OUT_OF_RANGE = -3,
} PhysicsRagdollStatus;

typedef struct CPH_PhysicsRagdollPartRequest {
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
} CPH_PhysicsRagdollPartRequest;

typedef struct CPH_PhysicsRagdollBodyHandleInfo {
    BodyHandle handle;
    bool alive;
} CPH_PhysicsRagdollBodyHandleInfo;

typedef struct CPH_PhysicsRagdollBodyDebugInfo {
    JPH_RVec3 position;
    JPH_Quat rotation;
    JPH_Vec3 linear_velocity;
} CPH_PhysicsRagdollBodyDebugInfo;

#ifdef __cplusplus
extern "C" {
#endif

JPH_Skeleton *CPH_PhysicsRagdollSkeletonCreate(void);
void CPH_PhysicsRagdollSkeletonDestroy(JPH_Skeleton *skeleton);
int CPH_PhysicsRagdollSkeletonAddJoint(JPH_Skeleton *skeleton, const char *name, int parent_index);
int CPH_PhysicsRagdollSkeletonGetJointIndex(JPH_Skeleton *skeleton, const char *name);
bool CPH_PhysicsRagdollSkeletonFinalize(JPH_Skeleton *skeleton);

void CPH_PhysicsRagdollSettingsDestroy(JPH_RagdollSettings *settings);
bool CPH_PhysicsRagdollSettingsStabilize(JPH_RagdollSettings *settings);
JPH_Shape *CPH_PhysicsRagdollFindOrCreateShape(CPH_PhysicsWorld *world, int shape_type,
                                                const float shape_params[4]);
PhysicsRagdollStatus CPH_PhysicsRagdollSettingsAddPart(JPH_RagdollSettings *settings,
                                                           JPH_Shape *shape,
                                                           const CPH_PhysicsRagdollPartRequest *request,
                                                           int *out_joint_count);

int CPH_PhysicsRagdollGetJointCount(JPH_Ragdoll *ragdoll);
bool CPH_PhysicsRagdollDriveToPose(JPH_Ragdoll *ragdoll, JPH_RVec3 root_position,
                                   const JPH_Mat4 *matrices, int joint_count);
int CPH_PhysicsRagdollGetBodyCount(JPH_Ragdoll *ragdoll);
void CPH_PhysicsRagdollCopyBodyHandles(CPH_PhysicsWorld *world, const uint32_t *body_slots,
                                       size_t body_count,
                                       CPH_PhysicsRagdollBodyHandleInfo *out_handles);
bool CPH_PhysicsRagdollCopyDebugInfo(CPH_PhysicsWorld *world, JPH_Ragdoll *ragdoll,
                                     CPH_PhysicsRagdollBodyDebugInfo *out_info, size_t body_count);
void CPH_PhysicsRagdollDeallocate(CPH_PhysicsWorld *world, JPH_Ragdoll *ragdoll,
                                const uint32_t *body_slots, size_t body_count);

#ifdef __cplusplus
}
#endif
