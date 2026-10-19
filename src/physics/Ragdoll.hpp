#pragma once

#include "core/Error.hpp"
#include "physics/PhysicsWorldState.hpp"

#include <expected>


namespace CPH {

enum class RagdollError : std::uint32_t {
    InvalidArgument = 1,
    InvalidShape    = 2,
    JointOutOfRange = 3,
};

struct PhysicsRagdollPartRequest {
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
};

struct PhysicsRagdollBodyHandleInfo {
    BodyHandle handle;
    bool alive;
};

struct PhysicsRagdollBodyDebugInfo {
    JPH_RVec3 position;
    JPH_Quat rotation;
    JPH_Vec3 linear_velocity;
};


JPH_Skeleton *PhysicsRagdollSkeletonCreate(void);
void PhysicsRagdollSkeletonDestroy(JPH_Skeleton *skeleton);
int PhysicsRagdollSkeletonAddJoint(JPH_Skeleton *skeleton, const char *name, int parent_index);
int PhysicsRagdollSkeletonGetJointIndex(JPH_Skeleton *skeleton, const char *name);
bool PhysicsRagdollSkeletonFinalize(JPH_Skeleton *skeleton);

void PhysicsRagdollSettingsDestroy(JPH_RagdollSettings *settings);
bool PhysicsRagdollSettingsStabilize(JPH_RagdollSettings *settings);
JPH_Shape *PhysicsRagdollFindOrCreateShape(PhysicsWorld *world, int shape_type,
                                                const float shape_params[4]);
[[nodiscard]] std::expected<void, Error>
PhysicsRagdollSettingsAddPart(JPH_RagdollSettings *settings, JPH_Shape *shape,
                              const PhysicsRagdollPartRequest *request, int *out_joint_count);

int PhysicsRagdollGetJointCount(JPH_Ragdoll *ragdoll);
bool PhysicsRagdollDriveToPose(JPH_Ragdoll *ragdoll, JPH_RVec3 root_position,
                                   const JPH_Mat4 *matrices, int joint_count);
int PhysicsRagdollGetBodyCount(JPH_Ragdoll *ragdoll);
void PhysicsRagdollCopyBodyHandles(PhysicsWorld *world, const uint32_t *body_slots,
                                       size_t body_count,
                                       PhysicsRagdollBodyHandleInfo *out_handles);
bool PhysicsRagdollCopyDebugInfo(PhysicsWorld *world, JPH_Ragdoll *ragdoll,
                                     PhysicsRagdollBodyDebugInfo *out_info, size_t body_count);
void PhysicsRagdollDeallocate(PhysicsWorld *world, JPH_Ragdoll *ragdoll,
                                const uint32_t *body_slots, size_t body_count);

} // namespace CPH
