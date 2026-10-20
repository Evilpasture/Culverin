#pragma once

#include "core/Error.hpp"
#include "jolt/Jolt.hpp"
#include <Jolt/Physics/Collision/Shape/Shape.h>
#include "physics/PhysicsParams.hpp"
#include <joltc.h>

#include <expected>
#include <span>


namespace CPH {

enum class RagdollError : std::uint32_t {
    InvalidArgument = 1,
    InvalidShape    = 2,
    JointOutOfRange = 3,
};

// The skeleton/settings/ragdoll objects below are their own subjects
// (no world involved); world-side ragdoll bookkeeping is declared as
// PhysicsWorld members in PhysicsWorldState.hpp.

JPH_Skeleton *PhysicsRagdollSkeletonCreate(void);
void PhysicsRagdollSkeletonDestroy(JPH_Skeleton *skeleton);
int PhysicsRagdollSkeletonAddJoint(JPH_Skeleton *skeleton, const char *name, int parent_index);
int PhysicsRagdollSkeletonGetJointIndex(JPH_Skeleton *skeleton, const char *name);
bool PhysicsRagdollSkeletonFinalize(JPH_Skeleton *skeleton);

void PhysicsRagdollSettingsDestroy(JPH_RagdollSettings *settings);
bool PhysicsRagdollSettingsStabilize(JPH_RagdollSettings *settings);
// Returns the joint count after the part is added.
[[nodiscard]] std::expected<int, Error>
PhysicsRagdollSettingsAddPart(JPH_RagdollSettings *settings, const JPH::Shape *shape,
                              const PhysicsRagdollPartRequest &request);

int PhysicsRagdollGetJointCount(JPH_Ragdoll *ragdoll);
bool PhysicsRagdollDriveToPose(JPH_Ragdoll *ragdoll, JPH_RVec3 root_position,
                                   std::span<const JPH_Mat4> matrices);
int PhysicsRagdollGetBodyCount(JPH_Ragdoll *ragdoll);

struct PhysicsWorld;
// World may be null when the owning world object is gone; then the
// ragdoll is destroyed without shadow cleanup.
void PhysicsRagdollDeallocate(PhysicsWorld *world, JPH_Ragdoll *ragdoll,
                              std::span<const std::uint32_t> body_slots);

} // namespace CPH
