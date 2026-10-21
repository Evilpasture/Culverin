#pragma once

#include "core/Error.hpp"
#include "jolt/Jolt.hpp"
#include <Jolt/Physics/Collision/Shape/Shape.h>
#include "physics/PhysicsParams.hpp"
#include <Jolt/Physics/Ragdoll/Ragdoll.h>
#include <Jolt/Skeleton/Skeleton.h>

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

// All three object types are Jolt RefTargets. The pointers below are
// non-owning views: creation AddRef's exactly once (matching what joltc's
// JPH_*_Create did) and the matching Destroy Release's it.
JPH::Skeleton *PhysicsRagdollSkeletonCreate(void);
void PhysicsRagdollSkeletonDestroy(JPH::Skeleton *skeleton);
int PhysicsRagdollSkeletonAddJoint(JPH::Skeleton *skeleton, const char *name, int parent_index);
int PhysicsRagdollSkeletonGetJointIndex(JPH::Skeleton *skeleton, const char *name);
bool PhysicsRagdollSkeletonFinalize(JPH::Skeleton *skeleton);

void PhysicsRagdollSettingsDestroy(JPH::RagdollSettings *settings);
bool PhysicsRagdollSettingsStabilize(JPH::RagdollSettings *settings);
// Returns the joint count after the part is added.
[[nodiscard]] std::expected<int, Error>
PhysicsRagdollSettingsAddPart(JPH::RagdollSettings *settings, const JPH::Shape *shape,
                              const PhysicsRagdollPartRequest &request);

int PhysicsRagdollGetJointCount(JPH::Ragdoll *ragdoll);
bool PhysicsRagdollDriveToPose(JPH::Ragdoll *ragdoll, JPH::RVec3 root_position,
                               std::span<const JPH::Mat44> matrices);
int PhysicsRagdollGetBodyCount(JPH::Ragdoll *ragdoll);

struct PhysicsWorld;
// World may be null when the owning world object is gone; then the
// ragdoll is destroyed without shadow cleanup.
void PhysicsRagdollDeallocate(PhysicsWorld *world, JPH::Ragdoll *ragdoll,
                              std::span<const std::uint32_t> body_slots);

} // namespace CPH
