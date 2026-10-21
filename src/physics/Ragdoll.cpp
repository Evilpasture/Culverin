#include "physics/Ragdoll.hpp"
#include "physics/PhysicsWorldState.hpp"
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Constraints/SwingTwistConstraint.h>
#include <Jolt/Skeleton/SkeletonPose.h>


namespace CPH {

JPH::Skeleton *PhysicsRagdollSkeletonCreate(void) {
    JPH::Skeleton *skeleton = new JPH::Skeleton();
    // joltc's JPH_Skeleton_Create returned an already-AddRef'd object; match it
    // so the Python side's ownership accounting is unchanged.
    skeleton->AddRef();
    return skeleton;
}

void PhysicsRagdollSkeletonDestroy(JPH::Skeleton *skeleton) {
    if (skeleton) {
        skeleton->Release();
    }
}

int PhysicsRagdollSkeletonAddJoint(JPH::Skeleton *skeleton, const char *name, int parent_index) {
    if (!skeleton || !name) {
        return -1;
    }
    return (int)skeleton->AddJoint(name, parent_index);
}

int PhysicsRagdollSkeletonGetJointIndex(JPH::Skeleton *skeleton, const char *name) {
    if (!skeleton || !name) {
        return -1;
    }
    return skeleton->GetJointIndex(name);
}

bool PhysicsRagdollSkeletonFinalize(JPH::Skeleton *skeleton) {
    if (!skeleton) {
        return false;
    }
    skeleton->CalculateParentJointIndices();
    return skeleton->AreJointsCorrectlyOrdered();
}

void PhysicsRagdollSettingsDestroy(JPH::RagdollSettings *settings) {
    if (settings) {
        settings->Release();
    }
}

bool PhysicsRagdollSettingsStabilize(JPH::RagdollSettings *settings) {
    return settings && settings->Stabilize();
}

std::expected<int, Error>
PhysicsRagdollSettingsAddPart(JPH::RagdollSettings *settings, const JPH::Shape *shape,
                              const PhysicsRagdollPartRequest &request) {
    if (!settings) {
        return std::unexpected(RagdollError::InvalidArgument);
    }
    if (!shape) {
        return std::unexpected(RagdollError::InvalidShape);
    }

    const JPH::Skeleton *skeleton = settings->GetSkeleton();
    if (!skeleton) {
        return std::unexpected(RagdollError::InvalidArgument);
    }
    const int joint_count = skeleton->GetJointCount();
    if (request.joint_index < 0 || request.joint_index >= joint_count) {
        return std::unexpected(RagdollError::JointOutOfRange);
    }

    if ((int)settings->mParts.size() <= request.joint_index) {
        settings->mParts.resize(joint_count);
    }

    JPH::RagdollSettings::Part &part = settings->mParts[request.joint_index];
    part.SetShape(shape);
    // joltc set CalculateInertia, not MassAndInertia: the mass is overridden but
    // the inertia tensor is still derived from the shape.
    part.mMassPropertiesOverride.mMass = request.mass;
    part.mOverrideMassProperties       = JPH::EOverrideMassProperties::CalculateInertia;
    part.mObjectLayer                  = 1;
    part.mMotionType                   = JPH::EMotionType::Dynamic;

    if (request.has_position) {
        part.mPosition = request.position;
    }

    if (request.parent_index >= 0) {
        // A fresh SwingTwistConstraintSettings carries Jolt's defaults, which is
        // exactly what the JoltC path produced: JPH_SwingTwistConstraintSettings_Init
        // memset the struct and then copied the defaults back out of a
        // default-constructed native settings object, and FromJolt mirrored every
        // field SetPartToParent later read. So default-constructing here and
        // applying only the overrides below is equivalent field for field.
        JPH::SwingTwistConstraintSettings *constraint = new JPH::SwingTwistConstraintSettings();
        // TODO(bug): the JoltC path wrote constraint.base.enabled = true, but
        // JPH_RagdollSettings_SetPartToParent never copied `base` across (nor
        // constraintPriority, the step-count overrides, drawConstraintSize or
        // userData), so that write was dead. It was harmless only because
        // ConstraintSettings::mEnabled defaults to true (Constraint.h:15). Left
        // unset here to match observable behaviour; setting it would be a no-op
        // today and a behaviour change if that default ever moved.
        constraint->mPosition1          = JPH::RVec3::sZero();
        constraint->mPosition2          = JPH::RVec3::sZero();
        constraint->mTwistAxis1         = request.axis;
        constraint->mTwistAxis2         = request.axis;
        constraint->mPlaneAxis1         = request.normal;
        constraint->mPlaneAxis2         = request.normal;
        constraint->mNormalHalfConeAngle = request.cone_angle;
        constraint->mPlaneHalfConeAngle  = request.cone_angle;
        constraint->mTwistMinAngle       = request.twist_min;
        constraint->mTwistMaxAngle       = request.twist_max;
        part.mToParent                   = constraint;
    }

    return joint_count;
}

int PhysicsRagdollGetJointCount(JPH::Ragdoll *ragdoll) {
    if (!ragdoll) {
        return 0;
    }
    const JPH::RagdollSettings *settings = ragdoll->GetRagdollSettings();
    if (!settings) {
        return 0;
    }
    const JPH::Skeleton *skeleton = settings->GetSkeleton();
    return skeleton ? skeleton->GetJointCount() : 0;
}

bool PhysicsRagdollDriveToPose(JPH::Ragdoll *ragdoll, JPH::RVec3 root_position,
                               std::span<const JPH::Mat44> matrices) {
    if (!ragdoll || matrices.size() > static_cast<std::size_t>(INT_MAX)) {
        return false;
    }

    const JPH::RagdollSettings *settings = ragdoll->GetRagdollSettings();
    if (!settings) {
        return false;
    }
    const JPH::Skeleton *skeleton = settings->GetSkeleton();
    if (!skeleton) {
        return false;
    }

    JPH::SkeletonPose pose;
    pose.SetSkeleton(skeleton);
    pose.SetRootOffset(root_position);
    ragdoll->SetPose(root_position, matrices.data(), true);
    // joltc clamped the copy to the pose's own joint count rather than resizing
    // or rejecting a mismatch; keep that, since SetSkeleton has already sized
    // mJointMatrices from the skeleton.
    JPH::SkeletonPose::Mat44Vector &pose_matrices = pose.GetJointMatrices();
    const int copy_count =
        (int)pose_matrices.size() < (int)matrices.size() ? (int)pose_matrices.size()
                                                        : (int)matrices.size();
    for (int i = 0; i < copy_count; i++) {
        pose_matrices[i] = matrices[i];
    }
    ragdoll->Activate(true);
    ragdoll->DriveToPoseUsingMotors(pose);
    return true;
}

int PhysicsRagdollGetBodyCount(JPH::Ragdoll *ragdoll) {
    return ragdoll ? (int)ragdoll->GetBodyCount() : 0;
}

void PhysicsWorld::CopyRagdollBodyHandles(std::span<const std::uint32_t> body_slots,
                                          std::span<PhysicsRagdollBodyHandleInfo> out_handles) {
    if (body_slots.size() != out_handles.size()) {
        return;
    }

    const std::scoped_lock shadow(shadow_lock);
    for (size_t i = 0; i < body_slots.size(); ++i) {
        out_handles[i]      = PhysicsRagdollBodyHandleInfo{.handle = BodyHandle{}, .alive = false};
        const uint32_t slot = body_slots[i];
        if (slot >= slot_capacity || !bodies.slot_states || !bodies.generations) {
            continue;
        }
        const uint8_t state = (bodies.slot_states[slot]).load(std::memory_order::acquire);
        if (state != std::to_underlying(SlotState::Alive)) {
            continue;
        }
        const uint32_t generation = (bodies.generations[slot]).load(std::memory_order::relaxed);
        out_handles[i] = PhysicsRagdollBodyHandleInfo{
            .handle = BodyHandle::FromParts(slot, generation),
            .alive  = true,
        };
    }
}

bool PhysicsWorld::CopyRagdollDebugInfo(JPH::Ragdoll &ragdoll,
                                        std::span<PhysicsRagdollBodyDebugInfo> out_info) {
    if (!body_interface) {
        return false;
    }
    const int ragdoll_body_count = (int)ragdoll.GetBodyCount();
    if (ragdoll_body_count < 0 || (size_t)ragdoll_body_count != out_info.size()) {
        return false;
    }

    // Same locking interface that `body_interface` aliases. ragdoll.GetBodyID
    // already hands back a JPH::BodyID, so the pack/unpack round-trip through
    // GetIndexAndSequenceNumber() is gone.
    JPH::BodyInterface &bi = physics_system.GetBodyInterface();

    const std::scoped_lock shadow(shadow_lock);
    for (int i = 0; i < ragdoll_body_count; ++i) {
        const JPH::BodyID body_id = ragdoll.GetBodyID(i);
        out_info[i].position        = bi.GetPosition(body_id);
        out_info[i].rotation        = bi.GetRotation(body_id);
        out_info[i].linear_velocity = bi.GetLinearVelocity(body_id);
    }
    return true;
}

void PhysicsRagdollDeallocate(PhysicsWorld *world, JPH::Ragdoll *ragdoll,
                              std::span<const std::uint32_t> body_slots) {
    if (!ragdoll) {
        return;
    }

    if (world) {
        auto shadow = world->LockShadowNotStepping();
        world->WaitUntilNotQuerying(shadow);
        ragdoll->RemoveFromPhysicsSystem(true);

        if (world->bodies.slot_states) {
            for (size_t i = 0; i < body_slots.size(); ++i) {
                const uint32_t slot = body_slots[i];
                if (slot >= world->slot_capacity) {
                    continue;
                }
                const uint8_t state =
                    (world->bodies.slot_states[slot]).load(std::memory_order::acquire);
                if (state == std::to_underlying(SlotState::Alive)) {
                    world->RemoveBodySlot(slot);
                }
            }
        }
        shadow.unlock();

        const std::scoped_lock trampoline(world->jph_trampoline_lock);
        ragdoll->Release();
        return;
    }

    ragdoll->Release();
}

} // namespace CPH
