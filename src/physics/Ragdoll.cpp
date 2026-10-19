#include "physics/Ragdoll.hpp"
#include "physics/CommandBuffer.hpp"
#include "physics/InternalQuery.hpp"


namespace CPH {

JPH_Skeleton *PhysicsRagdollSkeletonCreate(void) { return JPH_Skeleton_Create(); }

void PhysicsRagdollSkeletonDestroy(JPH_Skeleton *skeleton) {
    if (skeleton) {
        JPH_Skeleton_Destroy(skeleton);
    }
}

int PhysicsRagdollSkeletonAddJoint(JPH_Skeleton *skeleton, const char *name, int parent_index) {
    if (!skeleton || !name) {
        return -1;
    }
    return (int)JPH_Skeleton_AddJoint2(skeleton, name, parent_index);
}

int PhysicsRagdollSkeletonGetJointIndex(JPH_Skeleton *skeleton, const char *name) {
    if (!skeleton || !name) {
        return -1;
    }
    return JPH_Skeleton_GetJointIndex(skeleton, name);
}

bool PhysicsRagdollSkeletonFinalize(JPH_Skeleton *skeleton) {
    if (!skeleton) {
        return false;
    }
    JPH_Skeleton_CalculateParentJointIndices(skeleton);
    return JPH_Skeleton_AreJointsCorrectlyOrdered(skeleton);
}

void PhysicsRagdollSettingsDestroy(JPH_RagdollSettings *settings) {
    if (settings) {
        JPH_RagdollSettings_Destroy(settings);
    }
}

bool PhysicsRagdollSettingsStabilize(JPH_RagdollSettings *settings) {
    return settings && JPH_RagdollSettings_Stabilize(settings);
}

JPH_Shape *PhysicsRagdollFindOrCreateShape(PhysicsWorld *world, int shape_type,
                                                const float shape_params[4]) {
    if (!world || !shape_params) {
        return nullptr;
    }
    const std::scoped_lock shadow(world->shadow_lock);
    return FindOrCreateShapeLocked(world, shape_type, shape_params);
}

PhysicsRagdollStatus PhysicsRagdollSettingsAddPart(JPH_RagdollSettings *settings,
                                                           JPH_Shape *shape,
                                                           const PhysicsRagdollPartRequest *request,
                                                           int *out_joint_count) {
    if (!settings || !request || !out_joint_count) {
        return PHYSICS_RAGDOLL_STATUS_INVALID_ARGUMENT;
    }
    if (!shape) {
        return PHYSICS_RAGDOLL_STATUS_INVALID_SHAPE;
    }

    const JPH_Skeleton *skeleton = JPH_RagdollSettings_GetSkeleton(settings);
    if (!skeleton) {
        return PHYSICS_RAGDOLL_STATUS_INVALID_ARGUMENT;
    }
    const int joint_count = JPH_Skeleton_GetJointCount(skeleton);
    *out_joint_count      = joint_count;
    if (request->joint_index < 0 || request->joint_index >= joint_count) {
        return PHYSICS_RAGDOLL_STATUS_JOINT_OUT_OF_RANGE;
    }

    if (JPH_RagdollSettings_GetPartCount(settings) <= request->joint_index) {
        JPH_RagdollSettings_ResizeParts(settings, joint_count);
    }

    JPH_RagdollSettings_SetPartShape(settings, request->joint_index, shape);
    JPH_RagdollSettings_SetPartMassProperties(settings, request->joint_index, request->mass);
    JPH_RagdollSettings_SetPartObjectLayer(settings, request->joint_index, 1);
    JPH_RagdollSettings_SetPartMotionType(settings, request->joint_index, JPH_MotionType_Dynamic);

    if (request->has_position) {
        JPH_RagdollSettings_SetPartPosition(settings, request->joint_index, &request->position);
    }

    if (request->parent_index >= 0) {
        JPH_SwingTwistConstraintSettings constraint;
        JPH_SwingTwistConstraintSettings_Init(&constraint);
        constraint.base.enabled        = true;
        constraint.position1           = JPH_RVec3{0, 0, 0};
        constraint.position2           = JPH_RVec3{0, 0, 0};
        constraint.twistAxis1          = request->axis;
        constraint.twistAxis2          = request->axis;
        constraint.planeAxis1          = request->normal;
        constraint.planeAxis2          = request->normal;
        constraint.normalHalfConeAngle = request->cone_angle;
        constraint.planeHalfConeAngle  = request->cone_angle;
        constraint.twistMinAngle       = request->twist_min;
        constraint.twistMaxAngle       = request->twist_max;
        JPH_RagdollSettings_SetPartToParent(settings, request->joint_index, &constraint);
    }

    return PHYSICS_RAGDOLL_STATUS_OK;
}

int PhysicsRagdollGetJointCount(JPH_Ragdoll *ragdoll) {
    if (!ragdoll) {
        return 0;
    }
    const JPH_RagdollSettings *settings = JPH_Ragdoll_GetRagdollSettings(ragdoll);
    if (!settings) {
        return 0;
    }
    const JPH_Skeleton *skeleton = JPH_RagdollSettings_GetSkeleton(settings);
    return skeleton ? JPH_Skeleton_GetJointCount(skeleton) : 0;
}

bool PhysicsRagdollDriveToPose(JPH_Ragdoll *ragdoll, JPH_RVec3 root_position,
                                   const JPH_Mat4 *matrices, int joint_count) {
    if (!ragdoll || !matrices || joint_count < 0) {
        return false;
    }

    const JPH_RagdollSettings *settings = JPH_Ragdoll_GetRagdollSettings(ragdoll);
    if (!settings) {
        return false;
    }
    const JPH_Skeleton *skeleton = JPH_RagdollSettings_GetSkeleton(settings);
    if (!skeleton) {
        return false;
    }

    const JoltPtr<JPH_SkeletonPose, JPH_SkeletonPose_Destroy> pose(JPH_SkeletonPose_Create());
    if (!pose) {
        return false;
    }
    JPH_SkeletonPose_SetSkeleton(pose.get(), skeleton);
    JPH_SkeletonPose_SetRootOffset(pose.get(), &root_position);
    JPH_Ragdoll_SetPose2(ragdoll, &root_position, const_cast<JPH_Mat4 *>(matrices), true);
    JPH_SkeletonPose_SetJointMatrices(pose.get(), const_cast<JPH_Mat4 *>(matrices), joint_count);
    JPH_Ragdoll_Activate(ragdoll, true);
    JPH_Ragdoll_DriveToPoseUsingMotors(ragdoll, pose.get());
    return true;
}

int PhysicsRagdollGetBodyCount(JPH_Ragdoll *ragdoll) {
    return ragdoll ? JPH_Ragdoll_GetBodyCount(ragdoll) : 0;
}

void PhysicsRagdollCopyBodyHandles(PhysicsWorld *world, const uint32_t *body_slots,
                                       size_t body_count,
                                       PhysicsRagdollBodyHandleInfo *out_handles) {
    if (!world || (body_count && (!body_slots || !out_handles))) {
        return;
    }

    const std::scoped_lock shadow(world->shadow_lock);
    for (size_t i = 0; i < body_count; ++i) {
        out_handles[i]      = PhysicsRagdollBodyHandleInfo{.handle = 0, .alive = false};
        const uint32_t slot = body_slots[i];
        if (slot >= world->slot_capacity || !world->bodies.slot_states || !world->bodies.generations) {
            continue;
        }
        const uint8_t state = (world->bodies.slot_states[slot]).load(std::memory_order::acquire);
        if (state != std::to_underlying(SlotState::Alive)) {
            continue;
        }
        const uint32_t generation =
            (world->bodies.generations[slot]).load(std::memory_order::relaxed);
        out_handles[i] = PhysicsRagdollBodyHandleInfo{
            .handle = make_handle(slot, generation),
            .alive  = true,
        };
    }
}

bool PhysicsRagdollCopyDebugInfo(PhysicsWorld *world, JPH_Ragdoll *ragdoll,
                                     PhysicsRagdollBodyDebugInfo *out_info, size_t body_count) {
    if (!world || !ragdoll || (body_count && !out_info) || !world->body_interface) {
        return false;
    }
    const int ragdoll_body_count = JPH_Ragdoll_GetBodyCount(ragdoll);
    if (ragdoll_body_count < 0 || (size_t)ragdoll_body_count != body_count) {
        return false;
    }

    JPH_BodyInterface *body_interface = world->body_interface;
    const std::scoped_lock shadow(world->shadow_lock);
    for (int i = 0; i < ragdoll_body_count; ++i) {
        const JPH_BodyID body_id = JPH_Ragdoll_GetBodyID(ragdoll, i);
        JPH_BodyInterface_GetPosition(body_interface, body_id, &out_info[i].position);
        JPH_BodyInterface_GetRotation(body_interface, body_id, &out_info[i].rotation);
        JPH_BodyInterface_GetLinearVelocity(body_interface, body_id, &out_info[i].linear_velocity);
    }
    return true;
}

void PhysicsRagdollDeallocate(PhysicsWorld *world, JPH_Ragdoll *ragdoll,
                                const uint32_t *body_slots, size_t body_count) {
    if (!ragdoll) {
        return;
    }

    if (world) {
        auto shadow = LockShadowNotStepping(world);
        WaitUntilNotQuerying(world, shadow);
        JPH_Ragdoll_RemoveFromPhysicsSystem(ragdoll, true);

        if (body_slots && world->bodies.slot_states) {
            for (size_t i = 0; i < body_count; ++i) {
                const uint32_t slot = body_slots[i];
                if (slot >= world->slot_capacity) {
                    continue;
                }
                const uint8_t state =
                    (world->bodies.slot_states[slot]).load(std::memory_order::acquire);
                if (state == std::to_underlying(SlotState::Alive)) {
                    WorldRemoveBodySlot(world, slot);
                }
            }
        }
        shadow.unlock();

        const std::scoped_lock trampoline(world->jph_trampoline_lock);
        JPH_Ragdoll_Destroy(ragdoll);
        return;
    }

    JPH_Ragdoll_Destroy(ragdoll);
}

} // namespace CPH
