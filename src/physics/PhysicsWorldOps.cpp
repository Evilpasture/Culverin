#include "physics/PhysicsWorldOps.hpp"

#include "physics/CommandBuffer.hpp"
#include "physics/SlotState.hpp"

#include <utility>

namespace {

using namespace CPH::cmd;

bool is_immediate_jolt_action(PhysicsBodyOperationType type) {
    return type == PHYSICS_BODY_OP_APPLY_IMPULSE || type == PHYSICS_BODY_OP_APPLY_IMPULSE_AT ||
           type == PHYSICS_BODY_OP_APPLY_ANGULAR_IMPULSE || type == PHYSICS_BODY_OP_APPLY_FORCE ||
           type == PHYSICS_BODY_OP_APPLY_TORQUE;
}

std::uint32_t immediate_mask(PhysicsBodyOperationType type) {
    switch (type) {
    case PHYSICS_BODY_OP_APPLY_ANGULAR_IMPULSE:
    case PHYSICS_BODY_OP_SET_ANGULAR_VELOCITY:
    case PHYSICS_BODY_OP_SET_MOTION_TYPE:
        return kMaskImmStrict;
    default:
        return kMaskImmStandard;
    }
}

// Returns false only for an unknown operation type.
bool enqueue_operation(CPH_PhysicsWorld *world, std::uint32_t slot,
                       const CPH_PhysicsBodyOperation &op) {
    switch (op.type) {
    case PHYSICS_BODY_OP_APPLY_IMPULSE:
        CPH_EnqueueCommand(world, slot, ApplyImpulse{op.vector});
        return true;
    case PHYSICS_BODY_OP_APPLY_IMPULSE_AT:
        CPH_EnqueueCommand(world, slot, ApplyImpulseAt{op.vector, op.position});
        return true;
    case PHYSICS_BODY_OP_APPLY_ANGULAR_IMPULSE:
        CPH_EnqueueCommand(world, slot, ApplyAngularImpulse{op.vector});
        return true;
    case PHYSICS_BODY_OP_APPLY_FORCE:
        CPH_EnqueueCommand(world, slot, ApplyForce{op.vector});
        return true;
    case PHYSICS_BODY_OP_APPLY_TORQUE:
        CPH_EnqueueCommand(world, slot, ApplyTorque{op.vector});
        return true;
    case PHYSICS_BODY_OP_SET_POSITION:
        CPH_EnqueueCommand(world, slot, SetPosition{op.position});
        return true;
    case PHYSICS_BODY_OP_SET_ROTATION:
        CPH_EnqueueCommand(world, slot, SetRotation{op.rotation});
        return true;
    case PHYSICS_BODY_OP_SET_LINEAR_VELOCITY:
        CPH_EnqueueCommand(world, slot, SetLinearVelocity{op.vector});
        return true;
    case PHYSICS_BODY_OP_SET_ANGULAR_VELOCITY:
        CPH_EnqueueCommand(world, slot, SetAngularVelocity{op.vector});
        return true;
    case PHYSICS_BODY_OP_SET_MOTION_TYPE:
        CPH_EnqueueCommand(world, slot, SetMotionType{op.value});
        return true;
    case PHYSICS_BODY_OP_ACTIVATE:
        CPH_EnqueueCommand(world, slot, Activate{});
        return true;
    case PHYSICS_BODY_OP_DEACTIVATE:
        CPH_EnqueueCommand(world, slot, Deactivate{});
        return true;
    case PHYSICS_BODY_OP_SET_USER_DATA:
        // Shadow-only; nothing to flush (user data never reaches Jolt).
        return true;
    case PHYSICS_BODY_OP_SET_TRANSFORM:
        CPH_EnqueueCommand(world, slot, SetTransform{op.position, op.rotation});
        return true;
    case PHYSICS_BODY_OP_SET_CCD:
        CPH_EnqueueCommand(world, slot, SetCcd{op.value});
        return true;
    default:
        return false;
    }
}

void mirror_to_shadow(CPH_PhysicsWorld *world, std::uint32_t slot,
                      const CPH_PhysicsBodyOperation &op) {
    const std::uint32_t dense = world->bodies.slot_to_dense[slot];
    switch (op.type) {
    case PHYSICS_BODY_OP_SET_POSITION: {
        const CPH_PosStride position = {op.position.x, op.position.y, op.position.z, 0.0};
        world->bodies.positions[dense]      = position;
        world->bodies.prev_positions[dense] = position;
        break;
    }
    case PHYSICS_BODY_OP_SET_ROTATION: {
        const CPH_AuxStride rotation = {op.rotation.x, op.rotation.y, op.rotation.z,
                                        op.rotation.w};
        world->bodies.rotations[dense]      = rotation;
        world->bodies.prev_rotations[dense] = rotation;
        break;
    }
    case PHYSICS_BODY_OP_SET_LINEAR_VELOCITY:
        world->bodies.linear_velocities[dense] = {op.vector.x, op.vector.y, op.vector.z, 0.0F};
        break;
    case PHYSICS_BODY_OP_SET_ANGULAR_VELOCITY:
        world->bodies.angular_velocities[dense] = {op.vector.x, op.vector.y, op.vector.z, 0.0F};
        break;
    case PHYSICS_BODY_OP_SET_USER_DATA:
        world->bodies.user_data[dense] = op.user_data;
        break;
    case PHYSICS_BODY_OP_SET_TRANSFORM: {
        const CPH_PosStride position = {op.position.x, op.position.y, op.position.z, 0.0};
        const CPH_AuxStride rotation = {op.rotation.x, op.rotation.y, op.rotation.z,
                                        op.rotation.w};
        world->bodies.positions[dense]      = position;
        world->bodies.prev_positions[dense] = position;
        world->bodies.rotations[dense]      = rotation;
        world->bodies.prev_rotations[dense] = rotation;
        break;
    }
    default:
        break;
    }
}

void execute_now(CPH_PhysicsWorld *world, JPH_BodyID body_id,
                 const CPH_PhysicsBodyOperation &op) {
    JPH_BodyInterface *bi = world->body_interface;
    JPH_Vec3 vector       = op.vector;
    switch (op.type) {
    case PHYSICS_BODY_OP_APPLY_IMPULSE:
        JPH_BodyInterface_AddImpulse(bi, body_id, &vector);
        break;
    case PHYSICS_BODY_OP_APPLY_IMPULSE_AT: {
        JPH_RVec3 position = op.position;
        JPH_BodyInterface_AddImpulse2(bi, body_id, &vector, &position);
        break;
    }
    case PHYSICS_BODY_OP_APPLY_ANGULAR_IMPULSE:
        JPH_BodyInterface_AddAngularImpulse(bi, body_id, &vector);
        break;
    case PHYSICS_BODY_OP_APPLY_FORCE:
        JPH_BodyInterface_AddForce(bi, body_id, &vector);
        break;
    case PHYSICS_BODY_OP_APPLY_TORQUE:
        JPH_BodyInterface_AddTorque(bi, body_id, &vector);
        break;
    default:
        return;
    }
    JPH_BodyInterface_ActivateBody(bi, body_id);
}

} // namespace

PhysicsBodyOperationStatus
CPH_PhysicsWorldApplyBodyOperation(CPH_PhysicsWorld *world,
                                   const CPH_PhysicsBodyOperation *operation) {
    if (!world || !operation || operation->type > PHYSICS_BODY_OP_SET_CCD) {
        return PHYSICS_BODY_OPERATION_STATUS_INVALID_ARGUMENT;
    }
    const CPH_PhysicsBodyOperation &op = *operation;

    bool execute_immediately = false;
    JPH_BodyID body_id       = kJphInvalidBodyId;
    {
        const auto shadow = CPH_LockShadowNotStepping(world);

        std::uint32_t slot = 0;
        if (!unpack_handle(world, op.handle, &slot)) {
            return PHYSICS_BODY_OPERATION_STATUS_STALE_HANDLE;
        }
        const auto state = static_cast<SlotState>(
            world->bodies.slot_states[slot].load(std::memory_order::acquire));
        const CPH_SlotPredicate predicate = get_slot_predicate(state, immediate_mask(op.type));
        if (!predicate.is_executable()) {
            return PHYSICS_BODY_OPERATION_STATUS_STALE_HANDLE;
        }
        if (CPH_UNLIKELY(!CPH_EnsureCommandCapacity(world))) {
            return PHYSICS_BODY_OPERATION_STATUS_OUT_OF_MEMORY;
        }

        // Impulse-class ops on live bodies run straight against Jolt below
        // instead of being deferred; everything else defers.
        execute_immediately = is_immediate_jolt_action(op.type) && predicate.is_immediate;
        if (!execute_immediately && !enqueue_operation(world, slot, op)) {
            return PHYSICS_BODY_OPERATION_STATUS_INVALID_ARGUMENT;
        }
        mirror_to_shadow(world, slot, op);
        if (execute_immediately) {
            body_id = world->bodies.body_ids[world->bodies.slot_to_dense[slot]];
        }
    }

    if (execute_immediately) {
        execute_now(world, body_id, op);
    }
    return PHYSICS_BODY_OPERATION_STATUS_OK;
}

PhysicsBodyOperationStatus CPH_PhysicsWorldQueueBodyDestroy(CPH_PhysicsWorld *world,
                                                            BodyHandle handle) {
    if (!world) {
        return PHYSICS_BODY_OPERATION_STATUS_INVALID_ARGUMENT;
    }
    const auto shadow = CPH_LockShadowNotStepping(world);

    std::uint32_t slot = 0;
    if (!unpack_handle(world, handle, &slot)) {
        return PHYSICS_BODY_OPERATION_STATUS_STALE_HANDLE;
    }
    if (!CPH_EnsureCommandCapacity(world)) {
        return PHYSICS_BODY_OPERATION_STATUS_OUT_OF_MEMORY;
    }
    const auto state = static_cast<SlotState>(
        world->bodies.slot_states[slot].load(std::memory_order::acquire));
    if (get_slot_predicate(state, kMaskDestructible).is_executable()) {
        CPH_EnqueueCommand(world, slot, DestroyBody{});
        world->bodies.slot_states[slot].store(SLOT_PENDING_DESTROY, std::memory_order::release);
    }
    return PHYSICS_BODY_OPERATION_STATUS_OK;
}

PhysicsBodyOperationStatus CPH_PhysicsWorldQueueBodyDestroyBatch(CPH_PhysicsWorld *world,
                                                                  const BodyHandle *handles,
                                                                  std::size_t handle_count) {
    if (!world || (handle_count != 0 && !handles)) {
        return PHYSICS_BODY_OPERATION_STATUS_INVALID_ARGUMENT;
    }
    if (handle_count == 0) {
        return PHYSICS_BODY_OPERATION_STATUS_OK;
    }
    const auto shadow = CPH_LockShadowNotStepping(world);
    if (!CPH_EnsureCommandBulkCapacity(world, handle_count)) {
        return PHYSICS_BODY_OPERATION_STATUS_OUT_OF_MEMORY;
    }
    for (std::size_t i = 0; i < handle_count; ++i) {
        std::uint32_t slot = 0;
        if (!unpack_handle(world, handles[i], &slot)) {
            continue;
        }
        const auto state = static_cast<SlotState>(
            world->bodies.slot_states[slot].load(std::memory_order::acquire));
        if (get_slot_predicate(state, kMaskDestructible).is_executable()) {
            CPH_EnqueueCommand(world, slot, DestroyBody{});
            world->bodies.slot_states[slot].store(SLOT_PENDING_DESTROY,
                                                  std::memory_order::release);
        }
    }
    return PHYSICS_BODY_OPERATION_STATUS_OK;
}
