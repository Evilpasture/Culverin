#include "physics/PhysicsWorldOps.hpp"

#include "physics/CommandBuffer.hpp"
#include "physics/SlotState.hpp"

#include <utility>


namespace CPH {

namespace {

using namespace CPH::cmd;

bool is_immediate_jolt_action(PhysicsBodyOperationType type) {
    return type == PhysicsBodyOperationType::ApplyImpulse || type == PhysicsBodyOperationType::ApplyImpulseAt ||
           type == PhysicsBodyOperationType::ApplyAngularImpulse || type == PhysicsBodyOperationType::ApplyForce ||
           type == PhysicsBodyOperationType::ApplyTorque;
}

std::uint32_t immediate_mask(PhysicsBodyOperationType type) {
    switch (type) {
    case PhysicsBodyOperationType::ApplyAngularImpulse:
    case PhysicsBodyOperationType::SetAngularVelocity:
    case PhysicsBodyOperationType::SetMotionType:
        return kMaskImmStrict;
    default:
        return kMaskImmStandard;
    }
}

// Returns false only for an unknown operation type.
bool enqueue_operation(PhysicsWorld *world, std::uint32_t slot,
                       const PhysicsBodyOperation &op) {
    switch (op.type) {
    case PhysicsBodyOperationType::ApplyImpulse:
        EnqueueCommand(world, slot, ApplyImpulse{op.vector});
        return true;
    case PhysicsBodyOperationType::ApplyImpulseAt:
        EnqueueCommand(world, slot, ApplyImpulseAt{op.vector, op.position});
        return true;
    case PhysicsBodyOperationType::ApplyAngularImpulse:
        EnqueueCommand(world, slot, ApplyAngularImpulse{op.vector});
        return true;
    case PhysicsBodyOperationType::ApplyForce:
        EnqueueCommand(world, slot, ApplyForce{op.vector});
        return true;
    case PhysicsBodyOperationType::ApplyTorque:
        EnqueueCommand(world, slot, ApplyTorque{op.vector});
        return true;
    case PhysicsBodyOperationType::SetPosition:
        EnqueueCommand(world, slot, SetPosition{op.position});
        return true;
    case PhysicsBodyOperationType::SetRotation:
        EnqueueCommand(world, slot, SetRotation{op.rotation});
        return true;
    case PhysicsBodyOperationType::SetLinearVelocity:
        EnqueueCommand(world, slot, SetLinearVelocity{op.vector});
        return true;
    case PhysicsBodyOperationType::SetAngularVelocity:
        EnqueueCommand(world, slot, SetAngularVelocity{op.vector});
        return true;
    case PhysicsBodyOperationType::SetMotionType:
        EnqueueCommand(world, slot, SetMotionType{op.value});
        return true;
    case PhysicsBodyOperationType::Activate:
        EnqueueCommand(world, slot, Activate{});
        return true;
    case PhysicsBodyOperationType::Deactivate:
        EnqueueCommand(world, slot, Deactivate{});
        return true;
    case PhysicsBodyOperationType::SetUserData:
        // Shadow-only; nothing to flush (user data never reaches Jolt).
        return true;
    case PhysicsBodyOperationType::SetTransform:
        EnqueueCommand(world, slot, SetTransform{op.position, op.rotation});
        return true;
    case PhysicsBodyOperationType::SetCcd:
        EnqueueCommand(world, slot, SetCcd{op.value});
        return true;
    default:
        return false;
    }
}

void mirror_to_shadow(PhysicsWorld *world, std::uint32_t slot,
                      const PhysicsBodyOperation &op) {
    const std::uint32_t dense = world->bodies.slot_to_dense[slot];
    switch (op.type) {
    case PhysicsBodyOperationType::SetPosition: {
        const PosStride position = {op.position.x, op.position.y, op.position.z, 0.0};
        world->bodies.positions[dense]      = position;
        world->bodies.prev_positions[dense] = position;
        break;
    }
    case PhysicsBodyOperationType::SetRotation: {
        const AuxStride rotation = {op.rotation.x, op.rotation.y, op.rotation.z,
                                        op.rotation.w};
        world->bodies.rotations[dense]      = rotation;
        world->bodies.prev_rotations[dense] = rotation;
        break;
    }
    case PhysicsBodyOperationType::SetLinearVelocity:
        world->bodies.linear_velocities[dense] = {op.vector.x, op.vector.y, op.vector.z, 0.0F};
        break;
    case PhysicsBodyOperationType::SetAngularVelocity:
        world->bodies.angular_velocities[dense] = {op.vector.x, op.vector.y, op.vector.z, 0.0F};
        break;
    case PhysicsBodyOperationType::SetUserData:
        world->bodies.user_data[dense] = op.user_data;
        break;
    case PhysicsBodyOperationType::SetTransform: {
        const PosStride position = {op.position.x, op.position.y, op.position.z, 0.0};
        const AuxStride rotation = {op.rotation.x, op.rotation.y, op.rotation.z,
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

void execute_now(PhysicsWorld *world, JPH_BodyID body_id,
                 const PhysicsBodyOperation &op) {
    JPH_BodyInterface *bi = world->body_interface;
    JPH_Vec3 vector       = op.vector;
    switch (op.type) {
    case PhysicsBodyOperationType::ApplyImpulse:
        JPH_BodyInterface_AddImpulse(bi, body_id, &vector);
        break;
    case PhysicsBodyOperationType::ApplyImpulseAt: {
        JPH_RVec3 position = op.position;
        JPH_BodyInterface_AddImpulse2(bi, body_id, &vector, &position);
        break;
    }
    case PhysicsBodyOperationType::ApplyAngularImpulse:
        JPH_BodyInterface_AddAngularImpulse(bi, body_id, &vector);
        break;
    case PhysicsBodyOperationType::ApplyForce:
        JPH_BodyInterface_AddForce(bi, body_id, &vector);
        break;
    case PhysicsBodyOperationType::ApplyTorque:
        JPH_BodyInterface_AddTorque(bi, body_id, &vector);
        break;
    default:
        return;
    }
    JPH_BodyInterface_ActivateBody(bi, body_id);
}

} // namespace

PhysicsBodyOperationStatus
PhysicsWorldApplyBodyOperation(PhysicsWorld *world,
                                   const PhysicsBodyOperation *operation) {
    if (!world || !operation || operation->type > PhysicsBodyOperationType::SetCcd) {
        return PHYSICS_BODY_OPERATION_STATUS_INVALID_ARGUMENT;
    }
    const PhysicsBodyOperation &op = *operation;

    bool execute_immediately = false;
    JPH_BodyID body_id       = kJphInvalidBodyId;
    {
        const auto shadow = LockShadowNotStepping(world);

        std::uint32_t slot = 0;
        if (!unpack_handle(world, op.handle, &slot)) {
            return PHYSICS_BODY_OPERATION_STATUS_STALE_HANDLE;
        }
        const auto state = static_cast<SlotState>(
            world->bodies.slot_states[slot].load(std::memory_order::acquire));
        const SlotPredicate predicate = get_slot_predicate(std::to_underlying(state), immediate_mask(op.type));
        if (!predicate.is_executable()) {
            return PHYSICS_BODY_OPERATION_STATUS_STALE_HANDLE;
        }
        if (CPH_UNLIKELY(!EnsureCommandCapacity(world))) {
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

PhysicsBodyOperationStatus PhysicsWorldQueueBodyDestroy(PhysicsWorld *world,
                                                            BodyHandle handle) {
    if (!world) {
        return PHYSICS_BODY_OPERATION_STATUS_INVALID_ARGUMENT;
    }
    const auto shadow = LockShadowNotStepping(world);

    std::uint32_t slot = 0;
    if (!unpack_handle(world, handle, &slot)) {
        return PHYSICS_BODY_OPERATION_STATUS_STALE_HANDLE;
    }
    if (!EnsureCommandCapacity(world)) {
        return PHYSICS_BODY_OPERATION_STATUS_OUT_OF_MEMORY;
    }
    const auto state = static_cast<SlotState>(
        world->bodies.slot_states[slot].load(std::memory_order::acquire));
    if (get_slot_predicate(std::to_underlying(state), kMaskDestructible).is_executable()) {
        EnqueueCommand(world, slot, DestroyBody{});
        world->bodies.slot_states[slot].store(std::to_underlying(SlotState::PendingDestroy), std::memory_order::release);
    }
    return PHYSICS_BODY_OPERATION_STATUS_OK;
}

PhysicsBodyOperationStatus PhysicsWorldQueueBodyDestroyBatch(PhysicsWorld *world,
                                                                  const BodyHandle *handles,
                                                                  std::size_t handle_count) {
    if (!world || (handle_count != 0 && !handles)) {
        return PHYSICS_BODY_OPERATION_STATUS_INVALID_ARGUMENT;
    }
    if (handle_count == 0) {
        return PHYSICS_BODY_OPERATION_STATUS_OK;
    }
    const auto shadow = LockShadowNotStepping(world);
    if (!EnsureCommandBulkCapacity(world, handle_count)) {
        return PHYSICS_BODY_OPERATION_STATUS_OUT_OF_MEMORY;
    }
    for (std::size_t i = 0; i < handle_count; ++i) {
        std::uint32_t slot = 0;
        if (!unpack_handle(world, handles[i], &slot)) {
            continue;
        }
        const auto state = static_cast<SlotState>(
            world->bodies.slot_states[slot].load(std::memory_order::acquire));
        if (get_slot_predicate(std::to_underlying(state), kMaskDestructible).is_executable()) {
            EnqueueCommand(world, slot, DestroyBody{});
            world->bodies.slot_states[slot].store(std::to_underlying(SlotState::PendingDestroy),
                                                  std::memory_order::release);
        }
    }
    return PHYSICS_BODY_OPERATION_STATUS_OK;
}

} // namespace CPH
