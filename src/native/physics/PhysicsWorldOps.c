#include "physics/PhysicsWorldOps.h"
#include "physics/CommandBuffer.h"
#include "physics/SlotState.h"

static void lock_until_not_stepping(PhysicsWorld *world) {
    SHADOW_LOCK(&world->shadow_lock);
    if (!atomic_load_explicit(&world->is_stepping, memory_order_relaxed)) {
        return;
    }

    atomic_fetch_add_explicit(&world->waiting_threads, 1, memory_order_relaxed);
    while (atomic_load_explicit(&world->is_stepping, memory_order_relaxed)) {
        SHADOW_UNLOCK(&world->shadow_lock);
        NATIVE_MUTEX_LOCK(world->step_sync.mutex);
        while (atomic_load_explicit(&world->is_stepping, memory_order_relaxed)) {
            NATIVE_COND_WAIT(world->step_sync.cond, world->step_sync.mutex);
        }
        NATIVE_MUTEX_UNLOCK(world->step_sync.mutex);
        SHADOW_LOCK(&world->shadow_lock);
    }
    atomic_fetch_sub_explicit(&world->waiting_threads, 1, memory_order_relaxed);
}

static bool operation_is_immediate_jolt_action(PhysicsBodyOperationType type) {
    switch (type) {
    case PHYSICS_BODY_OP_APPLY_IMPULSE:
    case PHYSICS_BODY_OP_APPLY_IMPULSE_AT:
    case PHYSICS_BODY_OP_APPLY_ANGULAR_IMPULSE:
    case PHYSICS_BODY_OP_APPLY_FORCE:
    case PHYSICS_BODY_OP_APPLY_TORQUE:
        return true;
    default:
        return false;
    }
}

static uint32_t operation_immediate_mask(PhysicsBodyOperationType type) {
    switch (type) {
    case PHYSICS_BODY_OP_APPLY_ANGULAR_IMPULSE:
    case PHYSICS_BODY_OP_SET_ANGULAR_VELOCITY:
    case PHYSICS_BODY_OP_SET_MOTION_TYPE:
        return MASK_IMM_STRICT;
    default:
        return MASK_IMM_STANDARD;
    }
}

static bool write_operation_command(PhysicsWorld *world, uint32_t slot,
                                    const PhysicsBodyOperation *operation,
                                    SlotPredicate predicate) {
    PhysicsCommand *command     = &world->command_queue[world->command_count];
    const bool immediate_action = operation_is_immediate_jolt_action(operation->type);

    switch (operation->type) {
    case PHYSICS_BODY_OP_APPLY_IMPULSE:
        command->header  = CMD_HEADER(CMD_APPLY_IMPULSE, slot);
        command->vec3f.x = operation->vector.x;
        command->vec3f.y = operation->vector.y;
        command->vec3f.z = operation->vector.z;
        break;
    case PHYSICS_BODY_OP_APPLY_IMPULSE_AT:
        command->header        = CMD_HEADER(CMD_APPLY_IMPULSE_AT, slot);
        command->impulse_at.ix = operation->vector.x;
        command->impulse_at.iy = operation->vector.y;
        command->impulse_at.iz = operation->vector.z;
        command->impulse_at.px = operation->position.x;
        command->impulse_at.py = operation->position.y;
        command->impulse_at.pz = operation->position.z;
        break;
    case PHYSICS_BODY_OP_APPLY_ANGULAR_IMPULSE:
        command->header  = CMD_HEADER(CMD_APPLY_ANG_IMPULSE, slot);
        command->vec3f.x = operation->vector.x;
        command->vec3f.y = operation->vector.y;
        command->vec3f.z = operation->vector.z;
        break;
    case PHYSICS_BODY_OP_APPLY_FORCE:
        command->header  = CMD_HEADER(CMD_APPLY_FORCE, slot);
        command->vec3f.x = operation->vector.x;
        command->vec3f.y = operation->vector.y;
        command->vec3f.z = operation->vector.z;
        break;
    case PHYSICS_BODY_OP_APPLY_TORQUE:
        command->header  = CMD_HEADER(CMD_APPLY_TORQUE, slot);
        command->vec3f.x = operation->vector.x;
        command->vec3f.y = operation->vector.y;
        command->vec3f.z = operation->vector.z;
        break;
    case PHYSICS_BODY_OP_SET_POSITION:
        command->header = CMD_HEADER(CMD_SET_POS, slot);
        command->pos.x  = operation->position.x;
        command->pos.y  = operation->position.y;
        command->pos.z  = operation->position.z;
        break;
    case PHYSICS_BODY_OP_SET_ROTATION:
        command->header = CMD_HEADER(CMD_SET_ROT, slot);
        command->quat.x = operation->rotation.x;
        command->quat.y = operation->rotation.y;
        command->quat.z = operation->rotation.z;
        command->quat.w = operation->rotation.w;
        break;
    case PHYSICS_BODY_OP_SET_LINEAR_VELOCITY:
        command->header  = CMD_HEADER(CMD_SET_LINVEL, slot);
        command->vec3f.x = operation->vector.x;
        command->vec3f.y = operation->vector.y;
        command->vec3f.z = operation->vector.z;
        break;
    case PHYSICS_BODY_OP_SET_ANGULAR_VELOCITY:
        command->header  = CMD_HEADER(CMD_SET_ANGVEL, slot);
        command->vec3f.x = operation->vector.x;
        command->vec3f.y = operation->vector.y;
        command->vec3f.z = operation->vector.z;
        break;
    case PHYSICS_BODY_OP_SET_MOTION_TYPE:
        command->header             = CMD_HEADER(CMD_SET_MOTION, slot);
        command->motion.motion_type = operation->value;
        break;
    case PHYSICS_BODY_OP_SET_USER_DATA:
        command->header                  = CMD_HEADER(CMD_SET_USER_DATA, slot);
        command->user_data.user_data_val = operation->user_data;
        break;
    case PHYSICS_BODY_OP_ACTIVATE:
        command->header = CMD_HEADER(CMD_ACTIVATE, slot);
        break;
    case PHYSICS_BODY_OP_DEACTIVATE:
        command->header = CMD_HEADER(CMD_DEACTIVATE, slot);
        break;
    case PHYSICS_BODY_OP_SET_TRANSFORM:
        command->header       = CMD_HEADER(CMD_SET_TRNS, slot);
        command->transform.px = operation->position.x;
        command->transform.py = operation->position.y;
        command->transform.pz = operation->position.z;
        command->transform.rx = operation->rotation.x;
        command->transform.ry = operation->rotation.y;
        command->transform.rz = operation->rotation.z;
        command->transform.rw = operation->rotation.w;
        break;
    case PHYSICS_BODY_OP_SET_CCD:
        command->header             = CMD_HEADER(CMD_SET_CCD, slot);
        command->motion.motion_type = operation->value;
        break;
    default:
        return false;
    }

    world->command_count += immediate_action ? predicate.is_deferred : predicate.is_executable;
    return true;
}

static void mirror_operation_to_shadow(PhysicsWorld *world, uint32_t slot,
                                       const PhysicsBodyOperation *operation,
                                       SlotPredicate predicate) {
    if (!predicate.is_executable) {
        return;
    }

    const uint32_t dense = world->slot_to_dense[slot];
    switch (operation->type) {
    case PHYSICS_BODY_OP_SET_POSITION: {
        const PosStride position                    = {operation->position.x, operation->position.y,
                                                       operation->position.z, 0.0};
        ((PosStride *)world->positions)[dense]      = position;
        ((PosStride *)world->prev_positions)[dense] = position;
        break;
    }
    case PHYSICS_BODY_OP_SET_ROTATION: {
        const AuxStride rotation               = {operation->rotation.x, operation->rotation.y,
                                                  operation->rotation.z, operation->rotation.w};
        ((AuxStride *)world->rotations)[dense] = rotation;
        ((AuxStride *)world->prev_rotations)[dense] = rotation;
        break;
    }
    case PHYSICS_BODY_OP_SET_LINEAR_VELOCITY:
        ((AuxStride *)world->linear_velocities)[dense] =
            (AuxStride){operation->vector.x, operation->vector.y, operation->vector.z, 0.0f};
        break;
    case PHYSICS_BODY_OP_SET_ANGULAR_VELOCITY:
        ((AuxStride *)world->angular_velocities)[dense] =
            (AuxStride){operation->vector.x, operation->vector.y, operation->vector.z, 0.0f};
        break;
    case PHYSICS_BODY_OP_SET_USER_DATA:
        world->user_data[dense] = operation->user_data;
        break;
    case PHYSICS_BODY_OP_SET_TRANSFORM: {
        const PosStride position               = {operation->position.x, operation->position.y,
                                                  operation->position.z, 0.0};
        const AuxStride rotation               = {operation->rotation.x, operation->rotation.y,
                                                  operation->rotation.z, operation->rotation.w};
        ((PosStride *)world->positions)[dense] = position;
        ((PosStride *)world->prev_positions)[dense] = position;
        ((AuxStride *)world->rotations)[dense]      = rotation;
        ((AuxStride *)world->prev_rotations)[dense] = rotation;
        break;
    }
    default:
        break;
    }
}

static void execute_immediate_jolt_action(PhysicsWorld *world, JPH_BodyID body_id,
                                          const PhysicsBodyOperation *operation) {
    JPH_Vec3 vector    = operation->vector;
    JPH_RVec3 position = operation->position;
    switch (operation->type) {
    case PHYSICS_BODY_OP_APPLY_IMPULSE:
        JPH_BodyInterface_AddImpulse(world->body_interface, body_id, &vector);
        break;
    case PHYSICS_BODY_OP_APPLY_IMPULSE_AT:
        JPH_BodyInterface_AddImpulse2(world->body_interface, body_id, &vector, &position);
        break;
    case PHYSICS_BODY_OP_APPLY_ANGULAR_IMPULSE:
        JPH_BodyInterface_AddAngularImpulse(world->body_interface, body_id, &vector);
        break;
    case PHYSICS_BODY_OP_APPLY_FORCE:
        JPH_BodyInterface_AddForce(world->body_interface, body_id, &vector);
        break;
    case PHYSICS_BODY_OP_APPLY_TORQUE:
        JPH_BodyInterface_AddTorque(world->body_interface, body_id, &vector);
        break;
    default:
        return;
    }
    JPH_BodyInterface_ActivateBody(world->body_interface, body_id);
}

PhysicsBodyOperationStatus
physics_world_apply_body_operation(PhysicsWorld *world, const PhysicsBodyOperation *operation) {
    if (!world || !operation || operation->type > PHYSICS_BODY_OP_SET_CCD) {
        return PHYSICS_BODY_OPERATION_STATUS_INVALID_ARGUMENT;
    }

    lock_until_not_stepping(world);

    uint32_t slot = 0;
    if (!unpack_handle(world, operation->handle, &slot)) {
        SHADOW_UNLOCK(&world->shadow_lock);
        return PHYSICS_BODY_OPERATION_STATUS_STALE_HANDLE;
    }

    const uint8_t state = atomic_load_explicit(&world->slot_states[slot], memory_order_acquire);
    const SlotPredicate predicate =
        get_slot_predicate(state, operation_immediate_mask(operation->type));

    if (UNLIKELY(!ensure_command_capacity(world))) {
        SHADOW_UNLOCK(&world->shadow_lock);
        return PHYSICS_BODY_OPERATION_STATUS_OUT_OF_MEMORY;
    }

    if (!write_operation_command(world, slot, operation, predicate)) {
        SHADOW_UNLOCK(&world->shadow_lock);
        return PHYSICS_BODY_OPERATION_STATUS_INVALID_ARGUMENT;
    }
    mirror_operation_to_shadow(world, slot, operation, predicate);

    const bool execute_now =
        operation_is_immediate_jolt_action(operation->type) && predicate.is_immediate;
    const JPH_BodyID body_id =
        execute_now ? world->body_ids[world->slot_to_dense[slot]] : JPH_INVALID_BODY_ID;

    SHADOW_UNLOCK(&world->shadow_lock);

    if (!predicate.is_executable) {
        return PHYSICS_BODY_OPERATION_STATUS_STALE_HANDLE;
    }
    if (execute_now) {
        execute_immediate_jolt_action(world, body_id, operation);
    }
    return PHYSICS_BODY_OPERATION_STATUS_OK;
}

PhysicsBodyOperationStatus physics_world_queue_body_destroy(PhysicsWorld *world,
                                                            BodyHandle handle) {
    if (!world) {
        return PHYSICS_BODY_OPERATION_STATUS_INVALID_ARGUMENT;
    }

    lock_until_not_stepping(world);
    uint32_t slot = 0;
    if (!unpack_handle(world, handle, &slot)) {
        SHADOW_UNLOCK(&world->shadow_lock);
        return PHYSICS_BODY_OPERATION_STATUS_STALE_HANDLE;
    }
    const uint8_t state = atomic_load_explicit(&world->slot_states[slot], memory_order_acquire);
    const bool is_destructible = get_slot_predicate(state, MASK_DESTRUCTIBLE).is_executable;
    if (!ensure_command_capacity(world)) {
        SHADOW_UNLOCK(&world->shadow_lock);
        return PHYSICS_BODY_OPERATION_STATUS_OUT_OF_MEMORY;
    }

    PhysicsCommand *command = &world->command_queue[world->command_count];
    command->header         = CMD_HEADER(CMD_DESTROY_BODY, slot);
    world->command_count += is_destructible;
    if (is_destructible) {
        atomic_store_explicit(&world->slot_states[slot], SLOT_PENDING_DESTROY,
                              memory_order_release);
    }
    SHADOW_UNLOCK(&world->shadow_lock);
    return PHYSICS_BODY_OPERATION_STATUS_OK;
}

PhysicsBodyOperationStatus physics_world_queue_body_destroy_batch(PhysicsWorld *world,
                                                                  const BodyHandle *handles,
                                                                  size_t handle_count) {
    if (!world || (handle_count != 0 && !handles)) {
        return PHYSICS_BODY_OPERATION_STATUS_INVALID_ARGUMENT;
    }
    if (handle_count == 0) {
        return PHYSICS_BODY_OPERATION_STATUS_OK;
    }

    lock_until_not_stepping(world);
    if (!ensure_command_bulk_capacity(world, handle_count)) {
        SHADOW_UNLOCK(&world->shadow_lock);
        return PHYSICS_BODY_OPERATION_STATUS_OUT_OF_MEMORY;
    }

    PhysicsCommand *commands     = world->command_queue;
    size_t command_index         = world->command_count;
    CULV_ATOMIC(uint8_t) *states = world->slot_states;
    for (size_t i = 0; i < handle_count; i++) {
        uint32_t slot = 0;
        if (!unpack_handle(world, handles[i], &slot)) {
            continue;
        }
        const uint8_t state            = atomic_load_explicit(&states[slot], memory_order_acquire);
        const bool is_destructible     = get_slot_predicate(state, MASK_DESTRUCTIBLE).is_executable;
        commands[command_index].header = CMD_HEADER(CMD_DESTROY_BODY, slot);
        command_index += is_destructible;
        if (is_destructible) {
            atomic_store_explicit(&states[slot], SLOT_PENDING_DESTROY, memory_order_release);
        }
    }
    world->command_count = command_index;
    SHADOW_UNLOCK(&world->shadow_lock);
    return PHYSICS_BODY_OPERATION_STATUS_OK;
}
