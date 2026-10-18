#pragma once

#include "physics/PhysicsWorldState.h"

typedef enum PhysicsBodyOperationType : uint8_t {
    PHYSICS_BODY_OP_APPLY_IMPULSE,
    PHYSICS_BODY_OP_APPLY_IMPULSE_AT,
    PHYSICS_BODY_OP_APPLY_ANGULAR_IMPULSE,
    PHYSICS_BODY_OP_APPLY_FORCE,
    PHYSICS_BODY_OP_APPLY_TORQUE,
    PHYSICS_BODY_OP_SET_POSITION,
    PHYSICS_BODY_OP_SET_ROTATION,
    PHYSICS_BODY_OP_SET_LINEAR_VELOCITY,
    PHYSICS_BODY_OP_SET_ANGULAR_VELOCITY,
    PHYSICS_BODY_OP_SET_MOTION_TYPE,
    PHYSICS_BODY_OP_SET_USER_DATA,
    PHYSICS_BODY_OP_ACTIVATE,
    PHYSICS_BODY_OP_DEACTIVATE,
    PHYSICS_BODY_OP_SET_TRANSFORM,
    PHYSICS_BODY_OP_SET_CCD,
} PhysicsBodyOperationType;

typedef struct PhysicsBodyOperation {
    PhysicsBodyOperationType type;
    BodyHandle handle;
    JPH_RVec3 position;
    JPH_Vec3 vector;
    JPH_Quat rotation;
    uint64_t user_data;
    int value;
} PhysicsBodyOperation;

typedef enum PhysicsBodyOperationStatus : int {
    PHYSICS_BODY_OPERATION_STATUS_OK               = 0,
    PHYSICS_BODY_OPERATION_STATUS_OUT_OF_MEMORY    = -1,
    PHYSICS_BODY_OPERATION_STATUS_STALE_HANDLE     = -2,
    PHYSICS_BODY_OPERATION_STATUS_INVALID_ARGUMENT = -3,
} PhysicsBodyOperationStatus;

// Executes engine-side body mutations, command staging, and shadow-buffer
// mirroring. Call with the GIL released; stale-handle errors are translated by
// the CPython binding.
PhysicsBodyOperationStatus
physics_world_apply_body_operation(PhysicsWorld *world, const PhysicsBodyOperation *operation);
PhysicsBodyOperationStatus physics_world_queue_body_destroy(PhysicsWorld *world, BodyHandle handle);
PhysicsBodyOperationStatus physics_world_queue_body_destroy_batch(PhysicsWorld *world,
                                                                  const BodyHandle *handles,
                                                                  size_t handle_count);
