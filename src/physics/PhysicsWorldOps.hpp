#pragma once

#include "physics/PhysicsWorldState.hpp"

#include <cstddef>
#include <cstdint>

enum PhysicsBodyOperationType : std::uint8_t {
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
};

struct CPH_PhysicsBodyOperation {
    PhysicsBodyOperationType type;
    BodyHandle handle;
    JPH_RVec3 position;
    JPH_Vec3 vector;
    JPH_Quat rotation;
    std::uint64_t user_data;
    int value;
};

enum PhysicsBodyOperationStatus : int {
    PHYSICS_BODY_OPERATION_STATUS_OK               = 0,
    PHYSICS_BODY_OPERATION_STATUS_OUT_OF_MEMORY    = -1,
    PHYSICS_BODY_OPERATION_STATUS_STALE_HANDLE     = -2,
    PHYSICS_BODY_OPERATION_STATUS_INVALID_ARGUMENT = -3,
};

// Executes an engine-side body mutation: command staging, shadow-buffer
// mirroring, and immediate Jolt application for impulse-style ops on
// already-live bodies. Stale-handle errors are translated by the binding.
PhysicsBodyOperationStatus
CPH_PhysicsWorldApplyBodyOperation(CPH_PhysicsWorld *world,
                                   const CPH_PhysicsBodyOperation *operation);
PhysicsBodyOperationStatus CPH_PhysicsWorldQueueBodyDestroy(CPH_PhysicsWorld *world,
                                                            BodyHandle handle);
PhysicsBodyOperationStatus
CPH_PhysicsWorldQueueBodyDestroyBatch(CPH_PhysicsWorld *world, const BodyHandle *handles,
                                       std::size_t handle_count);
