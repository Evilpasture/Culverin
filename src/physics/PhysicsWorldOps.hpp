#pragma once

#include "physics/PhysicsWorldState.hpp"

#include <cstddef>
#include <cstdint>


namespace CPH {

enum class PhysicsBodyOperationType : std::uint8_t {
    ApplyImpulse,
    ApplyImpulseAt,
    ApplyAngularImpulse,
    ApplyForce,
    ApplyTorque,
    SetPosition,
    SetRotation,
    SetLinearVelocity,
    SetAngularVelocity,
    SetMotionType,
    SetUserData,
    Activate,
    Deactivate,
    SetTransform,
    SetCcd,
};

struct PhysicsBodyOperation {
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
PhysicsWorldApplyBodyOperation(PhysicsWorld *world,
                                   const PhysicsBodyOperation *operation);
PhysicsBodyOperationStatus PhysicsWorldQueueBodyDestroy(PhysicsWorld *world,
                                                            BodyHandle handle);
PhysicsBodyOperationStatus
PhysicsWorldQueueBodyDestroyBatch(PhysicsWorld *world, const BodyHandle *handles,
                                       std::size_t handle_count);

} // namespace CPH
