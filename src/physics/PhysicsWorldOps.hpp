#pragma once

#include "core/Error.hpp"
#include "physics/PhysicsWorldState.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>


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

enum class BodyOperationError : std::uint32_t {
    OutOfMemory     = 1,
    StaleHandle     = 2,
    InvalidArgument = 3,
};

// Executes an engine-side body mutation: command staging, shadow-buffer
// mirroring, and immediate Jolt application for impulse-style ops on
// already-live bodies. Stale-handle errors are translated by the binding.
[[nodiscard]] std::expected<void, Error>
PhysicsWorldApplyBodyOperation(PhysicsWorld *world,
                                   const PhysicsBodyOperation *operation);
[[nodiscard]] std::expected<void, Error> PhysicsWorldQueueBodyDestroy(PhysicsWorld *world,
                                                            BodyHandle handle);
[[nodiscard]] std::expected<void, Error>
PhysicsWorldQueueBodyDestroyBatch(PhysicsWorld *world, const BodyHandle *handles,
                                       std::size_t handle_count);

} // namespace CPH
