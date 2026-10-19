#pragma once

// Deferred mutation queue. Python-visible mutations are staged here and
// applied to Jolt by the stepper; that deferral is the thread-safety
// contract (producers never touch Jolt state concurrently with a step).
//
// Each command is a slot index plus a typed payload. The legacy encoding
// (64-byte union, bit-packed header, computed-goto dispatch) has been
// replaced by std::variant + std::visit: every command performs at least
// one Jolt BodyInterface call, which dwarfs dispatch cost by orders of
// magnitude, and the typed payloads own their Jolt settings so queue
// teardown can no longer leak them.

#include "jolt/JoltConfig.hpp"
#include "jolt/JoltUtils.hpp"
#include "physics/PhysicsParams.hpp"
#include <joltc.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <variant>

struct CPH_PhysicsWorld;

namespace CPH::cmd {

using BodySettingsPtr =
    JoltPtr<JPH_BodyCreationSettings, JPH_BodyCreationSettings_Destroy>;
using SoftBodySettingsPtr =
    JoltPtr<JPH_SoftBodyCreationSettings, JPH_SoftBodyCreationSettings_Destroy>;

struct CreateBody {
    BodySettingsPtr settings;
};
struct CreateSoftBody {
    SoftBodySettingsPtr settings;
    std::uint32_t num_vertices = 0;
};
struct DestroyBody {};
struct SetPosition {
    JPH_RVec3 position;
};
struct SetRotation {
    JPH_Quat rotation;
};
struct SetTransform {
    JPH_RVec3 position;
    JPH_Quat rotation;
};
struct SetLinearVelocity {
    JPH_Vec3 velocity;
};
struct SetAngularVelocity {
    JPH_Vec3 velocity;
};
struct SetMotionType {
    std::int32_t motion_type;
};
struct Activate {};
struct Deactivate {};
struct SetCcd {
    std::int32_t enabled;
};
struct ApplyImpulse {
    JPH_Vec3 impulse;
};
struct ApplyForce {
    JPH_Vec3 force;
};
struct ApplyTorque {
    JPH_Vec3 torque;
};
struct ApplyAngularImpulse {
    JPH_Vec3 impulse;
};
struct ApplyImpulseAt {
    JPH_Vec3 impulse;
    JPH_RVec3 position;
};
struct CreateConstraint {
    std::uint32_t body2_slot;
    std::uint32_t constraint_slot;
    int type;
    CPH_ConstraintParams params;
};

} // namespace CPH::cmd

struct PhysicsCommand {
    std::uint32_t slot = 0;
    std::variant<std::monostate, CPH::cmd::CreateBody, CPH::cmd::CreateSoftBody,
                 CPH::cmd::DestroyBody, CPH::cmd::SetPosition, CPH::cmd::SetRotation,
                 CPH::cmd::SetTransform, CPH::cmd::SetLinearVelocity,
                 CPH::cmd::SetAngularVelocity, CPH::cmd::SetMotionType, CPH::cmd::Activate,
                 CPH::cmd::Deactivate, CPH::cmd::SetCcd, CPH::cmd::ApplyImpulse,
                 CPH::cmd::ApplyForce, CPH::cmd::ApplyTorque, CPH::cmd::ApplyAngularImpulse,
                 CPH::cmd::ApplyImpulseAt, CPH::cmd::CreateConstraint>
        payload;
};

// Swap-remove `slot` from the dense body arrays and recycle it.
// Caller holds shadow_lock.
void CPH_WorldRemoveBodySlot(CPH_PhysicsWorld *world, std::uint32_t slot);

// Queue growth; caller holds shadow_lock and the world is not stepping.
[[nodiscard]] bool CPH_EnsureCommandCapacity(CPH_PhysicsWorld *world);
[[nodiscard]] bool CPH_EnsureCommandBulkCapacity(CPH_PhysicsWorld *world, std::size_t batch_size);

// Executes and clears each command. Caller holds jph_trampoline_lock.
void CPH_FlushCommandsInternal(CPH_PhysicsWorld *world, std::span<PhysicsCommand> queue);

// Drops pending commands, destroying payloads they own.
void CPH_ClearCommandQueue(CPH_PhysicsWorld *world);
