#include "physics/PhysicsWorldOps.hpp"

#include "physics/CommandBuffer.hpp"
#include "physics/PhysicsWorldState.hpp"
#include "physics/RotationSanitize.hpp"
#include "physics/SlotState.hpp"

#include <utility>


namespace CPH {

namespace {

using namespace CPH::cmd;

std::uint32_t ImmediateMask(PhysicsBodyOperationType type) {
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
bool EnqueueOperation(PhysicsWorld *world, std::uint32_t slot,
                       const PhysicsBodyOperation &op) {
    switch (op.type) {
    case PhysicsBodyOperationType::ApplyImpulse:
        world->EnqueueCommand(slot, ApplyImpulse{op.vector});
        return true;
    case PhysicsBodyOperationType::ApplyImpulseAt:
        world->EnqueueCommand(slot, ApplyImpulseAt{op.vector, op.position});
        return true;
    case PhysicsBodyOperationType::ApplyAngularImpulse:
        world->EnqueueCommand(slot, ApplyAngularImpulse{op.vector});
        return true;
    case PhysicsBodyOperationType::ApplyForce:
        world->EnqueueCommand(slot, ApplyForce{op.vector});
        return true;
    case PhysicsBodyOperationType::ApplyTorque:
        world->EnqueueCommand(slot, ApplyTorque{op.vector});
        return true;
    case PhysicsBodyOperationType::SetPosition:
        world->EnqueueCommand(slot, SetPosition{op.position});
        return true;
    case PhysicsBodyOperationType::SetRotation:
        world->EnqueueCommand(slot, SetRotation{op.rotation});
        return true;
    case PhysicsBodyOperationType::SetLinearVelocity:
        world->EnqueueCommand(slot, SetLinearVelocity{op.vector});
        return true;
    case PhysicsBodyOperationType::SetAngularVelocity:
        world->EnqueueCommand(slot, SetAngularVelocity{op.vector});
        return true;
    case PhysicsBodyOperationType::SetMotionType:
        world->EnqueueCommand(slot, SetMotionType{op.value});
        return true;
    case PhysicsBodyOperationType::Activate:
        world->EnqueueCommand(slot, Activate{});
        return true;
    case PhysicsBodyOperationType::Deactivate:
        world->EnqueueCommand(slot, Deactivate{});
        return true;
    case PhysicsBodyOperationType::SetUserData:
        // Shadow-only; nothing to flush (user data never reaches Jolt).
        return true;
    case PhysicsBodyOperationType::SetTransform:
        world->EnqueueCommand(slot, SetTransform{op.position, op.rotation});
        return true;
    case PhysicsBodyOperationType::SetCcd:
        world->EnqueueCommand(slot, SetCcd{op.value});
        return true;
    default:
        return false;
    }
}

void MirrorToShadow(PhysicsWorld *world, std::uint32_t slot,
                      const PhysicsBodyOperation &op) {
    const std::uint32_t dense = world->bodies.slotToDense[slot];
    // Set ops update the canonical bodies.* arrays (single source of truth for
    // Jolt sync and interpolation). The single PublishTransformSlot call below
    // mirrors the four exported channels into the published transform frames,
    // so a zero-copy memoryview observes the set immediately.
    switch (op.type) {
    case PhysicsBodyOperationType::SetPosition: {
        const PosStride position = {op.position.GetX(), op.position.GetY(), op.position.GetZ(), 0.0};
        world->bodies.positions[dense]      = position;
        world->bodies.prevPositions[dense] = position;
        break;
    }
    case PhysicsBodyOperationType::SetRotation: {
        const AuxStride rotation = {op.rotation.GetX(), op.rotation.GetY(), op.rotation.GetZ(),
                                        op.rotation.GetW()};
        world->bodies.rotations[dense]      = rotation;
        world->bodies.prevRotations[dense] = rotation;
        break;
    }
    case PhysicsBodyOperationType::SetLinearVelocity:
        world->bodies.linearVelocities[dense] = {op.vector.GetX(), op.vector.GetY(), op.vector.GetZ(), 0.0F};
        break;
    case PhysicsBodyOperationType::SetAngularVelocity:
        world->bodies.angularVelocities[dense] = {op.vector.GetX(), op.vector.GetY(), op.vector.GetZ(), 0.0F};
        break;
    case PhysicsBodyOperationType::SetUserData:
        // Shadow-only; nothing transform-related to publish.
        world->bodies.userData[dense] = op.userData;
        return;
    case PhysicsBodyOperationType::SetTransform: {
        const PosStride position = {op.position.GetX(), op.position.GetY(), op.position.GetZ(), 0.0};
        const AuxStride rotation = {op.rotation.GetX(), op.rotation.GetY(), op.rotation.GetZ(),
                                        op.rotation.GetW()};
        world->bodies.positions[dense]      = position;
        world->bodies.prevPositions[dense] = position;
        world->bodies.rotations[dense]      = rotation;
        world->bodies.prevRotations[dense] = rotation;
        break;
    }
    default:
        return;
    }
    world->PublishTransformSlot(dense);
}

} // namespace

std::expected<void, Error>
PhysicsWorld::ApplyBodyOperation(const PhysicsBodyOperation &op) {
    if (op.type > PhysicsBodyOperationType::SetCcd) {
        return std::unexpected(BodyOperationError::InvalidArgument);
    }

    // SetRotation and SetTransform carry an API-supplied quaternion that is
    // enqueued as a command and mirrored into the shadow buffers. Normalizing it
    // once, here, keeps those two in agreement and keeps Jolt's unit-length
    // precondition intact (see RotationSanitize.hpp). Every other operation
    // leaves `rotation` alone, so ops that never use it stay untouched.
    PhysicsBodyOperation sanitized = op;
    switch (sanitized.type) {
    case PhysicsBodyOperationType::SetRotation:
    case PhysicsBodyOperationType::SetTransform:
        sanitized.rotation = SanitizeRotation(op.rotation);
        break;
    default:
        break;
    }

    {
        const auto shadow = LockShadowNotStepping();

        const std::optional<std::uint32_t> resolved = Resolve(sanitized.handle);
        if (!resolved) {
            return std::unexpected(BodyOperationError::StaleHandle);
        }
        const std::uint32_t slot = *resolved;
        const auto state = static_cast<SlotState>(
            bodies.slotStates[slot].load(std::memory_order::acquire));
        const SlotPredicate predicate =
            GetSlotPredicate(std::to_underlying(state), ImmediateMask(sanitized.type));
        if (!predicate.is_executable()) {
            return std::unexpected(BodyOperationError::StaleHandle);
        }
        if (!EnsureCommandCapacity()) [[unlikely]] {
            return std::unexpected(BodyOperationError::OutOfMemory);
        }

        // Pure-asynchronous command queue (Step 3): every mutation -- even
        // impulses and forces -- is staged in the mailbox and applied by the
        // step thread before PhysicsSystem::Update(). No op reaches Jolt from
        // an arbitrary Python thread, so there is no immediate/deferred split
        // and no jph_trampoline_lock needed to protect it.
        if (!EnqueueOperation(this, slot, sanitized)) {
            return std::unexpected(BodyOperationError::InvalidArgument);
        }
        MirrorToShadow(this, slot, sanitized);
    }
    return {};
}

std::expected<void, Error> PhysicsWorld::QueueBodyDestroy(BodyHandle handle) {
    const auto shadow = LockShadowNotStepping();

    const std::optional<std::uint32_t> resolved = Resolve(handle);
    if (!resolved) {
        return std::unexpected(BodyOperationError::StaleHandle);
    }
    const std::uint32_t slot = *resolved;
    if (!EnsureCommandCapacity()) {
        return std::unexpected(BodyOperationError::OutOfMemory);
    }
    const auto state = static_cast<SlotState>(
        bodies.slotStates[slot].load(std::memory_order::acquire));
    if (GetSlotPredicate(std::to_underlying(state), kMaskDestructible).is_executable()) {
        EnqueueCommand(slot, DestroyBody{});
        bodies.slotStates[slot].store(std::to_underlying(SlotState::PendingDestroy), std::memory_order::release);
    }
    return {};
}

std::expected<void, Error>
PhysicsWorld::QueueBodyDestroyBatch(std::span<const BodyHandle> handles) {
    if (handles.empty()) {
        return {};
    }
    const auto shadow = LockShadowNotStepping();
    if (!EnsureCommandBulkCapacity(handles.size())) {
        return std::unexpected(BodyOperationError::OutOfMemory);
    }
    for (const BodyHandle handle : handles) {
        const std::optional<std::uint32_t> resolved = Resolve(handle);
        if (!resolved) {
            continue;
        }
        const std::uint32_t slot = *resolved;
        const auto state = static_cast<SlotState>(
            bodies.slotStates[slot].load(std::memory_order::acquire));
        if (GetSlotPredicate(std::to_underlying(state), kMaskDestructible).is_executable()) {
            EnqueueCommand(slot, DestroyBody{});
            bodies.slotStates[slot].store(std::to_underlying(SlotState::PendingDestroy),
                                           std::memory_order::release);
        }
    }
    return {};
}

} // namespace CPH
