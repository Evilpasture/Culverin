#include "physics/CommandBuffer.hpp"

#include "core/Buffer.hpp"
#include "physics/ConstraintFactory.hpp"
#include "physics/PhysicsWorldState.hpp"
#include "physics/SlotState.hpp"

#include <Jolt/Physics/Body/BodyLock.h>

#include <limits>
#include <utility>
#include <variant>


namespace CPH {

bool PhysicsWorld::EnsureCommandBulkCapacity(std::size_t batch_size) {
    // The mailbox owns the active buffer and grows it under its own mutex.
    return mailbox.Reserve(batch_size);
}

bool PhysicsWorld::EnsureCommandCapacity() {
    return EnsureCommandBulkCapacity(1);
}

void PhysicsWorld::RemoveBodySlot(std::uint32_t slot) {
    BodyBuffers &bodies = this->bodies;
    const std::uint32_t dense = bodies.slotToDense[slot];
    const std::uint32_t last =
        static_cast<std::uint32_t>(count.load(std::memory_order::acquire)) - 1;

    if (dense != last) {
        bodies.positions[dense]          = bodies.positions[last];
        bodies.prevPositions[dense]     = bodies.prevPositions[last];
        bodies.rotations[dense]          = bodies.rotations[last];
        bodies.prevRotations[dense]     = bodies.prevRotations[last];
        bodies.linearVelocities[dense]  = bodies.linearVelocities[last];
        bodies.angularVelocities[dense] = bodies.angularVelocities[last];
        bodies.bodyIds[dense]           = bodies.bodyIds[last];
        bodies.userData[dense]          = bodies.userData[last];
        bodies.categories[dense]         = bodies.categories[last];
        bodies.masks[dense]              = bodies.masks[last];
        bodies.materialIds[dense]       = bodies.materialIds[last];

        const std::uint32_t mover = bodies.denseToSlot[last];
        bodies.slotToDense[mover] = dense;
        bodies.denseToSlot[dense] = mover;

        // Move assignment releases the removed body's vertex buffer and
        // leaves the tail shadow empty (no double free).
        bodies.softShadows[dense] = std::move(bodies.softShadows[last]);

        // A swap-remove moved `last` into `dense`; mirror that transform into
        // the published frames so a zero-copy view sees the same packed dense
        // layout as the canonical buffers (the C arrays used to be one and the
        // same). Caller holds shadowLock with the world quiesced.
        PublishTransformSlot(dense);
    } else {
        bodies.softShadows[dense] = SoftBodyShadow{};
    }

    // Recycle the slot: bump generation so stale handles fail, mark empty,
    // push onto the free stack, then publish the new count (release: the
    // data moves above must be visible before a reader sees the count).
    bodies.generations[slot].fetch_add(1, std::memory_order::relaxed);
    bodies.slotStates[slot].store(std::to_underlying(SlotState::Empty), std::memory_order::relaxed);
    const std::size_t free_index = freeCount.fetch_add(1, std::memory_order::relaxed);
    bodies.freeSlots[free_index] = slot;
    count.fetch_sub(1, std::memory_order::release);
}

void PhysicsWorld::ClearCommandQueue() {
    // Destroy any pending payloads (body settings, etc.) via the mailbox.
    mailbox.Clear();
}

// --- Flush ---

namespace {

using namespace CPH::cmd;

struct FlushContext {
    PhysicsWorld *world;
    // The locking BodyInterface, which is what the JPH_BodyInterface * this
    // replaced was a reinterpret_cast of (PhysicsWorld.cpp:637).
    JPH::BodyInterface &bi;
    std::uint32_t slot;
    JPH::BodyID bid; // resolved body ID (invalid for creates)

    void operator()(std::monostate) const {}

    void operator()(CreateBody &cmd) const {
        const JPH::BodyID new_bid = world->physicsSystem.GetBodyInterface().CreateAndAddBody(
            *cmd.settings, JPH::EActivation::Activate);
        cmd.settings.reset();

        const std::scoped_lock shadow(world->shadowLock);
        if (new_bid.IsInvalid()) [[unlikely]] {
            world->RemoveBodySlot(slot);
            return;
        }
        world->bodies.bodyIds[world->bodies.slotToDense[slot]] = new_bid;
        publish_handle(slot, new_bid);
        world->bodies.slotStates[slot].store(std::to_underlying(SlotState::Alive), std::memory_order::release);
    }

    void operator()(CreateSoftBody &cmd) const {
        const JPH::BodyID new_bid = world->physicsSystem.GetBodyInterface().CreateAndAddSoftBody(
            *cmd.settings, JPH::EActivation::Activate);
        const std::scoped_lock shadow(world->shadowLock);
        if (new_bid.IsInvalid()) [[unlikely]] {
            world->RemoveBodySlot(slot);
        } else {
            const std::uint32_t dense = world->bodies.slotToDense[slot];
            world->bodies.bodyIds[dense] = new_bid;
            if (!world->bodies.softShadows[dense].vertices.allocate(cmd.numVertices)) {
                // OOM: flush has no error channel. The body still
                // simulates in Jolt; its (empty) shadow makes vertex
                // getters report 0 vertices instead of reading stale data.
            }
            publish_handle(slot, new_bid);
            world->bodies.slotStates[slot].store(std::to_underlying(SlotState::SoftBody), std::memory_order::release);
        }
        cmd.settings.reset();
    }

    void operator()(DestroyBody &) const {
        // Clear the body-pointer cache first so shadow sync cannot chase
        // a freed JPH_Body next frame.
        const std::uint32_t j_idx = bid.GetIndex();
        if (world->joltBodyPtrs && j_idx <= world->maxJoltBodies) {
            world->joltBodyPtrs[j_idx] = nullptr;
        }
        bi.RemoveBody(bid);
        bi.DestroyBody(bid);
        const std::scoped_lock shadow(world->shadowLock);
        world->RemoveBodySlot(slot);
    }

    void operator()(SetPosition &cmd) const {
        const bool active = bi.IsActive(bid);
        bi.SetPosition(bid, cmd.position,
                       active ? JPH::EActivation::DontActivate : JPH::EActivation::Activate);
    }

    void operator()(SetRotation &cmd) const {
        bi.SetRotation(bid, cmd.rotation, JPH::EActivation::Activate);
    }

    void operator()(SetTransform &cmd) const {
        bi.SetPositionAndRotation(bid, cmd.position, cmd.rotation,
                                  JPH::EActivation::Activate);
    }

    void operator()(SetLinearVelocity &cmd) const {
        bi.SetLinearVelocity(bid, cmd.velocity);
    }

    void operator()(SetAngularVelocity &cmd) const {
        bi.SetAngularVelocity(bid, cmd.velocity);
    }

    void operator()(SetMotionType &cmd) const {
        bi.SetMotionType(bid, static_cast<JPH::EMotionType>(cmd.motionType),
                         JPH::EActivation::Activate);
        // TODO(smell): the object layer is derived from the motion type rather
        // than carried on the command -- SetMotionType only has a motion_type
        // field. Static(0) -> ObjectLayer::Static(0), Kinematic(1)/Dynamic(2) ->
        // ObjectLayer::Dynamic(1). That happens to match Types.hpp:95-100 today
        // but conflates motion type with collision layer, and can never select
        // ObjectLayer::Vehicle. Preserved verbatim; fixing it means widening the
        // command, which is a behaviour change.
        bi.SetObjectLayer(bid,
                          static_cast<JPH::ObjectLayer>(cmd.motionType == 0 ? 0 : 1));
    }

    void operator()(Activate &) const { bi.ActivateBody(bid); }
    void operator()(Deactivate &) const { bi.DeactivateBody(bid); }

    void operator()(SetCcd &cmd) const {
        bi.SetMotionQuality(bid, cmd.enabled ? JPH::EMotionQuality::LinearCast
                                              : JPH::EMotionQuality::Discrete);
    }

    void operator()(ApplyImpulse &cmd) const {
        bi.AddImpulse(bid, cmd.impulse);
        bi.ActivateBody(bid);
    }

    void operator()(ApplyForce &cmd) const {
        bi.AddForce(bid, cmd.force);
        bi.ActivateBody(bid);
    }

    void operator()(ApplyTorque &cmd) const {
        bi.AddTorque(bid, cmd.torque);
        bi.ActivateBody(bid);
    }

    void operator()(ApplyAngularImpulse &cmd) const {
        bi.AddAngularImpulse(bid, cmd.impulse);
        bi.ActivateBody(bid);
    }

    void operator()(ApplyImpulseAt &cmd) const {
        bi.AddImpulse(bid, cmd.impulse, cmd.position);
        bi.ActivateBody(bid);
    }

    void operator()(cmd::CreateConstraint &cmd) const {
        const std::uint32_t c_slot = cmd.constraintSlot;
        ConstraintSlot &c_entry    = world->constraints[c_slot];

        // destroy_constraint() may have run while this was queued.
        if (c_entry.state.load(std::memory_order::acquire) != std::to_underlying(SlotState::PendingCreate)) {
            return;
        }

        const JPH::BodyID bid2 =
            world->bodies.bodyIds[world->bodies.slotToDense[cmd.body2Slot]];
        if (bid.IsInvalid() || bid2.IsInvalid()) {
            return;
        }

        const JPH::BodyLockInterface &li = world->physicsSystem.GetBodyLockInterface();
        // Lock each body in a separate scope: Jolt hashes BodyIDs onto a
        // small mutex array, so holding two write locks can recurse one
        // mutex and abort (EDEADLK). The world is quiesced here, so the
        // stable body pointers remain valid after unlocking.
        const auto lock_body = [&li](const JPH::BodyID &body_id) -> JPH::Body * {
            const JPH::BodyLockWrite lock(li, body_id);
            return (lock.Succeeded() && lock.GetBody().GetID() == body_id) ? &lock.GetBody()
                                                                           : nullptr;
        };
        JPH::Body *const b1 = lock_body(bid);
        JPH::Body *const b2 = lock_body(bid2);

        JPH::Constraint *constraint =
            (b1 && b2) ? CPH::CreateConstraint(static_cast<ConstraintType>(cmd.type), cmd.params,
                                              *b1, *b2)
                       : nullptr;
        if (constraint) {
            world->physicsSystem.AddConstraint(constraint);
            c_entry.constraint = constraint;
            c_entry.state.store(std::to_underlying(SlotState::Alive), std::memory_order::release);
        } else {
            c_entry.state.store(std::to_underlying(SlotState::Empty), std::memory_order::relaxed);
            world->freeConstraintSlots[world->freeConstraintCount++] = c_slot;
        }
    }

  private:
    void publish_handle(std::uint32_t body_slot, const JPH::BodyID &new_bid) const {
        const std::uint32_t j_idx = new_bid.GetIndex();
        if (world->idToHandleMap && j_idx <= world->maxJoltBodies) {
            const std::uint32_t gen =
                world->bodies.generations[body_slot].load(std::memory_order::relaxed);
            // Release: the body_ids write must be visible to any thread
            // that reads this handle.
            world->idToHandleMap[j_idx].store(BodyHandle::FromParts(body_slot, gen),
                                                 std::memory_order::release);
        }
    }
};

} // namespace

void PhysicsWorld::FlushCommands(std::span<PhysicsCommand> queue) {
    for (auto &command : queue) {
        const std::uint32_t slot = command.slot;
        const SlotState state =
            static_cast<SlotState>(bodies.slotStates[slot].load(std::memory_order::acquire));

        const bool is_create = std::holds_alternative<cmd::CreateBody>(command.payload) ||
                               std::holds_alternative<cmd::CreateSoftBody>(command.payload);
        const JPH::BodyID bid = bodies.bodyIds[bodies.slotToDense[slot]];

        // Stale commands (body destroyed while queued, create superseded)
        // are dropped; resetting the command still destroys payloads.
        if (std::to_underlying(state) < 32 && ((SlotStateBit(state) & kValidBidMask) != 0) &&
            (is_create || !bid.IsInvalid())) {
            std::visit(FlushContext{this, physicsSystem.GetBodyInterface(), slot, bid},
                       command.payload);
        }
        command = PhysicsCommand{};
    }
}

} // namespace CPH
