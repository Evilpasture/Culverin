#include "jolt/JoltUtils.hpp"
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

namespace {

constexpr std::size_t kInitialQueueCapacity = 64;

bool grow_queue(PhysicsWorld *world, std::size_t new_capacity) {
    // for_overwrite: queue slots past command_count are only ever
    // move-assigned into, never read — default construction (variant at
    // its first alternative) is all they need.
    //
    // Only the active queue grows. The spare keeps its current size:
    // detach is a plain swap, and the next enqueue grows the (now active)
    // ex-spare on demand if it is too small. Growing both queues here
    // would default-construct tens of thousands of variants that a
    // create-without-step workload never touches (legacy's spare was a
    // bare malloc whose pages were never faulted until used).
    return world->command_queue.reallocate_for_overwrite(new_capacity);
}

} // namespace

bool PhysicsWorld::EnsureCommandBulkCapacity(std::size_t batch_size) {
    if (batch_size > std::numeric_limits<std::size_t>::max() - command_count) {
        return false;
    }
    const std::size_t required = command_count + batch_size;
    if (CPH_UNLIKELY(required > command_queue.size())) {
        std::size_t new_capacity =
            command_queue.size() == 0 ? kInitialQueueCapacity : command_queue.size();
        while (new_capacity < required) {
            if (new_capacity > std::numeric_limits<std::size_t>::max() / 2) {
                return false;
            }
            new_capacity *= 2;
        }
        return grow_queue(this, new_capacity);
    }
    return true;
}

bool PhysicsWorld::EnsureCommandCapacity() {
    return EnsureCommandBulkCapacity(1);
}

void PhysicsWorld::RemoveBodySlot(std::uint32_t slot) {
    BodyBuffers &bodies = this->bodies;
    const std::uint32_t dense = bodies.slot_to_dense[slot];
    const std::uint32_t last =
        static_cast<std::uint32_t>(count.load(std::memory_order::acquire)) - 1;

    if (dense != last) {
        bodies.positions[dense]          = bodies.positions[last];
        bodies.prev_positions[dense]     = bodies.prev_positions[last];
        bodies.rotations[dense]          = bodies.rotations[last];
        bodies.prev_rotations[dense]     = bodies.prev_rotations[last];
        bodies.linear_velocities[dense]  = bodies.linear_velocities[last];
        bodies.angular_velocities[dense] = bodies.angular_velocities[last];
        bodies.body_ids[dense]           = bodies.body_ids[last];
        bodies.user_data[dense]          = bodies.user_data[last];
        bodies.categories[dense]         = bodies.categories[last];
        bodies.masks[dense]              = bodies.masks[last];
        bodies.material_ids[dense]       = bodies.material_ids[last];

        const std::uint32_t mover = bodies.dense_to_slot[last];
        bodies.slot_to_dense[mover] = dense;
        bodies.dense_to_slot[dense] = mover;

        // Move assignment releases the removed body's vertex buffer and
        // leaves the tail shadow empty (no double free).
        bodies.soft_shadows[dense] = std::move(bodies.soft_shadows[last]);
    } else {
        bodies.soft_shadows[dense] = SoftBodyShadow{};
    }

    // Recycle the slot: bump generation so stale handles fail, mark empty,
    // push onto the free stack, then publish the new count (release: the
    // data moves above must be visible before a reader sees the count).
    bodies.generations[slot].fetch_add(1, std::memory_order::relaxed);
    bodies.slot_states[slot].store(std::to_underlying(SlotState::Empty), std::memory_order::relaxed);
    const std::size_t free_index = free_count.fetch_add(1, std::memory_order::relaxed);
    bodies.free_slots[free_index] = slot;
    count.fetch_sub(1, std::memory_order::release);
}

void PhysicsWorld::ClearCommandQueue() {
    // Resetting each command destroys the payload (body settings, etc.)
    // it owns. The legacy loop freed only creation settings and leaked
    // heap constraint parameters.
    for (std::size_t i = 0; i < command_count; ++i) {
        command_queue[i] = PhysicsCommand{};
    }
    command_count = 0;
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
    JPH_BodyID bid; // resolved body ID (invalid for creates)

    // JPH_BodyID is typedef uint32_t; Jolt wants the packed word wrapped.
    [[nodiscard]] JPH::BodyID BodyId() const noexcept { return ToJoltBodyId(bid); }

    void operator()(std::monostate) const {}

    void operator()(CreateBody &cmd) const {
        const JPH_BodyID new_bid = world->physics_system.GetBodyInterface()
                                       .CreateAndAddBody(*cmd.settings, JPH::EActivation::Activate)
                                       .GetIndexAndSequenceNumber();
        cmd.settings.reset();

        const std::scoped_lock shadow(world->shadow_lock);
        if (CPH_UNLIKELY(new_bid == kJphInvalidBodyId)) {
            world->RemoveBodySlot(slot);
            return;
        }
        world->bodies.body_ids[world->bodies.slot_to_dense[slot]] = new_bid;
        publish_handle(slot, new_bid);
        world->bodies.slot_states[slot].store(std::to_underlying(SlotState::Alive), std::memory_order::release);
    }

    void operator()(CreateSoftBody &cmd) const {
        const JPH_BodyID new_bid =
            world->physics_system.GetBodyInterface()
                .CreateAndAddSoftBody(*cmd.settings, JPH::EActivation::Activate)
                .GetIndexAndSequenceNumber();
        const std::scoped_lock shadow(world->shadow_lock);
        if (CPH_UNLIKELY(new_bid == kJphInvalidBodyId)) {
            world->RemoveBodySlot(slot);
        } else {
            const std::uint32_t dense = world->bodies.slot_to_dense[slot];
            world->bodies.body_ids[dense] = new_bid;
            if (!world->bodies.soft_shadows[dense].vertices.allocate(cmd.num_vertices)) {
                // OOM: flush has no error channel. The body still
                // simulates in Jolt; its (empty) shadow makes vertex
                // getters report 0 vertices instead of reading stale data.
            }
            publish_handle(slot, new_bid);
            world->bodies.slot_states[slot].store(std::to_underlying(SlotState::SoftBody), std::memory_order::release);
        }
        cmd.settings.reset();
    }

    void operator()(DestroyBody &) const {
        // Clear the body-pointer cache first so shadow sync cannot chase
        // a freed JPH_Body next frame.
        const std::uint32_t j_idx = JPH_ID_TO_INDEX(bid);
        if (world->jolt_body_ptrs && j_idx <= world->max_jolt_bodies) {
            world->jolt_body_ptrs[j_idx] = nullptr;
        }
        bi.RemoveBody(BodyId());
        bi.DestroyBody(BodyId());
        const std::scoped_lock shadow(world->shadow_lock);
        world->RemoveBodySlot(slot);
    }

    void operator()(SetPosition &cmd) const {
        const bool active = bi.IsActive(BodyId());
        bi.SetPosition(BodyId(), ToJolt(cmd.position),
                       active ? JPH::EActivation::DontActivate : JPH::EActivation::Activate);
    }

    void operator()(SetRotation &cmd) const {
        bi.SetRotation(BodyId(), ToJolt(cmd.rotation), JPH::EActivation::Activate);
    }

    void operator()(SetTransform &cmd) const {
        bi.SetPositionAndRotation(BodyId(), ToJolt(cmd.position), ToJolt(cmd.rotation),
                                  JPH::EActivation::Activate);
    }

    void operator()(SetLinearVelocity &cmd) const {
        bi.SetLinearVelocity(BodyId(), ToJolt(cmd.velocity));
    }

    void operator()(SetAngularVelocity &cmd) const {
        bi.SetAngularVelocity(BodyId(), ToJolt(cmd.velocity));
    }

    void operator()(SetMotionType &cmd) const {
        bi.SetMotionType(BodyId(), static_cast<JPH::EMotionType>(cmd.motion_type),
                         JPH::EActivation::Activate);
        // TODO(smell): the object layer is derived from the motion type rather
        // than carried on the command -- SetMotionType only has a motion_type
        // field. Static(0) -> ObjectLayer::Static(0), Kinematic(1)/Dynamic(2) ->
        // ObjectLayer::Dynamic(1). That happens to match Types.hpp:95-100 today
        // but conflates motion type with collision layer, and can never select
        // ObjectLayer::Vehicle. Preserved verbatim; fixing it means widening the
        // command, which is a behaviour change.
        bi.SetObjectLayer(BodyId(),
                          static_cast<JPH::ObjectLayer>(cmd.motion_type == 0 ? 0 : 1));
    }

    void operator()(Activate &) const { bi.ActivateBody(BodyId()); }
    void operator()(Deactivate &) const { bi.DeactivateBody(BodyId()); }

    void operator()(SetCcd &cmd) const {
        bi.SetMotionQuality(BodyId(), cmd.enabled ? JPH::EMotionQuality::LinearCast
                                              : JPH::EMotionQuality::Discrete);
    }

    void operator()(ApplyImpulse &cmd) const {
        bi.AddImpulse(BodyId(), ToJolt(cmd.impulse));
        bi.ActivateBody(BodyId());
    }

    void operator()(ApplyForce &cmd) const {
        bi.AddForce(BodyId(), ToJolt(cmd.force));
        bi.ActivateBody(BodyId());
    }

    void operator()(ApplyTorque &cmd) const {
        bi.AddTorque(BodyId(), ToJolt(cmd.torque));
        bi.ActivateBody(BodyId());
    }

    void operator()(ApplyAngularImpulse &cmd) const {
        bi.AddAngularImpulse(BodyId(), ToJolt(cmd.impulse));
        bi.ActivateBody(BodyId());
    }

    void operator()(ApplyImpulseAt &cmd) const {
        bi.AddImpulse(BodyId(), ToJolt(cmd.impulse), ToJolt(cmd.position));
        bi.ActivateBody(BodyId());
    }

    void operator()(cmd::CreateConstraint &cmd) const {
        const std::uint32_t c_slot = cmd.constraint_slot;
        ConstraintSlot &c_entry    = world->constraints[c_slot];

        // destroy_constraint() may have run while this was queued.
        if (c_entry.state.load(std::memory_order::acquire) != std::to_underlying(SlotState::PendingCreate)) {
            return;
        }

        const JPH_BodyID bid2 =
            world->bodies.body_ids[world->bodies.slot_to_dense[cmd.body2_slot]];
        if (bid == kJphInvalidBodyId || bid2 == kJphInvalidBodyId) {
            return;
        }

        const JPH::BodyLockInterface &li = world->physics_system.GetBodyLockInterface();
        // Lock each body in a separate scope: Jolt hashes BodyIDs onto a
        // small mutex array, so holding two write locks can recurse one
        // mutex and abort (EDEADLK). The world is quiesced here, so the
        // stable body pointers remain valid after unlocking.
        const auto lock_body = [&li](JPH_BodyID id) -> JPH::Body * {
            const JPH::BodyID body_id(id);
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
            world->physics_system.AddConstraint(constraint);
            c_entry.constraint = constraint;
            c_entry.state.store(std::to_underlying(SlotState::Alive), std::memory_order::release);
        } else {
            c_entry.state.store(std::to_underlying(SlotState::Empty), std::memory_order::relaxed);
            world->free_constraint_slots[world->free_constraint_count++] = c_slot;
        }
    }

  private:
    void publish_handle(std::uint32_t body_slot, JPH_BodyID new_bid) const {
        const std::uint32_t j_idx = JPH_ID_TO_INDEX(new_bid);
        if (world->id_to_handle_map && j_idx <= world->max_jolt_bodies) {
            const std::uint32_t gen =
                world->bodies.generations[body_slot].load(std::memory_order::relaxed);
            // Release: the body_ids write must be visible to any thread
            // that reads this handle.
            world->id_to_handle_map[j_idx].store(make_handle(body_slot, gen),
                                                 std::memory_order::release);
        }
    }
};

} // namespace

void PhysicsWorld::FlushCommands(std::span<PhysicsCommand> queue) {
    for (auto &command : queue) {
        const std::uint32_t slot = command.slot;
        const SlotState state =
            static_cast<SlotState>(bodies.slot_states[slot].load(std::memory_order::acquire));

        const bool is_create = std::holds_alternative<cmd::CreateBody>(command.payload) ||
                               std::holds_alternative<cmd::CreateSoftBody>(command.payload);
        const JPH_BodyID bid = bodies.body_ids[bodies.slot_to_dense[slot]];

        // Stale commands (body destroyed while queued, create superseded)
        // are dropped; resetting the command still destroys payloads.
        if (std::to_underlying(state) < 32 && ((slot_state_bit(state) & kValidBidMask) != 0) &&
            (is_create || bid != kJphInvalidBodyId)) {
            std::visit(FlushContext{this, physics_system.GetBodyInterface(), slot, bid},
                       command.payload);
        }
        command = PhysicsCommand{};
    }
}

} // namespace CPH
