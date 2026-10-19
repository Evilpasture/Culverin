#include "physics/CommandBuffer.hpp"

#include "core/Buffer.hpp"
#include "physics/ConstraintFactory.hpp"
#include "physics/PhysicsWorldState.hpp"
#include "physics/SlotState.hpp"

#include <limits>
#include <utility>
#include <variant>

namespace {

constexpr std::size_t kInitialQueueCapacity = 64;

bool grow_queue(CPH_PhysicsWorld *world, std::size_t new_capacity) {
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

bool CPH_EnsureCommandBulkCapacity(CPH_PhysicsWorld *world, std::size_t batch_size) {
    if (batch_size > std::numeric_limits<std::size_t>::max() - world->command_count) {
        return false;
    }
    const std::size_t required = world->command_count + batch_size;
    if (CPH_UNLIKELY(required > world->command_queue.size())) {
        std::size_t new_capacity =
            world->command_queue.size() == 0 ? kInitialQueueCapacity : world->command_queue.size();
        while (new_capacity < required) {
            if (new_capacity > std::numeric_limits<std::size_t>::max() / 2) {
                return false;
            }
            new_capacity *= 2;
        }
        return grow_queue(world, new_capacity);
    }
    return true;
}

bool CPH_EnsureCommandCapacity(CPH_PhysicsWorld *world) {
    return CPH_EnsureCommandBulkCapacity(world, 1);
}

void CPH_WorldRemoveBodySlot(CPH_PhysicsWorld *world, std::uint32_t slot) {
    BodyBuffers &bodies = world->bodies;
    const std::uint32_t dense = bodies.slot_to_dense[slot];
    const std::uint32_t last =
        static_cast<std::uint32_t>(world->count.load(std::memory_order::acquire)) - 1;

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
        bodies.soft_shadows[dense] = CPH_SoftBodyShadow{};
    }

    // Recycle the slot: bump generation so stale handles fail, mark empty,
    // push onto the free stack, then publish the new count (release: the
    // data moves above must be visible before a reader sees the count).
    bodies.generations[slot].fetch_add(1, std::memory_order::relaxed);
    bodies.slot_states[slot].store(SLOT_EMPTY, std::memory_order::relaxed);
    const std::size_t free_index = world->free_count.fetch_add(1, std::memory_order::relaxed);
    bodies.free_slots[free_index] = slot;
    world->count.fetch_sub(1, std::memory_order::release);
}

void CPH_ClearCommandQueue(CPH_PhysicsWorld *world) {
    // Resetting each command destroys the payload (body settings, etc.)
    // it owns. The legacy loop freed only creation settings and leaked
    // heap constraint parameters.
    for (std::size_t i = 0; i < world->command_count; ++i) {
        world->command_queue[i] = PhysicsCommand{};
    }
    world->command_count = 0;
}

// --- Flush ---

namespace {

using namespace CPH::cmd;

struct FlushContext {
    CPH_PhysicsWorld *world;
    JPH_BodyInterface *bi;
    std::uint32_t slot;
    JPH_BodyID bid; // resolved body ID (invalid for creates)

    void operator()(std::monostate) const {}

    void operator()(CreateBody &cmd) const {
        const JPH_BodyID new_bid =
            JPH_BodyInterface_CreateAndAddBody(bi, cmd.settings.get(), JPH_Activation_Activate);
        cmd.settings.reset();

        const std::scoped_lock shadow(world->shadow_lock);
        if (CPH_UNLIKELY(new_bid == kJphInvalidBodyId)) {
            CPH_WorldRemoveBodySlot(world, slot);
            return;
        }
        world->bodies.body_ids[world->bodies.slot_to_dense[slot]] = new_bid;
        publish_handle(slot, new_bid);
        world->bodies.slot_states[slot].store(SLOT_ALIVE, std::memory_order::release);
    }

    void operator()(CreateSoftBody &cmd) const {
        const JPH_BodyID new_bid = JPH_BodyInterface_CreateAndAddSoftBody(bi, cmd.settings.get(),
                                                                          JPH_Activation_Activate);
        const std::scoped_lock shadow(world->shadow_lock);
        if (CPH_UNLIKELY(new_bid == kJphInvalidBodyId)) {
            CPH_WorldRemoveBodySlot(world, slot);
        } else {
            const std::uint32_t dense = world->bodies.slot_to_dense[slot];
            world->bodies.body_ids[dense] = new_bid;
            if (!world->bodies.soft_shadows[dense].vertices.allocate(cmd.num_vertices)) {
                // OOM: flush has no error channel. The body still
                // simulates in Jolt; its (empty) shadow makes vertex
                // getters report 0 vertices instead of reading stale data.
            }
            publish_handle(slot, new_bid);
            world->bodies.slot_states[slot].store(SLOT_SOFT_BODY, std::memory_order::release);
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
        JPH_BodyInterface_RemoveBody(bi, bid);
        JPH_BodyInterface_DestroyBody(bi, bid);
        const std::scoped_lock shadow(world->shadow_lock);
        CPH_WorldRemoveBodySlot(world, slot);
    }

    void operator()(SetPosition &cmd) const {
        const bool active = JPH_BodyInterface_IsActive(bi, bid);
        JPH_BodyInterface_SetPosition(
            bi, bid, &cmd.position,
            active ? JPH_Activation_DontActivate : JPH_Activation_Activate);
    }

    void operator()(SetRotation &cmd) const {
        JPH_BodyInterface_SetRotation(bi, bid, &cmd.rotation, JPH_Activation_Activate);
    }

    void operator()(SetTransform &cmd) const {
        JPH_BodyInterface_SetPositionAndRotation(bi, bid, &cmd.position, &cmd.rotation,
                                                 JPH_Activation_Activate);
    }

    void operator()(SetLinearVelocity &cmd) const {
        JPH_BodyInterface_SetLinearVelocity(bi, bid, &cmd.velocity);
    }

    void operator()(SetAngularVelocity &cmd) const {
        JPH_BodyInterface_SetAngularVelocity(bi, bid, &cmd.velocity);
    }

    void operator()(SetMotionType &cmd) const {
        JPH_BodyInterface_SetMotionType(bi, bid, static_cast<JPH_MotionType>(cmd.motion_type),
                                        JPH_Activation_Activate);
        JPH_BodyInterface_SetObjectLayer(
            bi, bid, static_cast<JPH_ObjectLayer>(cmd.motion_type == 0 ? 0 : 1));
    }

    void operator()(Activate &) const { JPH_BodyInterface_ActivateBody(bi, bid); }
    void operator()(Deactivate &) const { JPH_BodyInterface_DeactivateBody(bi, bid); }

    void operator()(SetCcd &cmd) const {
        JPH_BodyInterface_SetMotionQuality(
            bi, bid, cmd.enabled ? JPH_MotionQuality_LinearCast : JPH_MotionQuality_Discrete);
    }

    void operator()(ApplyImpulse &cmd) const {
        JPH_BodyInterface_AddImpulse(bi, bid, &cmd.impulse);
        JPH_BodyInterface_ActivateBody(bi, bid);
    }

    void operator()(ApplyForce &cmd) const {
        JPH_BodyInterface_AddForce(bi, bid, &cmd.force);
        JPH_BodyInterface_ActivateBody(bi, bid);
    }

    void operator()(ApplyTorque &cmd) const {
        JPH_BodyInterface_AddTorque(bi, bid, &cmd.torque);
        JPH_BodyInterface_ActivateBody(bi, bid);
    }

    void operator()(ApplyAngularImpulse &cmd) const {
        JPH_BodyInterface_AddAngularImpulse(bi, bid, &cmd.impulse);
        JPH_BodyInterface_ActivateBody(bi, bid);
    }

    void operator()(ApplyImpulseAt &cmd) const {
        JPH_BodyInterface_AddImpulse2(bi, bid, &cmd.impulse, &cmd.position);
        JPH_BodyInterface_ActivateBody(bi, bid);
    }

    void operator()(CreateConstraint &cmd) const {
        const std::uint32_t c_slot = cmd.constraint_slot;
        ConstraintSlot &c_entry    = world->constraints[c_slot];

        // destroy_constraint() may have run while this was queued.
        if (c_entry.state.load(std::memory_order::acquire) != SLOT_PENDING_CREATE) {
            return;
        }

        const JPH_BodyID bid2 =
            world->bodies.body_ids[world->bodies.slot_to_dense[cmd.body2_slot]];
        if (bid == kJphInvalidBodyId || bid2 == kJphInvalidBodyId) {
            return;
        }

        const JPH_BodyLockInterface *li =
            JPH_PhysicsSystem_GetBodyLockInterface(world->system);
        // Lock each body in a separate scope: Jolt hashes BodyIDs onto a
        // small mutex array, so holding two write locks can recurse one
        // mutex and abort (EDEADLK). The world is quiesced here, so the
        // stable body pointers remain valid after unlocking.
        const auto lock_body = [li](JPH_BodyID id) -> JPH_Body * {
            JPH_BodyLockWrite lock;
            JPH_BodyLockInterface_LockWrite(li, id, &lock);
            JPH_Body *body =
                (lock.body && JPH_Body_GetID(lock.body) == id) ? lock.body : nullptr;
            JPH_BodyLockInterface_UnlockWrite(li, &lock);
            return body;
        };
        JPH_Body *const b1 = lock_body(bid);
        JPH_Body *const b2 = lock_body(bid2);

        JPH_Constraint *constraint =
            (b1 && b2) ? CPH_CreateConstraint(static_cast<ConstraintType>(cmd.type), cmd.params,
                                              b1, b2)
                       : nullptr;
        if (constraint) {
            JPH_PhysicsSystem_AddConstraint(world->system, constraint);
            c_entry.constraint = constraint;
            c_entry.state.store(SLOT_ALIVE, std::memory_order::release);
        } else {
            c_entry.state.store(SLOT_EMPTY, std::memory_order::relaxed);
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

void CPH_FlushCommandsInternal(CPH_PhysicsWorld *world, std::span<PhysicsCommand> queue) {
    for (auto &command : queue) {
        const std::uint32_t slot = command.slot;
        const SlotState state =
            static_cast<SlotState>(world->bodies.slot_states[slot].load(std::memory_order::acquire));

        const bool is_create = std::holds_alternative<CreateBody>(command.payload) ||
                               std::holds_alternative<CreateSoftBody>(command.payload);
        const JPH_BodyID bid = world->bodies.body_ids[world->bodies.slot_to_dense[slot]];

        // Stale commands (body destroyed while queued, create superseded)
        // are dropped; resetting the command still destroys payloads.
        if (state < 32 && ((1U << state) & kValidBidMask) &&
            (is_create || bid != kJphInvalidBodyId)) {
            std::visit(FlushContext{world, world->body_interface, slot, bid}, command.payload);
        }
        command = PhysicsCommand{};
    }
}
