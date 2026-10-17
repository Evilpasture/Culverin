#include "culverin_command_buffer.h"
#include "culverin_constraint_factory.h"
#include "culverin_native_alloc.h"
#include "culverin_physics_world_state.h"
#include <string.h>

static constexpr size_t AVX_ALIGNMENT = 32;


void world_remove_body_slot(PhysicsWorld *world, uint32_t slot) {
    const uint32_t dense_idx = world->slot_to_dense[slot];

    // TSan Fix: Load count atomically to determine the last index
    const uint32_t last_dense =
        (uint32_t)atomic_load_explicit(&world->count, memory_order_acquire) - 1;

    if (world->soft_shadows && world->soft_shadows[dense_idx].vertices) {
        CulvNative_AlignedFree(world->soft_shadows[dense_idx].vertices);
        world->soft_shadows[dense_idx].vertices = nullptr;

        if (world->soft_shadows[dense_idx].velocities) {
            CulvNative_AlignedFree(world->soft_shadows[dense_idx].velocities);
            world->soft_shadows[dense_idx].velocities = nullptr;
        }
        if (world->soft_shadows[dense_idx].normals) {
            CulvNative_AlignedFree(world->soft_shadows[dense_idx].normals);
            world->soft_shadows[dense_idx].normals = nullptr;
        }
        world->soft_shadows[dense_idx].num_vertices = 0;
    }

    // THE SWAP-TO-DELETE (Dense Pack)
    if (dense_idx != last_dense) {
        CULV_PREFETCH_READ(&world->positions[last_dense]);
        CULV_PREFETCH_WRITE(&world->positions[dense_idx]);

        // --- GROUP 1: PHYSICS STATE (Non-atomic) ---
        ((PosStride * CULV_RESTRICT) world->positions)[dense_idx] =
            ((PosStride * CULV_RESTRICT) world->positions)[last_dense];
        ((PosStride * CULV_RESTRICT) world->prev_positions)[dense_idx] =
            ((PosStride * CULV_RESTRICT) world->prev_positions)[last_dense];

        ((AuxStride * CULV_RESTRICT) world->rotations)[dense_idx] =
            ((AuxStride *)world->rotations)[last_dense];
        ((AuxStride * CULV_RESTRICT) world->prev_rotations)[dense_idx] =
            ((AuxStride * CULV_RESTRICT) world->prev_rotations)[last_dense];

        ((AuxStride * CULV_RESTRICT) world->linear_velocities)[dense_idx] =
            ((AuxStride * CULV_RESTRICT) world->linear_velocities)[last_dense];
        ((AuxStride * CULV_RESTRICT) world->angular_velocities)[dense_idx] =
            ((AuxStride * CULV_RESTRICT) world->angular_velocities)[last_dense];

        // --- GROUP 2: METADATA (Non-atomic) ---
        world->body_ids[dense_idx]     = world->body_ids[last_dense];
        world->user_data[dense_idx]    = world->user_data[last_dense];
        world->categories[dense_idx]   = world->categories[last_dense];
        world->masks[dense_idx]        = world->masks[last_dense];
        world->material_ids[dense_idx] = world->material_ids[last_dense];

        // --- GROUP 3: THE MAP REWIRE ---
        const uint32_t mover_slot       = world->dense_to_slot[last_dense];
        world->slot_to_dense[mover_slot] = dense_idx;
        world->dense_to_slot[dense_idx]  = mover_slot;

        if (world->soft_shadows) {
            world->soft_shadows[dense_idx] = world->soft_shadows[last_dense];
            // Clear the old tail so we don't double-free later
            world->soft_shadows[last_dense].vertices     = nullptr;
            world->soft_shadows[last_dense].velocities   = nullptr;
            world->soft_shadows[last_dense].normals      = nullptr;
            world->soft_shadows[last_dense].num_vertices = 0;
        }
    }

    // HOUSEKEEPING

    // 1. Invalidate all existing body handles by incrementing the generation
    atomic_fetch_add_explicit(&world->generations[slot], 1, memory_order_relaxed);

    // 2. Mark the slot as empty atomically
    atomic_store_explicit(&world->slot_states[slot], SLOT_EMPTY, memory_order_relaxed);

    // 3. Push to free stack atomically
    // We fetch the current count, use it as index, and increment
    size_t f_idx            = atomic_fetch_add_explicit(&world->free_count, 1, memory_order_relaxed);
    world->free_slots[f_idx] = slot;

    // 4. Update the total world count atomically
    // We use memory_order_release to ensure all memory moves above are visible
    // to any thread reading the count (like the renderer).
    atomic_fetch_sub_explicit(&world->count, 1, memory_order_release);
}

// Internal helper to keep queues in sync
static bool grow_queues(PhysicsWorld *world, size_t new_cap) {
    if (new_cap > (SIZE_MAX / sizeof(PhysicsCommand))) {
        return false;
    }

    const size_t bytes = new_cap * sizeof(PhysicsCommand);
    PhysicsCommand *new_active =
        (PhysicsCommand *)CulvNative_AlignedMalloc(bytes, MEMORY_ALIGNMENT_SIZE);
    PhysicsCommand *new_spare =
        (PhysicsCommand *)CulvNative_AlignedMalloc(bytes, MEMORY_ALIGNMENT_SIZE);
    if (!new_active || !new_spare) {
        CulvNative_AlignedFree(new_active);
        CulvNative_AlignedFree(new_spare);
        return false;
    }

    // Only the active queue owns pending command payloads. The spare queue is
    // empty whenever growth is allowed (the world is not stepping).
    if (world->command_count > 0) {
        memcpy(new_active, world->command_queue,
               world->command_count * sizeof(PhysicsCommand));
    }

    CulvNative_AlignedFree(world->command_queue);
    CulvNative_AlignedFree(world->command_queue_spare);
    world->command_queue       = new_active;
    world->command_queue_spare = new_spare;
    world->command_capacity    = new_cap;
    return true;
}

CULV_NODISCARD
bool ensure_command_capacity(PhysicsWorld *world) {
    if (UNLIKELY(world->command_count >= world->command_capacity)) {
        if (world->command_capacity > SIZE_MAX / 2) {
            return false;
        }
        size_t new_cap = (world->command_capacity == 0) ? 64 : world->command_capacity * 2;
        return grow_queues(world, new_cap);
    }
    return true;
}

CULV_NODISCARD
bool ensure_command_bulk_capacity(PhysicsWorld *world, size_t batch_size) {
    if (batch_size > SIZE_MAX - world->command_count) {
        return false;
    }
    const size_t required = world->command_count + batch_size;
    if (UNLIKELY(required > world->command_capacity)) {
        size_t new_cap = (world->command_capacity == 0) ? 64 : world->command_capacity;
        while (new_cap < required) {
            if (new_cap > SIZE_MAX / 2) {
                return false;
            }
            new_cap *= 2;
        }
        return grow_queues(world, new_cap);
    }
    return true;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[gnu::hot, gnu::flatten, gnu::nonnull(1, 2)]]
void flush_commands_internal(PhysicsWorld *world, PhysicsCommand *CULV_RESTRICT queue,
                             size_t count) { // Check count before calling.
    queue                               = CULV_ASSUME_ALIGNED(queue, 64);
    JPH_BodyInterface *CULV_RESTRICT bi = world->body_interface;

    static const void *const dispatch_table[] = {
        [CMD_CREATE_BODY]       = &&op_CREATE_BODY,
        [CMD_DESTROY_BODY]      = &&op_DESTROY_BODY,
        [CMD_SET_POS]           = &&op_SET_POS,
        [CMD_SET_ROT]           = &&op_SET_ROT,
        [CMD_SET_TRNS]          = &&op_SET_TRNS,
        [CMD_SET_LINVEL]        = &&op_SET_LINVEL,
        [CMD_SET_ANGVEL]        = &&op_SET_ANGVEL,
        [CMD_SET_MOTION]        = &&op_SET_MOTION,
        [CMD_ACTIVATE]          = &&op_ACTIVATE,
        [CMD_DEACTIVATE]        = &&op_DEACTIVATE,
        [CMD_SET_USER_DATA]     = &&op_SET_USER_DATA,
        [CMD_SET_CCD]           = &&op_SET_CCD,
        [CMD_TELEPORT]          = &&op_TELEPORT,
        [CMD_APPLY_IMPULSE]     = &&op_APPLY_IMPULSE,
        [CMD_APPLY_FORCE]       = &&op_APPLY_FORCE,
        [CMD_APPLY_TORQUE]      = &&op_APPLY_TORQUE,
        [CMD_APPLY_ANG_IMPULSE] = &&op_APPLY_ANG_IMPULSE,
        [CMD_APPLY_IMPULSE_AT]  = &&op_APPLY_IMPULSE_AT,
        [CMD_CREATE_SOFT_BODY]  = &&op_CREATE_SOFT_BODY,
        [CMD_CREATE_CONSTRAINT] = &&op_CREATE_CONSTRAINT,
    };

    size_t i = 0;
    PhysicsCommand *cmd;
    uint32_t header;
    uint32_t slot;
    CULV_MAYBE_UNUSED uint32_t dense;
    CommandType type;
    SlotState state;
    JPH_BodyID bid;

op_NOP:
    DISPATCH();

op_CREATE_BODY: {
    JPH_BodyCreationSettings *const settings = cmd->create.settings;
    const JPH_BodyID new_bid =
        JPH_BodyInterface_CreateAndAddBody(bi, settings, JPH_Activation_Activate);

    JPH_BodyCreationSettings_Destroy(settings);

    SHADOW_LOCK(&world->shadow_lock);

    if (UNLIKELY(new_bid == JPH_INVALID_BODY_ID)) {
        // world_remove_body_slot handles atomic count/state updates internally
        world_remove_body_slot(world, slot);
    } else {
        // body_ids is non-atomic; protected by shadow_lock and the 'is_stepping' phase
        world->body_ids[world->slot_to_dense[slot]] = new_bid;

        const uint32_t j_idx = JPH_ID_TO_INDEX(new_bid);
        if (world->id_to_handle_map && j_idx <= world->max_jolt_bodies) {
            // TSan Fix: Load generation atomically
            const uint32_t gen =
                atomic_load_explicit(&world->generations[slot], memory_order_relaxed);

            // BodyHandle is CULV_ATOMIC(uint64_t)
            const BodyHandle h = make_handle(slot, gen);

            // TSan Fix: Extract raw uint64_t to avoid implicit seq_cst load overhead
            const uint64_t raw_h = h;

            // TSan Fix: Publish the new handle to the shared map atomically.
            // Release ensures the body_ids update above is visible to Query threads.
            atomic_store_explicit(&world->id_to_handle_map[j_idx], raw_h, memory_order_release);
        }

        // TSan Fix: Mark the body as ALIVE atomically.
        // Release creates a barrier: any thread that sees SLOT_ALIVE is guaranteed
        // to see the correctly initialized body_ids and id_to_handle_map data.
        atomic_store_explicit(&world->slot_states[slot], SLOT_ALIVE, memory_order_release);
    }

    SHADOW_UNLOCK(&world->shadow_lock);

    DISPATCH();
}

op_CREATE_SOFT_BODY: {
    JPH_SoftBodyCreationSettings *const settings = cmd->create_soft.settings;
    const uint32_t num_verts = cmd->create_soft.num_vertices; // O(1) Cache-local read!

    const JPH_BodyID new_bid =
        JPH_BodyInterface_CreateAndAddSoftBody(bi, settings, JPH_Activation_Activate);

    SHADOW_LOCK(&world->shadow_lock);

    if (UNLIKELY(new_bid == JPH_INVALID_BODY_ID)) {
        world_remove_body_slot(world, slot);
    } else {
        const uint32_t dense_idx  = world->slot_to_dense[slot];
        world->body_ids[dense_idx] = new_bid;

        // --- ALLOCATE SHADOW VERTEX BUFFER ---
        world->soft_shadows[dense_idx].num_vertices = num_verts;
        world->soft_shadows[dense_idx].vertices =
            (JPH_Real *)CulvNative_AlignedMalloc(num_verts * sizeof(PosStride), AVX_ALIGNMENT);
        world->soft_shadows[dense_idx].velocities = nullptr; // Initialize explicitly
        world->soft_shadows[dense_idx].normals    = nullptr; // Initialize explicitly

        // Populate standard handles
        const uint32_t j_idx = JPH_ID_TO_INDEX(new_bid);
        if (world->id_to_handle_map && j_idx <= world->max_jolt_bodies) {
            const uint32_t gen =
                atomic_load_explicit(&world->generations[slot], memory_order_relaxed);
            const BodyHandle h   = make_handle(slot, gen);
            const uint64_t raw_h = h;
            atomic_store_explicit(&world->id_to_handle_map[j_idx], raw_h, memory_order_release);
        }

        atomic_store_explicit(&world->slot_states[slot], SLOT_SOFT_BODY, memory_order_release);
    }

    JPH_SoftBodyCreationSettings_Destroy(settings); // Cleanup

    SHADOW_UNLOCK(&world->shadow_lock);
    DISPATCH();
}

op_DESTROY_BODY: {
    // If we don't clear this, the next frame's sync step might try to call
    // b->GetID() on a freed C++ pointer!
    const uint32_t j_idx = JPH_ID_TO_INDEX(bid);
    if (world->jolt_body_ptrs && j_idx <= world->max_jolt_bodies) {
        world->jolt_body_ptrs[j_idx] = nullptr;
    }
    JPH_BodyInterface_RemoveBody(bi, bid);
    JPH_BodyInterface_DestroyBody(bi, bid);
    SHADOW_LOCK(&world->shadow_lock);
    world_remove_body_slot(world, slot);
    SHADOW_UNLOCK(&world->shadow_lock);
    DISPATCH();
}

op_SET_POS: {
    JPH_STACK_ALLOC(JPH_RVec3, p);
    p->x        = cmd->pos.x;
    p->y        = cmd->pos.y;
    p->z        = cmd->pos.z;
    bool active = JPH_BodyInterface_IsActive(bi, bid);
    JPH_BodyInterface_SetPosition(
        bi, bid, p, (int)active ? JPH_Activation_DontActivate : JPH_Activation_Activate);
    DISPATCH();
}

op_SET_ROT: {
    JPH_STACK_ALLOC(JPH_Quat, q);
    q->x = cmd->quat.x;
    q->y = cmd->quat.y;
    q->z = cmd->quat.z;
    q->w = cmd->quat.w;
    JPH_BodyInterface_SetRotation(bi, bid, q, JPH_Activation_Activate);
    DISPATCH();
}

op_SET_TRNS: {
    JPH_STACK_ALLOC(JPH_RVec3, p);
    p->x = cmd->transform.px;
    p->y = cmd->transform.py;
    p->z = cmd->transform.pz;

    JPH_STACK_ALLOC(JPH_Quat, q);
    q->x = cmd->transform.rx;
    q->y = cmd->transform.ry;
    q->z = cmd->transform.rz;
    q->w = cmd->transform.rw;

    JPH_BodyInterface_SetPositionAndRotation(bi, bid, p, q, JPH_Activation_Activate);
    DISPATCH();
}

op_SET_LINVEL: {
    JPH_STACK_ALLOC(JPH_Vec3, v);
    v->x = cmd->vec3f.x;
    v->y = cmd->vec3f.y;
    v->z = cmd->vec3f.z;
    JPH_BodyInterface_SetLinearVelocity(bi, bid, v);
    DISPATCH();
}

op_SET_ANGVEL: {
    JPH_STACK_ALLOC(JPH_Vec3, v);
    v->x = cmd->vec3f.x;
    v->y = cmd->vec3f.y;
    v->z = cmd->vec3f.z;
    JPH_BodyInterface_SetAngularVelocity(bi, bid, v);
    DISPATCH();
}

op_SET_MOTION: {
    JPH_BodyInterface_SetMotionType(bi, bid, (JPH_MotionType)cmd->motion.motion_type,
                                    JPH_Activation_Activate);
    uint32_t layer = (cmd->motion.motion_type == 0) ? 0 : 1;
    JPH_BodyInterface_SetObjectLayer(bi, bid, (JPH_ObjectLayer)layer);
    DISPATCH();
}

op_ACTIVATE: {
    JPH_BodyInterface_ActivateBody(bi, bid);
    DISPATCH();
}

op_DEACTIVATE: {
    JPH_BodyInterface_DeactivateBody(bi, bid);
    DISPATCH();
}

op_SET_USER_DATA: { DISPATCH(); }

op_SET_CCD: {
    JPH_MotionQuality qual =
        cmd->motion.motion_type ? JPH_MotionQuality_LinearCast : JPH_MotionQuality_Discrete;
    JPH_BodyInterface_SetMotionQuality(bi, bid, qual);
    DISPATCH();
}

op_TELEPORT: { DISPATCH(); }

op_APPLY_IMPULSE: {
    JPH_STACK_ALLOC(JPH_Vec3, v);
    v->x = cmd->vec3f.x;
    v->y = cmd->vec3f.y;
    v->z = cmd->vec3f.z;
    JPH_BodyInterface_AddImpulse(bi, bid, v);
    JPH_BodyInterface_ActivateBody(bi, bid);
    DISPATCH();
}

op_APPLY_FORCE: {
    JPH_STACK_ALLOC(JPH_Vec3, v);
    v->x = cmd->vec3f.x;
    v->y = cmd->vec3f.y;
    v->z = cmd->vec3f.z;
    JPH_BodyInterface_AddForce(bi, bid, v);
    JPH_BodyInterface_ActivateBody(bi, bid);
    DISPATCH();
}

op_APPLY_TORQUE: {
    JPH_STACK_ALLOC(JPH_Vec3, v);
    v->x = cmd->vec3f.x;
    v->y = cmd->vec3f.y;
    v->z = cmd->vec3f.z;
    JPH_BodyInterface_AddTorque(bi, bid, v);
    JPH_BodyInterface_ActivateBody(bi, bid);
    DISPATCH();
}

op_APPLY_ANG_IMPULSE: {
    JPH_STACK_ALLOC(JPH_Vec3, v);
    v->x = cmd->vec3f.x;
    v->y = cmd->vec3f.y;
    v->z = cmd->vec3f.z;
    JPH_BodyInterface_AddAngularImpulse(bi, bid, v);
    JPH_BodyInterface_ActivateBody(bi, bid);
    DISPATCH();
}

op_APPLY_IMPULSE_AT: {
    JPH_STACK_ALLOC(JPH_Vec3, imp);
    imp->x = cmd->impulse_at.ix;
    imp->y = cmd->impulse_at.iy;
    imp->z = cmd->impulse_at.iz;

    JPH_STACK_ALLOC(JPH_RVec3, pos);
    pos->x = cmd->impulse_at.px;
    pos->y = cmd->impulse_at.py;
    pos->z = cmd->impulse_at.pz;

    JPH_BodyInterface_AddImpulse2(bi, bid, imp, pos);
    JPH_BodyInterface_ActivateBody(bi, bid);
    DISPATCH();
}

op_CREATE_CONSTRAINT: {
    uint32_t s1         = slot;
    uint32_t s2         = cmd->constraint.body2_slot;
    uint32_t c_slot     = cmd->constraint.constraint_slot;
    ConstraintParams *p = cmd->constraint.params;

    JPH_BodyID bid1 = world->body_ids[world->slot_to_dense[s1]];
    JPH_BodyID bid2 = world->body_ids[world->slot_to_dense[s2]];

    // CHECK: Did the user call destroy_constraint() while this was in the queue?
    state = atomic_load_explicit(&world->constraint_states[c_slot], memory_order_acquire);

    if (state == SLOT_PENDING_CREATE) {
        // Verify bodies still exist in Jolt
        if (bid1 != JPH_INVALID_BODY_ID && bid2 != JPH_INVALID_BODY_ID) {
            JPH_BodyLockWrite lock1;
            JPH_BodyLockWrite lock2;
            const JPH_BodyLockInterface *li = JPH_PhysicsSystem_GetBodyLockInterface(world->system);

            // Consistent locking order to prevent deadlocks
            if (bid1 < bid2) {
                JPH_BodyLockInterface_LockWrite(li, bid1, &lock1);
                JPH_BodyLockInterface_LockWrite(li, bid2, &lock2);
            } else {
                JPH_BodyLockInterface_LockWrite(li, bid2, &lock2);
                JPH_BodyLockInterface_LockWrite(li, bid1, &lock1);
            }

            JPH_Constraint *constraint = nullptr;
            if (lock1.body && lock2.body) {
                JPH_Body *b1 = (JPH_Body_GetID(lock1.body) == bid1) ? lock1.body : lock2.body;
                JPH_Body *b2 = (JPH_Body_GetID(lock1.body) == bid2) ? lock1.body : lock2.body;

                switch (cmd->constraint.type) {
                case CONSTRAINT_FIXED:
                    constraint = create_fixed(p, b1, b2);
                    break;
                case CONSTRAINT_POINT:
                    constraint = create_point(p, b1, b2);
                    break;
                case CONSTRAINT_HINGE:
                    constraint = create_hinge(p, b1, b2);
                    break;
                case CONSTRAINT_SLIDER:
                    constraint = create_slider(p, b1, b2);
                    break;
                case CONSTRAINT_CONE:
                    constraint = create_cone(p, b1, b2);
                    break;
                case CONSTRAINT_DISTANCE:
                    constraint = create_distance(p, b1, b2);
                    break;
                default:
                    break;
                }
            }

            JPH_BodyLockInterface_UnlockWrite(li, &lock1);
            JPH_BodyLockInterface_UnlockWrite(li, &lock2);

            if (constraint) {
                JPH_PhysicsSystem_AddConstraint(world->system, constraint);
                world->constraints[c_slot] = constraint;
                // Mark as ALIVE so future SET_TARGET commands can find it
                atomic_store_explicit(&world->constraint_states[c_slot], SLOT_ALIVE,
                                      memory_order_release);
            } else {
                // Failure: Return slot to pool
                atomic_store_explicit(&world->constraint_states[c_slot], SLOT_EMPTY,
                                      memory_order_relaxed);
                world->free_constraint_slots[world->free_constraint_count++] = c_slot;
            }
        }
    }
    CulvNative_Free(p); // Important: Free the heap allocation
    DISPATCH();
}
}

void clear_command_queue(PhysicsWorld *world) {
    if (!world->command_queue) {
        return;
    }

    for (size_t i = 0; i < world->command_count; i++) {
        PhysicsCommand *cmd = &world->command_queue[i];
        if (CMD_GET_TYPE(cmd->header) == CMD_CREATE_BODY) {
            if (cmd->create.settings) {
                JPH_BodyCreationSettings_Destroy(cmd->create.settings);
            }
        } else if (CMD_GET_TYPE(cmd->header) == CMD_CREATE_SOFT_BODY) {
            if (cmd->create_soft.settings) {
                JPH_SoftBodyCreationSettings_Destroy(cmd->create_soft.settings);
            }
        }
    }
    world->command_count = 0;
}