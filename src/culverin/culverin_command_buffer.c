#include "culverin_command_buffer.h"
#include "culverin_compiler_specifics.h"

/**
 * Internal helper to remove a body from the dense arrays.
 * Maintains a packed, contiguous array by swapping the last body into the hole.
 * MUST be called while holding SHADOW_LOCK.
 */
void world_remove_body_slot(PhysicsWorldObject *self, uint32_t slot) {
    uint32_t dense_idx = self->slot_to_dense[slot];
    auto last_dense    = (uint32_t)self->count - 1;
    JPH_BodyID bid     = self->body_ids[dense_idx];

    // 1. Cleanup Jolt Mapping
    if (bid != JPH_INVALID_BODY_ID) {
        uint32_t j_idx = JPH_ID_TO_INDEX(bid);
        if (self->id_to_handle_map && j_idx < self->max_jolt_bodies) {
            self->id_to_handle_map[j_idx] = 0;
        }
    }

    // 2. Swap-and-Pop
    if (dense_idx != last_dense) {
        // Type-safe Casts
        auto *pos      = (PosStride *)self->positions;
        auto *prev_pos = (PosStride *)self->prev_positions;

        auto *rot      = (AuxStride *)self->rotations;
        auto *prev_rot = (AuxStride *)self->prev_rotations;
        auto *lvel     = (AuxStride *)self->linear_velocities;
        auto *avel     = (AuxStride *)self->angular_velocities;

        // Struct Copy (Compiler handles size/alignment)
        pos[dense_idx]      = pos[last_dense];
        prev_pos[dense_idx] = prev_pos[last_dense];

        rot[dense_idx]      = rot[last_dense];
        prev_rot[dense_idx] = prev_rot[last_dense];
        lvel[dense_idx]     = lvel[last_dense];
        avel[dense_idx]     = avel[last_dense];

        // Metadata Copy
        self->body_ids[dense_idx]     = self->body_ids[last_dense];
        self->user_data[dense_idx]    = self->user_data[last_dense];
        self->categories[dense_idx]   = self->categories[last_dense];
        self->masks[dense_idx]        = self->masks[last_dense];
        self->material_ids[dense_idx] = self->material_ids[last_dense];

        // Fix Indirection
        uint32_t mover_slot             = self->dense_to_slot[last_dense];
        self->slot_to_dense[mover_slot] = dense_idx;
        self->dense_to_slot[dense_idx]  = mover_slot;
    }

    // 3. Finalize
    self->generations[slot]++;
    self->free_slots[self->free_count++] = slot;
    self->slot_states[slot]              = SLOT_EMPTY;
    self->count--;
    self->view_shape[0] = (Py_ssize_t)self->count;
}

// Helper to grow queue
CULV_NODISCARD
bool ensure_command_capacity(PhysicsWorldObject *self) {
    if (self->command_count >= self->command_capacity) {
        // Defensive: handle zero or uninitialized capacity
        size_t new_cap = (self->command_capacity == 0) ? 64 : self->command_capacity * 2;

        // Safety check: Prevent overflow on extreme counts
        if (new_cap > (SIZE_MAX / sizeof(PhysicsCommand))) {
            return false;
        }

        void *new_ptr = CULV_RAW_REALLOC(self->command_queue, new_cap * sizeof(PhysicsCommand));
        if (!new_ptr) {
            return false;
        }

        self->command_queue    = (PhysicsCommand *)new_ptr;
        self->command_capacity = new_cap;
    }
    return true;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void flush_commands_internal(PhysicsWorldObject *self, PhysicsCommand *CULV_RESTRICT queue,
                             size_t count) {
    if (UNLIKELY(count == 0)) {
        return;
    }

    // Tell Clang vectorizer that memory is aligned and safe
    queue                               = CULV_ASSUME_ALIGNED(queue, 8);
    JPH_BodyInterface *CULV_RESTRICT bi = self->body_interface;

    static const void *const dispatch_table[] = {[CMD_CREATE_BODY]       = &&op_CREATE_BODY,
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
                                                 [CMD_APPLY_IMPULSE_AT]  = &&op_APPLY_IMPULSE_AT};

    size_t i = 0;
    PhysicsCommand *cmd;
    uint32_t header;
    uint32_t slot;
    CULV_MAYBE_UNUSED uint32_t dense;
    CommandType type;
    SlotState state;
    JPH_BodyID bid;

// The entire "VM" fetch-decode loop in one perfectly inlined macro
#define NEXT_CMD()                                                                                 \
    while (i < count) {                                                                            \
        cmd    = &queue[i++];                                                                      \
        header = cmd->header;                                                                      \
        type   = CMD_GET_TYPE(header);                                                             \
        slot   = CMD_GET_SLOT(header);                                                             \
        state  = self->slot_states[slot];                                                          \
        bid    = JPH_INVALID_BODY_ID;                                                              \
        if (LIKELY(state == SLOT_ALIVE || state == SLOT_PENDING_CREATE)) {                         \
            bid = self->body_ids[self->slot_to_dense[slot]];                                       \
        }                                                                                          \
        if (LIKELY(type == CMD_CREATE_BODY || bid != JPH_INVALID_BODY_ID)) {                       \
            goto *dispatch_table[type];                                                            \
        }                                                                                          \
    }                                                                                              \
    return;

    // Kick off execution
    NEXT_CMD()

op_CREATE_BODY: {
    JPH_BodyCreationSettings *s = cmd->create.settings;
    JPH_BodyID new_bid = JPH_BodyInterface_CreateAndAddBody(bi, s, JPH_Activation_Activate);
    JPH_BodyCreationSettings_Destroy(s);

    if (UNLIKELY(new_bid == JPH_INVALID_BODY_ID)) {
        world_remove_body_slot(self, slot);
    } else {
        self->body_ids[self->slot_to_dense[slot]] = new_bid;
        uint32_t j_idx                            = JPH_ID_TO_INDEX(new_bid);
        if (self->id_to_handle_map && j_idx < self->max_jolt_bodies) {
            self->id_to_handle_map[j_idx] = make_handle(slot, self->generations[slot]);
        }
        self->slot_states[slot] = SLOT_ALIVE;
    }
    NEXT_CMD()
}

op_DESTROY_BODY: {
    JPH_BodyInterface_RemoveBody(bi, bid);
    JPH_BodyInterface_DestroyBody(bi, bid);
    world_remove_body_slot(self, slot);
    NEXT_CMD()
}

op_SET_POS: {
    JPH_STACK_ALLOC(JPH_RVec3, p);
    p->x        = cmd->pos.x;
    p->y        = cmd->pos.y;
    p->z        = cmd->pos.z;
    bool active = JPH_BodyInterface_IsActive(bi, bid);
    JPH_BodyInterface_SetPosition(
        bi, bid, p, (int)active ? JPH_Activation_DontActivate : JPH_Activation_Activate);
    NEXT_CMD()
}

op_SET_ROT: {
    JPH_STACK_ALLOC(JPH_Quat, q);
    q->x = cmd->quat.x;
    q->y = cmd->quat.y;
    q->z = cmd->quat.z;
    q->w = cmd->quat.w;
    JPH_BodyInterface_SetRotation(bi, bid, q, JPH_Activation_Activate);
    NEXT_CMD()
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
    NEXT_CMD()
}

op_SET_LINVEL: {
    JPH_Vec3 v = {cmd->vec3f.x, cmd->vec3f.y, cmd->vec3f.z};
    JPH_BodyInterface_SetLinearVelocity(bi, bid, &v);
    NEXT_CMD()
}

op_SET_ANGVEL: {
    JPH_Vec3 v = {cmd->vec3f.x, cmd->vec3f.y, cmd->vec3f.z};
    JPH_BodyInterface_SetAngularVelocity(bi, bid, &v);
    NEXT_CMD()
}

op_SET_MOTION: {
    JPH_BodyInterface_SetMotionType(bi, bid, (JPH_MotionType)cmd->motion.motion_type,
                                    JPH_Activation_Activate);
    uint32_t layer = (cmd->motion.motion_type == 0) ? 0 : 1;
    JPH_BodyInterface_SetObjectLayer(bi, bid, (JPH_ObjectLayer)layer);
    NEXT_CMD()
}

op_ACTIVATE: {
    JPH_BodyInterface_ActivateBody(bi, bid);
    NEXT_CMD()
}

op_DEACTIVATE: {
    JPH_BodyInterface_DeactivateBody(bi, bid);
    NEXT_CMD()
}

op_SET_USER_DATA: { NEXT_CMD() }

op_SET_CCD: {
    JPH_MotionQuality qual =
        cmd->motion.motion_type ? JPH_MotionQuality_LinearCast : JPH_MotionQuality_Discrete;
    JPH_BodyInterface_SetMotionQuality(bi, bid, qual);
    NEXT_CMD()
}

op_TELEPORT: { NEXT_CMD() }

op_APPLY_IMPULSE: {
    JPH_Vec3 v = {cmd->vec3f.x, cmd->vec3f.y, cmd->vec3f.z};
    JPH_BodyInterface_AddImpulse(bi, bid, &v);
    JPH_BodyInterface_ActivateBody(bi, bid);
    NEXT_CMD()
}

op_APPLY_FORCE: {
    JPH_Vec3 v = {cmd->vec3f.x, cmd->vec3f.y, cmd->vec3f.z};
    JPH_BodyInterface_AddForce(bi, bid, &v);
    JPH_BodyInterface_ActivateBody(bi, bid);
    NEXT_CMD()
}

op_APPLY_TORQUE: {
    JPH_Vec3 v = {cmd->vec3f.x, cmd->vec3f.y, cmd->vec3f.z};
    JPH_BodyInterface_AddTorque(bi, bid, &v);
    JPH_BodyInterface_ActivateBody(bi, bid);
    NEXT_CMD()
}

op_APPLY_ANG_IMPULSE: {
    JPH_Vec3 v = {cmd->vec3f.x, cmd->vec3f.y, cmd->vec3f.z};
    JPH_BodyInterface_AddAngularImpulse(bi, bid, &v);
    JPH_BodyInterface_ActivateBody(bi, bid);
    NEXT_CMD()
}

op_APPLY_IMPULSE_AT: {
    JPH_Vec3 imp  = {cmd->impulse_at.ix, cmd->impulse_at.iy, cmd->impulse_at.iz};
    JPH_RVec3 pos = {cmd->impulse_at.px, cmd->impulse_at.py, cmd->impulse_at.pz};
    JPH_BodyInterface_AddImpulse2(bi, bid, &imp, &pos);
    JPH_BodyInterface_ActivateBody(bi, bid);
    NEXT_CMD()
}
}

/**
 * Helper: Flushes pending commands while releasing shadow_lock to
 * avoid stalling the world during heavy Jolt operations.
 */
void sync_and_flush_internal(PhysicsWorldObject *self) {
    BLOCK_UNTIL_NOT_STEPPING(self);
    BLOCK_UNTIL_NOT_QUERYING(self);

    if (self->command_count == 0) {
        return;
    }

    atomic_store_explicit(&self->is_stepping, true, memory_order_relaxed);

    // --- Double Buffer Swap (Zero Allocations) ---
    PhysicsCommand *captured_queue = self->command_queue;
    size_t captured_count          = self->command_count;

    if (UNLIKELY(self->command_capacity > self->spare_capacity)) {
        self->command_queue_spare = (PhysicsCommand *)CULV_RAW_REALLOC(
            self->command_queue_spare, self->command_capacity * sizeof(PhysicsCommand));
        self->spare_capacity = self->command_capacity;
    }
    self->command_queue       = self->command_queue_spare;
    self->command_queue_spare = captured_queue;
    self->command_count       = 0;
    // -------------------------------------------------------

    SHADOW_UNLOCK(&self->shadow_lock);

    Py_BEGIN_ALLOW_THREADS NATIVE_MUTEX_LOCK(g_jph_trampoline_lock);

    flush_commands_internal(self, captured_queue, captured_count);

    // NO CULV_RAW_FREE HERE! We keep it allocated in 'spare' for the next queue loop.

    NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);
    Py_END_ALLOW_THREADS

        SHADOW_LOCK(&self->shadow_lock);

    atomic_store_explicit(&self->is_stepping, false, memory_order_release);
    NATIVE_MUTEX_LOCK(self->step_sync.mutex);
    NATIVE_COND_BROADCAST(self->step_sync.cond);
    NATIVE_MUTEX_UNLOCK(self->step_sync.mutex);
}

void clear_command_queue(PhysicsWorldObject *self) {
    if (!self->command_queue) {
        return;
    }

    for (size_t i = 0; i < self->command_count; i++) {
        PhysicsCommand *cmd = &self->command_queue[i];
        if (CMD_GET_TYPE(cmd->header) == CMD_CREATE_BODY) {
            // We own this pointer until it's consumed by Jolt
            if (cmd->create.settings) {
                JPH_BodyCreationSettings_Destroy(cmd->create.settings);
            }
        }
    }
    self->command_count = 0;
}
