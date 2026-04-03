#include "culverin_command_buffer.h"
#include "culverin_compiler_specifics.h"

void world_remove_body_slot(PhysicsWorldObject *self, uint32_t slot) {
    uint32_t dense_idx = self->slot_to_dense[slot];
    auto last_dense    = (uint32_t)self->count - 1;
    CULV_MAYBE_UNUSED JPH_BodyID bid     = self->body_ids[dense_idx];

    // test_contact_removal_lifecycle will fail if this snippet gets uncommented
    // if (bid != JPH_INVALID_BODY_ID) {
    //     uint32_t j_idx = JPH_ID_TO_INDEX(bid);
    //     if (self->id_to_handle_map && j_idx < self->max_jolt_bodies) {
    //         self->id_to_handle_map[j_idx] = 0;
    //     }
    // }

    if (dense_idx != last_dense) {
        auto *pos      = (PosStride *)self->positions;
        auto *prev_pos = (PosStride *)self->prev_positions;
        auto *rot      = (AuxStride *)self->rotations;
        auto *prev_rot = (AuxStride *)self->prev_rotations;
        auto *lvel     = (AuxStride *)self->linear_velocities;
        auto *avel     = (AuxStride *)self->angular_velocities;

        pos[dense_idx]      = pos[last_dense];
        prev_pos[dense_idx] = prev_pos[last_dense];

        rot[dense_idx]      = rot[last_dense];
        prev_rot[dense_idx] = prev_rot[last_dense];
        lvel[dense_idx]     = lvel[last_dense];
        avel[dense_idx]     = avel[last_dense];

        self->body_ids[dense_idx]     = self->body_ids[last_dense];
        self->user_data[dense_idx]    = self->user_data[last_dense];
        self->categories[dense_idx]   = self->categories[last_dense];
        self->masks[dense_idx]        = self->masks[last_dense];
        self->material_ids[dense_idx] = self->material_ids[last_dense];

        uint32_t mover_slot             = self->dense_to_slot[last_dense];
        self->slot_to_dense[mover_slot] = dense_idx;
        self->dense_to_slot[dense_idx]  = mover_slot;
    }

    self->generations[slot]++;
    self->free_slots[self->free_count++] = slot;
    self->slot_states[slot]              = SLOT_EMPTY;
    self->count--;
    self->view_shape[0] = (Py_ssize_t)self->count;
}

CULV_NODISCARD
bool ensure_command_capacity(PhysicsWorldObject *self) {
    if (UNLIKELY(self->command_count >= self->command_capacity)) {
        size_t new_cap = (self->command_capacity == 0) ? 64 : self->command_capacity * 2;
        if (UNLIKELY(new_cap > (SIZE_MAX / sizeof(PhysicsCommand)))) {
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

    queue = CULV_ASSUME_ALIGNED(queue, 64);
    JPH_BodyInterface *CULV_RESTRICT bi = self->body_interface;

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
        [CMD_APPLY_IMPULSE_AT]  = &&op_APPLY_IMPULSE_AT
    };

    size_t i = 0;
    PhysicsCommand *cmd;
    uint32_t header;
    uint32_t slot;
    CULV_MAYBE_UNUSED uint32_t dense;
    CommandType type;
    SlotState state;
    JPH_BodyID bid;

op_NEXT:
    DISPATCH();

op_CREATE_BODY: {
    JPH_BodyCreationSettings *s = cmd->create.settings;
    JPH_BodyID new_bid = JPH_BodyInterface_CreateAndAddBody(bi, s, JPH_Activation_Activate);
    
    // Future Optimization: Replace with an arena allocator to avoid cross-thread malloc locks
    JPH_BodyCreationSettings_Destroy(s);

    SHADOW_LOCK(&self->shadow_lock);

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

    SHADOW_UNLOCK(&self->shadow_lock);
    
    DISPATCH();
}

op_DESTROY_BODY: {
    JPH_BodyInterface_RemoveBody(bi, bid);
    JPH_BodyInterface_DestroyBody(bi, bid);
    world_remove_body_slot(self, slot);
    DISPATCH();
}

op_SET_POS: {
    JPH_STACK_ALLOC(JPH_RVec3, p);
    p->x = cmd->pos.x;
    p->y = cmd->pos.y;
    p->z = cmd->pos.z;
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

op_SET_USER_DATA: { 
    DISPATCH(); 
}

op_SET_CCD: {
    JPH_MotionQuality qual =
        cmd->motion.motion_type ? JPH_MotionQuality_LinearCast : JPH_MotionQuality_Discrete;
    JPH_BodyInterface_SetMotionQuality(bi, bid, qual);
    DISPATCH();
}

op_TELEPORT: { 
    DISPATCH(); 
}

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
}

void sync_and_flush_internal(PhysicsWorldObject *self) {
    BLOCK_UNTIL_NOT_STEPPING(self);
    BLOCK_UNTIL_NOT_QUERYING(self);

    if (self->command_count == 0) {
        return;
    }

    atomic_store_explicit(&self->is_stepping, true, memory_order_relaxed);

    PhysicsCommand *captured_queue = self->command_queue;
    size_t captured_count          = self->command_count;

    if (UNLIKELY(self->command_capacity > self->spare_capacity)) {
        void *new_spare = CULV_RAW_REALLOC(
            self->command_queue_spare, self->command_capacity * sizeof(PhysicsCommand));
        if (new_spare) {
            self->command_queue_spare = (PhysicsCommand *)new_spare;
            self->spare_capacity = self->command_capacity;
        } else {
            size_t temp = self->command_capacity;
            self->command_capacity = self->spare_capacity;
            self->spare_capacity = temp;
        }
    }
    self->command_queue       = self->command_queue_spare;
    self->command_queue_spare = captured_queue;
    self->command_count       = 0;

    SHADOW_UNLOCK(&self->shadow_lock);

    Py_BEGIN_ALLOW_THREADS NATIVE_MUTEX_LOCK(g_jph_trampoline_lock);

    flush_commands_internal(self, captured_queue, captured_count);

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
            if (cmd->create.settings) {
                JPH_BodyCreationSettings_Destroy(cmd->create.settings);
            }
        }
    }
    self->command_count = 0;
}