#include "culverin_command_buffer.h"
#include "culverin_compiler_specifics.h"
#include "culverin_physics_sync.h"
#include "culverin_physics_world_internal.h"
#include "culverin_threading.h"

static constexpr size_t AVX_ALIGNMENT = 32;

void world_remove_body_slot(PhysicsWorldObject *self, uint32_t slot) {
    const uint32_t dense_idx = self->slot_to_dense[slot];

    // TSan Fix: Load count atomically to determine the last index
    const uint32_t last_dense =
        (uint32_t)atomic_load_explicit(&self->count, memory_order_acquire) - 1;

    if (self->soft_shadows && self->soft_shadows[dense_idx].vertices) {
        CulvMem_RawFreeAligned(self->soft_shadows[dense_idx].vertices);
        self->soft_shadows[dense_idx].vertices = nullptr;

        if (self->soft_shadows[dense_idx].velocities) {
            CulvMem_RawFreeAligned(self->soft_shadows[dense_idx].velocities);
            self->soft_shadows[dense_idx].velocities = nullptr;
        }
        if (self->soft_shadows[dense_idx].normals) {
            CulvMem_RawFreeAligned(self->soft_shadows[dense_idx].normals);
            self->soft_shadows[dense_idx].normals = nullptr;
        }
        self->soft_shadows[dense_idx].num_vertices = 0;
    }

    // THE SWAP-TO-DELETE (Dense Pack)
    if (dense_idx != last_dense) {
        CULV_PREFETCH_READ(&self->positions[last_dense]);
        CULV_PREFETCH_WRITE(&self->positions[dense_idx]);

        // --- GROUP 1: PHYSICS STATE (Non-atomic) ---
        ((PosStride *)self->positions)[dense_idx] = ((PosStride *)self->positions)[last_dense];
        ((PosStride *)self->prev_positions)[dense_idx] =
            ((PosStride *)self->prev_positions)[last_dense];

        ((AuxStride *)self->rotations)[dense_idx] = ((AuxStride *)self->rotations)[last_dense];
        ((AuxStride *)self->prev_rotations)[dense_idx] =
            ((AuxStride *)self->prev_rotations)[last_dense];

        ((AuxStride *)self->linear_velocities)[dense_idx] =
            ((AuxStride *)self->linear_velocities)[last_dense];
        ((AuxStride *)self->angular_velocities)[dense_idx] =
            ((AuxStride *)self->angular_velocities)[last_dense];

        // --- GROUP 2: METADATA (Non-atomic) ---
        self->body_ids[dense_idx]     = self->body_ids[last_dense];
        self->user_data[dense_idx]    = self->user_data[last_dense];
        self->categories[dense_idx]   = self->categories[last_dense];
        self->masks[dense_idx]        = self->masks[last_dense];
        self->material_ids[dense_idx] = self->material_ids[last_dense];

        // --- GROUP 3: THE MAP REWIRE ---
        const uint32_t mover_slot       = self->dense_to_slot[last_dense];
        self->slot_to_dense[mover_slot] = dense_idx;
        self->dense_to_slot[dense_idx]  = mover_slot;

        if (self->soft_shadows) {
            self->soft_shadows[dense_idx] = self->soft_shadows[last_dense];
            // Clear the old tail so we don't double-free later
            self->soft_shadows[last_dense].vertices     = nullptr;
            self->soft_shadows[last_dense].velocities   = nullptr;
            self->soft_shadows[last_dense].normals      = nullptr;
            self->soft_shadows[last_dense].num_vertices = 0;
        }
    }

    // HOUSEKEEPING

    // 1. Invalidate all existing Python handles by incrementing generation
    atomic_fetch_add_explicit(&self->generations[slot], 1, memory_order_relaxed);

    // 2. Mark the slot as empty atomically
    atomic_store_explicit(&self->slot_states[slot], SLOT_EMPTY, memory_order_relaxed);

    // 3. Push to free stack atomically
    // We fetch the current count, use it as index, and increment
    size_t f_idx            = atomic_fetch_add_explicit(&self->free_count, 1, memory_order_relaxed);
    self->free_slots[f_idx] = slot;

    // 4. Update the total world count atomically
    // We use memory_order_release to ensure all memory moves above are visible
    // to any thread reading the count (like the renderer).
    atomic_fetch_sub_explicit(&self->count, 1, memory_order_release);
}

// Internal helper to keep queues in sync
static bool grow_queues(PhysicsWorldObject *self, size_t new_cap) {
    if (new_cap > (SIZE_MAX / sizeof(PhysicsCommand))) {
        return false;
    }

    // Grow the ACTIVE queue
    void *new_active = CULV_RAW_REALLOC(self->command_queue, new_cap * sizeof(PhysicsCommand));
    if (!new_active) {
        return false;
    }
    self->command_queue    = (PhysicsCommand *)new_active;
    self->command_capacity = new_cap;

    // Grow the SPARE queue to match immediately
    void *new_spare = CULV_RAW_REALLOC(self->command_queue_spare, new_cap * sizeof(PhysicsCommand));
    if (!new_spare) {
        // This is a rare partial-failure state. We can't easily roll back active,
        // but we can mark spare_capacity as smaller so step() knows it's not mirrored.
        // However, for high-perf, we assume if realloc 1 worked, 2 likely will.
        return false;
    }
    self->command_queue_spare = (PhysicsCommand *)new_spare;
    self->spare_capacity      = new_cap;

    return true;
}

CULV_NODISCARD
bool ensure_command_capacity(PhysicsWorldObject *self) {
    if (UNLIKELY(self->command_count >= self->command_capacity)) {
        size_t new_cap = (self->command_capacity == 0) ? 64 : self->command_capacity * 2;
        return grow_queues(self, new_cap);
    }
    return true;
}

CULV_NODISCARD
bool ensure_command_bulk_capacity(PhysicsWorldObject *self, size_t batch_size) {
    size_t required = self->command_count + batch_size;
    if (UNLIKELY(required > self->command_capacity)) {
        size_t new_cap = (self->command_capacity == 0) ? 64 : self->command_capacity * 2;
        while (new_cap < required) {
            new_cap *= 2;
        }
        return grow_queues(self, new_cap);
    }
    return true;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void flush_commands_internal(PhysicsWorldObject *self, PhysicsCommand *CULV_RESTRICT queue,
                             size_t count) {
    if (UNLIKELY(count == 0)) {
        return;
    }

    queue                               = CULV_ASSUME_ALIGNED(queue, 64);
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
                                                 [CMD_APPLY_IMPULSE_AT]  = &&op_APPLY_IMPULSE_AT,
                                                 [CMD_CREATE_SOFT_BODY]  = &&op_CREATE_SOFT_BODY};

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

    SHADOW_LOCK(&self->shadow_lock);

    if (UNLIKELY(new_bid == JPH_INVALID_BODY_ID)) {
        // world_remove_body_slot handles atomic count/state updates internally
        world_remove_body_slot(self, slot);
    } else {
        // body_ids is non-atomic; protected by shadow_lock and the 'is_stepping' phase
        self->body_ids[self->slot_to_dense[slot]] = new_bid;

        const uint32_t j_idx = JPH_ID_TO_INDEX(new_bid);
        if (self->id_to_handle_map && j_idx <= self->max_jolt_bodies) {
            // TSan Fix: Load generation atomically
            const uint32_t gen =
                atomic_load_explicit(&self->generations[slot], memory_order_relaxed);

            // BodyHandle is CULV_ATOMIC(uint64_t)
            const BodyHandle h = make_handle(slot, gen);

            // TSan Fix: Extract raw uint64_t to avoid implicit seq_cst load overhead
            const uint64_t raw_h = h;

            // TSan Fix: Publish the new handle to the shared map atomically.
            // Release ensures the body_ids update above is visible to Query threads.
            atomic_store_explicit(&self->id_to_handle_map[j_idx], raw_h, memory_order_release);
        }

        // TSan Fix: Mark the body as ALIVE atomically.
        // Release creates a barrier: any thread that sees SLOT_ALIVE is guaranteed
        // to see the correctly initialized body_ids and id_to_handle_map data.
        atomic_store_explicit(&self->slot_states[slot], SLOT_ALIVE, memory_order_release);
    }

    SHADOW_UNLOCK(&self->shadow_lock);

    DISPATCH();
}

op_CREATE_SOFT_BODY: {
    JPH_SoftBodyCreationSettings *const settings = cmd->create_soft.settings;
    const uint32_t num_verts = cmd->create_soft.num_vertices; // O(1) Cache-local read!

    PyObject *const py_shared = cmd->create_soft.user_data.obj;

    const JPH_BodyID new_bid =
        JPH_BodyInterface_CreateAndAddSoftBody(bi, settings, JPH_Activation_Activate);

    SHADOW_LOCK(&self->shadow_lock);

    if (UNLIKELY(new_bid == JPH_INVALID_BODY_ID)) {
        world_remove_body_slot(self, slot);
    } else {
        const uint32_t dense_idx  = self->slot_to_dense[slot];
        self->body_ids[dense_idx] = new_bid;

        // --- ALLOCATE SHADOW VERTEX BUFFER ---
        self->soft_shadows[dense_idx].num_vertices = num_verts;
        self->soft_shadows[dense_idx].vertices =
            (JPH_Real *)CulvMem_RawMallocAligned(num_verts * sizeof(PosStride), AVX_ALIGNMENT);
        self->soft_shadows[dense_idx].velocities = nullptr; // Initialize explicitly
        self->soft_shadows[dense_idx].normals    = nullptr; // Initialize explicitly

        // Populate standard handles
        const uint32_t j_idx = JPH_ID_TO_INDEX(new_bid);
        if (self->id_to_handle_map && j_idx <= self->max_jolt_bodies) {
            const uint32_t gen =
                atomic_load_explicit(&self->generations[slot], memory_order_relaxed);
            const BodyHandle h   = make_handle(slot, gen);
            const uint64_t raw_h = h;
            atomic_store_explicit(&self->id_to_handle_map[j_idx], raw_h, memory_order_release);
        }

        atomic_store_explicit(&self->slot_states[slot], SLOT_SOFT_BODY, memory_order_release);
    }

    JPH_SoftBodyCreationSettings_Destroy(settings); // Cleanup

    // --- RELEASE PROTECTION ---
    // The creation is finished; Python can now safely delete the shared settings if it wants.
    Py_DECREF(py_shared);

    SHADOW_UNLOCK(&self->shadow_lock);
    DISPATCH();
}

op_DESTROY_BODY: {
    JPH_BodyInterface_RemoveBody(bi, bid);
    JPH_BodyInterface_DestroyBody(bi, bid);
    SHADOW_LOCK(&self->shadow_lock);
    world_remove_body_slot(self, slot);
    SHADOW_UNLOCK(&self->shadow_lock);
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
        void *new_spare = CULV_RAW_REALLOC(self->command_queue_spare,
                                           self->command_capacity * sizeof(PhysicsCommand));
        if (new_spare) {
            self->command_queue_spare = (PhysicsCommand *)new_spare;
            self->spare_capacity      = self->command_capacity;
        } else {
            size_t temp            = self->command_capacity;
            self->command_capacity = self->spare_capacity;
            self->spare_capacity   = temp;
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
        } else if (CMD_GET_TYPE(cmd->header) == CMD_CREATE_SOFT_BODY) {
            if (cmd->create_soft.settings) {
                JPH_SoftBodyCreationSettings_Destroy(cmd->create_soft.settings);
            }
        }
    }
    self->command_count = 0;
}