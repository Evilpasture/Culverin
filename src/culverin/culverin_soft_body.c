#include "culverin_soft_body.h"
#include "culverin_physics_sync.h"
#include "culverin_physics_world_internal.h"

static constexpr uint32_t COLLISION_FILTER_ALL_CATEGORIES = 0xFFFF;
static constexpr uint32_t COLLISION_FILTER_ALL_MASKS      = 0xFFFF;

static uint64_t physics_world_commit_create_soft_locked(PhysicsWorldObject *self,
                                                        JPH_SoftBodyCreationSettings *settings,
                                                        uint32_t slot_state) {
    size_t current_count = atomic_load_explicit(&self->count, memory_order_acquire);
    size_t available     = atomic_load_explicit(&self->free_count, memory_order_acquire);

    if (UNLIKELY(current_count >= self->max_jolt_bodies)) {
        PyErr_Format(PyExc_RuntimeError, "PhysicsWorld limit reached: %u bodies",
                     self->max_jolt_bodies);
        return 0;
    }

    constexpr auto INITIAL_BODY_CAPACITY = 1024;

    if (UNLIKELY(available == 0 || current_count + 1 > self->capacity)) {
        size_t next_cap = (self->capacity == 0) ? INITIAL_BODY_CAPACITY : self->capacity * 2;
        if (next_cap > self->max_jolt_bodies) {
            next_cap = self->max_jolt_bodies;
        }
        if (PhysicsWorld_resize(self, next_cap) < 0) {
            return 0;
        }
        available = atomic_load_explicit(&self->free_count, memory_order_acquire);
    }

    if (UNLIKELY(!ensure_command_capacity(self))) {
        return 0;
    }

    uint32_t slot  = self->free_slots[--available];
    uint32_t dense = (uint32_t)atomic_fetch_add_explicit(&self->count, 1, memory_order_relaxed);
    atomic_store_explicit(&self->free_count, available, memory_order_release);

    uint32_t gen      = atomic_load_explicit(&self->generations[slot], memory_order_relaxed);
    BodyHandle handle = make_handle(slot, gen);
    uint64_t raw_h    = atomic_load_explicit(&handle, memory_order_relaxed);

    // CRITICAL FIX: Use the SoftBody specific setter
    JPH_SoftBodyCreationSettings_SetUserData(settings, raw_h);

    self->slot_to_dense[slot]  = dense;
    self->dense_to_slot[dense] = slot;
    self->body_ids[dense]      = JPH_INVALID_BODY_ID;
    atomic_store_explicit(&self->slot_states[slot], slot_state, memory_order_release);

    return raw_h;
}

PyType_DeclareSlot_StatusFromModule SoftBodySharedSettings_init(SoftBodySharedSettingsObject *self,
                                                                CULV_MAYBE_UNUSED PyObject *args,
                                                                CULV_MAYBE_UNUSED PyObject *kwds) {
    // 1. Create the native Jolt object
    self->settings     = JPH_SoftBodySharedSettings_Create();
    self->num_vertices = 0;

    if (!self->settings) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create Jolt SoftBodySharedSettings");
        return -1;
    }

    return 0;
}

PyType_DeclareSlot_VoidFromModule
SoftBodySharedSettings_dealloc(SoftBodySharedSettingsObject *self) {
    if (self->settings) {
        JPH_SoftBodySharedSettings_Destroy(self->settings);
        self->settings = nullptr;
    }

    PyTypeObject *tp = Py_TYPE(self);
    tp->tp_free((PyObject *)self);
    Py_DECREF(tp);
}

PyCFunction_DeclareMethodFromModule
SoftBodySharedSettings_add_vertex(SoftBodySharedSettingsObject *self, PyObject *const *args,
                                  Py_ssize_t nargs, PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));

    PyObject *o_pos = nullptr;
    float inv_mass  = 1.0f;

    void *targets[SbssAddVertex_COUNT] = {[IDX_SAV_POS]  = (void *)&o_pos,
                                          [IDX_SAV_MASS] = (void *)&inv_mass};

    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.SbssAddVertexParser, targets)) {
        return nullptr;
    }

    // Extraction from the PyObject* captured by FastParse
    JPH_Vec3 pos;
    if (!parse_vec3_direct(o_pos, &pos.x, &pos.y, &pos.z)) {
        return nullptr;
    }

    JPH_SoftBodySharedSettings_AddVertex(self->settings, &pos, inv_mass);
    self->num_vertices++;

    Py_RETURN_NONE;
}

PyCFunction_DeclareMethodFromModule
SoftBodySharedSettings_add_face(SoftBodySharedSettingsObject *self, PyObject *const *args,
                                Py_ssize_t nargs, PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));

    uint32_t v1;
    uint32_t v2;
    uint32_t v3;
    void *targets[SbssAddFace_COUNT] = {
        [IDX_SAF_V1] = (void *)&v1, [IDX_SAF_V2] = (void *)&v2, [IDX_SAF_V3] = (void *)&v3};

    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.SbssAddFaceParser, targets)) {
        return nullptr;
    }

    JPH_SoftBodySharedSettings_AddFace(self->settings, v1, v2, v3);
    Py_RETURN_NONE;
}

// Method: optimize()
// Crucial: This calculates the edge constraints and bending constraints.
PyCFunction_DeclareMethodFromModule
SoftBodySharedSettings_optimize(SoftBodySharedSettingsObject *self, PyObject *Py_UNUSED(args)) {
    JPH_SoftBodySharedSettings_Optimize(self->settings);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethodFromModule PhysicsWorld_create_soft_body(PhysicsWorldObject *self,
                                                                  PyObject *const *args,
                                                                  size_t nargsf,
                                                                  PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));

    // 1. FAST PARSE
    PyObject *o_shared = nullptr;
    PyObject *o_pos    = nullptr;
    PyObject *o_rot    = nullptr;
    uint64_t user_data = 0;
    uint32_t category  = COLLISION_FILTER_ALL_CATEGORIES;
    uint32_t mask      = COLLISION_FILTER_ALL_MASKS;

    void *targets[CreateSoftBody_COUNT] = {
        [IDX_CSB_SHARED] = (void *)&o_shared, [IDX_CSB_POS] = (void *)&o_pos,
        [IDX_CSB_ROT] = (void *)&o_rot,       [IDX_CSB_USER_DATA] = (void *)&user_data,
        [IDX_CSB_CAT] = (void *)&category,    [IDX_CSB_MASK] = (void *)&mask};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &st->parsers.CreateSoftBodyParser, targets)) {
        return nullptr;
    }

    // 2. VECTOR EXTRACTION
    JPH_Real px;
    JPH_Real py;
    JPH_Real pz;
    float rx;
    float ry;
    float rz;
    float rw;
    if (!parse_vec3_direct(o_pos, &px, &py, &pz) || !parse_quat_direct(o_rot, &rx, &ry, &rz, &rw)) {
        return nullptr;
    }

    VALIDATE_FINITE_VEC3(px, py, pz, "Position");
    VALIDATE_FINITE_QUAT(rx, ry, rz, rw, "Rotation");

    // 3. JOLT PREP
    JPH_SoftBodyCreationSettings *settings = JPH_SoftBodyCreationSettings_Create();
    auto *py_shared = (SoftBodySharedSettingsObject *)o_shared;
    Py_INCREF(o_shared); 

    constexpr auto VERTEX_RADIUS = 0.05f;

    JPH_SoftBodyCreationSettings_SetSharedSettings(settings, py_shared->settings);
    JPH_SoftBodyCreationSettings_SetVertexRadius(settings, VERTEX_RADIUS);

    JPH_RVec3 j_pos = {px, py, pz};
    JPH_Quat j_rot  = {rx, ry, rz, rw};
    
    // CRITICAL: Use SoftBody specific setters, NOT BodyCreationSettings_Set...
    JPH_SoftBodyCreationSettings_SetPosition(settings, &j_pos);
    JPH_SoftBodyCreationSettings_SetRotation(settings, &j_rot);
    JPH_SoftBodyCreationSettings_SetObjectLayer(settings, OBJECT_LAYER_DYNAMIC);
    JPH_SoftBodyCreationSettings_SetAllowSleeping(settings, true);

    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    // Use the NEW dedicated helper
    uint64_t raw_h = physics_world_commit_create_soft_locked(self, settings, SLOT_PENDING_CREATE);

    if (UNLIKELY(!raw_h)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        JPH_SoftBodyCreationSettings_Destroy(settings);
        Py_DECREF(o_shared);
        return (PyErr_Occurred()) ? nullptr : PyErr_NoMemory();
    }

    // Overwrite UserData in the actual SoftBody settings with the final generational handle
    JPH_SoftBodyCreationSettings_SetUserData(settings, raw_h);

    // 5. SHADOW BUFFER UPDATE (For Center of Mass)
    uint32_t slot  = (uint32_t)(raw_h & HANDLE_INDEX_MASK);
    uint32_t dense = self->slot_to_dense[slot];

    ((PosStride *)self->positions)[dense]      = (PosStride){px, py, pz, 0.0};
    ((PosStride *)self->prev_positions)[dense] = (PosStride){px, py, pz, 0.0};
    ((AuxStride *)self->rotations)[dense]      = (AuxStride){rx, ry, rz, rw};
    ((AuxStride *)self->prev_rotations)[dense] = (AuxStride){rx, ry, rz, rw};

    self->categories[dense] = category;
    self->masks[dense]      = mask;
    self->user_data[dense]  = user_data;

    self->view_shape[0] = (Py_ssize_t)atomic_load_explicit(&self->count, memory_order_relaxed);

    // 6. QUEUE COMMAND
    PhysicsCommand *cmd       = &self->command_queue[self->command_count++];
    cmd->header               = CMD_HEADER(CMD_CREATE_SOFT_BODY, slot);
    cmd->create_soft.settings = settings;
    cmd->create_soft.category = category;
    cmd->create_soft.mask     = mask;
    // Store the Python object pointer in the padding of the command so we can DECREF it later!
    // Since create_soft.user_data is uint64_t, we can use it to store the PyObject*
    cmd->create_soft.user_data    = (uintptr_t)o_shared;
    cmd->create_soft.num_vertices = py_shared->num_vertices;

    SHADOW_UNLOCK(&self->shadow_lock);
    return PyLong_FromUnsignedLongLong(raw_h);
}