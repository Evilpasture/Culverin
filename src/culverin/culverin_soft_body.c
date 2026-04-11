#include "culverin_soft_body.h"
#include "culverin_physics_sync.h"

static constexpr uint32_t COLLISION_FILTER_ALL_CATEGORIES = 0xFFFF;
static constexpr uint32_t COLLISION_FILTER_ALL_MASKS      = 0xFFFF;

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

    void *targets[SbssAddVertex_COUNT] = {
        [IDX_SAV_POS] = (void *)&o_pos, [IDX_SAV_MASS] = (void *)&inv_mass};

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

    // Extract C-struct from Python Wrapper (assuming you named it SoftBodySharedSettingsObject)
    auto *py_shared = (SoftBodySharedSettingsObject *)o_shared;

    // --- LIFETIME PROTECTION ---
    // We increase the Python refcount because the PhysicsCommand queue 
    // now effectively "owns" a piece of this object until the next step().
    Py_INCREF(o_shared); 

    JPH_SoftBodyCreationSettings_SetSharedSettings(settings, py_shared->settings);
    JPH_SoftBodyCreationSettings_SetVertexRadius(settings, 0.05f);

    // SoftBodyCreationSettings inherits from BodyCreationSettings, safe to cast
    JPH_RVec3 j_pos = {px, py, pz};
    JPH_Quat j_rot  = {rx, ry, rz, rw};
    JPH_BodyCreationSettings_SetPosition((JPH_BodyCreationSettings *)settings, &j_pos);
    JPH_BodyCreationSettings_SetRotation((JPH_BodyCreationSettings *)settings, &j_rot);

    // Set explicit Soft Body properties
    JPH_SoftBodyCreationSettings_SetUserData(settings, 0); // Will be set to handle below
    
    // --- Set Object Layer and Motion Type ---
    JPH_BodyCreationSettings_SetObjectLayer((JPH_BodyCreationSettings *)settings, OBJECT_LAYER_DYNAMIC);
    JPH_BodyCreationSettings_SetMotionType((JPH_BodyCreationSettings *)settings, JPH_MotionType_Dynamic);
    JPH_BodyCreationSettings_SetAllowSleeping((JPH_BodyCreationSettings *)settings, true);

    // 4. COMMIT PHASE
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    // Commit slot (SLOT_PENDING_CREATE will be upgraded to SLOT_SOFT_BODY in
    // flush_commands_internal)
    uint64_t raw_h = physics_world_commit_create_locked(self, (JPH_BodyCreationSettings *)settings,
                                                        SLOT_PENDING_CREATE);

    if (UNLIKELY(!raw_h)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        JPH_SoftBodyCreationSettings_Destroy(settings);
        Py_DECREF(o_shared); // Release protection
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
    PhysicsCommand *cmd        = &self->command_queue[self->command_count++];
    cmd->header                = CMD_HEADER(CMD_CREATE_SOFT_BODY, slot);
    cmd->create_soft.settings  = settings;
    cmd->create_soft.category  = category;
    cmd->create_soft.mask      = mask;
    // Store the Python object pointer in the padding of the command so we can DECREF it later!
    // Since create_soft.user_data is uint64_t, we can use it to store the PyObject*
    cmd->create_soft.user_data  = (uintptr_t)o_shared; 
    cmd->create_soft.num_vertices = py_shared->num_vertices;

    SHADOW_UNLOCK(&self->shadow_lock);
    return PyLong_FromUnsignedLongLong(raw_h);
}