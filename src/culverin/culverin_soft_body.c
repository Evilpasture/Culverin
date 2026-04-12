#include "culverin_soft_body.h"
#include "culverin_fast_build.h"
#include "culverin_physics_sync.h"
#include "culverin_physics_world_internal.h"

static constexpr uint32_t COLLISION_FILTER_ALL_CATEGORIES = 0xFFFF;
static constexpr uint32_t COLLISION_FILTER_ALL_MASKS      = 0xFFFF;

/**
 * HELPER: physics_world_commit_create_soft_locked
 * Separate path for soft bodies to avoid binary-incompatibility with Rigid Body settings.
 */
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

    // CRITICAL: Use the SoftBody specific setter (Binder ensures correct memory offset)
    JPH_SoftBodyCreationSettings_SetUserData(settings, raw_h);

    self->slot_to_dense[slot]  = dense;
    self->dense_to_slot[dense] = slot;
    self->body_ids[dense]      = JPH_INVALID_BODY_ID;
    atomic_store_explicit(&self->slot_states[slot], slot_state, memory_order_release);

    return raw_h;
}

// --- SharedSettings Lifecycle ---

PyType_DeclareSlot_StatusFromModule SoftBodySharedSettings_init(SoftBodySharedSettingsObject *self,
                                                                CULV_MAYBE_UNUSED PyObject *args,
                                                                CULV_MAYBE_UNUSED PyObject *kwds) {
    self->settings            = JPH_SoftBodySharedSettings_Create();
    self->num_vertices        = 0;
    self->constraints_created = false;
    self->optimized           = false;

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

// --- SharedSettings Topology Methods ---

PyCFunction_DeclareMethodFromModule
SoftBodySharedSettings_add_vertex(SoftBodySharedSettingsObject *self, PyObject *const *args,
                                  Py_ssize_t nargs, PyObject *kwnames) {
    CulverinState *st                  = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    PyObject *o_pos                    = nullptr;
    float inv_mass                     = 1.0f;
    void *targets[SbssAddVertex_COUNT] = {[IDX_SAV_POS]  = (void *)&o_pos,
                                          [IDX_SAV_MASS] = (void *)&inv_mass};

    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.SbssAddVertexParser, targets)) {
        return nullptr;
    }
    if (self->optimized) {
        PyErr_SetString(PyExc_RuntimeError, "Cannot modify settings after optimize()");
        return nullptr;
    }

    JPH_Vec3 pos;
    if (!parse_vec3_direct(o_pos, &pos.x, &pos.y, &pos.z)) {
        return nullptr;
    }

    JPH_SoftBodySharedSettings_AddVertex(self->settings, &pos, inv_mass);
    self->num_vertices++;
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethodFromModule
SoftBodySharedSettings_add_vertices(SoftBodySharedSettingsObject *self, PyObject *const *args,
                                    Py_ssize_t nargs, PyObject *kwnames) {
    CulverinState *st                    = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    PyObject *o_pos                      = nullptr;
    PyObject *o_mass                     = nullptr;
    void *targets[SbssAddVertices_COUNT] = {[IDX_SAVS_POS]  = (void *)&o_pos,
                                            [IDX_SAVS_MASS] = (void *)&o_mass};

    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.SbssAddVerticesParser, targets)) {
        return nullptr;
    }
    if (self->optimized) {
        PyErr_SetString(PyExc_RuntimeError, "Cannot modify settings after optimize()");
        return nullptr;
    }

    Py_buffer pos_view;
    if (PyObject_GetBuffer(o_pos, &pos_view, PyBUF_SIMPLE) != 0) {
        return nullptr;
    }

    // Must be flat float32 triplets
    if (pos_view.len % (3 * sizeof(float)) != 0) {
        PyBuffer_Release(&pos_view);
        PyErr_SetString(PyExc_ValueError,
                        "positions buffer must be a flat array of float32 triplets");
        return nullptr;
    }

    uint32_t count = (uint32_t)(pos_view.len / (3 * sizeof(float)));

    Py_buffer mass_view = {};
    float *masses       = nullptr;
    if (o_mass && o_mass != Py_None) {
        if (PyObject_GetBuffer(o_mass, &mass_view, PyBUF_SIMPLE) != 0) {
            PyBuffer_Release(&pos_view);
            return nullptr;
        }
        if (mass_view.len / sizeof(float) != count) {
            PyBuffer_Release(&pos_view);
            PyBuffer_Release(&mass_view);
            PyErr_SetString(PyExc_ValueError, "inv_masses buffer length must match vertex count");
            return nullptr;
        }
        masses = (float *)mass_view.buf;
    }

    JPH_SoftBodySharedSettings_AddVertices(self->settings, (const JPH_Vec3 *)pos_view.buf, masses,
                                           count);
    self->num_vertices += count;

    PyBuffer_Release(&pos_view);
    if (o_mass && o_mass != Py_None) {
        PyBuffer_Release(&mass_view);
    }

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
    if (self->optimized) {
        PyErr_SetString(PyExc_RuntimeError, "Cannot modify settings after optimize()");
        return nullptr;
    }

    if (v1 == v2 || v2 == v3 || v1 == v3) {
        PyErr_SetString(PyExc_ValueError, "Face must have 3 distinct vertex indices");
        return nullptr;
    }

    if (v1 >= self->num_vertices || v2 >= self->num_vertices || v3 >= self->num_vertices) {
        PyErr_Format(PyExc_IndexError, "Face vertex index out of range (have %u vertices)",
                     self->num_vertices);
        return nullptr;
    }

    JPH_SoftBodySharedSettings_AddFace(self->settings, v1, v2, v3);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethodFromModule
SoftBodySharedSettings_add_faces(SoftBodySharedSettingsObject *self, PyObject *const *args,
                                 Py_ssize_t nargs, PyObject *kwnames) {
    CulverinState *st                 = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    PyObject *o_ind                   = nullptr;
    void *targets[SbssAddFaces_COUNT] = {[IDX_SAFS_IND] = (void *)&o_ind};

    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.SbssAddFacesParser, targets)) {
        return nullptr;
    }
    if (self->optimized) {
        PyErr_SetString(PyExc_RuntimeError, "Cannot modify settings after optimize()");
        return nullptr;
    }

    Py_buffer ind_view;
    if (PyObject_GetBuffer(o_ind, &ind_view, PyBUF_SIMPLE) != 0) {
        return nullptr;
    }

    // Must be flat uint32 triplets
    if (ind_view.len % (3 * sizeof(uint32_t)) != 0) {
        PyBuffer_Release(&ind_view);
        PyErr_SetString(PyExc_ValueError, "indices buffer must be a flat array of uint32 triplets");
        return nullptr;
    }

    uint32_t face_count  = (uint32_t)(ind_view.len / (3 * sizeof(uint32_t)));
    const uint32_t *inds = (const uint32_t *)ind_view.buf;

    // Validate indices against total vertices to prevent Jolt crashes
    for (uint32_t i = 0; i < face_count * 3; i++) {
        if (inds[i] >= self->num_vertices) {
            PyBuffer_Release(&ind_view);
            PyErr_Format(PyExc_IndexError, "Face vertex index %u out of range (have %u vertices)",
                         inds[i], self->num_vertices);
            return nullptr;
        }
    }

    JPH_SoftBodySharedSettings_AddFaces(self->settings, inds, face_count);

    PyBuffer_Release(&ind_view);
    Py_RETURN_NONE;
}

// Method: settings.add_pinned_vertex(index) (METH_O)
PyCFunction_DeclareMethodFromModule
SoftBodySharedSettings_add_pinned_vertex(SoftBodySharedSettingsObject *self, PyObject *arg) {
    long index = PyLong_AsLong(arg);
    if (PyErr_Occurred()) {
        return nullptr;
    }

    if (self->optimized) {
        PyErr_SetString(PyExc_RuntimeError, "Cannot modify settings after optimize()");
        return nullptr;
    }
    if (index < 0 || (uint32_t)index >= self->num_vertices) {
        PyErr_SetString(PyExc_IndexError, "Vertex index out of range");
        return nullptr;
    }

    // Logic: Must be called BEFORE CreateConstraints
    JPH_SoftBodySharedSettings_AddPinnedVertex(self->settings, (uint32_t)index);
    Py_RETURN_NONE;
}

static inline bool inCasePythonUsers_PassStupidInts(int val, int max) {
    return (bool)((val) >= 0 && (val) < (max));
}

// Method: settings.create_constraints(compliance, bend_type=1)
PyCFunction_DeclareMethodFromModule
SoftBodySharedSettings_create_constraints(SoftBodySharedSettingsObject *self, PyObject *const *args,
                                          Py_ssize_t nargs, PyObject *kwnames) {
    CulverinState *st              = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    float compliance               = 0.0001f;
    JPH_SoftBodyBendType bend_type = JPH_SoftBodyBendType_Distance; // Default: Distance

    void *targets[2] = {&compliance, &bend_type};
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.SbssCreateConstraintsParser,
                           targets)) {
        return nullptr;
    }
    if (!inCasePythonUsers_PassStupidInts(bend_type, 3)) {
        PyErr_Format(PyExc_ValueError,
                     "bend_type must be 0 (None), 1 (Distance), or 2 (Dihedral), got %d",
                     bend_type);
        return nullptr;
    }
    if (self->constraints_created) {
        PyErr_SetString(PyExc_RuntimeError, "create_constraints already called");
        return nullptr;
    }
    self->constraints_created = true;

    // Binder handles both Edge compliance and Shear compliance
    JPH_SoftBodySharedSettings_CreateConstraints(self->settings, compliance, bend_type);
    Py_RETURN_NONE;
}

// Method: settings.optimize() (METH_NOARGS)
PyCFunction_DeclareMethodFromModule
SoftBodySharedSettings_optimize(SoftBodySharedSettingsObject *self,
                                CULV_MAYBE_UNUSED PyObject *args) {
    if (!self->constraints_created) {
        PyErr_SetString(PyExc_RuntimeError, "create_constraints must be called before optimize");
        return nullptr;
    }
    JPH_SoftBodySharedSettings_Optimize(self->settings);
    self->optimized = true;
    Py_RETURN_NONE;
}

// Method: settings.get_vertex_position(index) (METH_O) - Returns REST pose
PyCFunction_DeclareMethodFromModule
SoftBodySharedSettings_get_vertex_position(SoftBodySharedSettingsObject *self, PyObject *arg) {
    long index = PyLong_AsLong(arg);
    if (PyErr_Occurred()) {
        return nullptr;
    }

    if (self->num_vertices == 0) {
        PyErr_SetString(PyExc_RuntimeError, "No vertices added");
        return nullptr;
    }

    if (index < 0 || (uint32_t)index >= self->num_vertices) {
        PyErr_SetString(PyExc_IndexError, "Vertex index out of range");
        return nullptr;
    }

    JPH_Vec3 pos;
    JPH_SoftBodySharedSettings_GetVertexPosition(self->settings, (uint32_t)index, &pos);
    return FastBuild_Tuple(pos.x, pos.y, pos.z);
}

// --- PhysicsWorld Methods ---

PyCFunction_DeclareMethodFromModule PhysicsWorld_create_soft_body(PhysicsWorldObject *self,
                                                                  PyObject *const *args,
                                                                  size_t nargsf,
                                                                  PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));

    // 1. FAST PARSE
    PyObject *o_shared        = nullptr;
    PyObject *o_pos           = nullptr;
    PyObject *o_rot           = nullptr;
    uint64_t user_data        = 0;
    uint32_t category         = COLLISION_FILTER_ALL_CATEGORIES;
    uint32_t mask             = COLLISION_FILTER_ALL_MASKS;
    float pressure            = 0.0f;
    float vertex_radius       = 0.05f;
    float linear_damping      = 0.1f;
    uint32_t num_iterations   = 10;
    float max_linear_velocity = 500.0f;
    float gravity_factor      = 1.0f;
    float friction            = 0.2f;
    float restitution         = 0.0f;
    bool make_rot_identity    = false;

    void *targets[CreateSoftBody_COUNT] = {[IDX_CSB_SHARED]    = (void *)&o_shared,
                                           [IDX_CSB_POS]       = (void *)&o_pos,
                                           [IDX_CSB_ROT]       = (void *)&o_rot,
                                           [IDX_CSB_USER_DATA] = (void *)&user_data,
                                           [IDX_CSB_CAT]       = (void *)&category,
                                           [IDX_CSB_MASK]      = (void *)&mask,
                                           [IDX_CSB_PRESSURE]  = (void *)&pressure,
                                           [IDX_CSB_V_RADIUS]  = (void *)&vertex_radius,
                                           [IDX_CSB_LIN_DAMP]  = (void *)&linear_damping,
                                           [IDX_CSB_ITER]      = (void *)&num_iterations,
                                           [IDX_CSB_MAX_VEL]   = (void *)&max_linear_velocity,
                                           [IDX_CSB_GRAV]      = (void *)&gravity_factor,
                                           [IDX_CSB_FRIC]      = (void *)&friction,
                                           [IDX_CSB_REST]      = (void *)&restitution,
                                           [IDX_CSB_ROT_ID]    = (void *)&make_rot_identity};

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
    auto *py_shared                        = (SoftBodySharedSettingsObject *)o_shared;
    Py_INCREF(o_shared); // Ownership transfer to command queue

    JPH_SoftBodyCreationSettings_SetSharedSettings(settings, py_shared->settings);
    JPH_SoftBodyCreationSettings_SetPressure(settings, pressure);
    JPH_SoftBodyCreationSettings_SetVertexRadius(settings, vertex_radius);
    JPH_SoftBodyCreationSettings_SetLinearDamping(settings, linear_damping);
    JPH_SoftBodyCreationSettings_SetNumIterations(settings, num_iterations);
    JPH_SoftBodyCreationSettings_SetMaxLinearVelocity(settings, max_linear_velocity);
    JPH_SoftBodyCreationSettings_SetGravityFactor(settings, gravity_factor);
    JPH_SoftBodyCreationSettings_SetFriction(settings, friction);
    JPH_SoftBodyCreationSettings_SetRestitution(settings, restitution);
    JPH_SoftBodyCreationSettings_SetMakeRotationIdentity(settings, make_rot_identity);

    JPH_RVec3 j_pos = {px, py, pz};
    JPH_Quat j_rot  = {rx, ry, rz, rw};
    JPH_SoftBodyCreationSettings_SetPosition(settings, &j_pos);
    JPH_SoftBodyCreationSettings_SetRotation(settings, &j_rot);
    JPH_SoftBodyCreationSettings_SetObjectLayer(settings, OBJECT_LAYER_DYNAMIC);
    JPH_SoftBodyCreationSettings_SetAllowSleeping(settings, true);

    // 4. COMMIT
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint64_t raw_h = physics_world_commit_create_soft_locked(self, settings, SLOT_PENDING_CREATE);

    if (UNLIKELY(!raw_h)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        JPH_SoftBodyCreationSettings_Destroy(settings);
        Py_DECREF(o_shared);
        return nullptr;
    }

    // 5. SHADOW UPDATE
    uint32_t slot  = (uint32_t)(raw_h & HANDLE_INDEX_MASK);
    uint32_t dense = self->slot_to_dense[slot];

    ((PosStride *)self->positions)[dense]      = (PosStride){px, py, pz, 0.0};
    ((PosStride *)self->prev_positions)[dense] = (PosStride){px, py, pz, 0.0};
    ((AuxStride *)self->rotations)[dense]      = (AuxStride){rx, ry, rz, rw};
    ((AuxStride *)self->prev_rotations)[dense] = (AuxStride){rx, ry, rz, rw};
    self->categories[dense]                    = category;
    self->masks[dense]                         = mask;
    self->user_data[dense]                     = user_data;
    self->view_shape[0] = (Py_ssize_t)atomic_load_explicit(&self->count, memory_order_relaxed);

    // 6. QUEUE COMMAND
    PhysicsCommand *cmd            = &self->command_queue[self->command_count++];
    cmd->header                    = CMD_HEADER(CMD_CREATE_SOFT_BODY, slot);
    cmd->create_soft.settings      = settings;
    cmd->create_soft.category      = category;
    cmd->create_soft.mask          = mask;
    cmd->create_soft.user_data.ptr = o_shared;
    cmd->create_soft.num_vertices  = py_shared->num_vertices;

    SHADOW_UNLOCK(&self->shadow_lock);
    return PyLong_FromUnsignedLongLong(raw_h);
}