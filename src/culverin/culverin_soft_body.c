#include "culverin_soft_body.h"
#include "culverin_arg_indices.h"
#include "culverin_fast_build.h"
#include "culverin_module.h"
#include "culverin_python.h"

// --- SharedSettings Lifecycle ---

PyType_DeclareSlot_StatusFromModule SoftBodySharedSettings_init(SoftBodySharedSettingsObject *self,
                                                                CULV_MAYBE_UNUSED PyObject *args,
                                                                CULV_MAYBE_UNUSED PyObject *kwds) {
    self->settings            = JPH_SoftBodySharedSettings_Create();
    self->num_vertices        = 0;
    self->constraints_created = false;
    self->optimized           = false;

    if (!self->settings) {
        PyErr_NoMemory();
        return -1;
    }
    self->parsers =
        (SoftBodySharedSettingsParsers *)PyMem_Malloc(sizeof(SoftBodySharedSettingsParsers));
    if (!self->parsers) {
        JPH_SoftBodySharedSettings_Destroy(self->settings);
        PyErr_NoMemory();
        return -1;
    }
    culverin_init_sbss_parsers(self->parsers);
    return 0;
}

PyType_DeclareSlot_VoidFromModule
SoftBodySharedSettings_dealloc(SoftBodySharedSettingsObject *self) {
    if (self->settings) {
        JPH_SoftBodySharedSettings_Destroy(self->settings);
        self->settings = nullptr;
    }
    if (self->parsers) {
        culverin_free_sbss_parsers(self->parsers);
        PyMem_Free(self->parsers);
    }
    PyTypeObject *tp = Py_TYPE(self);
    tp->tp_free((PyObject *)self);
    Py_DECREF(tp);
}

// --- SharedSettings Topology Methods ---

PyCFunction_DeclareMethodFromModule
SoftBodySharedSettings_add_vertex(SoftBodySharedSettingsObject *self, PyObject *const *args,
                                  Py_ssize_t nargs, PyObject *kwnames) {
    constexpr float default_inv_mass = 1.0F;

    PyObject *o_pos = nullptr;
    PyObject *o_vel = nullptr;
    float inv_mass  = default_inv_mass;

    void *targets[SbssAddVertex_COUNT] = {[IDX_SAV_POS]  = (void *)&o_pos,
                                          [IDX_SAV_MASS] = (void *)&inv_mass,
                                          [IDX_SAV_VEL]  = (void *)&o_vel};

    if (!FastParse_Unified(args, nargs, kwnames, &self->parsers->SbssAddVertexParser, targets)) {
        return nullptr;
    }

    if (self->optimized) {
        PyErr_SetString(PyExc_RuntimeError, "Cannot modify settings after optimize()");
        return nullptr;
    }

    // 1. Parse Position (Required)
    JPH_Vec3 pos = {};
    if (!parse_vec3_direct(o_pos, &pos.x, &pos.y, &pos.z)) {
        return nullptr;
    }

    // 2. Parse Velocity (Optional)
    JPH_Vec3 vel = {};
    if (o_vel && o_vel != Py_None) {
        if (!parse_vec3_direct(o_vel, &vel.x, &vel.y, &vel.z)) {
            return nullptr;
        }
    }

    // 3. Construct the Jolt packed vertex structure
    const JPH_SoftVertex vertex = {.position = pos, .velocity = vel, .invMass = inv_mass};

    // 4. Dispatch and track
    JPH_SoftBodySharedSettings_AddVertex(self->settings, &vertex);
    self->num_vertices++;

    Py_RETURN_NONE;
}

PyCFunction_DeclareMethodFromModule
SoftBodySharedSettings_add_vertices(SoftBodySharedSettingsObject *self, PyObject *const *args,
                                    Py_ssize_t nargs, PyObject *kwnames) {
    auto o_pos  = (PyObject *)nullptr;
    auto o_mass = (PyObject *)nullptr;
    auto o_vel  = (PyObject *)nullptr;

    void *targets[SbssAddVertices_COUNT] = {[IDX_SAVS_POS]  = (void *)&o_pos,
                                            [IDX_SAVS_MASS] = (void *)&o_mass,
                                            [IDX_SAVS_VEL]  = (void *)&o_vel};

    if (!FastParse_Unified(args, nargs, kwnames, &self->parsers->SbssAddVerticesParser, targets)) {
        return nullptr;
    }
    if (self->optimized) {
        return PyErr_Format(PyExc_RuntimeError, "Cannot modify after optimize()");
    }

    // 1. Buffer Acquisitions
    Py_buffer pos_view;
    if (PyObject_GetBuffer(o_pos, &pos_view, PyBUF_SIMPLE) != 0) {
        return nullptr;
    }

    constexpr auto vec3_size = 3 * sizeof(float);
    if (pos_view.len % vec3_size != 0) {
        PyBuffer_Release(&pos_view);
        return PyErr_Format(PyExc_ValueError, "positions must be float32 triplets");
    }

    const auto count = (uint32_t)(pos_view.len / vec3_size);

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
            return PyErr_Format(PyExc_ValueError, "inv_masses length must match vertex count");
        }
        masses = (float *)mass_view.buf;
    }

    Py_buffer vel_view = {};
    float *velocities  = nullptr;
    if (o_vel && o_vel != Py_None) {
        if (PyObject_GetBuffer(o_vel, &vel_view, PyBUF_SIMPLE) != 0) {
            PyBuffer_Release(&pos_view);
            if (masses) {
                PyBuffer_Release(&mass_view);
            }
            return nullptr;
        }
        if (vel_view.len / vec3_size != count) {
            PyBuffer_Release(&pos_view);
            if (masses) {
                PyBuffer_Release(&mass_view);
            }
            PyBuffer_Release(&vel_view);
            return PyErr_Format(PyExc_ValueError, "velocities length must match vertex count");
        }
        velocities = (float *)vel_view.buf;
    }

    // 2. The Interleave Allocation
    auto staging = (JPH_SoftVertex *)PyMem_Malloc(sizeof(JPH_SoftVertex) * count);
    if (!staging) {
        PyBuffer_Release(&pos_view);
        if (masses) {
            PyBuffer_Release(&mass_view);
        }
        if (velocities) {
            PyBuffer_Release(&vel_view);
        }
        return PyErr_NoMemory();
    }

    // 3. High-Speed Packing Loop
    const auto src_pos = (const float *)pos_view.buf;
    for (size_t i = 0; i < count; ++i) {
        const size_t v_idx = i * 3;

        staging[i] = (JPH_SoftVertex){
            .position = {src_pos[v_idx], src_pos[v_idx + 1], src_pos[v_idx + 2]},
            .velocity = velocities ? (JPH_Vec3){velocities[v_idx], velocities[v_idx + 1],
                                                velocities[v_idx + 2]}
                                   : (JPH_Vec3){},
            .invMass  = masses ? masses[i] : 1.0F};
    }

    // 4. Jolt Dispatch & Cleanup
    JPH_SoftBodySharedSettings_AddVertices(self->settings, staging, count);
    self->num_vertices += count;

    PyMem_Free(staging);
    PyBuffer_Release(&pos_view);
    if (masses) {
        PyBuffer_Release(&mass_view);
    }
    if (velocities) {
        PyBuffer_Release(&vel_view);
    }

    Py_RETURN_NONE;
}

PyCFunction_DeclareMethodFromModule
SoftBodySharedSettings_add_face(SoftBodySharedSettingsObject *self, PyObject *const *args,
                                Py_ssize_t nargs, PyObject *kwnames) {
    uint32_t v1      = 0;
    uint32_t v2      = 0;
    uint32_t v3      = 0;
    uint32_t mat_idx = 0;

    void *targets[SbssAddFace_COUNT] = {
        [IDX_SAF_V1] = &v1, [IDX_SAF_V2] = &v2, [IDX_SAF_V3] = &v3, [IDX_SAF_MAT] = &mat_idx};

    if (!FastParse_Unified(args, nargs, kwnames, &self->parsers->SbssAddFaceParser, targets)) {
        return nullptr;
    }

    if (self->optimized) {
        PyErr_SetString(PyExc_RuntimeError, "Cannot modify settings after optimize()");
        return nullptr;
    }

    // 1. Degenerate Triangle Validation
    if (v1 == v2 || v2 == v3 || v1 == v3) {
        PyErr_SetString(PyExc_ValueError, "Face must have 3 distinct vertex indices");
        return nullptr;
    }

    // 2. Source of Truth Validation
    const auto vertex_count = JPH_SoftBodySharedSettings_GetVertexCount(self->settings);
    if (v1 >= vertex_count || v2 >= vertex_count || v3 >= vertex_count) {
        PyErr_Format(PyExc_IndexError,
                     "Face vertex index out of range (Indices: %u,%u,%u | Total: %u)", v1, v2, v3,
                     vertex_count);
        return nullptr;
    }

    // 3. Pack into Jolt's new AoS structure using designated initializers
    const JPH_SoftFace face = {
        .vertex1 = v1, .vertex2 = v2, .vertex3 = v3, .materialIndex = mat_idx};

    // 4. Dispatch to Jolt
    JPH_SoftBodySharedSettings_AddFace(self->settings, &face);

    Py_RETURN_NONE;
}

PyCFunction_DeclareMethodFromModule
SoftBodySharedSettings_add_faces(SoftBodySharedSettingsObject *self, PyObject *const *args,
                                 Py_ssize_t nargs, PyObject *kwnames) {
    auto o_ind = (PyObject *)nullptr;
    auto o_mat = (PyObject *)nullptr;

    void *targets[SbssAddFaces_COUNT] = {[IDX_SAFS_IND] = (void *)&o_ind,
                                         [IDX_SAFS_MAT] = (void *)&o_mat};

    if (!FastParse_Unified(args, nargs, kwnames, &self->parsers->SbssAddFacesParser, targets)) {
        return nullptr;
    }

    if (self->optimized) {
        PyErr_SetString(PyExc_RuntimeError, "Cannot modify settings after optimize()");
        return nullptr;
    }

    // 1. Index Buffer Acquisition
    Py_buffer ind_view;
    if (PyObject_GetBuffer(o_ind, &ind_view, PyBUF_SIMPLE) != 0) {
        return nullptr;
    }

    constexpr auto triplet_size = 3 * sizeof(uint32_t);
    if (ind_view.len % triplet_size != 0) {
        PyBuffer_Release(&ind_view);
        PyErr_SetString(PyExc_ValueError, "indices buffer must be uint32 triplets");
        return nullptr;
    }

    const auto face_count = (uint32_t)(ind_view.len / triplet_size);
    const auto inds       = (const uint32_t *)ind_view.buf;

    // 2. Optional Material Buffer Acquisition
    Py_buffer mat_view = {};
    auto mats          = (const uint32_t *)nullptr;
    if (o_mat && o_mat != Py_None) {
        if (PyObject_GetBuffer(o_mat, &mat_view, PyBUF_SIMPLE) != 0) {
            PyBuffer_Release(&ind_view);
            return nullptr;
        }
        if (mat_view.len / sizeof(uint32_t) != face_count) {
            PyBuffer_Release(&ind_view);
            PyBuffer_Release(&mat_view);
            PyErr_SetString(PyExc_ValueError, "materials length must match face count");
            return nullptr;
        }
        mats = (const uint32_t *)mat_view.buf;
    }

    // 3. Resource & Source-of-Truth Validation
    const auto vertex_count = JPH_SoftBodySharedSettings_GetVertexCount(self->settings);

    // 4. Interleave Packing Loop
    auto staging = (JPH_SoftFace *)PyMem_Malloc(sizeof(JPH_SoftFace) * face_count);
    if (staging == nullptr) {
        PyBuffer_Release(&ind_view);
        if (mats) {
            PyBuffer_Release(&mat_view);
        }
        return PyErr_NoMemory();
    }

    constexpr uint32_t STRIDE = 3;
    for (uint32_t i = 0; i < face_count; ++i) {
        const auto v1 = inds[(i * STRIDE) + 0];
        const auto v2 = inds[(i * STRIDE) + 1];
        const auto v3 = inds[(i * STRIDE) + 2];

        // Hardware-friendly bounds check
        if (v1 >= vertex_count || v2 >= vertex_count || v3 >= vertex_count) {
            PyMem_Free(staging);
            PyBuffer_Release(&ind_view);
            if (mats) {
                PyBuffer_Release(&mat_view);
            }
            PyErr_Format(PyExc_IndexError, "Face %u uses out-of-range vertex index", i);
            return nullptr;
        }

        // Interleave into Jolt's AoS layout
        staging[i] = (JPH_SoftFace){.vertex1       = v1,
                                    .vertex2       = v2,
                                    .vertex3       = v3,
                                    .materialIndex = (mats != nullptr) ? mats[i] : 0};
    }

    // 5. Atomic Push to Jolt
    JPH_SoftBodySharedSettings_AddFaces(self->settings, staging, face_count);

    // 6. Finalization
    PyMem_Free(staging);
    PyBuffer_Release(&ind_view);
    if (mats != nullptr) {
        PyBuffer_Release(&mat_view);
    }

    Py_RETURN_NONE;
}

// Method: settings.add_pinned_vertex(index) (METH_O)
/**
 * @deprecated Use add_vertices with an inverse mass of 0.0 instead.
 * This method is now a no-op because the underlying Jolt wrapper does not
 * support post-insertion vertex modification.
 */
[[gnu::deprecated("Use add_vertices with inv_mass=0.0 instead")]]
PyCFunction_DeclareMethodFromModule
SoftBodySharedSettings_add_pinned_vertex(SoftBodySharedSettingsObject *self, PyObject *arg) {
    // 1. Validate the argument type so we don't break existing logic flows
    const auto index_raw = PyLong_AsLong(arg);
    if (PyErr_Occurred()) {
        return nullptr;
    }

    // 2. Issue a Python-level DeprecationWarning
    // This allows Python users to see the fix required in their logs.
    const int warn_status = PyErr_WarnEx(PyExc_DeprecationWarning,
                                         "add_pinned_vertex() is deprecated and has no effect. "
                                         "Pin vertices by passing 0.0 mass in add_vertices().",
                                         1);

    if (warn_status < 0) {
        return nullptr; // User has warnings-as-errors enabled
    }

    // 3. Safety Check: Still perform bounds checking to help users debug their indices
    const auto vertex_count = JPH_SoftBodySharedSettings_GetVertexCount(self->settings);
    if (index_raw < 0 || (uint32_t)index_raw >= vertex_count) {
        PyErr_SetString(PyExc_IndexError, "Vertex index out of range");
        return nullptr;
    }

    // 4. Return None without modifying Jolt state
    Py_RETURN_NONE;
}
[[gnu::const]]
static inline bool check_bend_type(int val, int max) {
    return (bool)((val) >= 0 && (val) < (max));
}

// Method: settings.create_constraints(compliance, bend_type=1)
PyCFunction_DeclareMethodFromModule
SoftBodySharedSettings_create_constraints(SoftBodySharedSettingsObject *self, PyObject *const *args,
                                          Py_ssize_t nargs, PyObject *kwnames) {
    float compliance               = 0.0001f;
    JPH_SoftBodyBendType bend_type = JPH_SoftBodyBendType_Distance; // Default: Distance

    void *targets[2] = {&compliance, &bend_type};
    if (!FastParse_Unified(args, nargs, kwnames, &self->parsers->SbssCreateConstraintsParser,
                           targets)) {
        return nullptr;
    }
    if (!check_bend_type(bend_type, 3)) {
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
    const auto index_raw = PyLong_AsLong(arg);
    if (PyErr_Occurred()) {
        return nullptr;
    }

    // Safety: check vertex count using the updated API
    const auto vertex_count = JPH_SoftBodySharedSettings_GetVertexCount(self->settings);
    if (index_raw < 0 || (uint32_t)index_raw >= vertex_count) {
        PyErr_SetString(PyExc_IndexError, "Vertex index out of range");
        return nullptr;
    }

    const auto index = (uint32_t)index_raw;

    // Designated initialization for the return buffer
    JPH_SoftVertex vertex = {};

    // Use the new unified getter
    if (!JPH_SoftBodySharedSettings_GetVertex(self->settings, index, &vertex)) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to retrieve vertex data from Jolt");
        return nullptr;
    }

    // Return only the position component as requested
    return FastBuild_Tuple(vertex.position.x, vertex.position.y, vertex.position.z);
}

#define SBSS_FASTCALL(name) CULV_FEAT(SoftBodySharedSettings, name, METH_FASTCALL | METH_KEYWORDS)
#define SBSS_NOARGS(name) CULV_FEAT(SoftBodySharedSettings, name, METH_NOARGS)
#define SBSS_O(name) CULV_FEAT(SoftBodySharedSettings, name, METH_O)

PyType_Spec SoftBodySharedSettings_spec = {
    .name      = "culverin._culverin_c.SoftBodySharedSettingsObject",
    .basicsize = sizeof(SoftBodySharedSettingsObject),
    .flags     = Py_TPFLAGS_DEFAULT,
    .slots =
        (PyType_Slot[]){

            {.slot = Py_tp_new, .pfunc = PyType_GenericNew},
            {.slot = Py_tp_init, .pfunc = SoftBodySharedSettings_init},
            {.slot = Py_tp_dealloc, .pfunc = SoftBodySharedSettings_dealloc},
            {.slot = Py_tp_methods,
             .pfunc =
                 (PyMethodDef[]){

                     SBSS_FASTCALL(add_vertex),
                     SBSS_FASTCALL(add_vertices),
                     SBSS_O(add_pinned_vertex),
                     SBSS_O(get_vertex_position),
                     SBSS_FASTCALL(add_face),
                     SBSS_FASTCALL(add_faces),
                     SBSS_FASTCALL(create_constraints),
                     SBSS_NOARGS(optimize),
                     {}

                 }},
            {},
        }

};