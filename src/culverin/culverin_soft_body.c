#include "culverin_soft_body.h"
#include "culverin.h"
#include "culverin_arg_indices.h"
#include "culverin_fast_build.h"
#include "culverin_python.h"

static constexpr uint32_t COLLISION_FILTER_ALL_CATEGORIES = 0xFFFF;
static constexpr uint32_t COLLISION_FILTER_ALL_MASKS      = 0xFFFF;

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