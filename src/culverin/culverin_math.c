#include "culverin_math.h"
#include "culverin_arg_indices.h"
#include "culverin_python.h"
#include "fast_build.h"

typedef struct MathHolderObject {
    PyObject_HEAD MathParsers *parsers;
} MathHolderObject;

// ============================================================================
// FASTPARSE WRAPPERS
// ============================================================================

static PyObject *MathHolderObject_get_perspective(MathHolderObject *self, PyObject *const *args,
                                                  Py_ssize_t nargsf, PyObject *kwnames) {
    float fovy;
    float aspect;
    float near_p;
    float far_p;
    void *targets[MathPersp_COUNT] = {[IDX_MP_FOVY]   = &fovy,
                                      [IDX_MP_ASPECT] = &aspect,
                                      [IDX_MP_NEAR]   = &near_p,
                                      [IDX_MP_FAR]    = &far_p};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->MathPerspParser, targets)) {
        return nullptr;
    }

    float out[16];
    culverin_math_get_perspective(fovy, aspect, near_p, far_p, out);

    return FastBuild_Tuple(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8],
                           out[9], out[10], out[11], out[12], out[13], out[14], out[15]);
}

static PyObject *MathHolderObject_get_ortho(MathHolderObject *self, PyObject *const *args,
                                            Py_ssize_t nargsf, PyObject *kwnames) {
    float left;
    float right;
    float bottom;
    float top;
    float near_p;
    float far_p;
    void *targets[MathOrtho_COUNT] = {[IDX_MO_LEFT] = &left,     [IDX_MO_RIGHT] = &right,
                                      [IDX_MO_BOTTOM] = &bottom, [IDX_MO_TOP] = &top,
                                      [IDX_MO_NEAR] = &near_p,   [IDX_MO_FAR] = &far_p};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->MathOrthoParser, targets)) {
        return nullptr;
    }

    float out[16];
    culverin_math_get_ortho(left, right, bottom, top, near_p, far_p, out);

    return FastBuild_Tuple(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8],
                           out[9], out[10], out[11], out[12], out[13], out[14], out[15]);
}

static PyObject *MathHolderObject_get_look_at(MathHolderObject *self, PyObject *const *args,
                                              Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *eye_obj;
    PyObject *target_obj;
    PyObject *up_obj;
    void *targets[MathTrio_COUNT] = {[IDX_MT_0] = (void *)&eye_obj,
                                     [IDX_MT_1] = (void *)&target_obj,
                                     [IDX_MT_2] = (void *)&up_obj};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->MathTrioParser, targets)) {
        return nullptr;
    }

    // 1. Gatekeeper: Ensure we actually have tuples and they have 3 elements
    if (!PyTuple_Check(eye_obj) || !PyTuple_Check(target_obj) || !PyTuple_Check(up_obj)) {
        PyErr_SetString(PyExc_TypeError, "LookAt parameters must be tuples");
        return nullptr;
    }
    if (PyTuple_GET_SIZE(eye_obj) < 3 || PyTuple_GET_SIZE(target_obj) < 3 ||
        PyTuple_GET_SIZE(up_obj) < 3) {
        PyErr_SetString(PyExc_ValueError, "LookAt tuples must contain 3 elements");
        return nullptr;
    }

    // 2. SIMD Alignment: Pad to 4 floats and align to 16 bytes
    alignas(16) float eye[4]    = {0};
    alignas(16) float target[4] = {0};
    alignas(16) float up[4]     = {0};

    // 3. Fast Unpacking via Macros
    for (int i = 0; i < 3; ++i) {
        eye[i]    = (float)PyFloat_AsDouble(PyTuple_GET_ITEM(eye_obj, i));
        target[i] = (float)PyFloat_AsDouble(PyTuple_GET_ITEM(target_obj, i));
        up[i]     = (float)PyFloat_AsDouble(PyTuple_GET_ITEM(up_obj, i));
    }

    // Final sanity check for conversion errors (e.g., passing a string in the tuple)
    if (PyErr_Occurred()) {
        return nullptr;
    }

    alignas(16) float out[16];
    culverin_math_get_look_at(eye, target, up, out);

    return FastBuild_Tuple(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8],
                           out[9], out[10], out[11], out[12], out[13], out[14], out[15]);
}

static PyObject *MathHolderObject_get_trs(MathHolderObject *self, PyObject *const *args,
                                          Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *pos_obj;
    PyObject *rot_obj;
    PyObject *scale_obj;
    void *targets[MathTrio_COUNT] = {[IDX_MT_0] = (void *)&pos_obj,
                                     [IDX_MT_1] = (void *)&rot_obj,
                                     [IDX_MT_2] = (void *)&scale_obj};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->MathTrioParser, targets)) {
        return nullptr;
    }

    // 1. Guard the gates: Verify types and sizes once
    if (!PyTuple_Check(pos_obj) || !PyTuple_Check(rot_obj) || !PyTuple_Check(scale_obj)) {
        PyErr_SetString(PyExc_TypeError, "TRS inputs must be tuples");
        return nullptr;
    }
    if (PyTuple_GET_SIZE(pos_obj) < 3 || PyTuple_GET_SIZE(rot_obj) < 4 ||
        PyTuple_GET_SIZE(scale_obj) < 3) {
        PyErr_SetString(PyExc_ValueError, "Invalid tuple dimensions for TRS");
        return nullptr;
    }

    // 2. Aligned Stack Allocation (Crucial for SIMD)
    // We pad to 4 floats even for Vec3 to keep the 16-byte boundary clean
    alignas(16) float p[4];
    alignas(16) float r[4];
    alignas(16) float s[4];

    // 3. Fast Unpacking Loop
    // Position
    for (int i = 0; i < 3; ++i) {
        p[i] = (float)PyFloat_AsDouble(PyTuple_GET_ITEM(pos_obj, i));
    }
    // Rotation (Quaternion: x, y, z, w)
    for (int i = 0; i < 4; ++i) {
        r[i] = (float)PyFloat_AsDouble(PyTuple_GET_ITEM(rot_obj, i));
    }
    // Scale
    for (int i = 0; i < 3; ++i) {
        s[i] = (float)PyFloat_AsDouble(PyTuple_GET_ITEM(scale_obj, i));
    }

    // Check if any PyFloat_AsDouble failed (e.g. passed a string in the tuple)
    if (PyErr_Occurred()) {
        return nullptr;
    }

    alignas(16) float out[16];
    culverin_math_get_trs(p, r, s, out);

    return FastBuild_Tuple(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8],
                           out[9], out[10], out[11], out[12], out[13], out[14], out[15]);
}

static PyObject *MathHolderObject_get_trs_batch(MathHolderObject *self, PyObject *const *args,
                                                Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *pos_obj;
    PyObject *rot_obj;
    PyObject *scale_obj;
    void *targets[MathTrio_COUNT] = {[IDX_MT_0] = (void *)&pos_obj,
                                     [IDX_MT_1] = (void *)&rot_obj,
                                     [IDX_MT_2] = (void *)&scale_obj};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->MathTrioParser, targets)) {
        return nullptr;
    }

    Py_buffer p_buf;
    Py_buffer r_buf;
    Py_buffer s_buf;
    if (PyObject_GetBuffer(pos_obj, &p_buf, PyBUF_SIMPLE) < 0) {
        return nullptr;
    }
    if (PyObject_GetBuffer(rot_obj, &r_buf, PyBUF_SIMPLE) < 0) {
        PyBuffer_Release(&p_buf);
        return nullptr;
    }
    if (PyObject_GetBuffer(scale_obj, &s_buf, PyBUF_SIMPLE) < 0) {
        PyBuffer_Release(&p_buf);
        PyBuffer_Release(&r_buf);
        return nullptr;
    }

    Py_ssize_t count = p_buf.len / (3 * (Py_ssize_t)sizeof(float));
    PyObject *result = PyBytes_FromStringAndSize(nullptr, count * 16 * (Py_ssize_t)sizeof(float));
    if (result) {
        culverin_math_get_trs_batch(count, (float *)p_buf.buf, (float *)r_buf.buf,
                                    (float *)s_buf.buf, (float *)PyBytes_AsString(result));
    }

    PyBuffer_Release(&p_buf);
    PyBuffer_Release(&r_buf);
    PyBuffer_Release(&s_buf);
    return result;
}

void culverin_math_init_all_parsers(MathParsers *mp);
void culverin_math_free_all_parsers(MathParsers *mp);

static PyObject *MathHolderObject_new(PyTypeObject *type, CULV_MAYBE_UNUSED PyObject *args,
                                      CULV_MAYBE_UNUSED PyObject *kwds) {
    MathHolderObject *self = (MathHolderObject *)type->tp_alloc(type, 0);
    if (self == nullptr) {
        return nullptr;
    }
    self->parsers = (MathParsers *)PyMem_Malloc(sizeof(MathParsers));
    if (self->parsers == nullptr) {
        Py_DECREF(self);
        return PyErr_NoMemory();
    }

    // 3. Call your parser setup logic
    culverin_math_init_all_parsers(self->parsers);

    return (PyObject *)self;
}

static void MathHolderObject_dealloc(MathHolderObject *self) {
    if (self->parsers != nullptr) {
        culverin_math_free_all_parsers(self->parsers);
        PyMem_Free(self->parsers);
    }
    PyTypeObject *tp = Py_TYPE(self);
    tp->tp_free((PyObject *)self);
    Py_DECREF(tp);
}

// ============================================================================
// TYPE DEFINITION
// ============================================================================

#define MATH_FASTCALL(name) CULV_FEAT(MathHolderObject, name, METH_FASTCALL | METH_KEYWORDS)

PyType_Spec MathService_spec = {
    .name      = "culverin._culverin_c.MathService",
    .basicsize = sizeof(struct MathHolderObject),
    .flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .slots =
        (PyType_Slot[]){

            // 1. Add the Constructor logic
            {.slot = Py_tp_new, .pfunc = (void *)MathHolderObject_new},

            // 2. Add the Destructor logic
            {.slot = Py_tp_dealloc, .pfunc = (void *)MathHolderObject_dealloc},

            {.slot = Py_tp_methods,
             .pfunc =
                 (PyMethodDef[]){

                     MATH_FASTCALL(get_perspective),
                     MATH_FASTCALL(get_ortho),
                     MATH_FASTCALL(get_look_at),
                     MATH_FASTCALL(get_trs),
                     MATH_FASTCALL(get_trs_batch),
                     {},

                 }

            },
            {}

        },
};