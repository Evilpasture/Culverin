#include "culverin_math.h"
#include "culverin_arg_indices.h"
#include "culverin_python.h"
#include "fast_build.h"

typedef struct MathHolderObject {
    PyObject_HEAD
    MathParsers *parsers;
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

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &self->parsers->MathPerspParser,
                           targets)) {
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

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &self->parsers->MathOrthoParser,
                           targets)) {
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

    // Re-uses the MathTrio parser since LookAt, TRS, and Batch all share the exact same signature
    // (3 PyObject*)
    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &self->parsers->MathTrioParser,
                           targets)) {
        return nullptr;
    }

    float eye[3];
    float target[3];
    float up[3];
    for (int i = 0; i < 3; ++i) {
        eye[i]    = (float)PyFloat_AsDouble(PyTuple_GetItem(eye_obj, i));
        target[i] = (float)PyFloat_AsDouble(PyTuple_GetItem(target_obj, i));
        up[i]     = (float)PyFloat_AsDouble(PyTuple_GetItem(up_obj, i));
    }

    float out[16];
    culverin_math_get_look_at(eye, target, up, out);

    return FastBuild_Tuple(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8],
                           out[9], out[10], out[11], out[12], out[13], out[14], out[15]);
}

static PyObject *MathHolderObject_get_trs(MathHolderObject *self, PyObject *const *args, Py_ssize_t nargsf,
                                          PyObject *kwnames) {
    PyObject *pos_obj;
    PyObject *rot_obj;
    PyObject *scale_obj;
    void *targets[MathTrio_COUNT] = {[IDX_MT_0] = (void *)&pos_obj,
                                     [IDX_MT_1] = (void *)&rot_obj,
                                     [IDX_MT_2] = (void *)&scale_obj};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &self->parsers->MathTrioParser,
                           targets)) {
        return nullptr;
    }

    float p[3];
    float r[4];
    float s[3];
    for (int i = 0; i < 3; ++i) {
        p[i] = (float)PyFloat_AsDouble(PyTuple_GetItem(pos_obj, i));
    }
    for (int i = 0; i < 4; ++i) {
        r[i] = (float)PyFloat_AsDouble(PyTuple_GetItem(rot_obj, i));
    }
    for (int i = 0; i < 3; ++i) {
        s[i] = (float)PyFloat_AsDouble(PyTuple_GetItem(scale_obj, i));
    }

    float out[16];
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

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &self->parsers->MathTrioParser,
                           targets)) {
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

// ============================================================================
// TYPE DEFINITION
// ============================================================================


#define MATH_FASTCALL(name) CULV_FEAT(MathHolderObject, name, METH_FASTCALL | METH_KEYWORDS)

PyType_Spec MathService_spec = {
    .name      = "culverin._culverin_c.MathService",
    .basicsize = sizeof(struct MathHolderObject),
    .flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_IMMUTABLETYPE,
    .slots =
        (PyType_Slot[]){

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