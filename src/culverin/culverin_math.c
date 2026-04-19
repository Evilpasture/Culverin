#include "culverin_math.h"
#include "culverin_arg_indices.h"
#include "culverin_python.h"
#include "fast_build.h"

typedef struct MathHolderObject {
    PyObject_HEAD MathParsers *parsers;
} MathHolderObject;
static constexpr int sixteen_floats = 16;
static constexpr int simd_alignment = 16;
// Helper for extracting 16 floats from a Python Tuple
static inline bool unpack_mat44(PyObject *obj, float *out) {

    if (!PyTuple_Check(obj) || PyTuple_GET_SIZE(obj) < sixteen_floats) {
        return false;
    }
    for (int i = 0; i < sixteen_floats; ++i) {
        out[i] = (float)PyFloat_AsDouble(PyTuple_GET_ITEM(obj, i));
    }
    return (bool)!PyErr_Occurred();
}

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

    float out[sixteen_floats];
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
    void *targets[MathOrtho_COUNT] = {
        [IDX_MO_LEFT] = &left, [IDX_MO_RIGHT] = &right, [IDX_MO_BOTTOM] = &bottom,
        [IDX_MO_TOP] = &top,   [IDX_MO_NEAR] = &near_p, [IDX_MO_FAR] = &far_p};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->MathOrthoParser, targets)) {
        return nullptr;
    }

    float out[sixteen_floats];
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
    alignas(simd_alignment) float eye[4]    = {0};
    alignas(simd_alignment) float target[4] = {0};
    alignas(simd_alignment) float up[4]     = {0};

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

    alignas(simd_alignment) float out[sixteen_floats];
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
    alignas(simd_alignment) float p[4];
    alignas(simd_alignment) float r[4];
    alignas(simd_alignment) float s[4];

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

    alignas(simd_alignment) float out[sixteen_floats];
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
    PyObject *result =
        PyBytes_FromStringAndSize(nullptr, count * sixteen_floats * (Py_ssize_t)sizeof(float));
    if (result) {
        culverin_math_get_trs_batch(count, (float *)p_buf.buf, (float *)r_buf.buf,
                                    (float *)s_buf.buf, (float *)PyBytes_AsString(result));
    }

    PyBuffer_Release(&p_buf);
    PyBuffer_Release(&r_buf);
    PyBuffer_Release(&s_buf);
    return result;
}

// 1. Matrix Inverse
static PyObject *MathHolderObject_inverse(MathHolderObject *self, PyObject *const *args,
                                          Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *mat_obj;
    void *targets[MathMat_COUNT] = {[IDX_MMM_MAT] = (void *)&mat_obj};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &self->parsers->MathMatParser,
                           targets)) {
        return nullptr;
    }

    alignas(simd_alignment) float in[sixteen_floats];
    alignas(simd_alignment) float out[sixteen_floats];
    if (!unpack_mat44(mat_obj, in)) {
        PyErr_SetString(PyExc_TypeError, "Matrix must be a tuple of 16 floats");
        return nullptr;
    }

    culverin_math_mat44_inverse(in, out);
    return FastBuild_Tuple(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8],
                           out[9], out[10], out[11], out[12], out[13], out[14], out[15]);
}

// 2. Matrix Multiplication (A * B)
static PyObject *MathHolderObject_matmul(MathHolderObject *self, PyObject *const *args,
                                         Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *a_obj;
    PyObject *b_obj;
    void *targets[MathMatPair_COUNT] = {[IDX_MMP_A] = (void *)&a_obj, [IDX_MMP_B] = (void *)&b_obj};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->MathMatPairParser, targets)) {
        return nullptr;
    }

    alignas(simd_alignment) float a[sixteen_floats];
    alignas(simd_alignment) float b[sixteen_floats];
    alignas(simd_alignment) float out[sixteen_floats];
    if (!unpack_mat44(a_obj, a) || !unpack_mat44(b_obj, b)) {
        return nullptr;
    }

    culverin_math_mat44_mul(a, b, out);
    return FastBuild_Tuple(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8],
                           out[9], out[10], out[11], out[12], out[13], out[14], out[15]);
}

// 3. Vector Transform (Mat * Vec3)
static PyObject *MathHolderObject_transform_vec3(MathHolderObject *self, PyObject *const *args,
                                                 Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *mat_obj;
    PyObject *vec_obj;
    void *targets[MathMatVec_COUNT] = {
        [IDX_MMV_MAT] = (void *)&mat_obj, [IDX_MMV_VEC] = (void *)&vec_obj};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->MathMatVecParser, targets)) {
        return nullptr;
    }

    // Standard engine constants for readability or use raw numbers
    alignas(simd_alignment) float m[sixteen_floats];
    if (!unpack_mat44(mat_obj, m)) {
        PyErr_SetString(PyExc_TypeError, "Matrix must be a tuple of 16 floats");
        return nullptr;
    }

    if (!PyTuple_Check(vec_obj) || PyTuple_GET_SIZE(vec_obj) < 3) {
        PyErr_SetString(PyExc_ValueError, "Vector must be a tuple of at least 3 floats");
        return nullptr;
    }

    float v[3] = {(float)PyFloat_AsDouble(PyTuple_GET_ITEM(vec_obj, 0)),
                  (float)PyFloat_AsDouble(PyTuple_GET_ITEM(vec_obj, 1)),
                  (float)PyFloat_AsDouble(PyTuple_GET_ITEM(vec_obj, 2))};

    // Sanity check for float conversion errors
    if (PyErr_Occurred()) {
        return nullptr;
    }

    float out[3];
    culverin_math_transform_vec3(m, v, out);

    // FastBuild is significantly faster than Py_BuildValue
    return FastBuild_Tuple(out[0], out[1], out[2]);
}

// 4. Batch Matrix Multiply (Single * Batch)
// Efficient for: MVP_Batch = VP_Matrix * Model_Matrix_Bytes
static PyObject *MathHolderObject_matmul_batch(MathHolderObject *self, PyObject *const *args,
                                               Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *mat_obj;
    PyObject *batch_obj;
    void *targets[MathMatBatch_COUNT] = {
        [IDX_MMB_MAT] = (void *)&mat_obj, [IDX_MMB_BATCH] = (void *)&batch_obj};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->MathMatBatchParser, targets)) {
        return nullptr;
    }

    alignas(simd_alignment) float m[sixteen_floats];
    if (!unpack_mat44(mat_obj, m)) {
        return nullptr;
    }

    Py_buffer view;
    if (PyObject_GetBuffer(batch_obj, &view, PyBUF_SIMPLE) < 0) {
        return nullptr;
    }

    size_t count     = view.len / (sixteen_floats * sizeof(float));
    PyObject *result = PyBytes_FromStringAndSize(nullptr, view.len);
    if (result) {
        culverin_math_mat44_mul_batch(m, (float *)view.buf, count,
                                      (float *)PyBytes_AsString(result));
    }

    PyBuffer_Release(&view);
    return result;
}

// 5. Single AABB Culling
static PyObject *MathHolderObject_cull_aabb(MathHolderObject *self, PyObject *const *args,
                                            Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *vp_obj;
    PyObject *min_obj;
    PyObject *max_obj;
    void *targets[MathCull_COUNT] = {[IDX_MC_VP]  = (void *)&vp_obj,
                                     [IDX_MC_MIN] = (void *)&min_obj,
                                     [IDX_MC_MAX] = (void *)&max_obj};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->MathCullParser, targets)) {
        return nullptr;
    }

    alignas(simd_alignment) float vp[sixteen_floats];
    if (!unpack_mat44(vp_obj, vp)) {
        return nullptr;
    }

    // Fast unpack of 3-float tuples
    if (!PyTuple_Check(min_obj) || !PyTuple_Check(max_obj)) {
        return nullptr;
    }

    float a_min[3] = {(float)PyFloat_AsDouble(PyTuple_GET_ITEM(min_obj, 0)),
                      (float)PyFloat_AsDouble(PyTuple_GET_ITEM(min_obj, 1)),
                      (float)PyFloat_AsDouble(PyTuple_GET_ITEM(min_obj, 2))};
    float a_max[3] = {(float)PyFloat_AsDouble(PyTuple_GET_ITEM(max_obj, 0)),
                      (float)PyFloat_AsDouble(PyTuple_GET_ITEM(max_obj, 1)),
                      (float)PyFloat_AsDouble(PyTuple_GET_ITEM(max_obj, 2))};

    if (PyErr_Occurred()) {
        return nullptr;
    }

    int visible = culverin_math_cull_aabb(vp, a_min, a_max);
    return FastBuild_Value((bool)visible);
}

// 6. Batch AABB Culling (Essential for ECS Performance)
static PyObject *MathHolderObject_cull_aabb_batch(MathHolderObject *self, PyObject *const *args,
                                                  Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *vp_obj;
    PyObject *aabbs_obj;
    void *targets[MathCullBatch_COUNT] = {
        [IDX_MCB_VP] = (void *)&vp_obj, [IDX_MCB_AABBS] = (void *)&aabbs_obj};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->MathCullBatchParser, targets)) {
        return nullptr;
    }

    alignas(simd_alignment) float vp[sixteen_floats];
    if (!unpack_mat44(vp_obj, vp)) {
        return nullptr;
    }

    Py_buffer view;
    if (PyObject_GetBuffer(aabbs_obj, &view, PyBUF_SIMPLE) < 0) {
        return nullptr;
    }

    // aabb_data is expected as [minX, minY, minZ, maxX, maxY, maxZ, ...]
    Py_ssize_t count = view.len / (6 * (Py_ssize_t)sizeof(float));
    PyObject *result = PyByteArray_FromStringAndSize(nullptr, count);
    if (result) {
        culverin_math_cull_aabb_batch(vp, (float *)view.buf, count,
                                      (uint8_t *)PyByteArray_AsString(result));
    }

    PyBuffer_Release(&view);
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
                     MATH_FASTCALL(inverse),
                     MATH_FASTCALL(matmul),
                     MATH_FASTCALL(transform_vec3),
                     MATH_FASTCALL(matmul_batch),
                     MATH_FASTCALL(cull_aabb),
                     MATH_FASTCALL(cull_aabb_batch),
                     {},

                 }

            },
            {}

        },
};