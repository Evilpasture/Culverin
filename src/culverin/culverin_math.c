#include "culverin_math.h"
#include "culverin_arg_indices.h"
#include "culverin_compiler_specifics.h"
#include "culverin_python.h"
#include "fast_build.h"
#include "fast_parse.h"
#include <Python.h>

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

static inline bool unpack_quat(PyObject *obj, float *out) {
    if (!PyTuple_Check(obj) || PyTuple_GET_SIZE(obj) < 4) {
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        out[i] = (float)PyFloat_AsDouble(PyTuple_GET_ITEM(obj, i));
    }
    return (bool)!PyErr_Occurred();
}

static inline bool unpack_vec3(PyObject *obj, float *out) {
    if (!PyTuple_Check(obj) || PyTuple_GET_SIZE(obj) < 3) {
        return false;
    }
    for (int i = 0; i < 3; ++i) {
        out[i] = (float)PyFloat_AsDouble(PyTuple_GET_ITEM(obj, i));
    }
    return (bool)!PyErr_Occurred();
}

static inline bool unpack_viewport(PyObject *obj, int *out) {
    if (!PyTuple_Check(obj) || PyTuple_GET_SIZE(obj) < 4) {
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        out[i] = (int)PyLong_AsLong(PyTuple_GET_ITEM(obj, i));
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
    void *targets[MathPersp_COUNT] = {[IDX_MP_FOVY]   = (void *)&fovy,
                                      [IDX_MP_ASPECT] = (void *)&aspect,
                                      [IDX_MP_NEAR]   = (void *)&near_p,
                                      [IDX_MP_FAR]    = (void *)&far_p};

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
        [IDX_MO_LEFT] = (void *)&left,     [IDX_MO_RIGHT] = (void *)&right,
        [IDX_MO_BOTTOM] = (void *)&bottom, [IDX_MO_TOP] = (void *)&top,
        [IDX_MO_NEAR] = (void *)&near_p,   [IDX_MO_FAR] = (void *)&far_p};

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
    void *targets[MathLookAt_COUNT] = {[IDX_ML_EYE]    = (void *)&eye_obj,
                                       [IDX_ML_TARGET] = (void *)&target_obj,
                                       [IDX_ML_UP]     = (void *)&up_obj};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->MathLookAtParser, targets)) {
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
    void *targets[MathTRS_COUNT] = {[IDX_MTRS_T] = (void *)&pos_obj,
                                    [IDX_MTRS_R] = (void *)&rot_obj,
                                    [IDX_MTRS_S] = (void *)&scale_obj};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &self->parsers->MathTRSParser,
                           targets)) {
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
    void *targets[MathTRSBatch_COUNT] = {[IDX_MTRSB_T] = (void *)&pos_obj,
                                         [IDX_MTRSB_R] = (void *)&rot_obj,
                                         [IDX_MTRSB_S] = (void *)&scale_obj};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->MathTRSBatchParser, targets)) {
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
    void *targets[MathMatVec_COUNT] = {[IDX_MMV_MAT] = (void *)&mat_obj,
                                       [IDX_MMV_VEC] = (void *)&vec_obj};

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
    void *targets[MathMatBatch_COUNT] = {[IDX_MMB_MAT]   = (void *)&mat_obj,
                                         [IDX_MMB_BATCH] = (void *)&batch_obj};

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
    void *targets[MathCullBatch_COUNT] = {[IDX_MCB_VP]    = (void *)&vp_obj,
                                          [IDX_MCB_AABBS] = (void *)&aabbs_obj};

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

static PyObject *MathHolderObject_vec3_normalize_batch(MathHolderObject *self,
                                                       PyObject *const *args, Py_ssize_t nargsf,
                                                       PyObject *kwnames) {
    PyObject *vecs_obj;
    void *targets[MathVec3Batch_COUNT] = {[IDX_MVB_VECS] = (void *)&vecs_obj};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->MathVec3BatchParser, targets)) {
        return nullptr;
    }

    Py_buffer view;
    if (PyObject_GetBuffer(vecs_obj, &view, PyBUF_SIMPLE) < 0) {
        return nullptr; // PyErr_SetString already called by GetBuffer
    }

    // Verify we have enough bytes for float3 vectors
    const size_t vec3_bytes = 3 * sizeof(float);
    if (view.len % vec3_bytes != 0) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_ValueError, "Buffer length must be a multiple of 12 (3 floats)");
        return nullptr;
    }

    size_t count     = view.len / vec3_bytes;
    PyObject *result = PyBytes_FromStringAndSize(NULL, view.len);

    if (result) {
        culverin_math_vec3_normalize_batch((const float *)view.buf, count,
                                           (float *)PyBytes_AsString(result));
    }

    PyBuffer_Release(&view);
    return result;
}

static PyObject *MathHolderObject_quat_from_euler(MathHolderObject *self, PyObject *const *args,
                                                  Py_ssize_t nargsf, PyObject *kwnames) {
    float x;
    float y;
    float z;
    void *targets[MathEuler_COUNT] = {
        [IDX_ME_X] = (void *)&x, [IDX_ME_Y] = (void *)&y, [IDX_ME_Z] = (void *)&z};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->MathEulerParser, targets)) {
        return nullptr;
    }

    float out[4]; // quat: x, y, z, w
    culverin_math_quat_from_euler(x, y, z, out);

    // FastBuild_Tuple is optimized for returning fixed-size math results
    return FastBuild_Tuple(out[0], out[1], out[2], out[3]);
}

static PyObject *MathHolderObject_quat_to_euler(MathHolderObject *self, PyObject *const *args,
                                                Py_ssize_t nargsf, PyObject *kwnames) {
    float x;
    float y;
    float z;
    float w;
    void *targets[MathQuat_COUNT] = {[IDX_MQ_X] = (void *)&x,
                                     [IDX_MQ_Y] = (void *)&y,
                                     [IDX_MQ_Z] = (void *)&z,
                                     [IDX_MQ_W] = (void *)&w};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->MathQuatParser, targets)) {
        return nullptr;
    }

    float in_q[4] = {x, y, z, w};
    float out_euler[3];

    culverin_math_quat_to_euler(in_q, out_euler);

    return FastBuild_Tuple(out_euler[0], out_euler[1], out_euler[2]);
}

static PyObject *MathHolderObject_quat_slerp(MathHolderObject *self, PyObject *const *args,
                                             Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *q1_obj;
    PyObject *q2_obj;
    float t;
    void *targets[MathSlerp_COUNT] = {
        [IDX_MS_Q1] = (void *)&q1_obj, [IDX_MS_Q2] = (void *)&q2_obj, [IDX_MS_T] = (void *)&t};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->MathSlerpParser, targets)) {
        return nullptr;
    }

    float q1[4];
    float q2[4];
    float out[4];
    if (!unpack_quat(q1_obj, q1) || !unpack_quat(q2_obj, q2)) {
        PyErr_SetString(PyExc_TypeError, "Quaternions must be tuples of 4 floats");
        return nullptr;
    }

    culverin_math_quat_slerp(q1, q2, t, out);

    return FastBuild_Tuple(out[0], out[1], out[2], out[3]);
}

static PyObject *MathHolderObject_quat_mul(MathHolderObject *self, PyObject *const *args,
                                           Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *a_obj;
    PyObject *b_obj;
    void *targets[MathQuatPair_COUNT] = {[IDX_MQP_A] = (void *)&a_obj,
                                         [IDX_MQP_B] = (void *)&b_obj};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->MathQuatPairParser, targets)) {
        return nullptr;
    }

    float a[4];
    float b[4];
    float out[4];
    if (!unpack_quat(a_obj, a) || !unpack_quat(b_obj, b)) {
        PyErr_SetString(PyExc_TypeError, "Quaternions must be tuples of 4 floats");
        return nullptr;
    }

    culverin_math_quat_mul(a, b, out);

    return FastBuild_Tuple(out[0], out[1], out[2], out[3]);
}

static PyObject *MathHolderObject_vec3_lerp_batch(MathHolderObject *self, PyObject *const *args,
                                                  Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *a_obj;
    PyObject *b_obj;
    float alpha;
    void *targets[MathLerpBatch_COUNT] = {[IDX_MLB_VECS_A] = (void *)&a_obj,
                                          [IDX_MLB_VECS_B] = (void *)&b_obj,
                                          [IDX_MLB_ALPHA]  = (void *)&alpha};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->MathLerpBatchParser, targets)) {
        return nullptr;
    }

    Py_buffer view_a;
    Py_buffer view_b;
    if (PyObject_GetBuffer(a_obj, &view_a, PyBUF_SIMPLE) < 0) {
        return nullptr;
    }
    if (PyObject_GetBuffer(b_obj, &view_b, PyBUF_SIMPLE) < 0) {
        PyBuffer_Release(&view_a);
        return nullptr;
    }

    if (view_a.len != view_b.len || (view_a.len % 12 != 0)) {
        PyBuffer_Release(&view_a);
        PyBuffer_Release(&view_b);
        PyErr_SetString(PyExc_ValueError, "Buffers must be equal size and multiples of 12 bytes");
        return nullptr;
    }

    size_t count     = view_a.len / 12;
    PyObject *result = PyBytes_FromStringAndSize(NULL, view_a.len);

    if (result) {
        culverin_math_vec3_lerp_batch((const float *)view_a.buf, (const float *)view_b.buf, alpha,
                                      count, (float *)PyBytes_AsString(result));
    }

    PyBuffer_Release(&view_a);
    PyBuffer_Release(&view_b);
    return result;
}

static PyObject *MathHolderObject_quat_rotate_vec3(MathHolderObject *self, PyObject *const *args,
                                                   Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *q_obj;
    PyObject *v_obj;
    void *targets[MathQuatVec_COUNT] = {[IDX_MQV_Q] = (void *)&q_obj, [IDX_MQV_V] = (void *)&v_obj};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->MathQuatVecParser, targets)) {
        return nullptr;
    }

    float q[4];
    float v[3];
    float out[3];
    if (!unpack_quat(q_obj, q) || !unpack_vec3(v_obj, v)) {
        PyErr_SetString(PyExc_TypeError, "Inputs must be tuples of floats (Quat=4, Vec3=3)");
        return nullptr;
    }

    culverin_math_quat_rotate_vec3(q, v, out);

    return FastBuild_Tuple(out[0], out[1], out[2]);
}

static PyObject *MathHolderObject_quat_rotate_vec3_batch(MathHolderObject *self,
                                                         PyObject *const *args, Py_ssize_t nargsf,
                                                         PyObject *kwnames) {
    PyObject *q_obj;
    PyObject *vecs_obj;
    void *targets[MathQuatVecBatch_COUNT] = {[IDX_MQVB_Q]    = (void *)&q_obj,
                                             [IDX_MQVB_VECS] = (void *)&vecs_obj};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->MathQuatVecBatchParser, targets)) {
        return nullptr;
    }

    float q[4];
    if (!unpack_quat(q_obj, q)) {
        PyErr_SetString(PyExc_TypeError, "Quaternion must be a tuple of 4 floats");
        return nullptr;
    }

    Py_buffer view;
    if (PyObject_GetBuffer(vecs_obj, &view, PyBUF_SIMPLE) < 0) {
        return nullptr;
    }

    if (view.len % 12 != 0) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_ValueError,
                        "Vector buffer must be a multiple of 12 bytes (3 floats)");
        return nullptr;
    }

    size_t count     = view.len / 12;
    PyObject *result = PyBytes_FromStringAndSize(NULL, view.len);

    if (result) {
        culverin_math_quat_rotate_vec3_batch(q, (const float *)view.buf, count,
                                             (float *)PyBytes_AsString(result));
    }

    PyBuffer_Release(&view);
    return result;
}

static PyObject *MathHolderObject_quat_inverse(MathHolderObject *self, PyObject *const *args,
                                               Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *q_obj;
    void *targets[MathQuatOp_COUNT] = {[IDX_MQO_Q] = (void *)&q_obj};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->MathQuatOpParser, targets)) {
        return nullptr;
    }

    float q[4];
    float out[4];
    if (!unpack_quat(q_obj, q)) {
        PyErr_SetString(PyExc_TypeError, "Quaternion must be a tuple of 4 floats");
        return nullptr;
    }

    culverin_math_quat_inverse(q, out);

    return FastBuild_Tuple(out[0], out[1], out[2], out[3]);
}

static PyObject *MathHolderObject_project(MathHolderObject *self, PyObject *const *args,
                                          Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *v_obj;
    PyObject *mvp_obj;
    PyObject *vp_obj;
    void *targets[MathProject_COUNT] = {[IDX_MPR_V]   = (void *)&v_obj,
                                        [IDX_MPR_MVP] = (void *)&mvp_obj,
                                        [IDX_MPR_VP]  = (void *)&vp_obj};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->MathProjectParser, targets)) {
        return nullptr;
    }

    float v[3];
    float mvp[16];
    float out[3];
    int viewport[4];

    if (!unpack_vec3(v_obj, v) || !unpack_mat44(mvp_obj, mvp) ||
        !unpack_viewport(vp_obj, viewport)) {
        PyErr_SetString(PyExc_TypeError,
                        "Invalid arguments for project (Vec3, Mat44, ViewportTuple)");
        return nullptr;
    }

    culverin_math_project(v, mvp, viewport, out);

    return FastBuild_Tuple(out[0], out[1], out[2]);
}

static PyObject *MathHolderObject_unproject(MathHolderObject *self, PyObject *const *args,
                                            Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *v_obj;
    PyObject *mvp_obj;
    PyObject *vp_obj;
    void *targets[MathUnproject_COUNT] = {[IDX_MUP_V]   = (void *)&v_obj,
                                          [IDX_MUP_MVP] = (void *)&mvp_obj,
                                          [IDX_MUP_VP]  = (void *)&vp_obj};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->MathUnprojectParser, targets)) {
        return nullptr;
    }

    float v[3];
    float mvp[16];
    float out[3];
    int viewport[4];

    if (!unpack_vec3(v_obj, v) || !unpack_mat44(mvp_obj, mvp) ||
        !unpack_viewport(vp_obj, viewport)) {
        PyErr_SetString(PyExc_TypeError,
                        "Invalid arguments for unproject (Vec3, Mat44, ViewportTuple)");
        return nullptr;
    }

    culverin_math_unproject(v, mvp, viewport, out);

    return FastBuild_Tuple(out[0], out[1], out[2]);
}

static PyObject *MathHolderObject_quat_from_to(MathHolderObject *self, PyObject *const *args,
                                               Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *v1_obj;
    PyObject *v2_obj;
    void *targets[MathVecPair_COUNT] = {[IDX_MVP_V1] = (void *)&v1_obj,
                                        [IDX_MVP_V2] = (void *)&v2_obj};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->MathVecPairParser, targets)) {
        return nullptr;
    }

    float v1[3];
    float v2[3];
    float out[4];
    if (!unpack_vec3(v1_obj, v1) || !unpack_vec3(v2_obj, v2)) {
        PyErr_SetString(PyExc_TypeError, "v1 and v2 must be tuples of 3 floats");
        return nullptr;
    }

    culverin_math_quat_from_to(v1, v2, out);

    return FastBuild_Tuple(out[0], out[1], out[2], out[3]);
}

static PyObject *MathHolderObject_vec3_dot(MathHolderObject *self, PyObject *const *args,
                                           Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *v1_obj;
    PyObject *v2_obj;
    void *targets[MathVecPair_COUNT] = {[IDX_MVP_V1] = (void *)&v1_obj,
                                        [IDX_MVP_V2] = (void *)&v2_obj};

    // Reusing the parser from quat_from_to
    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->MathVecPairParser, targets)) {
        return nullptr;
    }

    float v1[3];
    float v2[3];
    if (!unpack_vec3(v1_obj, v1) || !unpack_vec3(v2_obj, v2)) {
        PyErr_SetString(PyExc_TypeError, "v1 and v2 must be tuples of 3 floats");
        return nullptr;
    }

    float result = culverin_math_vec3_dot(v1, v2);

    return PyFloat_FromDouble((double)result);
}

static PyObject *MathHolderObject_vec3_cross(MathHolderObject *self, PyObject *const *args,
                                             Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *v1_obj;
    PyObject *v2_obj;
    void *targets[MathVecPair_COUNT] = {[IDX_MVP_V1] = (void *)&v1_obj,
                                        [IDX_MVP_V2] = (void *)&v2_obj};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->MathVecPairParser, targets)) {
        return nullptr;
    }

    float v1[3];
    float v2[3];
    float out[3];
    if (!unpack_vec3(v1_obj, v1) || !unpack_vec3(v2_obj, v2)) {
        PyErr_SetString(PyExc_TypeError, "v1 and v2 must be tuples of 3 floats");
        return nullptr;
    }

    culverin_math_vec3_cross(v1, v2, out);

    return FastBuild_Tuple(out[0], out[1], out[2]);
}

static PyObject *MathHolderObject_intersect_ray_plane(MathHolderObject *self, PyObject *const *args,
                                                      Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *ro_obj;
    PyObject *rd_obj;
    PyObject *po_obj;
    PyObject *pn_obj;
    void *targets[MathRayPlane_COUNT] = {[IDX_RP_RO] = (void *)&ro_obj,
                                         [IDX_RP_RD] = (void *)&rd_obj,
                                         [IDX_RP_PO] = (void *)&po_obj,
                                         [IDX_RP_PN] = (void *)&pn_obj};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->MathRayPlaneParser, targets)) {
        return nullptr;
    }

    float ro[3];
    float rd[3];
    float po[3];
    float pn[3];
    float out_p[3];
    float out_t;
    if (!unpack_vec3(ro_obj, ro) || !unpack_vec3(rd_obj, rd) || !unpack_vec3(po_obj, po) ||
        !unpack_vec3(pn_obj, pn)) {
        return nullptr;
    }

    int hit = culverin_math_intersect_ray_plane(ro, rd, po, pn, &out_t, out_p);

    if (!hit) {
        return FastBuild_Tuple(Py_False, 0.0f, Py_None);
    }

    return FastBuild_Tuple(Py_True, out_t, FastBuild_Tuple(out_p[0], out_p[1], out_p[2]));
}

static PyObject *MathHolderObject_quat_get_axis_angle(MathHolderObject *self, PyObject *const *args,
                                                      Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *q_obj;
    // Reusing MathQuatOp which has [IDX_MQO_Q]
    void *targets[MathQuatOp_COUNT] = {[IDX_MQO_Q] = (void *)&q_obj};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->MathQuatOpParser, targets)) {
        return nullptr;
    }

    float q[4];
    float axis[3];
    float angle;
    if (!unpack_quat(q_obj, q)) {
        PyErr_SetString(PyExc_TypeError, "Quaternion must be a tuple of 4 floats");
        return nullptr;
    }

    culverin_math_quat_get_axis_angle(q, axis, &angle);

    return FastBuild_Tuple(FastBuild_Tuple(axis[0], axis[1], axis[2]), angle);
}

static PyObject *MathHolderObject_quat_from_axis_angle(MathHolderObject *self,
                                                       PyObject *const *args, Py_ssize_t nargsf,
                                                       PyObject *kwnames) {
    PyObject *axis_obj;
    float angle;
    void *targets[MathAxisAngle_COUNT] = {[IDX_MAA_AXIS]  = (void *)&axis_obj,
                                          [IDX_MAA_ANGLE] = (void *)&angle};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->MathAxisAngleParser, targets)) {
        return nullptr;
    }

    float axis[3];
    float out[4];
    if (!unpack_vec3(axis_obj, axis)) {
        PyErr_SetString(PyExc_TypeError, "Axis must be a tuple of 3 floats");
        return nullptr;
    }

    culverin_math_quat_from_axis_angle(axis, angle, out);

    return FastBuild_Tuple(out[0], out[1], out[2], out[3]);
}

static PyObject *MathHolderObject_vec3_distance_batch(MathHolderObject *self, PyObject *const *args,
                                                      Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *a_obj;
    PyObject *b_obj;
    void *targets[MathDistBatch_COUNT] = {[IDX_MDB_VECS_A] = (void *)&a_obj,
                                          [IDX_MDB_VECS_B] = (void *)&b_obj};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->MathDistBatchParser, targets)) {
        return nullptr;
    }

    Py_buffer view_a;
    Py_buffer view_b;
    if (PyObject_GetBuffer(a_obj, &view_a, PyBUF_SIMPLE) < 0) {
        return nullptr;
    }
    if (PyObject_GetBuffer(b_obj, &view_b, PyBUF_SIMPLE) < 0) {
        PyBuffer_Release(&view_a);
        return nullptr;
    }

    if (view_a.len != view_b.len || (view_a.len % 12 != 0)) {
        PyBuffer_Release(&view_a);
        PyBuffer_Release(&view_b);
        PyErr_SetString(PyExc_ValueError, "Buffers must be equal size and multiples of 12 bytes");
        return nullptr;
    }

    size_t count = view_a.len / 12;
    // Result is a flat array of floats (4 bytes per distance)
    PyObject *result = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)(count * sizeof(float)));

    if (result) {
        culverin_math_vec3_distance_batch((const float *)view_a.buf, (const float *)view_b.buf,
                                          count, (float *)PyBytes_AsString(result));
    }

    PyBuffer_Release(&view_a);
    PyBuffer_Release(&view_b);
    return result;
}

static PyObject *MathHolderObject_vec3_normalize(MathHolderObject *self, PyObject *const *args,
                                                 Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *v_obj;
    void *targets[MathVecOp_COUNT] = {[IDX_MVO_V] = (void *)&v_obj};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->MathVecOpParser, targets)) {
        return nullptr;
    }

    float v[3], out[3];
    if (!unpack_vec3(v_obj, v)) {
        PyErr_SetString(PyExc_TypeError, "v must be a tuple of 3 floats");
        return nullptr;
    }

    culverin_math_vec3_normalize(v, out);

    return FastBuild_Tuple(out[0], out[1], out[2]);
}

static PyObject *MathHolderObject_mat44_get_translation(MathHolderObject *self,
                                                        PyObject *const *args, Py_ssize_t nargsf,
                                                        PyObject *kwnames) {
    PyObject *mat_obj;
    void *targets[MathMat_COUNT] = {[IDX_MMM_MAT] = (void *)&mat_obj};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &self->parsers->MathMatParser,
                           targets)) {
        return nullptr;
    }

    alignas(simd_alignment) float in_mat[sixteen_floats];
    if (!unpack_mat44(mat_obj, in_mat)) {
        PyErr_SetString(PyExc_TypeError, "Matrix must be a tuple of 16 floats");
        return nullptr;
    }

    float out_vec[3];
    culverin_math_mat44_get_translation(in_mat, out_vec);

    return FastBuild_Tuple(out_vec[0], out_vec[1], out_vec[2]);
}

static PyObject *MathHolderObject_mat44_get_rotation(MathHolderObject *self, PyObject *const *args,
                                                     Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *mat_obj;
    void *targets[MathMat_COUNT] = {[IDX_MMM_MAT] = (void *)&mat_obj};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &self->parsers->MathMatParser,
                           targets)) {
        return nullptr;
    }

    alignas(simd_alignment) float in_mat[sixteen_floats];
    if (!unpack_mat44(mat_obj, in_mat)) {
        PyErr_SetString(PyExc_TypeError, "Matrix must be a tuple of 16 floats");
        return nullptr;
    }

    float out_quat[4];
    culverin_math_mat44_get_rotation(in_mat, out_quat);

    return FastBuild_Tuple(out_quat[0], out_quat[1], out_quat[2], out_quat[3]);
}

static PyObject *MathHolderObject_mat44_identity(CULV_MAYBE_UNUSED MathHolderObject *self,
                                                 PyObject *Py_UNUSED(ignored)) {
    float out[sixteen_floats];
    culverin_math_mat44_identity(out);

    return FastBuild_Tuple(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8],
                           out[9], out[10], out[11], out[12], out[13], out[14], out[15]);
}

static PyObject *MathHolderObject_vec3_reflect(MathHolderObject *self, PyObject *const *args,
                                               Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *v_obj, *n_obj;
    void *targets[MathReflect_COUNT] = {[IDX_MRF_V] = (void *)&v_obj, [IDX_MRF_N] = (void *)&n_obj};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->MathReflectParser, targets)) {
        return nullptr;
    }

    float v[3], n[3], out[3];
    if (!unpack_vec3(v_obj, v) || !unpack_vec3(n_obj, n)) {
        PyErr_SetString(PyExc_TypeError, "v and normal must be tuples of 3 floats");
        return nullptr;
    }

    culverin_math_vec3_reflect(v, n, out);

    return FastBuild_Tuple(out[0], out[1], out[2]);
}

static PyObject *MathHolderObject_vec3_distance(MathHolderObject *self, PyObject *const *args,
                                                Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *v1_obj, *v2_obj;
    void *targets[MathVecPair_COUNT] = {[IDX_MVP_V1] = (void *)&v1_obj,
                                        [IDX_MVP_V2] = (void *)&v2_obj};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->MathVecPairParser, targets)) {
        return nullptr;
    }

    float v1[3], v2[3];
    if (!unpack_vec3(v1_obj, v1) || !unpack_vec3(v2_obj, v2)) {
        PyErr_SetString(PyExc_TypeError, "v1 and v2 must be tuples of 3 floats");
        return nullptr;
    }

    float dist = culverin_math_vec3_distance(v1, v2);

    return PyFloat_FromDouble((double)dist);
}

static PyObject *MathHolderObject_quat_rotate_vec3_inverse(MathHolderObject *self,
                                                           PyObject *const *args, Py_ssize_t nargsf,
                                                           PyObject *kwnames) {
    PyObject *q_obj, *v_obj;
    void *targets[MathQuatVec_COUNT] = {[IDX_MQV_Q] = (void *)&q_obj, [IDX_MQV_V] = (void *)&v_obj};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->MathQuatVecParser, targets)) {
        return nullptr;
    }

    float q[4], v[3], out[3];
    if (!unpack_quat(q_obj, q) || !unpack_vec3(v_obj, v)) {
        PyErr_SetString(PyExc_TypeError, "Inputs must be tuples of floats (Quat=4, Vec3=3)");
        return nullptr;
    }

    culverin_math_quat_rotate_vec3_inverse(q, v, out);

    return FastBuild_Tuple(out[0], out[1], out[2]);
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
#define MATH_NOARGS(name) CULV_FEAT(MathHolderObject, name, METH_NOARGS)

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
                     MATH_FASTCALL(vec3_normalize_batch),
                     MATH_FASTCALL(quat_from_euler),
                     MATH_FASTCALL(quat_to_euler),
                     MATH_FASTCALL(quat_slerp),
                     MATH_FASTCALL(quat_mul),
                     MATH_FASTCALL(vec3_lerp_batch),
                     MATH_FASTCALL(quat_rotate_vec3),
                     MATH_FASTCALL(quat_rotate_vec3_batch),
                     MATH_FASTCALL(quat_inverse),
                     MATH_FASTCALL(project),
                     MATH_FASTCALL(unproject),
                     MATH_FASTCALL(quat_from_to),
                     MATH_FASTCALL(vec3_dot),
                     MATH_FASTCALL(vec3_cross),
                     MATH_FASTCALL(intersect_ray_plane),
                     MATH_FASTCALL(quat_get_axis_angle),
                     MATH_FASTCALL(quat_from_axis_angle),
                     MATH_FASTCALL(vec3_distance_batch),
                     MATH_FASTCALL(vec3_normalize),
                     MATH_FASTCALL(mat44_get_translation),
                     MATH_FASTCALL(mat44_get_rotation),
                     MATH_NOARGS(mat44_identity),
                     MATH_FASTCALL(vec3_reflect),
                     MATH_FASTCALL(vec3_distance),
                     MATH_FASTCALL(quat_rotate_vec3_inverse),
                     {},

                 }

            },
            {}

        },
};