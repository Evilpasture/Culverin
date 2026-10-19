#include "math/Math.hpp"
#include "bindings/FastBuild.hpp"
#include "bindings/PythonSupport.hpp"
#include "bindings/parsing/FastParser.hpp"
#include "core/CompilerSpecifics.h"
#include <Python.h>

namespace CPH::Python {

struct MathHolderObject {
    PyObject_HEAD CPH::Python::MathParsers *parsers;
};
static constexpr int kSixteenFloats = 16;
static constexpr int kSimdAlignment = 16;
// Helper for extracting 16 floats from a Python Tuple
static inline bool unpack_mat44(PyObject *obj, float *out) {

    if (!PyTuple_Check(obj) || PyTuple_GET_SIZE(obj) < kSixteenFloats) {
        return false;
    }
    for (int i = 0; i < kSixteenFloats; ++i) {
        out[i] = static_cast<float>(PyFloat_AsDouble(PyTuple_GET_ITEM(obj, i)));
    }
    return !PyErr_Occurred();
}

static inline bool unpack_quat(PyObject *obj, float *out) {
    if (!PyTuple_Check(obj) || PyTuple_GET_SIZE(obj) < 4) {
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        out[i] = static_cast<float>(PyFloat_AsDouble(PyTuple_GET_ITEM(obj, i)));
    }
    return !PyErr_Occurred();
}

static inline bool unpack_vec3(PyObject *obj, float *out) {
    if (!PyTuple_Check(obj) || PyTuple_GET_SIZE(obj) < 3) {
        return false;
    }
    for (int i = 0; i < 3; ++i) {
        out[i] = static_cast<float>(PyFloat_AsDouble(PyTuple_GET_ITEM(obj, i)));
    }
    return !PyErr_Occurred();
}

static inline bool unpack_viewport(PyObject *obj, int *out) {
    if (!PyTuple_Check(obj) || PyTuple_GET_SIZE(obj) < 4) {
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        out[i] = static_cast<int>(PyLong_AsLong(PyTuple_GET_ITEM(obj, i)));
    }
    return !PyErr_Occurred();
}

// ============================================================================
// FASTPARSE WRAPPERS
// ============================================================================

static PyObject *MathGetPerspective(MathHolderObject *self, PyObject *const *args,
                                           Py_ssize_t nargsf, PyObject *kwnames) {
    float fovy;
    float aspect;
    float near_p;
    float far_p;

    if (!self->parsers->MathPersp.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, fovy, aspect,
                                        near_p, far_p)) {
        return nullptr;
    }

    float out[kSixteenFloats];
    CPH_MathGetPerspective(fovy, aspect, near_p, far_p, out);

    return BuildTuple(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7],
                              out[8], out[9], out[10], out[11], out[12], out[13], out[14], out[15]);
}

static PyObject *MathGetOrtho(MathHolderObject *self, PyObject *const *args,
                                     Py_ssize_t nargsf, PyObject *kwnames) {
    float left;
    float right;
    float bottom;
    float top;
    float near_p;
    float far_p;

    if (!self->parsers->MathOrtho.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, left, right,
                                        bottom, top, near_p, far_p)) {
        return nullptr;
    }

    float out[kSixteenFloats];
    CPH_MathGetOrtho(left, right, bottom, top, near_p, far_p, out);

    return BuildTuple(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7],
                              out[8], out[9], out[10], out[11], out[12], out[13], out[14], out[15]);
}

static PyObject *MathGetLookAt(MathHolderObject *self, PyObject *const *args,
                                      Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *eye_obj;
    PyObject *target_obj;
    PyObject *up_obj;

    if (!self->parsers->MathLookAt.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, eye_obj,
                                         target_obj, up_obj)) {
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
    alignas(kSimdAlignment) float eye[4]    = {0};
    alignas(kSimdAlignment) float target[4] = {0};
    alignas(kSimdAlignment) float up[4]     = {0};

    // 3. Fast Unpacking via Macros
    for (int i = 0; i < 3; ++i) {
        eye[i]    = static_cast<float>(PyFloat_AsDouble(PyTuple_GET_ITEM(eye_obj, i)));
        target[i] = static_cast<float>(PyFloat_AsDouble(PyTuple_GET_ITEM(target_obj, i)));
        up[i]     = static_cast<float>(PyFloat_AsDouble(PyTuple_GET_ITEM(up_obj, i)));
    }

    // Final sanity check for conversion errors (e.g., passing a string in the tuple)
    if (PyErr_Occurred()) {
        return nullptr;
    }

    alignas(kSimdAlignment) float out[kSixteenFloats];
    CPH_MathGetLookAt(eye, target, up, out);

    return BuildTuple(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7],
                              out[8], out[9], out[10], out[11], out[12], out[13], out[14], out[15]);
}

static PyObject *MathGetTrs(MathHolderObject *self, PyObject *const *args,
                                   Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *pos_obj;
    PyObject *rot_obj;
    PyObject *scale_obj;

    if (!self->parsers->MathTRS.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, pos_obj, rot_obj,
                                      scale_obj)) {
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
    alignas(kSimdAlignment) float p[4];
    alignas(kSimdAlignment) float r[4];
    alignas(kSimdAlignment) float s[4];

    // 3. Fast Unpacking Loop
    // Position
    for (int i = 0; i < 3; ++i) {
        p[i] = static_cast<float>(PyFloat_AsDouble(PyTuple_GET_ITEM(pos_obj, i)));
    }
    // Rotation (Quaternion: x, y, z, w)
    for (int i = 0; i < 4; ++i) {
        r[i] = static_cast<float>(PyFloat_AsDouble(PyTuple_GET_ITEM(rot_obj, i)));
    }
    // Scale
    for (int i = 0; i < 3; ++i) {
        s[i] = static_cast<float>(PyFloat_AsDouble(PyTuple_GET_ITEM(scale_obj, i)));
    }

    // Check if any PyFloat_AsDouble failed (e.g. passed a string in the tuple)
    if (PyErr_Occurred()) {
        return nullptr;
    }

    alignas(kSimdAlignment) float out[kSixteenFloats];
    CPH_MathGetTrs(p, r, s, out);

    return BuildTuple(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7],
                              out[8], out[9], out[10], out[11], out[12], out[13], out[14], out[15]);
}

static PyObject *MathGetTrsBatch(MathHolderObject *self, PyObject *const *args,
                                        Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *pos_obj;
    PyObject *rot_obj;
    PyObject *scale_obj;

    if (!self->parsers->MathTRSBatch.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, pos_obj,
                                           rot_obj, scale_obj)) {
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

    Py_ssize_t count = p_buf.len / (3 * static_cast<Py_ssize_t>(sizeof(float)));
    PyObject *result = PyBytes_FromStringAndSize(
        nullptr, count * kSixteenFloats * static_cast<Py_ssize_t>(sizeof(float)));
    if (result) {
        CPH_MathGetTrsBatch(count, static_cast<float *>(p_buf.buf), static_cast<float *>(r_buf.buf),
                            static_cast<float *>(s_buf.buf),
                            reinterpret_cast<float *>(PyBytes_AsString(result)));
    }

    PyBuffer_Release(&p_buf);
    PyBuffer_Release(&r_buf);
    PyBuffer_Release(&s_buf);
    return result;
}

// 1. Matrix Inverse
static PyObject *MathInverse(MathHolderObject *self, PyObject *const *args,
                                    Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *mat_obj;

    if (!self->parsers->MathMat.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, mat_obj)) {
        return nullptr;
    }

    alignas(kSimdAlignment) float in[kSixteenFloats];
    alignas(kSimdAlignment) float out[kSixteenFloats];
    if (!unpack_mat44(mat_obj, in)) {
        PyErr_SetString(PyExc_TypeError, "Matrix must be a tuple of 16 floats");
        return nullptr;
    }

    CPH_MathMat44Inverse(in, out);
    return BuildTuple(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7],
                              out[8], out[9], out[10], out[11], out[12], out[13], out[14], out[15]);
}

// 2. Matrix Multiplication (A * B)
static PyObject *MathMatmul(MathHolderObject *self, PyObject *const *args,
                                   Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *a_obj;
    PyObject *b_obj;

    if (!self->parsers->MathMatPair.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, a_obj,
                                          b_obj)) {
        return nullptr;
    }

    alignas(kSimdAlignment) float a[kSixteenFloats];
    alignas(kSimdAlignment) float b[kSixteenFloats];
    alignas(kSimdAlignment) float out[kSixteenFloats];
    if (!unpack_mat44(a_obj, a) || !unpack_mat44(b_obj, b)) {
        return nullptr;
    }

    CPH_MathMat44Mul(a, b, out);
    return BuildTuple(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7],
                              out[8], out[9], out[10], out[11], out[12], out[13], out[14], out[15]);
}

// 3. Vector Transform (Mat * Vec3)
static PyObject *MathTransformVec3(MathHolderObject *self, PyObject *const *args,
                                          Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *mat_obj;
    PyObject *vec_obj;

    if (!self->parsers->MathMatVec.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, mat_obj,
                                         vec_obj)) {
        return nullptr;
    }

    // Standard engine constants for readability or use raw numbers
    alignas(kSimdAlignment) float m[kSixteenFloats];
    if (!unpack_mat44(mat_obj, m)) {
        PyErr_SetString(PyExc_TypeError, "Matrix must be a tuple of 16 floats");
        return nullptr;
    }

    if (!PyTuple_Check(vec_obj) || PyTuple_GET_SIZE(vec_obj) < 3) {
        PyErr_SetString(PyExc_ValueError, "Vector must be a tuple of at least 3 floats");
        return nullptr;
    }

    float v[3] = {static_cast<float>(PyFloat_AsDouble(PyTuple_GET_ITEM(vec_obj, 0))),
                  static_cast<float>(PyFloat_AsDouble(PyTuple_GET_ITEM(vec_obj, 1))),
                  static_cast<float>(PyFloat_AsDouble(PyTuple_GET_ITEM(vec_obj, 2)))};

    // Sanity check for float conversion errors
    if (PyErr_Occurred()) {
        return nullptr;
    }

    float out[3];
    CPH_MathTransformVec3(m, v, out);

    // FastBuild is significantly faster than Py_BuildValue
    return BuildTuple(out[0], out[1], out[2]);
}

// 4. Batch Matrix Multiply (Single * Batch)
// Efficient for: MVP_Batch = VP_Matrix * Model_Matrix_Bytes
static PyObject *MathMatmulBatch(MathHolderObject *self, PyObject *const *args,
                                        Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *mat_obj;
    PyObject *batch_obj;

    if (!self->parsers->MathMatBatch.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, mat_obj,
                                           batch_obj)) {
        return nullptr;
    }

    alignas(kSimdAlignment) float m[kSixteenFloats];
    if (!unpack_mat44(mat_obj, m)) {
        return nullptr;
    }

    Py_buffer view;
    if (PyObject_GetBuffer(batch_obj, &view, PyBUF_SIMPLE) < 0) {
        return nullptr;
    }

    size_t count     = view.len / (kSixteenFloats * sizeof(float));
    PyObject *result = PyBytes_FromStringAndSize(nullptr, view.len);
    if (result) {
        CPH_MathMat44MulBatch(m, static_cast<float *>(view.buf), count,
                              reinterpret_cast<float *>(PyBytes_AsString(result)));
    }

    PyBuffer_Release(&view);
    return result;
}

// 5. Single AABB Culling
static PyObject *MathCullAabb(MathHolderObject *self, PyObject *const *args,
                                     Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *vp_obj;
    PyObject *min_obj;
    PyObject *max_obj;

    if (!self->parsers->MathCull.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, vp_obj, min_obj,
                                       max_obj)) {
        return nullptr;
    }

    alignas(kSimdAlignment) float vp[kSixteenFloats];
    if (!unpack_mat44(vp_obj, vp)) {
        return nullptr;
    }

    // Fast unpack of 3-float tuples
    if (!PyTuple_Check(min_obj) || !PyTuple_Check(max_obj)) {
        return nullptr;
    }

    float a_min[3] = {static_cast<float>(PyFloat_AsDouble(PyTuple_GET_ITEM(min_obj, 0))),
                      static_cast<float>(PyFloat_AsDouble(PyTuple_GET_ITEM(min_obj, 1))),
                      static_cast<float>(PyFloat_AsDouble(PyTuple_GET_ITEM(min_obj, 2)))};
    float a_max[3] = {static_cast<float>(PyFloat_AsDouble(PyTuple_GET_ITEM(max_obj, 0))),
                      static_cast<float>(PyFloat_AsDouble(PyTuple_GET_ITEM(max_obj, 1))),
                      static_cast<float>(PyFloat_AsDouble(PyTuple_GET_ITEM(max_obj, 2)))};

    if (PyErr_Occurred()) {
        return nullptr;
    }

    int visible = CPH_MathCullAabb(vp, a_min, a_max);
    return BuildValue(static_cast<bool>(visible));
}

// 6. Batch AABB Culling (Essential for ECS Performance)
static PyObject *MathCullAabbBatch(MathHolderObject *self, PyObject *const *args,
                                          Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *vp_obj;
    PyObject *aabbs_obj;

    if (!self->parsers->MathCullBatch.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, vp_obj,
                                            aabbs_obj)) {
        return nullptr;
    }

    alignas(kSimdAlignment) float vp[kSixteenFloats];
    if (!unpack_mat44(vp_obj, vp)) {
        return nullptr;
    }

    Py_buffer view;
    if (PyObject_GetBuffer(aabbs_obj, &view, PyBUF_SIMPLE) < 0) {
        return nullptr;
    }

    // aabb_data is expected as [minX, minY, minZ, maxX, maxY, maxZ, ...]
    Py_ssize_t count = view.len / (6 * static_cast<Py_ssize_t>(sizeof(float)));
    PyObject *result = PyByteArray_FromStringAndSize(nullptr, count);
    if (result) {
        CPH_MathCullAabbBatch(vp, static_cast<float *>(view.buf), count,
                              reinterpret_cast<uint8_t *>(PyByteArray_AsString(result)));
    }

    PyBuffer_Release(&view);
    return result;
}

static PyObject *MathVec3NormalizeBatch(MathHolderObject *self, PyObject *const *args,
                                               Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *vecs_obj;

    if (!self->parsers->MathVec3Batch.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, vecs_obj)) {
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
    PyObject *result = PyBytes_FromStringAndSize(nullptr, view.len);

    if (result) {
        CPH_MathVec3NormalizeBatch(static_cast<const float *>(view.buf), count,
                                   reinterpret_cast<float *>(PyBytes_AsString(result)));
    }

    PyBuffer_Release(&view);
    return result;
}

static PyObject *MathQuatFromEuler(MathHolderObject *self, PyObject *const *args,
                                          Py_ssize_t nargsf, PyObject *kwnames) {
    float x;
    float y;
    float z;

    if (!self->parsers->MathEuler.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, x, y, z)) {
        return nullptr;
    }

    float out[4]; // quat: x, y, z, w
    CPH_MathQuatFromEuler(x, y, z, out);

    // BuildTuple is optimized for returning fixed-size math results
    return BuildTuple(out[0], out[1], out[2], out[3]);
}

static PyObject *MathQuatToEuler(MathHolderObject *self, PyObject *const *args,
                                        Py_ssize_t nargsf, PyObject *kwnames) {
    float x;
    float y;
    float z;
    float w;

    if (!self->parsers->MathQuat.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, x, y, z, w)) {
        return nullptr;
    }

    float in_q[4] = {x, y, z, w};
    float out_euler[3];

    CPH_MathQuatToEuler(in_q, out_euler);

    return BuildTuple(out_euler[0], out_euler[1], out_euler[2]);
}

static PyObject *MathQuatSlerp(MathHolderObject *self, PyObject *const *args,
                                      Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *q1_obj;
    PyObject *q2_obj;
    float t;

    if (!self->parsers->MathSlerp.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, q1_obj, q2_obj,
                                        t)) {
        return nullptr;
    }

    float q1[4];
    float q2[4];
    float out[4];
    if (!unpack_quat(q1_obj, q1) || !unpack_quat(q2_obj, q2)) {
        PyErr_SetString(PyExc_TypeError, "Quaternions must be tuples of 4 floats");
        return nullptr;
    }

    CPH_MathQuatSlerp(q1, q2, t, out);

    return BuildTuple(out[0], out[1], out[2], out[3]);
}

static PyObject *MathQuatMul(MathHolderObject *self, PyObject *const *args,
                                    Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *a_obj;
    PyObject *b_obj;

    if (!self->parsers->MathQuatPair.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, a_obj,
                                           b_obj)) {
        return nullptr;
    }

    float a[4];
    float b[4];
    float out[4];
    if (!unpack_quat(a_obj, a) || !unpack_quat(b_obj, b)) {
        PyErr_SetString(PyExc_TypeError, "Quaternions must be tuples of 4 floats");
        return nullptr;
    }

    CPH_MathQuatMul(a, b, out);

    return BuildTuple(out[0], out[1], out[2], out[3]);
}

static PyObject *MathVec3LerpBatch(MathHolderObject *self, PyObject *const *args,
                                          Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *a_obj;
    PyObject *b_obj;
    float alpha;

    if (!self->parsers->MathLerpBatch.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, a_obj, b_obj,
                                            alpha)) {
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
    PyObject *result = PyBytes_FromStringAndSize(nullptr, view_a.len);

    if (result) {
        CPH_MathVec3LerpBatch(static_cast<const float *>(view_a.buf),
                              static_cast<const float *>(view_b.buf), alpha, count,
                              reinterpret_cast<float *>(PyBytes_AsString(result)));
    }

    PyBuffer_Release(&view_a);
    PyBuffer_Release(&view_b);
    return result;
}

static PyObject *MathQuatRotateVec3(MathHolderObject *self, PyObject *const *args,
                                           Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *q_obj;
    PyObject *v_obj;

    if (!self->parsers->MathQuatVec.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, q_obj,
                                          v_obj)) {
        return nullptr;
    }

    float q[4];
    float v[3];
    float out[3];
    if (!unpack_quat(q_obj, q) || !unpack_vec3(v_obj, v)) {
        PyErr_SetString(PyExc_TypeError, "Inputs must be tuples of floats (Quat=4, Vec3=3)");
        return nullptr;
    }

    CPH_MathQuatRotateVec3(q, v, out);

    return BuildTuple(out[0], out[1], out[2]);
}

static PyObject *MathQuatRotateVec3Batch(MathHolderObject *self, PyObject *const *args,
                                                Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *q_obj;
    PyObject *vecs_obj;

    if (!self->parsers->MathQuatVecBatch.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, q_obj,
                                               vecs_obj)) {
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
    PyObject *result = PyBytes_FromStringAndSize(nullptr, view.len);

    if (result) {
        CPH_MathQuatRotateVec3Batch(q, static_cast<const float *>(view.buf), count,
                                    reinterpret_cast<float *>(PyBytes_AsString(result)));
    }

    PyBuffer_Release(&view);
    return result;
}

static PyObject *MathQuatInverse(MathHolderObject *self, PyObject *const *args,
                                        Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *q_obj;

    if (!self->parsers->MathQuatOp.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, q_obj)) {
        return nullptr;
    }

    float q[4];
    float out[4];
    if (!unpack_quat(q_obj, q)) {
        PyErr_SetString(PyExc_TypeError, "Quaternion must be a tuple of 4 floats");
        return nullptr;
    }

    CPH_MathQuatInverse(q, out);

    return BuildTuple(out[0], out[1], out[2], out[3]);
}

static PyObject *MathProject(MathHolderObject *self, PyObject *const *args,
                                    Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *v_obj;
    PyObject *mvp_obj;
    PyObject *vp_obj;

    if (!self->parsers->MathProject.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, v_obj, mvp_obj,
                                          vp_obj)) {
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

    CPH_MathProject(v, mvp, viewport, out);

    return BuildTuple(out[0], out[1], out[2]);
}

static PyObject *MathUnproject(MathHolderObject *self, PyObject *const *args,
                                      Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *v_obj;
    PyObject *mvp_obj;
    PyObject *vp_obj;

    if (!self->parsers->MathUnproject.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, v_obj,
                                            mvp_obj, vp_obj)) {
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

    CPH_MathUnproject(v, mvp, viewport, out);

    return BuildTuple(out[0], out[1], out[2]);
}

static PyObject *MathQuatFromTo(MathHolderObject *self, PyObject *const *args,
                                       Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *v1_obj;
    PyObject *v2_obj;

    if (!self->parsers->MathVecPair.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, v1_obj,
                                          v2_obj)) {
        return nullptr;
    }

    float v1[3];
    float v2[3];
    float out[4];
    if (!unpack_vec3(v1_obj, v1) || !unpack_vec3(v2_obj, v2)) {
        PyErr_SetString(PyExc_TypeError, "v1 and v2 must be tuples of 3 floats");
        return nullptr;
    }

    CPH_MathQuatFromTo(v1, v2, out);

    return BuildTuple(out[0], out[1], out[2], out[3]);
}

static PyObject *MathVec3Dot(MathHolderObject *self, PyObject *const *args,
                                    Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *v1_obj;
    PyObject *v2_obj;

    // Reusing the parser from quat_from_to
    if (!self->parsers->MathVecPair.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, v1_obj,
                                          v2_obj)) {
        return nullptr;
    }

    float v1[3];
    float v2[3];
    if (!unpack_vec3(v1_obj, v1) || !unpack_vec3(v2_obj, v2)) {
        PyErr_SetString(PyExc_TypeError, "v1 and v2 must be tuples of 3 floats");
        return nullptr;
    }

    float result = CPH_MathVec3Dot(v1, v2);

    return PyFloat_FromDouble(static_cast<double>(result));
}

static PyObject *MathVec3Cross(MathHolderObject *self, PyObject *const *args,
                                      Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *v1_obj;
    PyObject *v2_obj;

    if (!self->parsers->MathVecPair.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, v1_obj,
                                          v2_obj)) {
        return nullptr;
    }

    float v1[3];
    float v2[3];
    float out[3];
    if (!unpack_vec3(v1_obj, v1) || !unpack_vec3(v2_obj, v2)) {
        PyErr_SetString(PyExc_TypeError, "v1 and v2 must be tuples of 3 floats");
        return nullptr;
    }

    CPH_MathVec3Cross(v1, v2, out);

    return BuildTuple(out[0], out[1], out[2]);
}

static PyObject *MathIntersectRayPlane(MathHolderObject *self, PyObject *const *args,
                                              Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *ro_obj;
    PyObject *rd_obj;
    PyObject *po_obj;
    PyObject *pn_obj;

    if (!self->parsers->MathRayPlane.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, ro_obj,
                                           rd_obj, po_obj, pn_obj)) {
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

    int hit = CPH_MathIntersectRayPlane(ro, rd, po, pn, &out_t, out_p);

    if (!hit) {
        return BuildTuple(Py_False, 0.0f, Py_None);
    }

    return BuildTuple(Py_True, out_t, BuildTuple(out_p[0], out_p[1], out_p[2]));
}

static PyObject *MathQuatGetAxisAngle(MathHolderObject *self, PyObject *const *args,
                                             Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *q_obj;

    if (!self->parsers->MathQuatOp.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, q_obj)) {
        return nullptr;
    }

    float q[4];
    float axis[3];
    float angle;
    if (!unpack_quat(q_obj, q)) {
        PyErr_SetString(PyExc_TypeError, "Quaternion must be a tuple of 4 floats");
        return nullptr;
    }

    CPH_MathQuatGetAxisAngle(q, axis, &angle);

    return BuildTuple(BuildTuple(axis[0], axis[1], axis[2]), angle);
}

static PyObject *MathQuatFromAxisAngle(MathHolderObject *self, PyObject *const *args,
                                              Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *axis_obj;
    float angle;

    if (!self->parsers->MathAxisAngle.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, axis_obj,
                                            angle)) {
        return nullptr;
    }

    float axis[3];
    float out[4];
    if (!unpack_vec3(axis_obj, axis)) {
        PyErr_SetString(PyExc_TypeError, "Axis must be a tuple of 3 floats");
        return nullptr;
    }

    CPH_MathQuatFromAxisAngle(axis, angle, out);

    return BuildTuple(out[0], out[1], out[2], out[3]);
}

static PyObject *MathVec3DistanceBatch(MathHolderObject *self, PyObject *const *args,
                                              Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *a_obj;
    PyObject *b_obj;

    if (!self->parsers->MathDistBatch.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, a_obj,
                                            b_obj)) {
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
    PyObject *result =
        PyBytes_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(count * sizeof(float)));

    if (result) {
        CPH_MathVec3DistanceBatch(static_cast<const float *>(view_a.buf),
                                  static_cast<const float *>(view_b.buf), count,
                                  reinterpret_cast<float *>(PyBytes_AsString(result)));
    }

    PyBuffer_Release(&view_a);
    PyBuffer_Release(&view_b);
    return result;
}

static PyObject *MathVec3Normalize(MathHolderObject *self, PyObject *const *args,
                                          Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *v_obj;

    if (!self->parsers->MathVecOp.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, v_obj)) {
        return nullptr;
    }

    float v[3];
    float out[3];
    if (!unpack_vec3(v_obj, v)) {
        PyErr_SetString(PyExc_TypeError, "v must be a tuple of 3 floats");
        return nullptr;
    }

    CPH_MathVec3Normalize(v, out);

    return BuildTuple(out[0], out[1], out[2]);
}

static PyObject *MathMat44GetTranslation(MathHolderObject *self, PyObject *const *args,
                                                Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *mat_obj;

    if (!self->parsers->MathMat.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, mat_obj)) {
        return nullptr;
    }

    alignas(kSimdAlignment) float in_mat[kSixteenFloats];
    if (!unpack_mat44(mat_obj, in_mat)) {
        PyErr_SetString(PyExc_TypeError, "Matrix must be a tuple of 16 floats");
        return nullptr;
    }

    float out_vec[3];
    CPH_MathMat44GetTranslation(in_mat, out_vec);

    return BuildTuple(out_vec[0], out_vec[1], out_vec[2]);
}

static PyObject *MathMat44GetRotation(MathHolderObject *self, PyObject *const *args,
                                             Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *mat_obj;

    if (!self->parsers->MathMat.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, mat_obj)) {
        return nullptr;
    }

    alignas(kSimdAlignment) float in_mat[kSixteenFloats];
    if (!unpack_mat44(mat_obj, in_mat)) {
        PyErr_SetString(PyExc_TypeError, "Matrix must be a tuple of 16 floats");
        return nullptr;
    }

    float out_quat[4];
    CPH_MathMat44GetRotation(in_mat, out_quat);

    return BuildTuple(out_quat[0], out_quat[1], out_quat[2], out_quat[3]);
}

static PyObject *MathMat44Identity([[maybe_unused]] MathHolderObject *self,
                                          PyObject *ignored [[maybe_unused]]) {
    float out[kSixteenFloats];
    CPH_MathMat44Identity(out);

    return BuildTuple(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7],
                              out[8], out[9], out[10], out[11], out[12], out[13], out[14], out[15]);
}

static PyObject *MathVec3Reflect(MathHolderObject *self, PyObject *const *args,
                                        Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *v_obj;
    PyObject *n_obj;

    if (!self->parsers->MathReflect.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, v_obj,
                                          n_obj)) {
        return nullptr;
    }

    float v[3];
    float n[3];
    float out[3];
    if (!unpack_vec3(v_obj, v) || !unpack_vec3(n_obj, n)) {
        PyErr_SetString(PyExc_TypeError, "v and normal must be tuples of 3 floats");
        return nullptr;
    }

    CPH_MathVec3Reflect(v, n, out);

    return BuildTuple(out[0], out[1], out[2]);
}

static PyObject *MathVec3Distance(MathHolderObject *self, PyObject *const *args,
                                         Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *v1_obj;
    PyObject *v2_obj;

    if (!self->parsers->MathVecPair.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, v1_obj,
                                          v2_obj)) {
        return nullptr;
    }

    float v1[3];
    float v2[3];
    if (!unpack_vec3(v1_obj, v1) || !unpack_vec3(v2_obj, v2)) {
        PyErr_SetString(PyExc_TypeError, "v1 and v2 must be tuples of 3 floats");
        return nullptr;
    }

    float dist = CPH_MathVec3Distance(v1, v2);

    return PyFloat_FromDouble(static_cast<double>(dist));
}

static PyObject *MathQuatRotateVec3Inverse(MathHolderObject *self, PyObject *const *args,
                                                  Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *q_obj;
    PyObject *v_obj;

    if (!self->parsers->MathQuatVec.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, q_obj,
                                          v_obj)) {
        return nullptr;
    }

    float q[4];
    float v[3];
    float out[3];
    if (!unpack_quat(q_obj, q) || !unpack_vec3(v_obj, v)) {
        PyErr_SetString(PyExc_TypeError, "Inputs must be tuples of floats (Quat=4, Vec3=3)");
        return nullptr;
    }

    CPH_MathQuatRotateVec3Inverse(q, v, out);

    return BuildTuple(out[0], out[1], out[2]);
}

// --- Wrapper for single tuple (x, y, z) -> (x, y, z, w) ---
static PyObject *MathEulerToQuat(MathHolderObject *self, PyObject *const *args,
                                        Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *euler_obj;

    if (!self->parsers->MathEulerVec.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, euler_obj)) {
        return nullptr;
    }

    float euler[3];
    float q[4];
    if (!unpack_vec3(euler_obj, euler)) {
        PyErr_SetString(PyExc_TypeError, "Euler must be a tuple of 3 floats");
        return nullptr;
    }

    CPH_MathEulerToQuat(euler, q);
    return BuildTuple(q[0], q[1], q[2], q[3]);
}

// --- Wrapper for batch conversion (Bytes/Buffer) ---
static PyObject *MathEulerToQuatBatch(MathHolderObject *self, PyObject *const *args,
                                             Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *eulers_obj;

    if (!self->parsers->MathEulerBatch.Parse(args, PyVectorcall_NARGS(nargsf), kwnames,
                                             eulers_obj)) {
        return nullptr;
    }

    Py_buffer view;
    if (PyObject_GetBuffer(eulers_obj, &view, PyBUF_SIMPLE) < 0) {
        return nullptr;
    }

    if (view.len % 12 != 0) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_ValueError, "Buffer must be multiple of 12 bytes (3 floats)");
        return nullptr;
    }

    size_t count     = view.len / 12;
    PyObject *result = PyBytes_FromStringAndSize(
        nullptr, static_cast<Py_ssize_t>(count * 16)); // 4 floats per quat

    if (result) {
        CPH_MathEulerToQuatBatch(static_cast<const float *>(view.buf), count,
                                 reinterpret_cast<float *>(PyBytes_AsString(result)));
    }

    PyBuffer_Release(&view);
    return result;
}
static PyObject *MathNew(PyTypeObject *type, [[maybe_unused]] PyObject *args,
                                [[maybe_unused]] PyObject *kwds) {
    MathHolderObject *self = reinterpret_cast<MathHolderObject *>(type->tp_alloc(type, 0));
    if (self == nullptr) {
        return nullptr;
    }
    self->parsers = CPH::Python::NewParserGroup<CPH::Python::MathParsers>();
    if (self->parsers == nullptr) {
        Py_DECREF(self);
        return PyErr_NoMemory();
    }

    // 3. Call your parser setup logic

    return reinterpret_cast<PyObject *>(self);
}

static void MathDealloc(MathHolderObject *self) {
    if (self->parsers != nullptr) {

        delete self->parsers;
    }
    PyTypeObject *tp = Py_TYPE(self);
    tp->tp_free(reinterpret_cast<PyObject *>(self));
    Py_DECREF(tp);
}

// ============================================================================
// TYPE DEFINITION
// ============================================================================

static PyMethodDef kMathMethods[] = {

    CPH::Python::FastCallMethod("get_perspective", MathGetPerspective),
    CPH::Python::FastCallMethod("get_ortho", MathGetOrtho),
    CPH::Python::FastCallMethod("get_look_at", MathGetLookAt),
    CPH::Python::FastCallMethod("get_trs", MathGetTrs),
    CPH::Python::FastCallMethod("get_trs_batch", MathGetTrsBatch),
    CPH::Python::FastCallMethod("inverse", MathInverse),
    CPH::Python::FastCallMethod("matmul", MathMatmul),
    CPH::Python::FastCallMethod("transform_vec3", MathTransformVec3),
    CPH::Python::FastCallMethod("matmul_batch", MathMatmulBatch),
    CPH::Python::FastCallMethod("cull_aabb", MathCullAabb),
    CPH::Python::FastCallMethod("cull_aabb_batch", MathCullAabbBatch),
    CPH::Python::FastCallMethod("vec3_normalize_batch", MathVec3NormalizeBatch),
    CPH::Python::FastCallMethod("quat_from_euler", MathQuatFromEuler),
    CPH::Python::FastCallMethod("quat_to_euler", MathQuatToEuler),
    CPH::Python::FastCallMethod("quat_slerp", MathQuatSlerp),
    CPH::Python::FastCallMethod("quat_mul", MathQuatMul),
    CPH::Python::FastCallMethod("vec3_lerp_batch", MathVec3LerpBatch),
    CPH::Python::FastCallMethod("quat_rotate_vec3", MathQuatRotateVec3),
    CPH::Python::FastCallMethod("quat_rotate_vec3_batch", MathQuatRotateVec3Batch),
    CPH::Python::FastCallMethod("quat_inverse", MathQuatInverse),
    CPH::Python::FastCallMethod("project", MathProject),
    CPH::Python::FastCallMethod("unproject", MathUnproject),
    CPH::Python::FastCallMethod("quat_from_to", MathQuatFromTo),
    CPH::Python::FastCallMethod("vec3_dot", MathVec3Dot),
    CPH::Python::FastCallMethod("vec3_cross", MathVec3Cross),
    CPH::Python::FastCallMethod("intersect_ray_plane", MathIntersectRayPlane),
    CPH::Python::FastCallMethod("quat_get_axis_angle", MathQuatGetAxisAngle),
    CPH::Python::FastCallMethod("quat_from_axis_angle", MathQuatFromAxisAngle),
    CPH::Python::FastCallMethod("vec3_distance_batch", MathVec3DistanceBatch),
    CPH::Python::FastCallMethod("vec3_normalize", MathVec3Normalize),
    CPH::Python::FastCallMethod("mat44_get_translation", MathMat44GetTranslation),
    CPH::Python::FastCallMethod("mat44_get_rotation", MathMat44GetRotation),
    CPH::Python::NoArgsMethod("mat44_identity", MathMat44Identity),
    CPH::Python::FastCallMethod("vec3_reflect", MathVec3Reflect),
    CPH::Python::FastCallMethod("vec3_distance", MathVec3Distance),
    CPH::Python::FastCallMethod("quat_rotate_vec3_inverse", MathQuatRotateVec3Inverse),
    CPH::Python::FastCallMethod("euler_to_quat", MathEulerToQuat),
    CPH::Python::FastCallMethod("euler_to_quat_batch", MathEulerToQuatBatch),
    {nullptr, nullptr, 0, nullptr}};

static PyType_Slot MathServiceSlots[] = {
    {.slot = Py_tp_new, .pfunc = reinterpret_cast<void *>(MathNew)},
    {.slot = Py_tp_dealloc, .pfunc = reinterpret_cast<void *>(MathDealloc)},
    {.slot = Py_tp_methods, .pfunc = reinterpret_cast<void *>(kMathMethods)},
    {},
};

PyType_Spec MathServiceSpec = {
    .name      = "culverin._culverin_c.MathService",
    .basicsize = sizeof(struct MathHolderObject),
    .flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .slots     = MathServiceSlots,
};

} // namespace CPH::Python
