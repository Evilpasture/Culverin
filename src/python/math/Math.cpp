#include "math/Math.hpp"
#include "bindings/BufferView.hpp"
#include "bindings/FastBuild.hpp"
#include "bindings/PythonSupport.hpp"
#include "bindings/parsing/Parser.hpp"
#include "core/Compiler.hpp"
#include <Python.h>
#include "bindings/Result.hpp"

namespace CPH::Python {

struct MathHolderObject {
    PyObject_HEAD CPH::Python::MathParsers *parsers;
};
static constexpr int kSixteenFloats = 16;
static constexpr int kSimdAlignment = 16;
// TODO: a non-tuple argument makes these two helpers return false without
// setting an exception, so the caller's `return nullptr` reaches Python as
// "SystemError: returned NULL without setting an exception". Pre-existing
// (unrelated to the buffer handling below) and deliberately not fixed here:
// the error text a caller sees would change.
// Helper for extracting 16 floats from a Python Tuple
static inline bool UnpackMat44(PyObject *obj, float *out) {

    if (!PyTuple_Check(obj) || PyTuple_GET_SIZE(obj) < kSixteenFloats) {
        return false;
    }
    for (int i = 0; i < kSixteenFloats; ++i) {
        out[i] = static_cast<float>(PyFloat_AsDouble(PyTuple_GET_ITEM(obj, i)));
    }
    return !PyErr_Occurred();
}

static inline bool UnpackQuat(PyObject *obj, float *out) {
    if (!PyTuple_Check(obj) || PyTuple_GET_SIZE(obj) < 4) {
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        out[i] = static_cast<float>(PyFloat_AsDouble(PyTuple_GET_ITEM(obj, i)));
    }
    return !PyErr_Occurred();
}

static inline bool UnpackVec3(PyObject *obj, float *out) {
    if (!PyTuple_Check(obj) || PyTuple_GET_SIZE(obj) < 3) {
        return false;
    }
    for (int i = 0; i < 3; ++i) {
        out[i] = static_cast<float>(PyFloat_AsDouble(PyTuple_GET_ITEM(obj, i)));
    }
    return !PyErr_Occurred();
}

static inline bool UnpackViewport(PyObject *obj, int *out) {
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

static BoundResult MathGetPerspective(MathHolderObject *self, PyObject *const *args,
                                           Py_ssize_t nargsf, PyObject *kwnames) {
    float fovy;
    float aspect;
    float near_p;
    float far_p;

    const ParseResult parsed =
        self->parsers->mathPersp.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, fovy, aspect,
                                        near_p, far_p);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return FailCurrent();
    }

    float out[kSixteenFloats];
    CPH::MathGetPerspective(fovy, aspect, near_p, far_p, out);

    return Adopt(BuildTuple(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7],
                              out[8], out[9], out[10], out[11], out[12], out[13], out[14], out[15]));
}

static BoundResult MathGetOrtho(MathHolderObject *self, PyObject *const *args,
                                     Py_ssize_t nargsf, PyObject *kwnames) {
    float left;
    float right;
    float bottom;
    float top;
    float near_p;
    float far_p;

    const ParseResult parsed =
        self->parsers->mathOrtho.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, left, right,
                                        bottom, top, near_p, far_p);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return FailCurrent();
    }

    float out[kSixteenFloats];
    CPH::MathGetOrtho(left, right, bottom, top, near_p, far_p, out);

    return Adopt(BuildTuple(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7],
                              out[8], out[9], out[10], out[11], out[12], out[13], out[14], out[15]));
}

static BoundResult MathGetLookAt(MathHolderObject *self, PyObject *const *args,
                                      Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *eye_obj;
    PyObject *target_obj;
    PyObject *up_obj;

    const ParseResult parsed =
        self->parsers->mathLookAt.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, eye_obj,
                                         target_obj, up_obj);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return FailCurrent();
    }

    // 1. Gatekeeper: Ensure we actually have tuples and they have 3 elements
    if (!PyTuple_Check(eye_obj) || !PyTuple_Check(target_obj) || !PyTuple_Check(up_obj)) {
        PyErr_SetString(PyExc_TypeError, "LookAt parameters must be tuples");
        return FailCurrent();
    }
    if (PyTuple_GET_SIZE(eye_obj) < 3 || PyTuple_GET_SIZE(target_obj) < 3 ||
        PyTuple_GET_SIZE(up_obj) < 3) {
        PyErr_SetString(PyExc_ValueError, "LookAt tuples must contain 3 elements");
        return FailCurrent();
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
        return FailCurrent();
    }

    alignas(kSimdAlignment) float out[kSixteenFloats];
    CPH::MathGetLookAt(eye, target, up, out);

    return Adopt(BuildTuple(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7],
                              out[8], out[9], out[10], out[11], out[12], out[13], out[14], out[15]));
}

static BoundResult MathGetTrs(MathHolderObject *self, PyObject *const *args,
                                   Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *pos_obj;
    PyObject *rot_obj;
    PyObject *scale_obj;

    const ParseResult parsed =
        self->parsers->mathTrs.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, pos_obj, rot_obj,
                                      scale_obj);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return FailCurrent();
    }

    // 1. Guard the gates: Verify types and sizes once
    if (!PyTuple_Check(pos_obj) || !PyTuple_Check(rot_obj) || !PyTuple_Check(scale_obj)) {
        PyErr_SetString(PyExc_TypeError, "TRS inputs must be tuples");
        return FailCurrent();
    }
    if (PyTuple_GET_SIZE(pos_obj) < 3 || PyTuple_GET_SIZE(rot_obj) < 4 ||
        PyTuple_GET_SIZE(scale_obj) < 3) {
        PyErr_SetString(PyExc_ValueError, "Invalid tuple dimensions for TRS");
        return FailCurrent();
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
        return FailCurrent();
    }

    alignas(kSimdAlignment) float out[kSixteenFloats];
    CPH::MathGetTrs(p, r, s, out);

    return Adopt(BuildTuple(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7],
                              out[8], out[9], out[10], out[11], out[12], out[13], out[14], out[15]));
}

static BoundResult MathGetTrsBatch(MathHolderObject *self, PyObject *const *args,
                                        Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *pos_obj;
    PyObject *rot_obj;
    PyObject *scale_obj;

    const ParseResult parsed =
        self->parsers->mathTrsBatch.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, pos_obj,
                                           rot_obj, scale_obj);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return FailCurrent();
    }

    const auto positions = BufferView::Acquire(pos_obj);
    if (!positions) {
        return FailCurrent();
    }
    const auto rotations = BufferView::Acquire(rot_obj);
    if (!rotations) {
        return FailCurrent();
    }
    const auto scales = BufferView::Acquire(scale_obj);
    if (!scales) {
        return FailCurrent();
    }

    const std::span<const float> packed_positions = positions->Float32();
    const size_t                 count            = packed_positions.size() / 3;
    PyObject                    *result           = PyBytes_FromStringAndSize(
        nullptr, static_cast<Py_ssize_t>(count * kSixteenFloats * sizeof(float)));
    if (result) {
        CPH::MathGetTrsBatch(count, packed_positions.data(), rotations->Float32().data(),
                            scales->Float32().data(),
                            reinterpret_cast<float *>(PyBytes_AsString(result)));
    }
    return Adopt(result);
}

// 1. Matrix Inverse
static BoundResult MathInverse(MathHolderObject *self, PyObject *const *args,
                                    Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *mat_obj;

    const ParseResult parsed =
        self->parsers->mathMat.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, mat_obj);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return FailCurrent();
    }

    alignas(kSimdAlignment) float in[kSixteenFloats];
    alignas(kSimdAlignment) float out[kSixteenFloats];
    if (!UnpackMat44(mat_obj, in)) {
        PyErr_SetString(PyExc_TypeError, "Matrix must be a tuple of 16 floats");
        return FailCurrent();
    }

    CPH::MathMat44Inverse(in, out);
    return Adopt(BuildTuple(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7],
                              out[8], out[9], out[10], out[11], out[12], out[13], out[14], out[15]));
}

// 2. Matrix Multiplication (A * B)
static BoundResult MathMatmul(MathHolderObject *self, PyObject *const *args,
                                   Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *a_obj;
    PyObject *b_obj;

    const ParseResult parsed =
        self->parsers->mathMatPair.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, a_obj,
                                          b_obj);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return FailCurrent();
    }

    alignas(kSimdAlignment) float a[kSixteenFloats];
    alignas(kSimdAlignment) float b[kSixteenFloats];
    alignas(kSimdAlignment) float out[kSixteenFloats];
    if (!UnpackMat44(a_obj, a) || !UnpackMat44(b_obj, b)) {
        return FailCurrent();
    }

    CPH::MathMat44Mul(a, b, out);
    return Adopt(BuildTuple(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7],
                              out[8], out[9], out[10], out[11], out[12], out[13], out[14], out[15]));
}

// 3. Vector Transform (Mat * Vec3)
static BoundResult MathTransformVec3(MathHolderObject *self, PyObject *const *args,
                                          Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *mat_obj;
    PyObject *vec_obj;

    const ParseResult parsed =
        self->parsers->mathMatVec.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, mat_obj,
                                         vec_obj);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return FailCurrent();
    }

    // Standard engine constants for readability or use raw numbers
    alignas(kSimdAlignment) float m[kSixteenFloats];
    if (!UnpackMat44(mat_obj, m)) {
        PyErr_SetString(PyExc_TypeError, "Matrix must be a tuple of 16 floats");
        return FailCurrent();
    }

    if (!PyTuple_Check(vec_obj) || PyTuple_GET_SIZE(vec_obj) < 3) {
        PyErr_SetString(PyExc_ValueError, "Vector must be a tuple of at least 3 floats");
        return FailCurrent();
    }

    float v[3] = {static_cast<float>(PyFloat_AsDouble(PyTuple_GET_ITEM(vec_obj, 0))),
                  static_cast<float>(PyFloat_AsDouble(PyTuple_GET_ITEM(vec_obj, 1))),
                  static_cast<float>(PyFloat_AsDouble(PyTuple_GET_ITEM(vec_obj, 2)))};

    // Sanity check for float conversion errors
    if (PyErr_Occurred()) {
        return FailCurrent();
    }

    float out[3];
    CPH::MathTransformVec3(m, v, out);

    // FastBuild is significantly faster than Py_BuildValue
    return Adopt(BuildTuple(out[0], out[1], out[2]));
}

// 4. Batch Matrix Multiply (Single * Batch)
// Efficient for: MVP_Batch = VP_Matrix * Model_Matrix_Bytes
static BoundResult MathMatmulBatch(MathHolderObject *self, PyObject *const *args,
                                        Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *mat_obj;
    PyObject *batch_obj;

    const ParseResult parsed =
        self->parsers->mathMatBatch.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, mat_obj,
                                           batch_obj);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return FailCurrent();
    }

    alignas(kSimdAlignment) float m[kSixteenFloats];
    if (!UnpackMat44(mat_obj, m)) {
        return FailCurrent();
    }

    const auto batch = BufferView::Acquire(batch_obj);
    if (!batch) {
        return FailCurrent();
    }

    const std::span<const float> packed_matrices = batch->Float32();
    const size_t                 count           = packed_matrices.size() / kSixteenFloats;
    PyObject                    *result = PyBytes_FromStringAndSize(
        nullptr, static_cast<Py_ssize_t>(batch->Bytes().size()));
    if (result) {
        CPH::MathMat44MulBatch(m, packed_matrices.data(), count,
                              reinterpret_cast<float *>(PyBytes_AsString(result)));
    }
    return Adopt(result);
}

// 5. Single AABB Culling
static BoundResult MathCullAabb(MathHolderObject *self, PyObject *const *args,
                                     Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *vp_obj;
    PyObject *min_obj;
    PyObject *max_obj;

    const ParseResult parsed =
        self->parsers->mathCull.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, vp_obj, min_obj,
                                       max_obj);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return FailCurrent();
    }

    alignas(kSimdAlignment) float vp[kSixteenFloats];
    if (!UnpackMat44(vp_obj, vp)) {
        return FailCurrent();
    }

    // Fast unpack of 3-float tuples
    if (!PyTuple_Check(min_obj) || !PyTuple_Check(max_obj)) {
        return FailCurrent();
    }

    float a_min[3] = {static_cast<float>(PyFloat_AsDouble(PyTuple_GET_ITEM(min_obj, 0))),
                      static_cast<float>(PyFloat_AsDouble(PyTuple_GET_ITEM(min_obj, 1))),
                      static_cast<float>(PyFloat_AsDouble(PyTuple_GET_ITEM(min_obj, 2)))};
    float a_max[3] = {static_cast<float>(PyFloat_AsDouble(PyTuple_GET_ITEM(max_obj, 0))),
                      static_cast<float>(PyFloat_AsDouble(PyTuple_GET_ITEM(max_obj, 1))),
                      static_cast<float>(PyFloat_AsDouble(PyTuple_GET_ITEM(max_obj, 2)))};

    if (PyErr_Occurred()) {
        return FailCurrent();
    }

    int visible = CPH::MathCullAabb(vp, a_min, a_max);
    return Adopt(BuildValue(static_cast<bool>(visible)));
}

// 6. Batch AABB Culling (Essential for ECS Performance)
static BoundResult MathCullAabbBatch(MathHolderObject *self, PyObject *const *args,
                                          Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *vp_obj;
    PyObject *aabbs_obj;

    const ParseResult parsed =
        self->parsers->mathCullBatch.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, vp_obj,
                                            aabbs_obj);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return FailCurrent();
    }

    alignas(kSimdAlignment) float vp[kSixteenFloats];
    if (!UnpackMat44(vp_obj, vp)) {
        return FailCurrent();
    }

    const auto aabbs = BufferView::Acquire(aabbs_obj);
    if (!aabbs) {
        return FailCurrent();
    }

    // aabb_data is expected as [minX, minY, minZ, maxX, maxY, maxZ, ...]
    const std::span<const float> packed_boxes = aabbs->Float32();
    const size_t                 count        = packed_boxes.size() / 6;
    PyObject *result = PyByteArray_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(count));
    if (result) {
        CPH::MathCullAabbBatch(vp, packed_boxes.data(), count,
                              reinterpret_cast<uint8_t *>(PyByteArray_AsString(result)));
    }
    return Adopt(result);
}

static BoundResult MathVec3NormalizeBatch(MathHolderObject *self, PyObject *const *args,
                                               Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *vecs_obj;

    const ParseResult parsed =
        self->parsers->mathVec3Batch.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, vecs_obj);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return FailCurrent();
    }

    const auto vectors = BufferView::Acquire(vecs_obj);
    if (!vectors) {
        return FailCurrent(); // PyErr_SetString already called by GetBuffer
    }

    // Verify we have enough bytes for float3 vectors
    constexpr size_t kVec3Bytes = 3 * sizeof(float);
    if (!vectors->RequireMultipleOf(kVec3Bytes,
                                    "Buffer length must be a multiple of 12 (3 floats)")) {
        return FailCurrent();
    }

    const std::span<const float> packed_vectors = vectors->Float32();
    const size_t                 count          = packed_vectors.size() / 3;
    PyObject                    *result         = PyBytes_FromStringAndSize(
        nullptr, static_cast<Py_ssize_t>(vectors->Bytes().size()));

    if (result) {
        CPH::MathVec3NormalizeBatch(packed_vectors.data(), count,
                                   reinterpret_cast<float *>(PyBytes_AsString(result)));
    }
    return Adopt(result);
}

static BoundResult MathQuatFromEuler(MathHolderObject *self, PyObject *const *args,
                                          Py_ssize_t nargsf, PyObject *kwnames) {
    float x;
    float y;
    float z;

    const ParseResult parsed =
        self->parsers->mathEuler.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, x, y, z);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return FailCurrent();
    }

    float out[4]; // quat: x, y, z, w
    CPH::MathQuatFromEuler(x, y, z, out);

    // BuildTuple is optimized for returning fixed-size math results
    return Adopt(BuildTuple(out[0], out[1], out[2], out[3]));
}

static BoundResult MathQuatToEuler(MathHolderObject *self, PyObject *const *args,
                                        Py_ssize_t nargsf, PyObject *kwnames) {
    float x;
    float y;
    float z;
    float w;

    const ParseResult parsed =
        self->parsers->mathQuat.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, x, y, z, w);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return FailCurrent();
    }

    float in_q[4] = {x, y, z, w};
    float out_euler[3];

    CPH::MathQuatToEuler(in_q, out_euler);

    return Adopt(BuildTuple(out_euler[0], out_euler[1], out_euler[2]));
}

static BoundResult MathQuatSlerp(MathHolderObject *self, PyObject *const *args,
                                      Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *q1_obj;
    PyObject *q2_obj;
    float t;

    const ParseResult parsed =
        self->parsers->mathSlerp.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, q1_obj, q2_obj,
                                        t);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return FailCurrent();
    }

    float q1[4];
    float q2[4];
    float out[4];
    if (!UnpackQuat(q1_obj, q1) || !UnpackQuat(q2_obj, q2)) {
        PyErr_SetString(PyExc_TypeError, "Quaternions must be tuples of 4 floats");
        return FailCurrent();
    }

    CPH::MathQuatSlerp(q1, q2, t, out);

    return Adopt(BuildTuple(out[0], out[1], out[2], out[3]));
}

static BoundResult MathQuatMul(MathHolderObject *self, PyObject *const *args,
                                    Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *a_obj;
    PyObject *b_obj;

    const ParseResult parsed =
        self->parsers->mathQuatPair.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, a_obj,
                                           b_obj);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return FailCurrent();
    }

    float a[4];
    float b[4];
    float out[4];
    if (!UnpackQuat(a_obj, a) || !UnpackQuat(b_obj, b)) {
        PyErr_SetString(PyExc_TypeError, "Quaternions must be tuples of 4 floats");
        return FailCurrent();
    }

    CPH::MathQuatMul(a, b, out);

    return Adopt(BuildTuple(out[0], out[1], out[2], out[3]));
}

static BoundResult MathVec3LerpBatch(MathHolderObject *self, PyObject *const *args,
                                          Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *a_obj;
    PyObject *b_obj;
    float alpha;

    const ParseResult parsed =
        self->parsers->mathLerpBatch.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, a_obj, b_obj,
                                            alpha);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return FailCurrent();
    }

    const auto a_vectors = BufferView::Acquire(a_obj);
    if (!a_vectors) {
        return FailCurrent();
    }
    const auto b_vectors = BufferView::Acquire(b_obj);
    if (!b_vectors) {
        return FailCurrent();
    }

    constexpr const char *kShapeError = "Buffers must be equal size and multiples of 12 bytes";
    if (!a_vectors->RequireSameSizeAs(*b_vectors, kShapeError) ||
        !a_vectors->RequireMultipleOf(12, kShapeError)) {
        return FailCurrent();
    }

    const std::span<const float> packed_a = a_vectors->Float32();
    const size_t                 count    = packed_a.size() / 3;
    PyObject                    *result   = PyBytes_FromStringAndSize(
        nullptr, static_cast<Py_ssize_t>(a_vectors->Bytes().size()));

    if (result) {
        CPH::MathVec3LerpBatch(packed_a.data(), b_vectors->Float32().data(), alpha, count,
                              reinterpret_cast<float *>(PyBytes_AsString(result)));
    }
    return Adopt(result);
}

static BoundResult MathQuatRotateVec3(MathHolderObject *self, PyObject *const *args,
                                           Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *q_obj;
    PyObject *v_obj;

    const ParseResult parsed =
        self->parsers->mathQuatVec.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, q_obj,
                                          v_obj);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return FailCurrent();
    }

    float q[4];
    float v[3];
    float out[3];
    if (!UnpackQuat(q_obj, q) || !UnpackVec3(v_obj, v)) {
        PyErr_SetString(PyExc_TypeError, "Inputs must be tuples of floats (Quat=4, Vec3=3)");
        return FailCurrent();
    }

    CPH::MathQuatRotateVec3(q, v, out);

    return Adopt(BuildTuple(out[0], out[1], out[2]));
}

static BoundResult MathQuatRotateVec3Batch(MathHolderObject *self, PyObject *const *args,
                                                Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *q_obj;
    PyObject *vecs_obj;

    const ParseResult parsed =
        self->parsers->mathQuatVecBatch.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, q_obj,
                                               vecs_obj);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return FailCurrent();
    }

    float q[4];
    if (!UnpackQuat(q_obj, q)) {
        PyErr_SetString(PyExc_TypeError, "Quaternion must be a tuple of 4 floats");
        return FailCurrent();
    }

    const auto vectors = BufferView::Acquire(vecs_obj);
    if (!vectors) {
        return FailCurrent();
    }

    if (!vectors->RequireMultipleOf(12, "Vector buffer must be a multiple of 12 bytes (3 floats)")) {
        return FailCurrent();
    }

    const std::span<const float> packed_vectors = vectors->Float32();
    const size_t                 count          = packed_vectors.size() / 3;
    PyObject                    *result         = PyBytes_FromStringAndSize(
        nullptr, static_cast<Py_ssize_t>(vectors->Bytes().size()));

    if (result) {
        CPH::MathQuatRotateVec3Batch(q, packed_vectors.data(), count,
                                    reinterpret_cast<float *>(PyBytes_AsString(result)));
    }
    return Adopt(result);
}

static BoundResult MathQuatInverse(MathHolderObject *self, PyObject *const *args,
                                        Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *q_obj;

    const ParseResult parsed =
        self->parsers->mathQuatOp.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, q_obj);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return FailCurrent();
    }

    float q[4];
    float out[4];
    if (!UnpackQuat(q_obj, q)) {
        PyErr_SetString(PyExc_TypeError, "Quaternion must be a tuple of 4 floats");
        return FailCurrent();
    }

    CPH::MathQuatInverse(q, out);

    return Adopt(BuildTuple(out[0], out[1], out[2], out[3]));
}

static BoundResult MathProject(MathHolderObject *self, PyObject *const *args,
                                    Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *v_obj;
    PyObject *mvp_obj;
    PyObject *vp_obj;

    const ParseResult parsed =
        self->parsers->mathProject.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, v_obj, mvp_obj,
                                          vp_obj);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return FailCurrent();
    }

    float v[3];
    float mvp[16];
    float out[3];
    int viewport[4];

    if (!UnpackVec3(v_obj, v) || !UnpackMat44(mvp_obj, mvp) ||
        !UnpackViewport(vp_obj, viewport)) {
        PyErr_SetString(PyExc_TypeError,
                        "Invalid arguments for project (Vec3, Mat44, ViewportTuple)");
        return FailCurrent();
    }

    CPH::MathProject(v, mvp, viewport, out);

    return Adopt(BuildTuple(out[0], out[1], out[2]));
}

static BoundResult MathUnproject(MathHolderObject *self, PyObject *const *args,
                                      Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *v_obj;
    PyObject *mvp_obj;
    PyObject *vp_obj;

    const ParseResult parsed =
        self->parsers->mathUnproject.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, v_obj,
                                            mvp_obj, vp_obj);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return FailCurrent();
    }

    float v[3];
    float mvp[16];
    float out[3];
    int viewport[4];

    if (!UnpackVec3(v_obj, v) || !UnpackMat44(mvp_obj, mvp) ||
        !UnpackViewport(vp_obj, viewport)) {
        PyErr_SetString(PyExc_TypeError,
                        "Invalid arguments for unproject (Vec3, Mat44, ViewportTuple)");
        return FailCurrent();
    }

    CPH::MathUnproject(v, mvp, viewport, out);

    return Adopt(BuildTuple(out[0], out[1], out[2]));
}

static BoundResult MathQuatFromTo(MathHolderObject *self, PyObject *const *args,
                                       Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *v1_obj;
    PyObject *v2_obj;

    const ParseResult parsed =
        self->parsers->mathVecPair.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, v1_obj,
                                          v2_obj);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return FailCurrent();
    }

    float v1[3];
    float v2[3];
    float out[4];
    if (!UnpackVec3(v1_obj, v1) || !UnpackVec3(v2_obj, v2)) {
        PyErr_SetString(PyExc_TypeError, "v1 and v2 must be tuples of 3 floats");
        return FailCurrent();
    }

    CPH::MathQuatFromTo(v1, v2, out);

    return Adopt(BuildTuple(out[0], out[1], out[2], out[3]));
}

static BoundResult MathVec3Dot(MathHolderObject *self, PyObject *const *args,
                                    Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *v1_obj;
    PyObject *v2_obj;

    // Reusing the parser from quat_from_to
    const ParseResult parsed =
        self->parsers->mathVecPair.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, v1_obj,
                                          v2_obj);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return FailCurrent();
    }

    float v1[3];
    float v2[3];
    if (!UnpackVec3(v1_obj, v1) || !UnpackVec3(v2_obj, v2)) {
        PyErr_SetString(PyExc_TypeError, "v1 and v2 must be tuples of 3 floats");
        return FailCurrent();
    }

    float result = CPH::MathVec3Dot(v1, v2);

    return Adopt(PyFloat_FromDouble(static_cast<double>(result)));
}

static BoundResult MathVec3Cross(MathHolderObject *self, PyObject *const *args,
                                      Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *v1_obj;
    PyObject *v2_obj;

    const ParseResult parsed =
        self->parsers->mathVecPair.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, v1_obj,
                                          v2_obj);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return FailCurrent();
    }

    float v1[3];
    float v2[3];
    float out[3];
    if (!UnpackVec3(v1_obj, v1) || !UnpackVec3(v2_obj, v2)) {
        PyErr_SetString(PyExc_TypeError, "v1 and v2 must be tuples of 3 floats");
        return FailCurrent();
    }

    CPH::MathVec3Cross(v1, v2, out);

    return Adopt(BuildTuple(out[0], out[1], out[2]));
}

static BoundResult MathIntersectRayPlane(MathHolderObject *self, PyObject *const *args,
                                              Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *ro_obj;
    PyObject *rd_obj;
    PyObject *po_obj;
    PyObject *pn_obj;

    const ParseResult parsed =
        self->parsers->mathRayPlane.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, ro_obj,
                                           rd_obj, po_obj, pn_obj);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return FailCurrent();
    }

    float ro[3];
    float rd[3];
    float po[3];
    float pn[3];
    float out_p[3];
    float out_t;
    if (!UnpackVec3(ro_obj, ro) || !UnpackVec3(rd_obj, rd) || !UnpackVec3(po_obj, po) ||
        !UnpackVec3(pn_obj, pn)) {
        return FailCurrent();
    }

    int hit = CPH::MathIntersectRayPlane(ro, rd, po, pn, &out_t, out_p);

    if (!hit) {
        return Adopt(BuildTuple(Py_False, 0.0f, Py_None));
    }

    return Adopt(BuildTuple(Py_True, out_t, BuildTuple(out_p[0], out_p[1], out_p[2])));
}

static BoundResult MathQuatGetAxisAngle(MathHolderObject *self, PyObject *const *args,
                                             Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *q_obj;

    const ParseResult parsed =
        self->parsers->mathQuatOp.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, q_obj);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return FailCurrent();
    }

    float q[4];
    float axis[3];
    float angle;
    if (!UnpackQuat(q_obj, q)) {
        PyErr_SetString(PyExc_TypeError, "Quaternion must be a tuple of 4 floats");
        return FailCurrent();
    }

    CPH::MathQuatGetAxisAngle(q, axis, &angle);

    return Adopt(BuildTuple(BuildTuple(axis[0], axis[1], axis[2]), angle));
}

static BoundResult MathQuatFromAxisAngle(MathHolderObject *self, PyObject *const *args,
                                              Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *axis_obj;
    float angle;

    const ParseResult parsed =
        self->parsers->mathAxisAngle.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, axis_obj,
                                            angle);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return FailCurrent();
    }

    float axis[3];
    float out[4];
    if (!UnpackVec3(axis_obj, axis)) {
        PyErr_SetString(PyExc_TypeError, "Axis must be a tuple of 3 floats");
        return FailCurrent();
    }

    CPH::MathQuatFromAxisAngle(axis, angle, out);

    return Adopt(BuildTuple(out[0], out[1], out[2], out[3]));
}

static BoundResult MathVec3DistanceBatch(MathHolderObject *self, PyObject *const *args,
                                              Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *a_obj;
    PyObject *b_obj;

    const ParseResult parsed =
        self->parsers->mathDistBatch.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, a_obj,
                                            b_obj);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return FailCurrent();
    }

    const auto a_vectors = BufferView::Acquire(a_obj);
    if (!a_vectors) {
        return FailCurrent();
    }
    const auto b_vectors = BufferView::Acquire(b_obj);
    if (!b_vectors) {
        return FailCurrent();
    }

    constexpr const char *kShapeError = "Buffers must be equal size and multiples of 12 bytes";
    if (!a_vectors->RequireSameSizeAs(*b_vectors, kShapeError) ||
        !a_vectors->RequireMultipleOf(12, kShapeError)) {
        return FailCurrent();
    }

    const std::span<const float> packed_a = a_vectors->Float32();
    const size_t                 count    = packed_a.size() / 3;
    // Result is a flat array of floats (4 bytes per distance)
    PyObject *result =
        PyBytes_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(count * sizeof(float)));

    if (result) {
        CPH::MathVec3DistanceBatch(packed_a.data(), b_vectors->Float32().data(), count,
                                  reinterpret_cast<float *>(PyBytes_AsString(result)));
    }
    return Adopt(result);
}

static BoundResult MathVec3Normalize(MathHolderObject *self, PyObject *const *args,
                                          Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *v_obj;

    const ParseResult parsed =
        self->parsers->mathVecOp.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, v_obj);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return FailCurrent();
    }

    float v[3];
    float out[3];
    if (!UnpackVec3(v_obj, v)) {
        PyErr_SetString(PyExc_TypeError, "v must be a tuple of 3 floats");
        return FailCurrent();
    }

    CPH::MathVec3Normalize(v, out);

    return Adopt(BuildTuple(out[0], out[1], out[2]));
}

static BoundResult MathMat44GetTranslation(MathHolderObject *self, PyObject *const *args,
                                                Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *mat_obj;

    const ParseResult parsed =
        self->parsers->mathMat.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, mat_obj);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return FailCurrent();
    }

    alignas(kSimdAlignment) float in_mat[kSixteenFloats];
    if (!UnpackMat44(mat_obj, in_mat)) {
        PyErr_SetString(PyExc_TypeError, "Matrix must be a tuple of 16 floats");
        return FailCurrent();
    }

    float out_vec[3];
    CPH::MathMat44GetTranslation(in_mat, out_vec);

    return Adopt(BuildTuple(out_vec[0], out_vec[1], out_vec[2]));
}

static BoundResult MathMat44GetRotation(MathHolderObject *self, PyObject *const *args,
                                             Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *mat_obj;

    const ParseResult parsed =
        self->parsers->mathMat.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, mat_obj);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return FailCurrent();
    }

    alignas(kSimdAlignment) float in_mat[kSixteenFloats];
    if (!UnpackMat44(mat_obj, in_mat)) {
        PyErr_SetString(PyExc_TypeError, "Matrix must be a tuple of 16 floats");
        return FailCurrent();
    }

    float out_quat[4];
    CPH::MathMat44GetRotation(in_mat, out_quat);

    return Adopt(BuildTuple(out_quat[0], out_quat[1], out_quat[2], out_quat[3]));
}

static BoundResult MathMat44Identity([[maybe_unused]] MathHolderObject *self,
                                          PyObject *ignored [[maybe_unused]]) {
    float out[kSixteenFloats];
    CPH::MathMat44Identity(out);

    return Adopt(BuildTuple(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7],
                              out[8], out[9], out[10], out[11], out[12], out[13], out[14], out[15]));
}

static BoundResult MathVec3Reflect(MathHolderObject *self, PyObject *const *args,
                                        Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *v_obj;
    PyObject *n_obj;

    const ParseResult parsed =
        self->parsers->mathReflect.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, v_obj,
                                          n_obj);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return FailCurrent();
    }

    float v[3];
    float n[3];
    float out[3];
    if (!UnpackVec3(v_obj, v) || !UnpackVec3(n_obj, n)) {
        PyErr_SetString(PyExc_TypeError, "v and normal must be tuples of 3 floats");
        return FailCurrent();
    }

    CPH::MathVec3Reflect(v, n, out);

    return Adopt(BuildTuple(out[0], out[1], out[2]));
}

static BoundResult MathVec3Distance(MathHolderObject *self, PyObject *const *args,
                                         Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *v1_obj;
    PyObject *v2_obj;

    const ParseResult parsed =
        self->parsers->mathVecPair.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, v1_obj,
                                          v2_obj);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return FailCurrent();
    }

    float v1[3];
    float v2[3];
    if (!UnpackVec3(v1_obj, v1) || !UnpackVec3(v2_obj, v2)) {
        PyErr_SetString(PyExc_TypeError, "v1 and v2 must be tuples of 3 floats");
        return FailCurrent();
    }

    float dist = CPH::MathVec3Distance(v1, v2);

    return Adopt(PyFloat_FromDouble(static_cast<double>(dist)));
}

static BoundResult MathQuatRotateVec3Inverse(MathHolderObject *self, PyObject *const *args,
                                                  Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *q_obj;
    PyObject *v_obj;

    const ParseResult parsed =
        self->parsers->mathQuatVec.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, q_obj,
                                          v_obj);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return FailCurrent();
    }

    float q[4];
    float v[3];
    float out[3];
    if (!UnpackQuat(q_obj, q) || !UnpackVec3(v_obj, v)) {
        PyErr_SetString(PyExc_TypeError, "Inputs must be tuples of floats (Quat=4, Vec3=3)");
        return FailCurrent();
    }

    CPH::MathQuatRotateVec3Inverse(q, v, out);

    return Adopt(BuildTuple(out[0], out[1], out[2]));
}

// --- Wrapper for single tuple (x, y, z) -> (x, y, z, w) ---
static BoundResult MathEulerToQuat(MathHolderObject *self, PyObject *const *args,
                                        Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *euler_obj;

    const ParseResult parsed =
        self->parsers->mathEulerVec.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, euler_obj);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return FailCurrent();
    }

    float euler[3];
    float q[4];
    if (!UnpackVec3(euler_obj, euler)) {
        PyErr_SetString(PyExc_TypeError, "Euler must be a tuple of 3 floats");
        return FailCurrent();
    }

    CPH::MathEulerToQuat(euler, q);
    return Adopt(BuildTuple(q[0], q[1], q[2], q[3]));
}

// --- Wrapper for batch conversion (Bytes/Buffer) ---
static BoundResult MathEulerToQuatBatch(MathHolderObject *self, PyObject *const *args,
                                             Py_ssize_t nargsf, PyObject *kwnames) {
    PyObject *eulers_obj;

    const ParseResult parsed =
        self->parsers->mathEulerBatch.Parse(args, PyVectorcall_NARGS(nargsf), kwnames,
                                             eulers_obj);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return FailCurrent();
    }

    const auto eulers = BufferView::Acquire(eulers_obj);
    if (!eulers) {
        return FailCurrent();
    }

    if (!eulers->RequireMultipleOf(12, "Buffer must be multiple of 12 bytes (3 floats)")) {
        return FailCurrent();
    }

    const std::span<const float> packed_eulers = eulers->Float32();
    const size_t                 count         = packed_eulers.size() / 3;
    PyObject                    *result        = PyBytes_FromStringAndSize(
        nullptr, static_cast<Py_ssize_t>(count * 16)); // 4 floats per quat

    if (result) {
        CPH::MathEulerToQuatBatch(packed_eulers.data(), count,
                                 reinterpret_cast<float *>(PyBytes_AsString(result)));
    }
    return Adopt(result);
}
static BoundResult MathNew(PyTypeObject *type, [[maybe_unused]] PyObject *args,
                             [[maybe_unused]] PyObject *kwds) {
    MathHolderObject *self = reinterpret_cast<MathHolderObject *>(type->tp_alloc(type, 0));
    if (self == nullptr) {
        return FailCurrent();
    }
    self->parsers = CPH::Python::NewParserGroup<CPH::Python::MathParsers>();
    if (self->parsers == nullptr) {
        Py_DECREF(self);
        return Adopt(PyErr_NoMemory());
    }

    // 3. Call your parser setup logic

    return Adopt(reinterpret_cast<PyObject *>(self));
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

    CPH::Python::Method<&MathGetPerspective>("get_perspective"),
    CPH::Python::Method<&MathGetOrtho>("get_ortho"),
    CPH::Python::Method<&MathGetLookAt>("get_look_at"),
    CPH::Python::Method<&MathGetTrs>("get_trs"),
    CPH::Python::Method<&MathGetTrsBatch>("get_trs_batch"),
    CPH::Python::Method<&MathInverse>("inverse"),
    CPH::Python::Method<&MathMatmul>("matmul"),
    CPH::Python::Method<&MathTransformVec3>("transform_vec3"),
    CPH::Python::Method<&MathMatmulBatch>("matmul_batch"),
    CPH::Python::Method<&MathCullAabb>("cull_aabb"),
    CPH::Python::Method<&MathCullAabbBatch>("cull_aabb_batch"),
    CPH::Python::Method<&MathVec3NormalizeBatch>("vec3_normalize_batch"),
    CPH::Python::Method<&MathQuatFromEuler>("quat_from_euler"),
    CPH::Python::Method<&MathQuatToEuler>("quat_to_euler"),
    CPH::Python::Method<&MathQuatSlerp>("quat_slerp"),
    CPH::Python::Method<&MathQuatMul>("quat_mul"),
    CPH::Python::Method<&MathVec3LerpBatch>("vec3_lerp_batch"),
    CPH::Python::Method<&MathQuatRotateVec3>("quat_rotate_vec3"),
    CPH::Python::Method<&MathQuatRotateVec3Batch>("quat_rotate_vec3_batch"),
    CPH::Python::Method<&MathQuatInverse>("quat_inverse"),
    CPH::Python::Method<&MathProject>("project"),
    CPH::Python::Method<&MathUnproject>("unproject"),
    CPH::Python::Method<&MathQuatFromTo>("quat_from_to"),
    CPH::Python::Method<&MathVec3Dot>("vec3_dot"),
    CPH::Python::Method<&MathVec3Cross>("vec3_cross"),
    CPH::Python::Method<&MathIntersectRayPlane>("intersect_ray_plane"),
    CPH::Python::Method<&MathQuatGetAxisAngle>("quat_get_axis_angle"),
    CPH::Python::Method<&MathQuatFromAxisAngle>("quat_from_axis_angle"),
    CPH::Python::Method<&MathVec3DistanceBatch>("vec3_distance_batch"),
    CPH::Python::Method<&MathVec3Normalize>("vec3_normalize"),
    CPH::Python::Method<&MathMat44GetTranslation>("mat44_get_translation"),
    CPH::Python::Method<&MathMat44GetRotation>("mat44_get_rotation"),
    CPH::Python::NoArgsMethod<&MathMat44Identity>("mat44_identity"),
    CPH::Python::Method<&MathVec3Reflect>("vec3_reflect"),
    CPH::Python::Method<&MathVec3Distance>("vec3_distance"),
    CPH::Python::Method<&MathQuatRotateVec3Inverse>("quat_rotate_vec3_inverse"),
    CPH::Python::Method<&MathEulerToQuat>("euler_to_quat"),
    CPH::Python::Method<&MathEulerToQuatBatch>("euler_to_quat_batch"),
    {nullptr, nullptr, 0, nullptr}};

static PyType_Slot MathServiceSlots[] = {
    CPH::Python::NewSlot<&MathNew>(),
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
