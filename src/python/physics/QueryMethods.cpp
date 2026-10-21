#include "physics/QueryMethods.hpp"
#include "bindings/PythonCommon.hpp"
#include "bindings/FastBuild.hpp"
#include "bindings/parsing/Parsers.hpp"
#include "core/Compiler.hpp"
#include "jolt/Jolt.hpp"

namespace CPH::Python {

static constexpr size_t kCqmVec3StrideBytes = 3 * sizeof(float);

static PyObject *QueryStatusError(Error error, const char *message) {
    if (error.Code().Is(QueryError::OutOfMemory)) {
        return PyErr_NoMemory();
    }

    if (error.Code().Is(QueryError::ShapeFailure)) {
        PyErr_SetString(PyExc_RuntimeError, message);
    } else if (error.Code().Is(QueryError::InvalidArgument)) {
        PyErr_SetString(PyExc_ValueError, message);
    } else {
        PyErr_Format(PyExc_RuntimeError, "Native query failed with error %u",
                     static_cast<unsigned int>(error.Code().Value()));
    }
    return nullptr;
}

static PyObject *OverlapHitsToList(const PhysicsQueryHitBuffer *hits) {
    if (hits->count > static_cast<size_t>(PY_SSIZE_T_MAX)) {
        PyErr_SetString(PyExc_OverflowError, "Overlap query returned too many results");
        return nullptr;
    }

    PyObject *result = PyList_New(static_cast<Py_ssize_t>(hits->count));
    if (!result) {
        return nullptr;
    }

    for (size_t i = 0; i < hits->count; ++i) {
        PyObject *handle =
            PyLong_FromUnsignedLongLong(hits->hits[i].Raw());
        if (!handle) {
            Py_DECREF(result);
            return nullptr;
        }
        PyList_SET_ITEM(result, static_cast<Py_ssize_t>(i), handle);
    }
    return result;
}

[[nodiscard]] PyObject * WorldOverlapSphere(PhysicsWorldObject *self,
                                                              PyObject *const *args, size_t nargsf,
                                                              PyObject *kwnames) {
    PyObject *o_center = nullptr;
    float radius       = 1.0f;

    const ParseResult parsed =
        self->parsers->OverlapSphere.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, o_center,
                                            radius);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    JPH::Real cx;
    JPH::Real cy;
    JPH::Real cz;
    if (ParseResult parsed = ParseVec3Into(o_center, cx, cy, cz); !parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    const JPH::RVec3 center(cx, cy, cz);
    PhysicsQueryHitBuffer hits;

    std::expected<void, Error> status;
    Py_BEGIN_ALLOW_THREADS status = self->physics.OverlapSphere(center, radius, hits);
    Py_END_ALLOW_THREADS

        if (!status) {
        return QueryStatusError(status.error(), "Failed to resolve sphere shape");
    }

    PyObject *result = OverlapHitsToList(&hits);
    return result;
}

[[nodiscard]] PyObject * WorldOverlapAabb(PhysicsWorldObject *self,
                                                            PyObject *const *args, size_t nargsf,
                                                            PyObject *kwnames) {
    PyObject *o_min = nullptr;
    PyObject *o_max = nullptr;

    const ParseResult parsed =
        self->parsers->OverlapAABB.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, o_min,
                                          o_max);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    JPH::Real min_x;
    JPH::Real min_y;
    JPH::Real min_z;
    JPH::Real max_x;
    JPH::Real max_y;
    JPH::Real max_z;
    if (ParseResult parsed = ParseVec3Into(o_min, min_x, min_y, min_z); !parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }
    if (ParseResult parsed = ParseVec3Into(o_max, max_x, max_y, max_z); !parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    const JPH::RVec3 min(min_x, min_y, min_z);
    const JPH::RVec3 max(max_x, max_y, max_z);
    PhysicsQueryHitBuffer hits;

    std::expected<void, Error> status;
    Py_BEGIN_ALLOW_THREADS status = self->physics.OverlapAabb(min, max, hits);
    Py_END_ALLOW_THREADS

        if (!status) {
        return QueryStatusError(status.error(), "overlap_aabb received invalid arguments");
    }

    PyObject *result = OverlapHitsToList(&hits);
    return result;
}

[[nodiscard]] PyObject * WorldRaycast(PhysicsWorldObject *self,
                                                        PyObject *const *args, size_t nargsf,
                                                        PyObject *kwnames) {
    PyObject *o_start      = nullptr;
    PyObject *o_direction  = nullptr;
    float max_distance     = kPhysicsQueryDefaultMaxDistance;
    uint64_t ignore_handle = 0;

    const ParseResult parsed =
        self->parsers->Raycast.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, o_start,
                                      o_direction, max_distance, ignore_handle);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    JPH::Real start_x;
    JPH::Real start_y;
    JPH::Real start_z;
    float direction_x;
    float direction_y;
    float direction_z;
    if (ParseResult parsed = ParseVec3Into(o_start, start_x, start_y, start_z); !parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }
    if (ParseResult parsed = ParseVec3Into(o_direction, direction_x, direction_y, direction_z); !parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    const JPH::RVec3 origin(start_x, start_y, start_z);
    const JPH::Vec3 direction(direction_x, direction_y, direction_z);
    std::expected<PhysicsRaycastHit, Error> status;
    Py_BEGIN_ALLOW_THREADS status =
        self->physics.Raycast(origin, direction, max_distance,
                                static_cast<BodyHandle>(ignore_handle));
    Py_END_ALLOW_THREADS

        if (!status) {
        return QueryStatusError(status.error(), "raycast received invalid arguments");
    }
    const PhysicsRaycastHit &hit = *status;
    if (!hit.has_hit) {
        Py_RETURN_NONE;
    }

    return BuildTuple(hit.handle.Raw(), hit.fraction,
                              BuildTuple(hit.normal.GetX(), hit.normal.GetY(), hit.normal.GetZ()));
}

[[nodiscard]] PyObject * WorldRaycastBatch(PhysicsWorldObject *self,
                                                             PyObject *const *args, size_t nargsf,
                                                             PyObject *kwnames) {
    PyObject *o_starts     = nullptr;
    PyObject *o_directions = nullptr;
    float max_distance     = kPhysicsQueryDefaultMaxDistance;

    const ParseResult parsed =
        self->parsers->RayBatch.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, o_starts,
                                       o_directions, max_distance);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    Py_buffer starts_buffer     = {};
    Py_buffer directions_buffer = {};
    if (PyObject_GetBuffer(o_starts, &starts_buffer, PyBUF_SIMPLE) < 0) [[unlikely]] {
        return nullptr;
    }
    if (PyObject_GetBuffer(o_directions, &directions_buffer, PyBUF_SIMPLE) < 0) [[unlikely]] {
        PyBuffer_Release(&starts_buffer);
        return nullptr;
    }

    if (starts_buffer.len != directions_buffer.len ||
                     starts_buffer.len % static_cast<Py_ssize_t>(kCqmVec3StrideBytes) != 0) [[unlikely]] {
        PyErr_SetString(PyExc_ValueError, "Buffer size mismatch");
        PyBuffer_Release(&starts_buffer);
        PyBuffer_Release(&directions_buffer);
        return nullptr;
    }

    const size_t count =
        static_cast<size_t>(starts_buffer.len / static_cast<Py_ssize_t>(kCqmVec3StrideBytes));
    if (count == 0) {
        PyBuffer_Release(&starts_buffer);
        PyBuffer_Release(&directions_buffer);
        return PyBytes_FromStringAndSize(nullptr, 0);
    }

    if (count > static_cast<size_t>(PY_SSIZE_T_MAX) / sizeof(RayCastBatchResult)) {
        PyErr_SetString(PyExc_OverflowError, "Batch raycast result is too large");
        PyBuffer_Release(&starts_buffer);
        PyBuffer_Release(&directions_buffer);
        return nullptr;
    }

    const Py_ssize_t output_size = static_cast<Py_ssize_t>(count * sizeof(RayCastBatchResult));
    PyObject *result_bytes       = PyBytes_FromStringAndSize(nullptr, output_size);
    if (!result_bytes) [[unlikely]] {
        PyBuffer_Release(&starts_buffer);
        PyBuffer_Release(&directions_buffer);
        return nullptr;
    }

    RayCastBatchResult *results =
        reinterpret_cast<RayCastBatchResult *>(PyBytes_AsString(result_bytes));
    if (!results) [[unlikely]] {
        Py_DECREF(result_bytes);
        PyBuffer_Release(&starts_buffer);
        PyBuffer_Release(&directions_buffer);
        return nullptr;
    }

    std::expected<void, Error> status;
    Py_BEGIN_ALLOW_THREADS status = self->physics.RaycastBatch(
        {static_cast<const float *>(starts_buffer.buf), count * 3},
        {static_cast<const float *>(directions_buffer.buf), count * 3}, max_distance,
        {results, count});
    Py_END_ALLOW_THREADS

        PyBuffer_Release(&starts_buffer);
    PyBuffer_Release(&directions_buffer);

    if (!status) {
        Py_DECREF(result_bytes);
        return QueryStatusError(status.error(), "raycast_batch received invalid buffers");
    }
    return result_bytes;
}

[[nodiscard]] PyObject * WorldShapecast(PhysicsWorldObject *self,
                                                          PyObject *const *args, size_t nargsf,
                                                          PyObject *kwnames) {
    int shape_type         = 0;
    PyObject *o_position   = nullptr;
    PyObject *o_rotation   = nullptr;
    PyObject *o_direction  = nullptr;
    PyObject *o_size       = nullptr;
    uint64_t ignore_handle = 0;

    const ParseResult parsed =
        self->parsers->Shapecast.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, shape_type,
                                        o_position, o_rotation, o_direction, o_size,
                                        ignore_handle);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    JPH::Real position_x;
    JPH::Real position_y;
    JPH::Real position_z;
    float rotation_x;
    float rotation_y;
    float rotation_z;
    float rotation_w;
    float direction_x;
    float direction_y;
    float direction_z;
    float size[4];
    if (ParseResult parsed = ParseVec3Into(o_position, position_x, position_y, position_z); !parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }
    if (ParseResult parsed = ParseQuatInto(o_rotation, rotation_x, rotation_y, rotation_z, rotation_w); !parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }
    if (ParseResult parsed = ParseVec3Into(o_direction, direction_x, direction_y, direction_z); !parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }
    ParseBodySize(o_size, size);

    const JPH::RVec3 position(position_x, position_y, position_z);
    const JPH::Quat rotation(rotation_x, rotation_y, rotation_z, rotation_w);
    const JPH::Vec3 direction(direction_x, direction_y, direction_z);
    std::expected<PhysicsShapecastHit, Error> status;
    Py_BEGIN_ALLOW_THREADS status =
        self->physics.Shapecast(shape_type, position, rotation, direction, size,
                                  static_cast<BodyHandle>(ignore_handle));
    Py_END_ALLOW_THREADS

        if (!status) {
        return QueryStatusError(status.error(), "Invalid shape parameters or cache failure");
    }
    const PhysicsShapecastHit &hit = *status;
    if (!hit.has_hit) {
        Py_RETURN_NONE;
    }

    return BuildTuple(hit.handle.Raw(), hit.fraction,
                              BuildTuple(hit.point.GetX(), hit.point.GetY(), hit.point.GetZ()),
                              BuildTuple(hit.normal.GetX(), hit.normal.GetY(), hit.normal.GetZ()));
}

} // namespace CPH::Python
