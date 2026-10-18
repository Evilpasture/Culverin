#include "bindings/physics/QueryMethods.h"
#include "bindings/BindingCommon.h"
#include "bindings/FastBuild.h"
#include "bindings/parsing/ArgIndices.h"
#include "bindings/parsing/Parsers.h"
#include "core/CompilerSpecifics.h"

static constexpr size_t CQM_VEC3_STRIDE_BYTES = 3 * sizeof(float);

static PyObject *query_status_error(PhysicsQueryStatus status, const char *message) {
    if (status == PHYSICS_QUERY_STATUS_OUT_OF_MEMORY) {
        return PyErr_NoMemory();
    }

    if (status == PHYSICS_QUERY_STATUS_SHAPE_FAILURE) {
        PyErr_SetString(PyExc_RuntimeError, message);
    } else if (status == PHYSICS_QUERY_STATUS_INVALID_ARGUMENT) {
        PyErr_SetString(PyExc_ValueError, message);
    } else {
        PyErr_Format(PyExc_RuntimeError, "Native query failed with status %d", (int)status);
    }
    return nullptr;
}

static PyObject *overlap_hits_to_list(const PhysicsQueryHitBuffer *hits) {
    if (hits->count > (size_t)PY_SSIZE_T_MAX) {
        PyErr_SetString(PyExc_OverflowError, "Overlap query returned too many results");
        return nullptr;
    }

    PyObject *result = PyList_New((Py_ssize_t)hits->count);
    if (!result) {
        return nullptr;
    }

    for (size_t i = 0; i < hits->count; ++i) {
        PyObject *handle = PyLong_FromUnsignedLongLong((unsigned long long)hits->hits[i]);
        if (!handle) {
            Py_DECREF(result);
            return nullptr;
        }
        PyList_SET_ITEM(result, (Py_ssize_t)i, handle);
    }
    return result;
}

PyCFunction_DeclareMethodFromModule PhysicsWorld_overlap_sphere(PhysicsWorldObject *self,
                                                                PyObject *const *args,
                                                                size_t nargsf, PyObject *kwnames) {
    PyObject *o_center                                      = nullptr;
    float radius                                            = 1.0f;
    const void *const restrict targets[OverlapSphere_COUNT] = {
        [IDX_OS_CENTER] = (const void *const restrict)&o_center,
        [IDX_OS_RADIUS] = (const void *const restrict)&radius,
    };

    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->OverlapSphereParser, targets)) {
        return nullptr;
    }

    JPH_Real cx;
    JPH_Real cy;
    JPH_Real cz;
    if (!parse_vec3_direct(o_center, &cx, &cy, &cz)) {
        return nullptr;
    }

    const JPH_RVec3 center = {cx, cy, cz};
    PhysicsQueryHitBuffer hits;
    physics_query_hit_buffer_init(&hits);

    PhysicsQueryStatus status;
    Py_BEGIN_ALLOW_THREADS status =
        physics_world_overlap_sphere(&self->physics, center, radius, &hits);
    Py_END_ALLOW_THREADS

        if (status != PHYSICS_QUERY_STATUS_OK) {
        physics_query_hit_buffer_destroy(&hits);
        return query_status_error(status, "Failed to resolve sphere shape");
    }

    PyObject *result = overlap_hits_to_list(&hits);
    physics_query_hit_buffer_destroy(&hits);
    return result;
}

PyCFunction_DeclareMethodFromModule PhysicsWorld_overlap_aabb(PhysicsWorldObject *self,
                                                              PyObject *const *args, size_t nargsf,
                                                              PyObject *kwnames) {
    PyObject *o_min                                       = nullptr;
    PyObject *o_max                                       = nullptr;
    const void *const restrict targets[OverlapAABB_COUNT] = {
        [IDX_OA_MIN] = (const void *const restrict)&o_min,
        [IDX_OA_MAX] = (const void *const restrict)&o_max,
    };

    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->OverlapAABBParser, targets)) {
        return nullptr;
    }

    JPH_Real min_x;
    JPH_Real min_y;
    JPH_Real min_z;
    JPH_Real max_x;
    JPH_Real max_y;
    JPH_Real max_z;
    if (!parse_vec3_direct(o_min, &min_x, &min_y, &min_z) ||
        !parse_vec3_direct(o_max, &max_x, &max_y, &max_z)) {
        return nullptr;
    }

    const JPH_RVec3 min = {min_x, min_y, min_z};
    const JPH_RVec3 max = {max_x, max_y, max_z};
    PhysicsQueryHitBuffer hits;
    physics_query_hit_buffer_init(&hits);

    PhysicsQueryStatus status;
    Py_BEGIN_ALLOW_THREADS status = physics_world_overlap_aabb(&self->physics, min, max, &hits);
    Py_END_ALLOW_THREADS

        if (status != PHYSICS_QUERY_STATUS_OK) {
        physics_query_hit_buffer_destroy(&hits);
        return query_status_error(status, "overlap_aabb received invalid arguments");
    }

    PyObject *result = overlap_hits_to_list(&hits);
    physics_query_hit_buffer_destroy(&hits);
    return result;
}

PyCFunction_DeclareMethodFromModule PhysicsWorld_raycast(PhysicsWorldObject *self,
                                                         PyObject *const *args, size_t nargsf,
                                                         PyObject *kwnames) {
    PyObject *o_start                                 = nullptr;
    PyObject *o_direction                             = nullptr;
    float max_distance                                = PHYSICS_QUERY_DEFAULT_MAX_DISTANCE;
    uint64_t ignore_handle                            = 0;
    const void *const restrict targets[Raycast_COUNT] = {
        [IDX_RAY_START] = (const void *const restrict)&o_start,
        [IDX_RAY_DIR]   = (const void *const restrict)&o_direction,
        [IDX_RAY_DIST]  = (const void *const restrict)&max_distance,
        [IDX_RAY_IGN]   = (const void *const restrict)&ignore_handle,
    };

    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->RaycastParser, targets)) {
        return nullptr;
    }

    JPH_Real start_x;
    JPH_Real start_y;
    JPH_Real start_z;
    float direction_x;
    float direction_y;
    float direction_z;
    if (!parse_vec3_direct(o_start, &start_x, &start_y, &start_z) ||
        !parse_vec3_direct(o_direction, &direction_x, &direction_y, &direction_z)) {
        return nullptr;
    }

    const JPH_RVec3 origin   = {start_x, start_y, start_z};
    const JPH_Vec3 direction = {direction_x, direction_y, direction_z};
    PhysicsRaycastHit hit    = {};

    PhysicsQueryStatus status;
    Py_BEGIN_ALLOW_THREADS status = physics_world_raycast(
        &self->physics, origin, direction, max_distance, (BodyHandle)ignore_handle, &hit);
    Py_END_ALLOW_THREADS

        if (status != PHYSICS_QUERY_STATUS_OK) {
        return query_status_error(status, "raycast received invalid arguments");
    }
    if (!hit.has_hit) {
        Py_RETURN_NONE;
    }

    return CULV_PY_BUILD_TUPLE((unsigned long long)hit.handle, hit.fraction,
                               CULV_PY_BUILD_TUPLE(hit.normal.x, hit.normal.y, hit.normal.z));
}

PyCFunction_DeclareMethodFromModule PhysicsWorld_raycast_batch(PhysicsWorldObject *self,
                                                               PyObject *const *args, size_t nargsf,
                                                               PyObject *kwnames) {
    PyObject *o_starts                                 = nullptr;
    PyObject *o_directions                             = nullptr;
    float max_distance                                 = PHYSICS_QUERY_DEFAULT_MAX_DISTANCE;
    const void *const restrict targets[RayBatch_COUNT] = {
        [IDX_RB_STARTS] = (const void *const restrict)&o_starts,
        [IDX_RB_DIRS]   = (const void *const restrict)&o_directions,
        [IDX_RB_DIST]   = (const void *const restrict)&max_distance,
    };

    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->RayBatchParser, targets)) {
        return nullptr;
    }

    Py_buffer starts_buffer     = {};
    Py_buffer directions_buffer = {};
    if (UNLIKELY(PyObject_GetBuffer(o_starts, &starts_buffer, PyBUF_SIMPLE) < 0)) {
        return nullptr;
    }
    if (UNLIKELY(PyObject_GetBuffer(o_directions, &directions_buffer, PyBUF_SIMPLE) < 0)) {
        PyBuffer_Release(&starts_buffer);
        return nullptr;
    }

    if (UNLIKELY(starts_buffer.len != directions_buffer.len ||
                 starts_buffer.len % (Py_ssize_t)CQM_VEC3_STRIDE_BYTES != 0)) {
        PyErr_SetString(PyExc_ValueError, "Buffer size mismatch");
        PyBuffer_Release(&starts_buffer);
        PyBuffer_Release(&directions_buffer);
        return nullptr;
    }

    const size_t count = (size_t)(starts_buffer.len / (Py_ssize_t)CQM_VEC3_STRIDE_BYTES);
    if (count == 0) {
        PyBuffer_Release(&starts_buffer);
        PyBuffer_Release(&directions_buffer);
        return PyBytes_FromStringAndSize(nullptr, 0);
    }

    if (count > (size_t)PY_SSIZE_T_MAX / sizeof(RayCastBatchResult)) {
        PyErr_SetString(PyExc_OverflowError, "Batch raycast result is too large");
        PyBuffer_Release(&starts_buffer);
        PyBuffer_Release(&directions_buffer);
        return nullptr;
    }

    const Py_ssize_t output_size = (Py_ssize_t)(count * sizeof(RayCastBatchResult));
    PyObject *result_bytes       = PyBytes_FromStringAndSize(nullptr, output_size);
    if (UNLIKELY(!result_bytes)) {
        PyBuffer_Release(&starts_buffer);
        PyBuffer_Release(&directions_buffer);
        return nullptr;
    }

    RayCastBatchResult *results = (RayCastBatchResult *)PyBytes_AsString(result_bytes);
    if (UNLIKELY(!results)) {
        Py_DECREF(result_bytes);
        PyBuffer_Release(&starts_buffer);
        PyBuffer_Release(&directions_buffer);
        return nullptr;
    }

    PhysicsQueryStatus status;
    Py_BEGIN_ALLOW_THREADS status = physics_world_raycast_batch(
        &self->physics, (const float *)starts_buffer.buf, (const float *)directions_buffer.buf,
        count, max_distance, results);
    Py_END_ALLOW_THREADS

        PyBuffer_Release(&starts_buffer);
    PyBuffer_Release(&directions_buffer);

    if (status != PHYSICS_QUERY_STATUS_OK) {
        Py_DECREF(result_bytes);
        return query_status_error(status, "raycast_batch received invalid buffers");
    }
    return result_bytes;
}

PyCFunction_DeclareMethodFromModule PhysicsWorld_shapecast(PhysicsWorldObject *self,
                                                           PyObject *const *args, size_t nargsf,
                                                           PyObject *kwnames) {
    int shape_type                                      = 0;
    PyObject *o_position                                = nullptr;
    PyObject *o_rotation                                = nullptr;
    PyObject *o_direction                               = nullptr;
    PyObject *o_size                                    = nullptr;
    uint64_t ignore_handle                              = 0;
    const void *const restrict targets[Shapecast_COUNT] = {
        [IDX_SC_SHAPE]  = (const void *const restrict)&shape_type,
        [IDX_SC_POS]    = (const void *const restrict)&o_position,
        [IDX_SC_ROT]    = (const void *const restrict)&o_rotation,
        [IDX_SC_DIR]    = (const void *const)&o_direction,
        [IDX_SC_SIZE]   = (const void *const)&o_size,
        [IDX_SC_IGNORE] = (const void *const)&ignore_handle,
    };

    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->ShapecastParser, targets)) {
        return nullptr;
    }

    JPH_Real position_x;
    JPH_Real position_y;
    JPH_Real position_z;
    float rotation_x;
    float rotation_y;
    float rotation_z;
    float rotation_w;
    float direction_x;
    float direction_y;
    float direction_z;
    float size[4];
    if (!parse_vec3_direct(o_position, &position_x, &position_y, &position_z) ||
        !parse_quat_direct(o_rotation, &rotation_x, &rotation_y, &rotation_z, &rotation_w) ||
        !parse_vec3_direct(o_direction, &direction_x, &direction_y, &direction_z)) {
        return nullptr;
    }
    parse_body_size(o_size, size);

    const JPH_RVec3 position = {position_x, position_y, position_z};
    const JPH_Quat rotation  = {rotation_x, rotation_y, rotation_z, rotation_w};
    const JPH_Vec3 direction = {direction_x, direction_y, direction_z};
    PhysicsShapecastHit hit  = {};

    PhysicsQueryStatus status;
    Py_BEGIN_ALLOW_THREADS status =
        physics_world_shapecast(&self->physics, shape_type, position, rotation, direction, size,
                                (BodyHandle)ignore_handle, &hit);
    Py_END_ALLOW_THREADS

        if (status != PHYSICS_QUERY_STATUS_OK) {
        return query_status_error(status, "Invalid shape parameters or cache failure");
    }
    if (!hit.has_hit) {
        Py_RETURN_NONE;
    }

    return CULV_PY_BUILD_TUPLE((unsigned long long)hit.handle, hit.fraction,
                               CULV_PY_BUILD_TUPLE(hit.point.x, hit.point.y, hit.point.z),
                               CULV_PY_BUILD_TUPLE(hit.normal.x, hit.normal.y, hit.normal.z));
}
