#include "bindings/physics/WorldAdapters.h"
#include "bindings/ModuleState.h"
#include "bindings/physics/PhysicsSync.h"
#include "physics/PhysicsWorld.h"

// Baked scene tuple layout produced by culverin.bake_scene.
enum {
    BAKED_INDEX_POS = 1,
    BAKED_INDEX_ROT,
    BAKED_INDEX_SHAPE,
    BAKED_INDEX_MOT,
    BAKED_INDEX_LAYER,
    BAKED_INDEX_USERDATA,
    BAKED_TUPLE_SIZE,
};

[[nodiscard]]
int CPH_GetBakedSceneCount(PyObject *baked, size_t *out_count) {
    if (out_count == nullptr || !PyTuple_Check(baked) ||
        PyTuple_GET_SIZE(baked) < BAKED_TUPLE_SIZE) {
        PyErr_SetString(PyExc_ValueError, "Invalid baked scene tuple");
        return -1;
    }

    const size_t count = PyLong_AsSize_t(PyTuple_GET_ITEM(baked, 0));
    if (PyErr_Occurred()) {
        return -1;
    }
    *out_count = count;
    return 0;
}

static int baked_buffer_size(size_t count, size_t bytes_per_body, size_t *out_size) {
    if (count > SIZE_MAX / bytes_per_body ||
        count > static_cast<size_t>(PY_SSIZE_T_MAX) / bytes_per_body) {
        PyErr_SetString(PyExc_OverflowError, "Baked scene buffer size is too large");
        return -1;
    }
    *out_size = count * bytes_per_body;
    return 0;
}

static int get_baked_buffer(PyObject *baked, Py_ssize_t index, size_t expected_size,
                            const char *name, const void **out_buffer) {
    PyObject *item = PyTuple_GetItem(baked, index);
    if (item == nullptr) {
        return -1;
    }

    char *buffer    = nullptr;
    Py_ssize_t size = 0;
    if (PyBytes_AsStringAndSize(item, &buffer, &size) < 0) {
        return -1;
    }
    if (static_cast<size_t>(size) < expected_size) {
        PyErr_Format(PyExc_ValueError, "Baked %s buffer too small. Expected %zu bytes, got %zd.",
                     name, expected_size, size);
        return -1;
    }

    *out_buffer = buffer;
    return 0;
}

[[nodiscard]]
int CPH_PyWorld_Resize(CPH_PhysicsWorldObject *self, size_t new_capacity) {
    CPH_PhysicsWorld *world = &self->physics;
    if (atomic_load_explicit(&world->view_export_count, memory_order_relaxed) > 0) {
        PyErr_SetString(PyExc_BufferError, "Cannot resize while memoryview is active.");
        return -1;
    }

    CPH_BLOCK_UNTIL_NOT_STEPPING(self);
    CPH_BLOCK_UNTIL_NOT_QUERYING(self);

    const PhysicsWorldStatus status = CPH_PhysicsWorldResize(world, new_capacity);
    if (status == PHYSICS_WORLD_STATUS_BUFFER_EXPORTED) {
        PyErr_SetString(PyExc_BufferError, "Cannot resize while memoryview is active.");
        return -1;
    }
    if (status != PHYSICS_WORLD_STATUS_OK) {
        PyErr_NoMemory();
        return -1;
    }
    return 0;
}

void CPH_PyWorld_FreeMembers(CPH_PhysicsWorldObject *self) {
    CPH_PhysicsWorldFreeNative(&self->physics);
}

// helper: Initialize settings via Python helper
[[nodiscard]]
int CPH_InitSettings(CPH_PhysicsWorldObject *self, PyObject *settings_dict,
                     CPH_GravityVector *gravity, CPH_WorldSettings *settings) {
    PyObject *st_module = PyType_GetModule(Py_TYPE(self));
    CPH_ModuleState *st = CPH_GetModuleState(st_module);
    PyObject *val_func  = PyObject_GetAttrString(st->helper, "validate_settings");
    if (!val_func) {
        return -1;
    }

    PyObject *norm =
        PyObject_CallFunctionObjArgs(val_func, settings_dict ? settings_dict : Py_None, nullptr);
    Py_DECREF(val_func);
    if (!norm) {
        return -1;
    }

    int ok = PyArg_ParseTuple(norm, "ffffiiiiiii", &gravity->gx, &gravity->gy, &gravity->gz,
                              &settings->penetration_slop, &settings->max_bodies,
                              &settings->max_pairs, &settings->max_contact_constraints,
                              &settings->temp_allocator_size, &settings->max_physics_jobs,
                              &settings->max_physics_barriers, &settings->num_threads);
    Py_DECREF(norm);
    return ok ? 0 : -1;
}

// Decode Python-owned byte buffers, then hand the plain data contract to native physics.
[[nodiscard]]
int CPH_LoadBakedScene(CPH_PhysicsWorldObject *self, PyObject *baked) {
    size_t count;
    if (CPH_GetBakedSceneCount(baked, &count) < 0) {
        return -1;
    }

    size_t positions_size;
    size_t rotations_size;
    size_t shapes_size;
    size_t byte_array_size;
    size_t user_data_size;
    if (baked_buffer_size(count, sizeof(CPH_PosStride), &positions_size) < 0 ||
        baked_buffer_size(count, sizeof(CPH_AuxStride), &rotations_size) < 0 ||
        baked_buffer_size(count, CULV_BAKED_SHAPE_STRIDE * sizeof(float), &shapes_size) < 0 ||
        baked_buffer_size(count, sizeof(uint8_t), &byte_array_size) < 0 ||
        baked_buffer_size(count, sizeof(uint64_t), &user_data_size) < 0) {
        return -1;
    }

    const void *positions;
    const void *rotations;
    const void *shapes;
    const void *motion_types;
    const void *object_layers;
    const void *user_data;
    if (get_baked_buffer(baked, BAKED_INDEX_POS, positions_size, "position", &positions) < 0 ||
        get_baked_buffer(baked, BAKED_INDEX_ROT, rotations_size, "rotation", &rotations) < 0 ||
        get_baked_buffer(baked, BAKED_INDEX_SHAPE, shapes_size, "shape", &shapes) < 0 ||
        get_baked_buffer(baked, BAKED_INDEX_MOT, byte_array_size, "motion type", &motion_types) <
            0 ||
        get_baked_buffer(baked, BAKED_INDEX_LAYER, byte_array_size, "object layer",
                         &object_layers) < 0 ||
        get_baked_buffer(baked, BAKED_INDEX_USERDATA, user_data_size, "user data", &user_data) <
            0) {
        return -1;
    }

    const CPH_BakedSceneData scene = {
        .count         = count,
        .positions     = static_cast<const JPH_Real *>(positions),
        .rotations     = static_cast<const float *>(rotations),
        .shapes        = static_cast<const float *>(shapes),
        .motion_types  = static_cast<const uint8_t *>(motion_types),
        .object_layers = static_cast<const uint8_t *>(object_layers),
        .user_data     = static_cast<const uint64_t *>(user_data),
    };

    const PhysicsWorldStatus status = CPH_PhysicsWorldLoadBakedScene(&self->physics, &scene);
    switch (status) {
    case PHYSICS_WORLD_STATUS_OK:
        return 0;
    case PHYSICS_WORLD_STATUS_OUT_OF_MEMORY:
        PyErr_NoMemory();
        break;
    case PHYSICS_WORLD_STATUS_LIMIT_REACHED:
        PyErr_Format(PyExc_ValueError,
                     "Initial scene contains %zu bodies, exceeding max_bodies (%u).", count,
                     self->physics.max_jolt_bodies);
        break;
    case PHYSICS_WORLD_STATUS_INVALID_ARGUMENT:
        PyErr_SetString(PyExc_ValueError, "Invalid baked scene data");
        break;
    default:
        PyErr_SetString(PyExc_RuntimeError, "Failed to load baked scene");
        break;
    }
    return -1;
}
