#include "bindings/physics/WorldAdapters.h"
#include "physics/PhysicsWorldCore.h"
#include "bindings/ModuleState.h"
#include "bindings/physics/PhysicsSync.h"

// Baked scene formats (tuple indices and per-object strides)
static constexpr int BAKED_INDEX_POS      = 1;
static constexpr int BAKED_INDEX_ROT      = 2;
static constexpr int BAKED_INDEX_SHAPE    = 3;
static constexpr int BAKED_INDEX_MOT      = 4;
static constexpr int BAKED_INDEX_LAYER    = 5;
static constexpr int BAKED_INDEX_USERDATA = 6;
static constexpr int BAKED_POS_STRIDE     = 4;
static constexpr int BAKED_ROT_STRIDE     = 4;
static constexpr int BAKED_SHAPE_STRIDE   = 5;

CULV_NODISCARD
int PhysicsWorld_resize(PhysicsWorldObject *self, size_t new_capacity) {
    PhysicsWorld *world = &self->physics;
    if (atomic_load_explicit(&world->view_export_count, memory_order_relaxed) > 0) {
        PyErr_SetString(PyExc_BufferError, "Cannot resize while memoryview is active.");
        return -1;
    }

    BLOCK_UNTIL_NOT_STEPPING(self);
    BLOCK_UNTIL_NOT_QUERYING(self);

    const PhysicsWorldStatus status = physics_world_resize(world, new_capacity);
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


void PhysicsWorld_free_members(PhysicsWorldObject *self) {
    physics_world_free_native(&self->physics);
}

// helper: Initialize settings via Python helper
CULV_NODISCARD
int init_settings(PhysicsWorldObject *self, PyObject *settings_dict, GravityVector *gravity,
                  WorldSettings *settings) {
    PyObject *st_module = PyType_GetModule(Py_TYPE(self));
    CulverinState *st   = get_culverin_state(st_module);
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

// helper: Iterate over baked Python data to create initial Jolt bodies
CULV_NODISCARD
int load_baked_scene(PhysicsWorldObject *self, PyObject *baked) {
    // TSan Fix: Load current count atomically
    size_t current_count = atomic_load_explicit(&self->physics.count, memory_order_acquire);

    PyObject *pos_bytes_obj = PyTuple_GetItem(baked, BAKED_INDEX_POS);
    Py_ssize_t pos_len      = PyBytes_Size(pos_bytes_obj);

    if (pos_len < (Py_ssize_t)(current_count * sizeof(PosStride))) {
        PyErr_Format(PyExc_ValueError,
                     "Baked position buffer too small. Expected %zu bytes, got %zd.",
                     current_count * sizeof(PosStride), pos_len);
        return -1;
    }

    // 1. EXTRACT RAW BYTES AS PACKED ARRAYS
    JPH_Real *baked_pos = (JPH_Real *)PyBytes_AsString(PyTuple_GetItem(baked, BAKED_INDEX_POS));
    float *baked_rot    = (float *)PyBytes_AsString(PyTuple_GetItem(baked, BAKED_INDEX_ROT));
    float *baked_shape  = (float *)PyBytes_AsString(PyTuple_GetItem(baked, BAKED_INDEX_SHAPE));
    unsigned char *u_mot =
        (unsigned char *)PyBytes_AsString(PyTuple_GetItem(baked, BAKED_INDEX_MOT));
    unsigned char *u_layer =
        (unsigned char *)PyBytes_AsString(PyTuple_GetItem(baked, BAKED_INDEX_LAYER));
    uint64_t *u_data = (uint64_t *)PyBytes_AsString(PyTuple_GetItem(baked, BAKED_INDEX_USERDATA));

    if (!baked_pos || !baked_rot || !baked_shape || !u_mot || !u_layer || !u_data) {
        PyErr_SetString(PyExc_ValueError, "Invalid or truncated baked data bytes");
        return -1;
    }

    int result = 0;
    SHADOW_LOCK(&self->physics.shadow_lock);

    JPH_BodyInterface *bi = self->physics.body_interface;
    auto shadow_pos       = (PosStride *)self->physics.positions;
    auto shadow_rot       = (AuxStride *)self->physics.rotations;

    for (size_t i = 0; i < current_count; i++) {
        // A. Shape Lookup
        float *s_data    = &baked_shape[i * BAKED_SHAPE_STRIDE];
        float params[4]  = {s_data[1], s_data[2], s_data[3], s_data[4]};
        JPH_Shape *shape = find_or_create_shape_locked(&self->physics, (int)s_data[0], params);

        if (UNLIKELY(!shape)) {
            result = -1;
            break;
        }

        // B. EXTRACT POSITION (Shadow buffers are non-atomic)
        shadow_pos[i].x = baked_pos[i * BAKED_POS_STRIDE + 0];
        shadow_pos[i].y = baked_pos[i * BAKED_POS_STRIDE + 1];
        shadow_pos[i].z = baked_pos[i * BAKED_POS_STRIDE + 2];
        shadow_pos[i].w = 0.0;

        shadow_rot[i].x = baked_rot[i * BAKED_ROT_STRIDE + 0];
        shadow_rot[i].y = baked_rot[i * BAKED_ROT_STRIDE + 1];
        shadow_rot[i].z = baked_rot[i * BAKED_ROT_STRIDE + 2];
        shadow_rot[i].w = baked_rot[i * BAKED_ROT_STRIDE + 3];

        // C. Jolt Create Settings
        JPH_RVec3 j_pos = {shadow_pos[i].x, shadow_pos[i].y, shadow_pos[i].z};
        JPH_Quat j_rot  = {shadow_rot[i].x, shadow_rot[i].y, shadow_rot[i].z, shadow_rot[i].w};

        JPH_BodyCreationSettings *creation = JPH_BodyCreationSettings_Create3(
            shape, &j_pos, &j_rot, (JPH_MotionType)u_mot[i], (JPH_ObjectLayer)u_layer[i]);

        // TSan Fix: Initialize the atomic generation for this slot
        atomic_store_explicit(&self->physics.generations[i], 1, memory_order_relaxed);

        // BodyHandle is CULV_ATOMIC(uint64_t). We create it locally.
        BodyHandle handle = make_handle((uint32_t)i, 1);

        // OPTIMIZATION: Use explicit relaxed load to avoid seq_cst penalty for Jolt
        uint64_t raw_h = handle;
        JPH_BodyCreationSettings_SetUserData(creation, raw_h);

        if (u_mot[i] == 2) {
            JPH_BodyCreationSettings_SetAllowSleeping(creation, true);
        }

        self->physics.body_ids[i] =
            JPH_BodyInterface_CreateAndAddBody(bi, creation, JPH_Activation_Activate);

        uint32_t j_idx = JPH_ID_TO_INDEX(self->physics.body_ids[i]);
        if (self->physics.id_to_handle_map && j_idx < self->physics.max_jolt_bodies) {
            // TSan Fix: Store to shared map using relaxed (world is not yet simulating)
            atomic_store_explicit(&self->physics.id_to_handle_map[j_idx], raw_h, memory_order_relaxed);
        }

        self->physics.slot_to_dense[i] = (uint32_t)i;
        self->physics.dense_to_slot[i] = (uint32_t)i;

        // TSan Fix: Atomic state update
        atomic_store_explicit(&self->physics.slot_states[i], SLOT_ALIVE, memory_order_relaxed);

        self->physics.user_data[i] = u_data[i];
        JPH_BodyCreationSettings_Destroy(creation);
    }

    SHADOW_UNLOCK(&self->physics.shadow_lock);
    return result;
}
