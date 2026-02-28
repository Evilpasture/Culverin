#include "culverin_ragdoll.h"
#include "culverin_arg_indices.h"
#include "culverin_math.h"
#include "culverin_physics_world_internal.h"

// Default mass
static constexpr float RAGDOLL_DEFAULT_PART_MASS = 10.0f;

// Ragdoll constraint angle limits (in radians)
static constexpr float RAGDOLL_DEFAULT_TWIST_MIN = -0.1f;
static constexpr float RAGDOLL_DEFAULT_TWIST_MAX = 0.1f;

// Jolt Physics collision masks (all layers/categories)
static constexpr uint32_t JOLT_ALL_LAYER_BITS = 0xFFFF;

// Buffer allocation increments
static constexpr size_t RAGDOLL_BODY_BUFFER_INCREMENT = 1024;

// JPH_Mat4 size in bytes (16 floats)
static constexpr size_t JPH_MAT4_SIZE_BYTES = 64;

PyCFunction_DeclareMethodFromModule Skeleton_add_joint(SkeletonObject *self, PyObject *const *args,
                                                       Py_ssize_t nargs, PyObject *kwnames) {
    // 1. Setup Targets
    PyObject *name_obj = NULL;
    int parent_idx     = -1; // Default to root

    void *targets[AddJoint_COUNT];
    targets[IDX_AJ_NAME]   = &name_obj;
    targets[IDX_AJ_PARENT] = &parent_idx;

    // 2. High Speed Parse
    if (!FastParse_Unified(args, nargs, kwnames, &AddJointParser, targets)) {
        return NULL;
    }

    // 3. Logic
    const char *name = PyUnicode_AsUTF8(name_obj);
    if (!name)
        return NULL;

    int idx = (int)JPH_Skeleton_AddJoint2(self->skeleton, name, parent_idx);
    return PyLong_FromLong(idx);
}

PyCFunction_DeclareMethodFromModule Skeleton_get_joint_index(SkeletonObject *self,
                                                             PyObject *const *args,
                                                             Py_ssize_t nargs, PyObject *kwnames) {
    // 1. Setup Targets
    PyObject *name_obj = NULL;
    void *targets[GetJointIdx_COUNT];
    targets[IDX_GJI_NAME] = &name_obj;

    // 2. High Speed Parse
    if (!FastParse_Unified(args, nargs, kwnames, &GetJointIdxParser, targets)) {
        return NULL;
    }

    // 3. Logic
    const char *name = PyUnicode_AsUTF8(name_obj);
    if (!name)
        return NULL;

    int idx = JPH_Skeleton_GetJointIndex(self->skeleton, name);
    return PyLong_FromLong(idx);
}

PyCFunction_DeclareMethodFromModule Skeleton_finalize(SkeletonObject *self,
                                                      PyObject *Py_UNUSED(args)) {
    JPH_Skeleton_CalculateParentJointIndices(self->skeleton);
    if (!JPH_Skeleton_AreJointsCorrectlyOrdered(self->skeleton)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Skeleton joints are out of order (parent must be added before child)");
        return NULL;
    }
    Py_RETURN_NONE;
}

// --- Ragdoll Settings Implementation ---

PyCFunction_DeclareMethodFromModule PhysicsWorld_create_ragdoll_settings(PhysicsWorldObject *self,
                                                                         PyObject *const *args,
                                                                         Py_ssize_t nargs,
                                                                         PyObject *kwnames) {
    // --- 1. FAST ARGUMENT PARSING ---
    PyObject *py_skel_obj = NULL;
    void *targets[RagdollSettings_COUNT];
    targets[IDX_RS_SKELETON] = &py_skel_obj;

    if (!FastParse_Unified(args, nargs, kwnames, &RagdollSettingsParser, targets)) {
        return NULL;
    }

    // --- 2. TYPE VALIDATION ---
    PyObject *module  = PyType_GetModule(Py_TYPE(self));
    CulverinState *st = get_culverin_state(module);

    // Manual type check (replaces O! format string)
    if (!PyObject_TypeCheck(py_skel_obj, (PyTypeObject *)st->SkeletonType)) {
        PyErr_SetString(PyExc_TypeError, "skeleton must be a Skeleton object");
        return NULL;
    }
    SkeletonObject *py_skel = (SkeletonObject *)py_skel_obj;

    // --- 3. OBJECT CREATION ---
    RagdollSettingsObject *obj = (RagdollSettingsObject *)PyObject_New(
        RagdollSettingsObject, (PyTypeObject *)st->RagdollSettingsType);
    if (!obj) {
        return NULL;
    }

    // Initialize Jolt settings and link skeleton
    obj->settings = JPH_RagdollSettings_Create();
    JPH_RagdollSettings_SetSkeleton(obj->settings, py_skel->skeleton);

    obj->world = self;
    Py_INCREF(self);

    return (PyObject *)obj;
}

PyCFunction_DeclareMethodFromModule RagdollSettings_add_part(RagdollSettingsObject *self,
                                                             PyObject *const *args,
                                                             Py_ssize_t nargs, PyObject *kwnames) {
    // 1. Setup Defaults
    int joint_idx     = 0;
    int parent_idx    = -1;
    int shape_type    = 0;
    float mass        = RAGDOLL_DEFAULT_PART_MASS;
    PyObject *py_size = NULL;
    PyObject *py_pos  = NULL;
    float twist_min   = RAGDOLL_DEFAULT_TWIST_MIN;
    float twist_max   = RAGDOLL_DEFAULT_TWIST_MAX;
    float cone_angle  = 0.0f;

    // Default orientation: Axis=X, Normal=Y
    Vec3f axis   = {.x = 1.0f, .y = 0.0f, .z = 0.0f};
    Vec3f normal = {.x = 0.0f, .y = 1.0f, .z = 0.0f};

    void *targets[RagdollAddPart_COUNT];
    targets[IDX_RAP_JOINT]     = &joint_idx;
    targets[IDX_RAP_SHAPE]     = &shape_type;
    targets[IDX_RAP_SIZE]      = &py_size;
    targets[IDX_RAP_MASS]      = &mass;
    targets[IDX_RAP_PARENT]    = &parent_idx;
    targets[IDX_RAP_TWIST_MIN] = &twist_min;
    targets[IDX_RAP_TWIST_MAX] = &twist_max;
    targets[IDX_RAP_CONE]      = &cone_angle;
    targets[IDX_RAP_AXIS]      = &axis;   // Converter calls parse_vec3_f32
    targets[IDX_RAP_NORMAL]    = &normal; // Converter calls parse_vec3_f32
    targets[IDX_RAP_POS]       = &py_pos;

    // 2. High Speed Parse
    if (!FastParse_Unified(args, nargs, kwnames, &RagdollAddPartParser, targets)) {
        return NULL;
    }

    // 3. Shape Acquisition (Using your existing helper)
    float s[4];
    parse_body_size(py_size, s); // From culverin_parsers.c

    JPH_Shape *shape = NULL;
    Py_BEGIN_ALLOW_THREADS NATIVE_MUTEX_LOCK(g_jph_trampoline_lock);
    SHADOW_LOCK(&self->world->shadow_lock);
    shape = find_or_create_shape_locked(self->world, shape_type, s);
    SHADOW_UNLOCK(&self->world->shadow_lock);
    NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);
    Py_END_ALLOW_THREADS;

    if (!shape)
        return PyErr_Format(PyExc_ValueError, "Invalid shape configuration");

    // 4. Validation & Resizing
    JPH_Skeleton *skel = (JPH_Skeleton *)JPH_RagdollSettings_GetSkeleton(self->settings);
    int skel_count     = JPH_Skeleton_GetJointCount(skel);
    if (joint_idx < 0 || joint_idx >= skel_count) {
        return PyErr_Format(PyExc_IndexError, "Joint index %d out of bounds", joint_idx);
    }

    if (JPH_RagdollSettings_GetPartCount(self->settings) <= joint_idx) {
        JPH_RagdollSettings_ResizeParts(self->settings, skel_count);
    }

    // 5. Apply Core Part Settings
    JPH_RagdollSettings_SetPartShape(self->settings, joint_idx, shape);
    JPH_RagdollSettings_SetPartMassProperties(self->settings, joint_idx, mass);
    JPH_RagdollSettings_SetPartObjectLayer(self->settings, joint_idx, 1);
    JPH_RagdollSettings_SetPartMotionType(self->settings, joint_idx, JPH_MotionType_Dynamic);

    // 6. Handle Position (Optional)
    if (py_pos && py_pos != Py_None) {
        PosStride p_stride;
        if (parse_py_vec3_pos(py_pos, &p_stride)) {
            JPH_RVec3 p = {.x = p_stride.x, .y = p_stride.y, .z = p_stride.z};
            JPH_RagdollSettings_SetPartPosition(self->settings, joint_idx, &p);
        }
    }

    // 7. Handle Parent Constraint
    if (parent_idx >= 0) {
        JPH_SwingTwistConstraintSettings cs;
        JPH_SwingTwistConstraintSettings_Init(&cs);
        cs.base.enabled = true;

        // Identity positions for local bind
        cs.position1 = (JPH_RVec3){0, 0, 0};
        cs.position2 = (JPH_RVec3){0, 0, 0};

        cs.twistAxis1 = (JPH_Vec3){axis.x, axis.y, axis.z};
        cs.twistAxis2 = (JPH_Vec3){axis.x, axis.y, axis.z};
        cs.planeAxis1 = (JPH_Vec3){normal.x, normal.y, normal.z};
        cs.planeAxis2 = (JPH_Vec3){normal.x, normal.y, normal.z};

        cs.normalHalfConeAngle = cone_angle;
        cs.planeHalfConeAngle  = cone_angle;
        cs.twistMinAngle       = twist_min;
        cs.twistMaxAngle       = twist_max;

        JPH_RagdollSettings_SetPartToParent(self->settings, joint_idx, &cs);
    }

    Py_RETURN_NONE;
}

PyCFunction_DeclareMethodFromModule RagdollSettings_stabilize(RagdollSettingsObject *self,
                                                              PyObject *Py_UNUSED(args)) {
    if (JPH_RagdollSettings_Stabilize(self->settings)) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

PyCFunction_DeclareMethodFromModule PhysicsWorld_create_ragdoll(PhysicsWorldObject *self,
                                                                PyObject *const *args,
                                                                Py_ssize_t nargs,
                                                                PyObject *kwnames) {
    // --- 1. FAST ARGUMENT PARSING ---
    PyObject *settings_obj = NULL;
    PosStride pos          = {.x = 0, .y = 0, .z = 0};
    AuxStride rot          = {.x = 0, .y = 0, .z = 0, .w = 1.0f};
    uint64_t user_data     = 0;
    uint32_t category      = JOLT_ALL_LAYER_BITS;
    uint32_t mask          = JOLT_ALL_LAYER_BITS;
    uint32_t material_id   = 0;

    void *targets[CreateRagdoll_COUNT];
    targets[IDX_CR_SETTINGS] = &settings_obj;
    targets[IDX_CR_POS]      = &pos;
    targets[IDX_CR_ROT]      = &rot;
    targets[IDX_CR_USER]     = &user_data;
    targets[IDX_CR_CAT]      = &category;
    targets[IDX_CR_MASK]     = &mask;
    targets[IDX_CR_MAT]      = &material_id;

    if (!FastParse_Unified(args, nargs, kwnames, &CreateRagdollParser, targets)) {
        return NULL;
    }

    // Type Safety Check (Replaces original O! logic)
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    if (!PyObject_TypeCheck(settings_obj, (PyTypeObject *)st->RagdollSettingsType)) {
        PyErr_SetString(PyExc_TypeError, "settings must be a RagdollSettings object");
        return NULL;
    }
    auto *py_settings = (RagdollSettingsObject *)settings_obj;

    // --- 2. JOLT PREPARATION (Logic Preserved) ---
    JPH_Ragdoll *j_rag         = NULL;
    JPH_Mat4 *neutral_matrices = NULL;
    size_t body_count          = 0;

    Py_BEGIN_ALLOW_THREADS NATIVE_MUTEX_LOCK(g_jph_trampoline_lock);

    JPH_RagdollSettings_CalculateBodyIndexToConstraintIndex(py_settings->settings);
    JPH_RagdollSettings_CalculateConstraintIndexToBodyIdxPair(py_settings->settings);

    j_rag = JPH_RagdollSettings_CreateRagdoll(py_settings->settings, self->system, 0, user_data);

    if (!j_rag) {
        NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);
        Py_BLOCK_THREADS;
        return PyErr_Format(PyExc_RuntimeError, "Jolt failed to create Ragdoll instance");
    }

    auto joint_count =
        (size_t)JPH_Skeleton_GetJointCount(JPH_RagdollSettings_GetSkeleton(py_settings->settings));
    neutral_matrices = (JPH_Mat4 *)CULV_RAW_MALLOC(joint_count * sizeof(JPH_Mat4));

    JPH_RVec3 zero_root = {0, 0, 0};
    JPH_Ragdoll_GetPose2(j_rag, &zero_root, neutral_matrices, true);

    JPH_Quat root_q = {.x = rot.x, .y = rot.y, .z = rot.z, .w = rot.w};
    JPH_STACK_ALLOC(JPH_Mat4, rot_matrix);
    manual_mat4_from_quat(&root_q, rot_matrix);

    for (size_t i = 0; i < joint_count; i++) {
        JPH_STACK_ALLOC(JPH_Mat4, result);
        manual_mat4_multiply(rot_matrix, &neutral_matrices[i], result);
        neutral_matrices[i] = *result;
    }

    JPH_RVec3 root_pos = {pos.x, pos.y, pos.z};
    JPH_Ragdoll_SetPose2(j_rag, &root_pos, neutral_matrices, true);
    JPH_Ragdoll_AddToPhysicsSystem(j_rag, JPH_Activation_Activate, true);

    body_count = (size_t)JPH_Ragdoll_GetBodyCount(j_rag);
    NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);
    Py_END_ALLOW_THREADS;

    // --- 3. PYTHON OBJECT CREATION ---
    auto *obj = (RagdollObject *)PyObject_New(RagdollObject, (PyTypeObject *)st->RagdollType);
    if (!obj) {
        Py_BEGIN_ALLOW_THREADS NATIVE_MUTEX_LOCK(g_jph_trampoline_lock);
        JPH_Ragdoll_Destroy(j_rag);
        NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);
        Py_END_ALLOW_THREADS CULV_RAW_FREE(neutral_matrices);
        return NULL;
    }

    obj->ragdoll = j_rag;
    obj->world   = self;
    Py_INCREF(self);
    obj->body_count = body_count;
    obj->body_slots = (uint32_t *)CULV_RAW_MALLOC(body_count * sizeof(uint32_t));

    // --- 4. SHADOW BUFFER WARM-UP ---
    SHADOW_LOCK(&self->shadow_lock);
    if (self->free_count < body_count) {
        if (PhysicsWorld_resize(self, self->capacity + body_count + RAGDOLL_BODY_BUFFER_INCREMENT) <
            0) {
            SHADOW_UNLOCK(&self->shadow_lock);
            JPH_Ragdoll_Destroy(j_rag);
            CULV_RAW_FREE(neutral_matrices);
            Py_DECREF(obj);
            return NULL;
        }
    }

    JPH_BodyInterface *bi = self->body_interface;
    auto *shadow_pos      = (PosStride *)self->positions;
    auto *shadow_ppos     = (PosStride *)self->prev_positions;
    auto *shadow_rot      = (AuxStride *)self->rotations;
    auto *shadow_prot     = (AuxStride *)self->prev_rotations;
    auto *shadow_lvel     = (AuxStride *)self->linear_velocities;
    auto *shadow_avel     = (AuxStride *)self->angular_velocities;

    for (size_t i = 0; i < body_count; i++) {
        JPH_BodyID bid     = JPH_Ragdoll_GetBodyID(j_rag, (int)i);
        uint32_t slot      = self->free_slots[--self->free_count];
        obj->body_slots[i] = slot;
        auto dense         = (uint32_t)self->count++;

        JPH_RVec3 world_p;
        JPH_Quat world_q;
        JPH_BodyInterface_GetPosition(bi, bid, &world_p);
        JPH_BodyInterface_GetRotation(bi, bid, &world_q);

        shadow_pos[dense]  = (PosStride){.x = world_p.x, .y = world_p.y, .z = world_p.z};
        shadow_ppos[dense] = shadow_pos[dense];

        shadow_rot[dense] =
            (AuxStride){.x = world_q.x, .y = world_q.y, .z = world_q.z, .w = world_q.w};
        shadow_prot[dense] = shadow_rot[dense];

        shadow_lvel[dense] = (AuxStride){};
        shadow_avel[dense] = (AuxStride){};

        self->body_ids[dense]      = bid;
        self->slot_to_dense[slot]  = dense;
        self->dense_to_slot[dense] = slot;
        self->slot_states[slot]    = SLOT_ALIVE;
        self->user_data[dense]     = user_data;
        self->categories[dense]    = category;
        self->masks[dense]         = mask;
        self->material_ids[dense]  = material_id;

        uint32_t j_idx = JPH_ID_TO_INDEX(bid);
        if (self->id_to_handle_map && j_idx < self->max_jolt_bodies) {
            self->id_to_handle_map[j_idx] = make_handle(slot, self->generations[slot]);
        }
        JPH_BodyInterface_SetUserData(bi, bid,
                                      (uint64_t)make_handle(slot, self->generations[slot]));
    }

    self->view_shape[0] = (Py_ssize_t)self->count;
    SHADOW_UNLOCK(&self->shadow_lock);

    CULV_RAW_FREE(neutral_matrices);
    return (PyObject *)obj;
}

PyCFunction_DeclareMethodFromModule Ragdoll_drive_to_pose(RagdollObject *self,
                                                          PyObject *const *args, Py_ssize_t nargs,
                                                          PyObject *kwnames) {
    // 1. FAST ARGUMENT PARSING
    PosStride root_p      = {.x = 0, .y = 0, .z = 0};
    AuxStride root_q      = {.x = 0, .y = 0, .z = 0, .w = 1.0f};
    PyObject *py_matrices = NULL;

    void *targets[RagdollDrive_COUNT];
    targets[IDX_RD_POS]  = &root_p;
    targets[IDX_RD_ROT]  = &root_q;
    targets[IDX_RD_MATS] = &py_matrices;

    if (!FastParse_Unified(args, nargs, kwnames, &RagdollDriveParser, targets)) {
        return NULL;
    }

    if (UNLIKELY(!self->ragdoll || !self->world)) {
        Py_RETURN_NONE;
    }

    // 2. RESOURCE ACQUISITION
    const JPH_RagdollSettings *settings = JPH_Ragdoll_GetRagdollSettings(self->ragdoll);
    JPH_Skeleton *skel                  = (JPH_Skeleton *)JPH_RagdollSettings_GetSkeleton(settings);
    int joint_count                     = JPH_Skeleton_GetJointCount(skel);

    // Validate Buffer Size
    Py_buffer view;
    if (PyObject_GetBuffer(py_matrices, &view, PyBUF_SIMPLE) < 0) {
        return NULL;
    }

    size_t required_size = (size_t)joint_count * JPH_MAT4_SIZE_BYTES;
    if (UNLIKELY((size_t)view.len < required_size)) {
        PyBuffer_Release(&view);
        return PyErr_Format(PyExc_ValueError,
                            "Matrices buffer too small. Expected %zu bytes for %d joints, got %zd",
                            required_size, joint_count, view.len);
    }

    JPH_Mat4 *matrices = (JPH_Mat4 *)view.buf;

    // 3. POSE SETUP
    JPH_SkeletonPose *pose = JPH_SkeletonPose_Create();
    JPH_SkeletonPose_SetSkeleton(pose, skel);

    JPH_STACK_ALLOC(JPH_RVec3, r_pos);
    r_pos->x = root_p.x;
    r_pos->y = root_p.y;
    r_pos->z = root_p.z;
    JPH_SkeletonPose_SetRootOffset(pose, r_pos);

    // 4. PHYSICS EXECUTION (Shadow Locked)
    SHADOW_LOCK(&self->world->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self->world);

    // Set the current world-space pose directly (Teleport)
    JPH_Ragdoll_SetPose2(self->ragdoll, r_pos, matrices, true);

    // Configure and execute the motor drive
    JPH_SkeletonPose_SetJointMatrices(pose, matrices, joint_count);
    JPH_Ragdoll_Activate(self->ragdoll, true);
    JPH_Ragdoll_DriveToPoseUsingMotors(self->ragdoll, pose);

    SHADOW_UNLOCK(&self->world->shadow_lock);

    // 5. CLEANUP
    PyBuffer_Release(&view);
    JPH_SkeletonPose_Destroy(pose);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethodFromModule Ragdoll_get_body_ids(RagdollObject *self,
                                                         PyObject *Py_UNUSED(args)) {
    // Helper to get the Body Handles of the parts so users can manipulate
    // specific limbs
    PyObject *list = PyList_New((Py_ssize_t)self->body_count);
    SHADOW_LOCK(&self->world->shadow_lock);
    for (size_t i = 0; i < self->body_count; i++) {
        uint32_t slot = self->body_slots[i];
        if (self->world->slot_states[slot] == SLOT_ALIVE) {
            uint32_t gen = self->world->generations[slot];
            PyList_SET_ITEM(list, i, PyLong_FromUnsignedLongLong(make_handle(slot, gen)));
        } else {
            Py_INCREF(Py_None);
            PyList_SET_ITEM(list, i, Py_None);
        }
    }
    SHADOW_UNLOCK(&self->world->shadow_lock);
    return list;
}

PyCFunction_DeclareMethodFromModule Ragdoll_get_debug_info(RagdollObject *self,
                                                           PyObject *Py_UNUSED(ignored)) {
    if (!self->ragdoll || !self->world) {
        Py_RETURN_NONE;
    }

    int body_count = JPH_Ragdoll_GetBodyCount(self->ragdoll);
    PyObject *list = PyList_New(body_count);

    JPH_BodyInterface *bi = self->world->body_interface;

    SHADOW_LOCK(&self->world->shadow_lock);
    for (int i = 0; i < body_count; i++) {
        JPH_BodyID bid = JPH_Ragdoll_GetBodyID(self->ragdoll, i);

        JPH_STACK_ALLOC(JPH_RVec3, pos);
        JPH_STACK_ALLOC(JPH_Quat, rot);
        JPH_STACK_ALLOC(JPH_Vec3, vel);

        JPH_BodyInterface_GetPosition(bi, bid, pos);
        JPH_BodyInterface_GetRotation(bi, bid, rot);
        JPH_BodyInterface_GetLinearVelocity(bi, bid, vel);

        PyObject *dict = PyDict_New();
        PyDict_SetItemString(dict, "index", PyLong_FromLong(i));
        PyDict_SetItemString(dict, "pos", Py_BuildValue("(ddd)", pos->x, pos->y, pos->z));
        PyDict_SetItemString(
            dict, "vel", Py_BuildValue("(ddd)", (double)vel->x, (double)vel->y, (double)vel->z));

        PyList_SET_ITEM(list, i, dict);
    }
    SHADOW_UNLOCK(&self->world->shadow_lock);

    return list;
}

PyType_DeclareSlot_VoidFromModule Skeleton_dealloc(SkeletonObject *self) {
    if (self->skeleton) {
        JPH_Skeleton_Destroy(self->skeleton);
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

PyType_DeclareSlot_ObjectFromModule Skeleton_new(PyTypeObject *type, PyObject *Py_UNUSED(args),
                                                 PyObject *Py_UNUSED(kwds)) {
    auto *self = (SkeletonObject *)type->tp_alloc(type, 0);
    if (self) {
        self->skeleton = JPH_Skeleton_Create();
        if (!self->skeleton) {
            Py_DECREF(self);
            return PyErr_NoMemory();
        }
    }
    return (PyObject *)self;
}

PyType_DeclareSlot_VoidFromModule RagdollSettings_dealloc(RagdollSettingsObject *self) {
    if (self->settings) {
        JPH_RagdollSettings_Destroy(self->settings);
    }
    Py_XDECREF(self->world);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

// --- Ragdoll Instance Implementation ---
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyType_DeclareSlot_VoidFromModule Ragdoll_dealloc(RagdollObject *self) {
    if (self->world && self->ragdoll) {
        SHADOW_LOCK(&self->world->shadow_lock);

        BLOCK_UNTIL_NOT_STEPPING(self->world);
        BLOCK_UNTIL_NOT_QUERYING(self->world);

        JPH_Ragdoll_RemoveFromPhysicsSystem(self->ragdoll, true);

        // Validate pointers before iteration to prevent corruption
        if (self->body_slots && self->world->slot_states) {
            for (size_t i = 0; i < self->body_count; i++) {
                uint32_t slot = self->body_slots[i];

                // Boundary check
                if (slot >= self->world->slot_capacity) {
                    continue;
                }

                if (self->world->slot_states[slot] != SLOT_ALIVE) {
                    continue;
                }
                world_remove_body_slot(self->world, slot);
            }
        }
        SHADOW_UNLOCK(&self->world->shadow_lock);
    }

    if (self->ragdoll) {
        JPH_Ragdoll_Destroy(self->ragdoll);
    }
    if (self->body_slots) {
        CULV_RAW_FREE(self->body_slots);
        self->body_slots = NULL; // Prevent double-free in weird recursion cases
    }

    Py_XDECREF(self->world);
    Py_TYPE(self)->tp_free((PyObject *)self);
}