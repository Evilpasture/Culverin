#include "culverin_ragdoll.h"
#include "culverin_arg_indices.h"
#include "culverin_fast_build.h"
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
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. Setup Targets
    PyObject *name_obj = nullptr;
    int parent_idx     = -1; // Default to root

    void *targets[AddJoint_COUNT];
    targets[IDX_AJ_NAME]   = (void *)&name_obj;
    targets[IDX_AJ_PARENT] = (void *)&parent_idx;

    // 2. High Speed Parse
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.AddJointParser, targets)) {
        return nullptr;
    }

    // 3. Logic
    const char *name = PyUnicode_AsUTF8(name_obj);
    if (!name) {
        return nullptr;
    }

    int idx = (int)JPH_Skeleton_AddJoint2(self->skeleton, name, parent_idx);
    return PyLong_FromLong(idx);
}

PyCFunction_DeclareMethodFromModule Skeleton_get_joint_index(SkeletonObject *self,
                                                             PyObject *const *args,
                                                             Py_ssize_t nargs, PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. Setup Targets
    PyObject *name_obj = nullptr;
    void *targets[GetJointIdx_COUNT];
    targets[IDX_GJI_NAME] = (void *)&name_obj;

    // 2. High Speed Parse
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.GetJointIdxParser, targets)) {
        return nullptr;
    }

    // 3. Logic
    const char *name = PyUnicode_AsUTF8(name_obj);
    if (!name) {
        return nullptr;
    }

    int idx = JPH_Skeleton_GetJointIndex(self->skeleton, name);
    return PyLong_FromLong(idx);
}

PyCFunction_DeclareMethodFromModule Skeleton_finalize(SkeletonObject *self,
                                                      PyObject *Py_UNUSED(args)) {
    JPH_Skeleton_CalculateParentJointIndices(self->skeleton);
    if (!JPH_Skeleton_AreJointsCorrectlyOrdered(self->skeleton)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Skeleton joints are out of order (parent must be added before child)");
        return nullptr;
    }
    Py_RETURN_NONE;
}

// --- Ragdoll Settings Implementation ---

PyCFunction_DeclareMethodFromModule PhysicsWorld_create_ragdoll_settings(PhysicsWorldObject *self,
                                                                         PyObject *const *args,
                                                                         Py_ssize_t nargs,
                                                                         PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // --- 1. FAST ARGUMENT PARSING ---
    PyObject *py_skel_obj = nullptr;
    void *targets[RagdollSettings_COUNT];
    targets[IDX_RS_SKELETON] = (void *)&py_skel_obj;

    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.RagdollSettingsParser, targets)) {
        return nullptr;
    }

    // --- 2. TYPE VALIDATION ---
    PyObject *module = PyType_GetModule(Py_TYPE(self));

    // Manual type check (replaces O! format string)
    if (!PyObject_TypeCheck(py_skel_obj, (PyTypeObject *)st->SkeletonType)) {
        PyErr_SetString(PyExc_TypeError, "skeleton must be a Skeleton object");
        return nullptr;
    }
    SkeletonObject *py_skel = (SkeletonObject *)py_skel_obj;

    // --- 3. OBJECT CREATION ---
    RagdollSettingsObject *obj = (RagdollSettingsObject *)PyObject_New(
        RagdollSettingsObject, (PyTypeObject *)st->RagdollSettingsType);
    if (!obj) {
        return nullptr;
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
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. Setup Defaults
    int joint_idx     = 0;
    int parent_idx    = -1;
    int shape_type    = 0;
    float mass        = RAGDOLL_DEFAULT_PART_MASS;
    PyObject *py_size = nullptr;
    PyObject *py_pos  = nullptr;
    float twist_min   = RAGDOLL_DEFAULT_TWIST_MIN;
    float twist_max   = RAGDOLL_DEFAULT_TWIST_MAX;
    float cone_angle  = 0.0f;

    // Default orientation: Axis=X, Normal=Y
    Vec3f axis   = {.x = 1.0f, .y = 0.0f, .z = 0.0f};
    Vec3f normal = {.x = 0.0f, .y = 1.0f, .z = 0.0f};

    void *targets[RagdollAddPart_COUNT];
    targets[IDX_RAP_JOINT]     = (void *)&joint_idx;
    targets[IDX_RAP_SHAPE]     = (void *)&shape_type;
    targets[IDX_RAP_SIZE]      = (void *)&py_size;
    targets[IDX_RAP_MASS]      = (void *)&mass;
    targets[IDX_RAP_PARENT]    = (void *)&parent_idx;
    targets[IDX_RAP_TWIST_MIN] = (void *)&twist_min;
    targets[IDX_RAP_TWIST_MAX] = (void *)&twist_max;
    targets[IDX_RAP_CONE]      = (void *)&cone_angle;
    targets[IDX_RAP_AXIS]      = (void *)&axis;   // Converter calls parse_vec3_f32
    targets[IDX_RAP_NORMAL]    = (void *)&normal; // Converter calls parse_vec3_f32
    targets[IDX_RAP_POS]       = (void *)&py_pos;

    // 2. High Speed Parse
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.RagdollAddPartParser, targets)) {
        return nullptr;
    }

    // 3. Shape Acquisition (Using your existing helper)
    float s[4];
    parse_body_size(py_size, s); // From culverin_parsers.c

    JPH_Shape *shape = nullptr;
    Py_BEGIN_ALLOW_THREADS NATIVE_MUTEX_LOCK(g_jph_trampoline_lock);
    SHADOW_LOCK(&self->world->shadow_lock);
    shape = find_or_create_shape_locked(self->world, shape_type, s);
    SHADOW_UNLOCK(&self->world->shadow_lock);
    NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);
    Py_END_ALLOW_THREADS;

    if (!shape) {
        return PyErr_Format(PyExc_ValueError, "Invalid shape configuration");
    }

    // 4. Validation & Resizing
    auto *skel     = JPH_RagdollSettings_GetSkeleton(self->settings);
    int skel_count = JPH_Skeleton_GetJointCount(skel);
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
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // --- 1. FAST ARGUMENT PARSING ---
    PyObject *settings_obj = nullptr;
    PosStride pos          = {.x = 0, .y = 0, .z = 0};
    AuxStride rot          = {.x = 0, .y = 0, .z = 0, .w = 1.0f};
    uint64_t user_data     = 0;
    uint32_t category      = JOLT_ALL_LAYER_BITS;
    uint32_t mask          = JOLT_ALL_LAYER_BITS;
    uint32_t material_id   = 0;

    void *targets[CreateRagdoll_COUNT];
    targets[IDX_CR_SETTINGS] = (void *)&settings_obj;
    targets[IDX_CR_POS]      = (void *)&pos;
    targets[IDX_CR_ROT]      = (void *)&rot;
    targets[IDX_CR_USER]     = (void *)&user_data;
    targets[IDX_CR_CAT]      = (void *)&category;
    targets[IDX_CR_MASK]     = (void *)&mask;
    targets[IDX_CR_MAT]      = (void *)&material_id;

    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.CreateRagdollParser, targets)) {
        return nullptr;
    }

    // Type Safety Check (Replaces original O! logic)
    if (!PyObject_TypeCheck(settings_obj, (PyTypeObject *)st->RagdollSettingsType)) {
        PyErr_SetString(PyExc_TypeError, "settings must be a RagdollSettings object");
        return nullptr;
    }
    auto *py_settings = (RagdollSettingsObject *)settings_obj;

    // --- 2. JOLT PREPARATION (Logic Preserved) ---
    JPH_Ragdoll *j_rag         = nullptr;
    JPH_Mat4 *neutral_matrices = nullptr;
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
        return nullptr;
    }

    obj->ragdoll = j_rag;
    obj->world   = self;
    Py_INCREF(self);
    obj->body_count = body_count;
    obj->body_slots = (uint32_t *)CULV_RAW_MALLOC(body_count * sizeof(uint32_t));

    // --- 4. SHADOW BUFFER WARM-UP ---
    SHADOW_LOCK(&self->shadow_lock);
    if (atomic_load_explicit(&self->free_count, memory_order_acquire) < body_count) {
        if (PhysicsWorld_resize(self, self->capacity + body_count + RAGDOLL_BODY_BUFFER_INCREMENT) <
            0) {
            SHADOW_UNLOCK(&self->shadow_lock);
            SHADOW_UNLOCK(&self->shadow_lock);
            JPH_Ragdoll_Destroy(j_rag);
            CULV_RAW_FREE(neutral_matrices);
            Py_DECREF(obj);
            return nullptr;
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
        JPH_BodyID bid = JPH_Ragdoll_GetBodyID(j_rag, (int)i);

        // TSan Fix: Pop from free stack atomically
        size_t f_idx  = atomic_fetch_sub_explicit(&self->free_count, 1, memory_order_relaxed) - 1;
        uint32_t slot = self->free_slots[f_idx];
        obj->body_slots[i] = slot;

        // TSan Fix: Increment dense count atomically
        auto dense = (uint32_t)atomic_fetch_add_explicit(&self->count, 1, memory_order_relaxed);

        JPH_RVec3 world_p;
        JPH_Quat world_q;
        JPH_BodyInterface_GetPosition(bi, bid, &world_p);
        JPH_BodyInterface_GetRotation(bi, bid, &world_q);

        shadow_pos[dense]  = (PosStride){world_p.x, world_p.y, world_p.z, 0.0};
        shadow_ppos[dense] = shadow_pos[dense];
        shadow_rot[dense]  = (AuxStride){world_q.x, world_q.y, world_q.z, world_q.w};
        shadow_prot[dense] = shadow_rot[dense];
        shadow_lvel[dense] = (AuxStride){};
        shadow_avel[dense] = (AuxStride){};

        self->body_ids[dense]      = bid;
        self->slot_to_dense[slot]  = dense;
        self->dense_to_slot[dense] = slot;

        // TSan Fix: Fetch generation atomically
        uint32_t gen   = atomic_load_explicit(&self->generations[slot], memory_order_relaxed);
        BodyHandle h   = make_handle(slot, gen);
        uint64_t raw_h = atomic_load_explicit(&h, memory_order_relaxed);

        uint32_t j_idx = JPH_ID_TO_INDEX(bid);
        if (self->id_to_handle_map && j_idx < self->max_jolt_bodies) {
            // TSan Fix: Store handle to shared map atomically (Release ensures shadow writes are
            // visible)
            atomic_store_explicit(&self->id_to_handle_map[j_idx], raw_h, memory_order_release);
        }
        JPH_BodyInterface_SetUserData(bi, bid, raw_h);

        // TSan Fix: Publish body as ALIVE atomically
        atomic_store_explicit(&self->slot_states[slot], SLOT_ALIVE, memory_order_release);

        self->user_data[dense]    = user_data;
        self->categories[dense]   = category;
        self->masks[dense]        = mask;
        self->material_ids[dense] = material_id;
    }

    // TSan Fix: Update view shape with atomic count
    self->view_shape[0] = (Py_ssize_t)atomic_load_explicit(&self->count, memory_order_relaxed);
    SHADOW_UNLOCK(&self->shadow_lock);

    CULV_RAW_FREE(neutral_matrices);
    return (PyObject *)obj;
}

PyCFunction_DeclareMethodFromModule Ragdoll_drive_to_pose(RagdollObject *self,
                                                          PyObject *const *args, Py_ssize_t nargs,
                                                          PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. FAST ARGUMENT PARSING
    PosStride root_p      = {.x = 0, .y = 0, .z = 0};
    AuxStride root_q      = {.x = 0, .y = 0, .z = 0, .w = 1.0f};
    PyObject *py_matrices = nullptr;

    void *targets[RagdollDrive_COUNT];
    targets[IDX_RD_POS]  = (void *)&root_p;
    targets[IDX_RD_ROT]  = (void *)&root_q;
    targets[IDX_RD_MATS] = (void *)&py_matrices;

    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.RagdollDriveParser, targets)) {
        return nullptr;
    }

    if (UNLIKELY(!self->ragdoll || !self->world)) {
        Py_RETURN_NONE;
    }

    // 2. RESOURCE ACQUISITION
    const JPH_RagdollSettings *settings = JPH_Ragdoll_GetRagdollSettings(self->ragdoll);
    auto *skel                          = JPH_RagdollSettings_GetSkeleton(settings);
    int joint_count                     = JPH_Skeleton_GetJointCount(skel);

    // Validate Buffer Size
    Py_buffer view;
    if (PyObject_GetBuffer(py_matrices, &view, PyBUF_SIMPLE) < 0) {
        return nullptr;
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

PyCFunction_DeclareMethodFromModule Ragdoll_get_body_handles(RagdollObject *self,
                                                             PyObject *Py_UNUSED(args)) {
    // Helper to get the Body Handles of the parts so users can manipulate
    // specific limbs
    PyObject *list = PyList_New((Py_ssize_t)self->body_count);
    if (!list) {
        return nullptr;
    }

    SHADOW_LOCK(&self->world->shadow_lock);

    for (size_t i = 0; i < self->body_count; i++) {
        uint32_t slot = self->body_slots[i];

        // TSan Fix: Atomic load of the slot state
        uint8_t state = atomic_load_explicit(&self->world->slot_states[slot], memory_order_acquire);

        if (state == SLOT_ALIVE) {
            // TSan Fix: Atomic load of the generation
            uint32_t gen =
                atomic_load_explicit(&self->world->generations[slot], memory_order_relaxed);

            BodyHandle h = make_handle(slot, gen);

            // Extract raw uint64 from the atomic BodyHandle for Python
            uint64_t raw_h = atomic_load_explicit(&h, memory_order_relaxed);
            PyList_SET_ITEM(list, i, PyLong_FromUnsignedLongLong(raw_h));
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
    if (!list) {
        return nullptr;
    }

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

        // Build the dictionary using FastBuild
        // Format: "index", i, "pos", (x,y,z), "vel", (vx,vy,vz)
        PyObject *dict = FastBuild_Dict("index", i, "pos", FastBuild_Tuple(pos->x, pos->y, pos->z),
                                        "vel", FastBuild_Tuple(vel->x, vel->y, vel->z));

        if (!dict) {
            Py_DECREF(list);
            SHADOW_UNLOCK(&self->world->shadow_lock);
            return nullptr;
        }

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

        // Guard against simulation or in-flight queries
        BLOCK_UNTIL_NOT_STEPPING(self->world);
        BLOCK_UNTIL_NOT_QUERYING(self->world);

        JPH_Ragdoll_RemoveFromPhysicsSystem(self->ragdoll, true);

        if (self->body_slots && self->world->slot_states) {
            for (size_t i = 0; i < self->body_count; i++) {
                uint32_t slot = self->body_slots[i];

                if (slot >= self->world->slot_capacity) {
                    continue;
                }

                // TSan Fix: Atomic load of slot state.
                // Acquire ensures we see all initialization data before calling remove.
                uint8_t state =
                    atomic_load_explicit(&self->world->slot_states[slot], memory_order_acquire);

                if (state != SLOT_ALIVE) {
                    continue;
                }

                // Note: world_remove_body_slot has been refactored
                // to handle atomic count/generation/state updates.
                world_remove_body_slot(self->world, slot);
            }
        }
        SHADOW_UNLOCK(&self->world->shadow_lock);
    }

    if (self->ragdoll) {
        // Jolt Destruction must happen outside the Shadow Lock to prevent deadlock
        // with Jolt internal pool managers.
        JPH_Ragdoll_Destroy(self->ragdoll);
    }

    if (self->body_slots) {
        CULV_RAW_FREE(self->body_slots);
        self->body_slots = nullptr;
    }

    Py_XDECREF(self->world);
    Py_TYPE(self)->tp_free((PyObject *)self);
}