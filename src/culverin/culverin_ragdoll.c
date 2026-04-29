#include "culverin_ragdoll.h"
#include "culverin.h"
#include "culverin_arg_indices.h"
#include "culverin_fast_build.h"
#include "culverin_module.h"
#include "culverin_physics_sync.h"
#include "culverin_physics_world_internal.h"
#include "culverin_python.h"

// Default mass
static constexpr float RAGDOLL_DEFAULT_PART_MASS = 10.0f;

// Ragdoll constraint angle limits (in radians)
static constexpr float RAGDOLL_DEFAULT_TWIST_MIN = -0.1f;
static constexpr float RAGDOLL_DEFAULT_TWIST_MAX = 0.1f;

PyCFunction_DeclareMethodFromModule Skeleton_add_joint(SkeletonObject *self, PyObject *const *args,
                                                       Py_ssize_t nargs, PyObject *kwnames) {
    // 1. Setup Targets
    PyObject *name_obj = nullptr;
    int parent_idx     = -1; // Default to root

    void *targets[AddJoint_COUNT] = {
        [IDX_AJ_NAME]   = (void *)&name_obj,
        [IDX_AJ_PARENT] = (void *)&parent_idx,
    };

    // 2. High Speed Parse
    if (!FastParse_Unified(args, nargs, kwnames, &self->parsers->AddJointParser, targets)) {
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
    // 1. Setup Targets
    PyObject *name_obj               = nullptr;
    void *targets[GetJointIdx_COUNT] = {
        [IDX_GJI_NAME] = (void *)&name_obj,
    };

    // 2. High Speed Parse
    if (!FastParse_Unified(args, nargs, kwnames, &self->parsers->GetJointIdxParser, targets)) {
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

PyCFunction_DeclareMethodFromModule RagdollSettings_stabilize(RagdollSettingsObject *self,
                                                              PyObject *Py_UNUSED(args)) {
    if (JPH_RagdollSettings_Stabilize(self->settings)) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

PyCFunction_DeclareMethodFromModule Ragdoll_drive_to_pose(RagdollObject *self,
                                                          PyObject *const *args, Py_ssize_t nargs,
                                                          PyObject *kwnames) {
    // 1. FAST ARGUMENT PARSING
    PosStride root_p      = {};
    AuxStride root_q      = {.w = 1.0f};
    PyObject *py_matrices = nullptr;

    void *targets[RagdollDrive_COUNT] = {
        [IDX_RD_POS]  = (void *)&root_p,
        [IDX_RD_ROT]  = (void *)&root_q,
        [IDX_RD_MATS] = (void *)&py_matrices,
    };

    if (!FastParse_Unified(args, nargs, kwnames, &self->parsers->RagdollDriveParser, targets)) {
        return nullptr;
    }

    if (UNLIKELY(!self->ragdoll || !self->world)) {
        Py_RETURN_NONE;
    }

    // 2. RESOURCE ACQUISITION
    const JPH_RagdollSettings *settings = JPH_Ragdoll_GetRagdollSettings(self->ragdoll);
    auto skel                           = JPH_RagdollSettings_GetSkeleton(settings);
    int joint_count                     = JPH_Skeleton_GetJointCount(skel);

    // Validate Buffer Size
    Py_buffer view;
    if (PyObject_GetBuffer(py_matrices, &view, PyBUF_SIMPLE) < 0) {
        return nullptr;
    }

    size_t required_size = (size_t)joint_count * sizeof(JPH_Mat4);
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
            uint64_t raw_h = h;
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
void culverin_free_skeleton_parsers(SkeletonParsers *sp);
PyType_DeclareSlot_VoidFromModule Skeleton_dealloc(SkeletonObject *self) {
    if (self->skeleton) {
        JPH_Skeleton_Destroy(self->skeleton);
    }
    if (self->parsers) {
        culverin_free_skeleton_parsers(self->parsers);
        PyMem_Free(self->parsers);
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}
void culverin_init_skeleton_parsers(SkeletonParsers *sp);
PyType_DeclareSlot_ObjectFromModule Skeleton_new(PyTypeObject *type, PyObject *Py_UNUSED(args),
                                                 PyObject *Py_UNUSED(kwds)) {
    auto self = (SkeletonObject *)type->tp_alloc(type, 0);
    if (self) {
        self->skeleton = JPH_Skeleton_Create();
        if (!self->skeleton) {
            Py_DECREF(self);
            return PyErr_NoMemory();
        }
        self->parsers = (SkeletonParsers *)PyMem_Malloc(sizeof(SkeletonParsers));
        if (!self->parsers) {
            JPH_Skeleton_Destroy(self->skeleton);
            Py_DECREF(self);
            return PyErr_NoMemory();
        }
        culverin_init_skeleton_parsers(self->parsers);
    }
    return (PyObject *)self;
}
void culverin_free_ragdoll_settings_parsers(RagdollSettingsParsers *rsp);
PyType_DeclareSlot_VoidFromModule RagdollSettings_dealloc(RagdollSettingsObject *self) {
    if (self->settings) {
        JPH_RagdollSettings_Destroy(self->settings);
    }
    if (self->parsers) {
        culverin_free_ragdoll_settings_parsers(self->parsers);
        PyMem_Free(self->parsers);
    }
    Py_XDECREF(self->world);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

// --- Ragdoll Instance Implementation ---
void culverin_free_ragdoll_parsers(RagdollParsers *rp);
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

    if (self->parsers) {
        culverin_free_ragdoll_parsers(self->parsers);
        PyMem_Free(self->parsers);
    }

    Py_XDECREF(self->world);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

PyCFunction_DeclareMethodFromModule RagdollSettings_add_part(RagdollSettingsObject *self,
                                                             PyObject *const *args,
                                                             Py_ssize_t nargs, PyObject *kwnames) {
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

    void *targets[RagdollAddPart_COUNT] = {
        [IDX_RAP_JOINT] = (void *)&joint_idx,     [IDX_RAP_SHAPE] = (void *)&shape_type,
        [IDX_RAP_SIZE] = (void *)&py_size,        [IDX_RAP_MASS] = (void *)&mass,
        [IDX_RAP_PARENT] = (void *)&parent_idx,   [IDX_RAP_TWIST_MIN] = (void *)&twist_min,
        [IDX_RAP_TWIST_MAX] = (void *)&twist_max, [IDX_RAP_CONE] = (void *)&cone_angle,
        [IDX_RAP_AXIS]   = (void *)&axis,   // Converter calls parse_vec3_f32
        [IDX_RAP_NORMAL] = (void *)&normal, // Converter calls parse_vec3_f32
        [IDX_RAP_POS]    = (void *)&py_pos,
    };

    // 2. High Speed Parse
    if (!FastParse_Unified(args, nargs, kwnames, &self->parsers->RagdollAddPartParser, targets)) {
        return nullptr;
    }

    // 3. Shape Acquisition (Using your existing helper)
    float s[4];
    parse_body_size(py_size, s); // From culverin_parsers.c

    JPH_Shape *shape = nullptr;
    Py_BEGIN_ALLOW_THREADS NATIVE_MUTEX_LOCK(self->world->jph_trampoline_lock);
    SHADOW_LOCK(&self->world->shadow_lock);
    shape = find_or_create_shape_locked(self->world, shape_type, s);
    SHADOW_UNLOCK(&self->world->shadow_lock);
    NATIVE_MUTEX_UNLOCK(self->world->jph_trampoline_lock);
    Py_END_ALLOW_THREADS;

    if (!shape) {
        return PyErr_Format(PyExc_ValueError, "Invalid shape configuration");
    }

    // 4. Validation & Resizing
    auto skel      = JPH_RagdollSettings_GetSkeleton(self->settings);
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

#define RD_FASTCALL(name) CULV_FEAT(Ragdoll, name, METH_FASTCALL | METH_KEYWORDS)
#define RD_NOARGS(name) CULV_FEAT(Ragdoll, name, METH_NOARGS)

#define SKEL_FASTCALL(name) CULV_FEAT(Skeleton, name, METH_FASTCALL | METH_KEYWORDS)
#define SKEL_NOARGS(name) CULV_FEAT(Skeleton, name, METH_NOARGS)

#define RDS_FASTCALL(name) CULV_FEAT(RagdollSettings, name, METH_FASTCALL | METH_KEYWORDS)
#define RDS_NOARGS(name) CULV_FEAT(RagdollSettings, name, METH_NOARGS)

PyType_Spec RagdollSettings_spec = {
    .name      = "culverin._culverin_c.RagdollSettings",
    .basicsize = sizeof(RagdollSettingsObject),
    .flags     = Py_TPFLAGS_DEFAULT,
    .slots =
        (PyType_Slot[]){

            {.slot = Py_tp_dealloc, .pfunc = RagdollSettings_dealloc},
            {.slot = Py_tp_methods,
             .pfunc =
                 (PyMethodDef[]){

                     RDS_FASTCALL(add_part), RDS_NOARGS(stabilize), {}

                 }},
            {},

        },
};

PyType_Spec Skeleton_spec = {
    .name      = "culverin._culverin_c.Skeleton",
    .basicsize = sizeof(SkeletonObject),
    .flags     = Py_TPFLAGS_DEFAULT,
    .slots =
        (PyType_Slot[]){

            {.slot = Py_tp_new, .pfunc = Skeleton_new},
            {.slot = Py_tp_dealloc, .pfunc = Skeleton_dealloc},
            {.slot = Py_tp_methods,
             .pfunc =
                 (PyMethodDef[]){

                     SKEL_FASTCALL(add_joint),
                     SKEL_FASTCALL(get_joint_index),
                     SKEL_NOARGS(finalize),
                     {}

                 }},
            {},

        },
};

PyType_Spec Ragdoll_spec = {
    .name      = "culverin._culverin_c.Ragdoll",
    .basicsize = sizeof(RagdollObject),
    .flags     = Py_TPFLAGS_DEFAULT,
    .slots =
        (PyType_Slot[]){
            {.slot = Py_tp_dealloc, .pfunc = Ragdoll_dealloc},
            {.slot = Py_tp_methods,
             .pfunc =
                 (PyMethodDef[]){

                     RD_FASTCALL(drive_to_pose),
                     RD_NOARGS(get_body_handles),
                     RD_NOARGS(get_debug_info),
                     {}

                 }},
            {},

        },
};