#include "ragdolls/Ragdoll.h"
#include "bindings/BindingCommon.h"
#include "bindings/FastBuild.h"
#include "bindings/PythonSupport.h"
#include "bindings/parsing/ArgIndices.h"
#include "bindings/physics/PhysicsSync.h"
#include "core/CompilerSpecifics.h"
#include "physics/Ragdoll.h"

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

    const void *const restrict targets[AddJoint_COUNT] = {
        [IDX_AJ_NAME]   = (const void *const restrict)&name_obj,
        [IDX_AJ_PARENT] = (const void *const restrict)&parent_idx,
    };

    // 2. High Speed Parse
    if (!CULV_FAST_PARSE_UNIFIED(args, nargs, kwnames, &self->parsers->AddJointParser, targets)) {
        return nullptr;
    }

    // 3. Logic
    const char *name = PyUnicode_AsUTF8(name_obj);
    if (!name) {
        return nullptr;
    }

    const int idx = physics_ragdoll_skeleton_add_joint(self->skeleton, name, parent_idx);
    return PyLong_FromLong(idx);
}

PyCFunction_DeclareMethodFromModule Skeleton_get_joint_index(SkeletonObject *self,
                                                             PyObject *const *args,
                                                             Py_ssize_t nargs, PyObject *kwnames) {
    // 1. Setup Targets
    PyObject *name_obj                                    = nullptr;
    const void *const restrict targets[GetJointIdx_COUNT] = {
        [IDX_GJI_NAME] = (const void *const restrict)&name_obj,
    };

    // 2. High Speed Parse
    if (!CULV_FAST_PARSE_UNIFIED(args, nargs, kwnames, &self->parsers->GetJointIdxParser,
                                 targets)) {
        return nullptr;
    }

    // 3. Logic
    const char *name = PyUnicode_AsUTF8(name_obj);
    if (!name) {
        return nullptr;
    }

    const int idx = physics_ragdoll_skeleton_get_joint_index(self->skeleton, name);
    return PyLong_FromLong(idx);
}

PyCFunction_DeclareMethodFromModule Skeleton_finalize(SkeletonObject *self,
                                                      PyObject *args [[maybe_unused]]) {
    if (!physics_ragdoll_skeleton_finalize(self->skeleton)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Skeleton joints are out of order (parent must be added before child)");
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethodFromModule RagdollSettings_stabilize(RagdollSettingsObject *self,
                                                              PyObject *args [[maybe_unused]]) {
    if (physics_ragdoll_settings_stabilize(self->settings)) {
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

    const void *const restrict targets[RagdollDrive_COUNT] = {
        [IDX_RD_POS]  = (const void *const restrict)&root_p,
        [IDX_RD_ROT]  = (const void *const restrict)&root_q,
        [IDX_RD_MATS] = (const void *const restrict)&py_matrices,
    };

    if (!CULV_FAST_PARSE_UNIFIED(args, nargs, kwnames, &self->parsers->RagdollDriveParser,
                                 targets)) {
        return nullptr;
    }

    if (UNLIKELY(!self->ragdoll || !self->world)) {
        Py_RETURN_NONE;
    }

    // 2. RESOURCE ACQUISITION
    const int joint_count = physics_ragdoll_get_joint_count(self->ragdoll);

    // Validate Buffer Size
    Py_buffer view;
    if (PyObject_GetBuffer(py_matrices, &view, PyBUF_SIMPLE) < 0) {
        return nullptr;
    }

    const size_t required_size = (size_t)joint_count * sizeof(JPH_Mat4);
    if (UNLIKELY((size_t)view.len < required_size)) {
        PyBuffer_Release(&view);
        return PyErr_Format(PyExc_ValueError,
                            "Matrices buffer too small. Expected %zu bytes for %d joints, got %zd",
                            required_size, joint_count, view.len);
    }

    JPH_Mat4 *matrices            = (JPH_Mat4 *)view.buf;
    const JPH_RVec3 root_position = {root_p.x, root_p.y, root_p.z};

    // 3. PHYSICS EXECUTION (Shadow Locked)
    SHADOW_LOCK(&self->world->physics.shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self->world);
    const bool pose_applied =
        physics_ragdoll_drive_to_pose(self->ragdoll, root_position, matrices, joint_count);
    SHADOW_UNLOCK(&self->world->physics.shadow_lock);

    // 4. CLEANUP
    PyBuffer_Release(&view);
    if (!pose_applied) {
        return PyErr_NoMemory();
    }
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethodFromModule Ragdoll_get_body_handles(RagdollObject *self,
                                                             PyObject *args [[maybe_unused]]) {
    if (self->body_count > (size_t)PY_SSIZE_T_MAX ||
        self->body_count > SIZE_MAX / sizeof(PhysicsRagdollBodyHandleInfo)) {
        return PyErr_NoMemory();
    }

    const Py_ssize_t body_count = (Py_ssize_t)self->body_count;
    PyObject *list              = PyList_New(body_count);
    if (!list) {
        return nullptr;
    }

    PhysicsRagdollBodyHandleInfo *handles =
        body_count ? PyMem_Malloc((size_t)body_count * sizeof(*handles)) : nullptr;
    if (body_count && !handles) {
        Py_DECREF(list);
        return PyErr_NoMemory();
    }

    physics_ragdoll_copy_body_handles(&self->world->physics, self->body_slots, (size_t)body_count,
                                      handles);
    for (Py_ssize_t i = 0; i < body_count; ++i) {
        if (handles[i].alive) {
            PyObject *handle = PyLong_FromUnsignedLongLong((uint64_t)handles[i].handle);
            if (!handle) {
                PyMem_Free(handles);
                Py_DECREF(list);
                return nullptr;
            }
            PyList_SET_ITEM(list, i, handle);
        } else {
            Py_INCREF(Py_None);
            PyList_SET_ITEM(list, i, Py_None);
        }
    }

    PyMem_Free(handles);
    return list;
}

PyCFunction_DeclareMethodFromModule Ragdoll_get_debug_info(RagdollObject *self,
                                                           PyObject *ignored [[maybe_unused]]) {
    if (!self->ragdoll || !self->world) {
        Py_RETURN_NONE;
    }

    const int body_count = physics_ragdoll_get_body_count(self->ragdoll);
    if (body_count < 0 || (size_t)body_count > SIZE_MAX / sizeof(PhysicsRagdollBodyDebugInfo)) {
        return PyErr_NoMemory();
    }

    PyObject *list = PyList_New(body_count);
    if (!list) {
        return nullptr;
    }

    PhysicsRagdollBodyDebugInfo *info =
        body_count ? PyMem_Malloc((size_t)body_count * sizeof(*info)) : nullptr;
    if (body_count && !info) {
        Py_DECREF(list);
        return PyErr_NoMemory();
    }
    if (!physics_ragdoll_copy_debug_info(&self->world->physics, self->ragdoll, info,
                                         (size_t)body_count)) {
        PyMem_Free(info);
        Py_DECREF(list);
        return PyErr_Format(PyExc_RuntimeError, "Unable to read ragdoll body state");
    }

    for (int i = 0; i < body_count; ++i) {
        const PhysicsRagdollBodyDebugInfo *part = &info[i];
        PyObject *dict                          = CULV_PY_BUILD_DICT(
            "index", i, "pos",
            CULV_PY_BUILD_TUPLE(part->position.x, part->position.y, part->position.z), "vel",
            CULV_PY_BUILD_TUPLE(part->linear_velocity.x, part->linear_velocity.y,
                                part->linear_velocity.z));

        if (!dict) {
            PyMem_Free(info);
            Py_DECREF(list);
            return nullptr;
        }

        PyList_SET_ITEM(list, i, dict);
    }

    PyMem_Free(info);
    return list;
}

PyType_DeclareSlot_VoidFromModule Skeleton_dealloc(SkeletonObject *self) {
    if (self->skeleton) {
        physics_ragdoll_skeleton_destroy(self->skeleton);
    }
    if (self->parsers) {
        culverin_free_skeleton_parsers(self->parsers);
        PyMem_Free(self->parsers);
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

PyType_DeclareSlot_ObjectFromModule
Skeleton_new(PyTypeObject *type, PyObject *args [[maybe_unused]], PyObject *kwds [[maybe_unused]]) {
    auto self = (SkeletonObject *)type->tp_alloc(type, 0);
    if (self) {
        self->skeleton = physics_ragdoll_skeleton_create();
        if (!self->skeleton) {
            Py_DECREF(self);
            return PyErr_NoMemory();
        }
        self->parsers = (SkeletonParsers *)PyMem_Malloc(sizeof(SkeletonParsers));
        if (!self->parsers) {
            physics_ragdoll_skeleton_destroy(self->skeleton);
            self->skeleton = nullptr;
            Py_DECREF(self);
            return PyErr_NoMemory();
        }
        culverin_init_skeleton_parsers(self->parsers);
    }
    return (PyObject *)self;
}

PyType_DeclareSlot_VoidFromModule RagdollSettings_dealloc(RagdollSettingsObject *self) {
    if (self->settings) {
        physics_ragdoll_settings_destroy(self->settings);
    }
    if (self->parsers) {
        culverin_free_ragdoll_settings_parsers(self->parsers);
        PyMem_Free(self->parsers);
    }
    Py_XDECREF(self->world);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

// --- Ragdoll Instance Implementation ---

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyType_DeclareSlot_VoidFromModule Ragdoll_dealloc(RagdollObject *self) {
    if (self->ragdoll) {
        PhysicsWorld *world = self->world ? &self->world->physics : nullptr;
        Py_BEGIN_ALLOW_THREADS physics_ragdoll_deallocate(world, self->ragdoll, self->body_slots,
                                                          self->body_count);
        Py_END_ALLOW_THREADS self->ragdoll = nullptr;
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

    const void *const restrict targets[RagdollAddPart_COUNT] = {
        [IDX_RAP_JOINT]     = (const void *const restrict)&joint_idx,
        [IDX_RAP_SHAPE]     = (const void *const restrict)&shape_type,
        [IDX_RAP_SIZE]      = (const void *const restrict)&py_size,
        [IDX_RAP_MASS]      = (const void *const restrict)&mass,
        [IDX_RAP_PARENT]    = (const void *const restrict)&parent_idx,
        [IDX_RAP_TWIST_MIN] = (const void *const restrict)&twist_min,
        [IDX_RAP_TWIST_MAX] = (const void *const restrict)&twist_max,
        [IDX_RAP_CONE]      = (const void *const restrict)&cone_angle,
        [IDX_RAP_AXIS]      = (const void *const restrict)&axis,   // Converter calls parse_vec3_f32
        [IDX_RAP_NORMAL]    = (const void *const restrict)&normal, // Converter calls parse_vec3_f32
        [IDX_RAP_POS]       = (const void *const restrict)&py_pos,
    };

    // 2. High Speed Parse
    if (!CULV_FAST_PARSE_UNIFIED(args, nargs, kwnames, &self->parsers->RagdollAddPartParser,
                                 targets)) {
        return nullptr;
    }

    // 3. Parse native shape and pose parameters, then delegate engine operations.
    float shape_params[4];
    parse_body_size(py_size, shape_params);

    JPH_Shape *shape             = nullptr;
    Py_BEGIN_ALLOW_THREADS shape = physics_ragdoll_find_or_create_shape(
        self->world ? &self->world->physics : nullptr, shape_type, shape_params);
    Py_END_ALLOW_THREADS if (!shape) {
        return PyErr_Format(PyExc_ValueError, "Invalid shape configuration");
    }

    bool has_position  = false;
    JPH_RVec3 position = {0, 0, 0};
    if (py_pos && py_pos != Py_None) {
        PosStride p_stride = {};
        if (parse_py_vec3_pos(py_pos, &p_stride)) {
            position     = (JPH_RVec3){p_stride.x, p_stride.y, p_stride.z};
            has_position = true;
        }
    }

    const PhysicsRagdollPartRequest request = {
        .joint_index  = joint_idx,
        .parent_index = parent_idx,
        .mass         = mass,
        .twist_min    = twist_min,
        .twist_max    = twist_max,
        .cone_angle   = cone_angle,
        .axis         = {axis.x, axis.y, axis.z},
        .normal       = {normal.x, normal.y, normal.z},
        .has_position = has_position,
        .position     = position,
    };
    int joint_count = 0;
    const PhysicsRagdollStatus status =
        physics_ragdoll_settings_add_part(self->settings, shape, &request, &joint_count);

    if (status == PHYSICS_RAGDOLL_STATUS_INVALID_SHAPE) {
        return PyErr_Format(PyExc_ValueError, "Invalid shape configuration");
    }
    if (status == PHYSICS_RAGDOLL_STATUS_JOINT_OUT_OF_RANGE) {
        return PyErr_Format(PyExc_IndexError, "Joint index %d out of bounds", joint_idx);
    }
    if (status != PHYSICS_RAGDOLL_STATUS_OK) {
        return PyErr_Format(PyExc_RuntimeError, "Invalid ragdoll settings state");
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