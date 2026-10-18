#include "ragdolls/Ragdoll.h"
#include "bindings/BindingCommon.h"
#include "bindings/FastBuild.h"
#include "bindings/PythonSupport.h"
#include "bindings/parsing/ArgIndices.h"
#include "bindings/physics/PhysicsSync.h"
#include "core/CompilerSpecifics.h"
#include "physics/Ragdoll.h"

// Default mass
static constexpr float kRagdollDefaultPartMass = 10.0f;

// Ragdoll constraint angle limits (in radians)
static constexpr float kRagdollDefaultTwistMin = -0.1f;
static constexpr float kRagdollDefaultTwistMax = 0.1f;

PyCFunction_DeclareMethodFromModule CPH_PySkeleton_AddJoint(CPH_SkeletonObject *self, PyObject *const *args,
                                                       size_t nargsf, PyObject *kwnames) {
    const Py_ssize_t nargs = (Py_ssize_t)PyVectorcall_NARGS(nargsf);
    // 1. Setup Targets
    PyObject *name_obj = nullptr;
    int parent_idx     = -1; // Default to root

    const void *const restrict targets[AddJoint_COUNT] = {
        [IDX_AJ_NAME]   = (const void *const restrict)&name_obj,
        [IDX_AJ_PARENT] = (const void *const restrict)&parent_idx,
    };

    // 2. High Speed Parse
    if (!CPH_FAST_PARSE_UNIFIED(args, nargs, kwnames, &self->parsers->AddJointParser, targets)) {
        return nullptr;
    }

    // 3. Logic
    const char *name = PyUnicode_AsUTF8(name_obj);
    if (!name) {
        return nullptr;
    }

    const int idx = CPH_PhysicsRagdollSkeletonAddJoint(self->skeleton, name, parent_idx);
    return PyLong_FromLong(idx);
}

PyCFunction_DeclareMethodFromModule CPH_PySkeleton_GetJointIndex(CPH_SkeletonObject *self,
                                                             PyObject *const *args,
                                                             size_t nargsf, PyObject *kwnames) {
    const Py_ssize_t nargs = (Py_ssize_t)PyVectorcall_NARGS(nargsf);
    // 1. Setup Targets
    PyObject *name_obj                                    = nullptr;
    const void *const restrict targets[GetJointIdx_COUNT] = {
        [IDX_GJI_NAME] = (const void *const restrict)&name_obj,
    };

    // 2. High Speed Parse
    if (!CPH_FAST_PARSE_UNIFIED(args, nargs, kwnames, &self->parsers->GetJointIdxParser,
                                 targets)) {
        return nullptr;
    }

    // 3. Logic
    const char *name = PyUnicode_AsUTF8(name_obj);
    if (!name) {
        return nullptr;
    }

    const int idx = CPH_PhysicsRagdollSkeletonGetJointIndex(self->skeleton, name);
    return PyLong_FromLong(idx);
}

PyCFunction_DeclareMethodFromModule CPH_PySkeleton_Finalize(CPH_SkeletonObject *self,
                                                      PyObject *args [[maybe_unused]]) {
    if (!CPH_PhysicsRagdollSkeletonFinalize(self->skeleton)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Skeleton joints are out of order (parent must be added before child)");
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethodFromModule CPH_PyRagdollSettings_Stabilize(CPH_RagdollSettingsObject *self,
                                                              PyObject *args [[maybe_unused]]) {
    if (CPH_PhysicsRagdollSettingsStabilize(self->settings)) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

PyCFunction_DeclareMethodFromModule CPH_PyRagdoll_DriveToPose(CPH_RagdollObject *self,
                                                          PyObject *const *args, size_t nargsf,
                                                          PyObject *kwnames) {
    const Py_ssize_t nargs = (Py_ssize_t)PyVectorcall_NARGS(nargsf);
    // 1. FAST ARGUMENT PARSING
    CPH_PosStride root_p      = {};
    CPH_AuxStride root_q      = {.w = 1.0f};
    PyObject *py_matrices = nullptr;

    const void *const restrict targets[RagdollDrive_COUNT] = {
        [IDX_RD_POS]  = (const void *const restrict)&root_p,
        [IDX_RD_ROT]  = (const void *const restrict)&root_q,
        [IDX_RD_MATS] = (const void *const restrict)&py_matrices,
    };

    if (!CPH_FAST_PARSE_UNIFIED(args, nargs, kwnames, &self->parsers->RagdollDriveParser,
                                 targets)) {
        return nullptr;
    }

    if (CPH_UNLIKELY(!self->ragdoll || !self->world)) {
        Py_RETURN_NONE;
    }

    // 2. RESOURCE ACQUISITION
    const int joint_count = CPH_PhysicsRagdollGetJointCount(self->ragdoll);

    // Validate Buffer Size
    Py_buffer view;
    if (PyObject_GetBuffer(py_matrices, &view, PyBUF_SIMPLE) < 0) {
        return nullptr;
    }

    const size_t required_size = (size_t)joint_count * sizeof(JPH_Mat4);
    if (CPH_UNLIKELY((size_t)view.len < required_size)) {
        PyBuffer_Release(&view);
        return PyErr_Format(PyExc_ValueError,
                            "Matrices buffer too small. Expected %zu bytes for %d joints, got %zd",
                            required_size, joint_count, view.len);
    }

    JPH_Mat4 *matrices            = (JPH_Mat4 *)view.buf;
    const JPH_RVec3 root_position = {root_p.x, root_p.y, root_p.z};

    // 3. PHYSICS EXECUTION (Shadow Locked)
    CPH_SHADOW_LOCK(&self->world->physics.shadow_lock);
    CPH_BLOCK_UNTIL_NOT_STEPPING(self->world);
    const bool pose_applied =
        CPH_PhysicsRagdollDriveToPose(self->ragdoll, root_position, matrices, joint_count);
    CPH_SHADOW_UNLOCK(&self->world->physics.shadow_lock);

    // 4. CLEANUP
    PyBuffer_Release(&view);
    if (!pose_applied) {
        return PyErr_NoMemory();
    }
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethodFromModule CPH_PyRagdoll_GetBodyHandles(CPH_RagdollObject *self,
                                                             PyObject *args [[maybe_unused]]) {
    if (self->body_count > (size_t)PY_SSIZE_T_MAX ||
        self->body_count > SIZE_MAX / sizeof(CPH_PhysicsRagdollBodyHandleInfo)) {
        return PyErr_NoMemory();
    }

    const Py_ssize_t body_count = (Py_ssize_t)self->body_count;
    PyObject *list              = PyList_New(body_count);
    if (!list) {
        return nullptr;
    }

    CPH_PhysicsRagdollBodyHandleInfo *handles =
        body_count ? CPH_PyAlloc((size_t)body_count * sizeof(*handles)) : nullptr;
    if (body_count && !handles) {
        Py_DECREF(list);
        return PyErr_NoMemory();
    }

    CPH_PhysicsRagdollCopyBodyHandles(&self->world->physics, self->body_slots, (size_t)body_count,
                                      handles);
    for (Py_ssize_t i = 0; i < body_count; ++i) {
        if (handles[i].alive) {
            PyObject *handle = PyLong_FromUnsignedLongLong((uint64_t)handles[i].handle);
            if (!handle) {
                CPH_PyFree(handles);
                Py_DECREF(list);
                return nullptr;
            }
            PyList_SET_ITEM(list, i, handle);
        } else {
            Py_INCREF(Py_None);
            PyList_SET_ITEM(list, i, Py_None);
        }
    }

    CPH_PyFree(handles);
    return list;
}

PyCFunction_DeclareMethodFromModule CPH_PyRagdoll_GetDebugInfo(CPH_RagdollObject *self,
                                                           PyObject *ignored [[maybe_unused]]) {
    if (!self->ragdoll || !self->world) {
        Py_RETURN_NONE;
    }

    const int body_count = CPH_PhysicsRagdollGetBodyCount(self->ragdoll);
    if (body_count < 0 || (size_t)body_count > SIZE_MAX / sizeof(CPH_PhysicsRagdollBodyDebugInfo)) {
        return PyErr_NoMemory();
    }

    PyObject *list = PyList_New(body_count);
    if (!list) {
        return nullptr;
    }

    CPH_PhysicsRagdollBodyDebugInfo *info =
        body_count ? CPH_PyAlloc((size_t)body_count * sizeof(*info)) : nullptr;
    if (body_count && !info) {
        Py_DECREF(list);
        return PyErr_NoMemory();
    }
    if (!CPH_PhysicsRagdollCopyDebugInfo(&self->world->physics, self->ragdoll, info,
                                         (size_t)body_count)) {
        CPH_PyFree(info);
        Py_DECREF(list);
        return PyErr_Format(PyExc_RuntimeError, "Unable to read ragdoll body state");
    }

    for (int i = 0; i < body_count; ++i) {
        const CPH_PhysicsRagdollBodyDebugInfo *part = &info[i];
        PyObject *dict                          = CPH_PY_BUILD_DICT(
            "index", i, "pos",
            CPH_PY_BUILD_TUPLE(part->position.x, part->position.y, part->position.z), "vel",
            CPH_PY_BUILD_TUPLE(part->linear_velocity.x, part->linear_velocity.y,
                                part->linear_velocity.z));

        if (!dict) {
            CPH_PyFree(info);
            Py_DECREF(list);
            return nullptr;
        }

        PyList_SET_ITEM(list, i, dict);
    }

    CPH_PyFree(info);
    return list;
}

PyType_DeclareSlot_VoidFromModule CPH_PySkeleton_Dealloc(CPH_SkeletonObject *self) {
    if (self->skeleton) {
        CPH_PhysicsRagdollSkeletonDestroy(self->skeleton);
    }
    if (self->parsers) {
        CPH_CulverinFreeSkeletonParsers(self->parsers);
        CPH_PyFree(self->parsers);
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

PyType_DeclareSlot_ObjectFromModule
CPH_PySkeleton_New(PyTypeObject *type, PyObject *args [[maybe_unused]], PyObject *kwds [[maybe_unused]]) {
    auto self = (CPH_SkeletonObject *)type->tp_alloc(type, 0);
    if (self) {
        self->skeleton = CPH_PhysicsRagdollSkeletonCreate();
        if (!self->skeleton) {
            Py_DECREF(self);
            return PyErr_NoMemory();
        }
        self->parsers = (CPH_SkeletonParsers *)CPH_PyAlloc(sizeof(CPH_SkeletonParsers));
        if (!self->parsers) {
            CPH_PhysicsRagdollSkeletonDestroy(self->skeleton);
            self->skeleton = nullptr;
            Py_DECREF(self);
            return PyErr_NoMemory();
        }
        CPH_CulverinInitSkeletonParsers(self->parsers);
    }
    return (PyObject *)self;
}

PyType_DeclareSlot_VoidFromModule CPH_PyRagdollSettings_Dealloc(CPH_RagdollSettingsObject *self) {
    if (self->settings) {
        CPH_PhysicsRagdollSettingsDestroy(self->settings);
    }
    if (self->parsers) {
        CPH_CulverinFreeRagdollSettingsParsers(self->parsers);
        CPH_PyFree(self->parsers);
    }
    Py_XDECREF(self->world);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

// --- Ragdoll Instance Implementation ---

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyType_DeclareSlot_VoidFromModule CPH_PyRagdoll_Dealloc(CPH_RagdollObject *self) {
    if (self->ragdoll) {
        CPH_PhysicsWorld *world = self->world ? &self->world->physics : nullptr;
        Py_BEGIN_ALLOW_THREADS CPH_PhysicsRagdollDeallocate(world, self->ragdoll, self->body_slots,
                                                          self->body_count);
        Py_END_ALLOW_THREADS self->ragdoll = nullptr;
    }

    if (self->body_slots) {
        CPH_PyFree(self->body_slots);
        self->body_slots = nullptr;
    }

    if (self->parsers) {
        CPH_CulverinFreeRagdollParsers(self->parsers);
        CPH_PyFree(self->parsers);
    }

    Py_XDECREF(self->world);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

PyCFunction_DeclareMethodFromModule CPH_PyRagdollSettings_AddPart(CPH_RagdollSettingsObject *self,
                                                             PyObject *const *args,
                                                             size_t nargsf, PyObject *kwnames) {
    const Py_ssize_t nargs = (Py_ssize_t)PyVectorcall_NARGS(nargsf);
    // 1. Setup Defaults
    int joint_idx     = 0;
    int parent_idx    = -1;
    int shape_type    = 0;
    float mass        = kRagdollDefaultPartMass;
    PyObject *py_size = nullptr;
    PyObject *py_pos  = nullptr;
    float twist_min   = kRagdollDefaultTwistMin;
    float twist_max   = kRagdollDefaultTwistMax;
    float cone_angle  = 0.0f;

    // Default orientation: Axis=X, Normal=Y
    CPH_Vec3f axis   = {.x = 1.0f, .y = 0.0f, .z = 0.0f};
    CPH_Vec3f normal = {.x = 0.0f, .y = 1.0f, .z = 0.0f};

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
    if (!CPH_FAST_PARSE_UNIFIED(args, nargs, kwnames, &self->parsers->RagdollAddPartParser,
                                 targets)) {
        return nullptr;
    }

    // 3. Parse native shape and pose parameters, then delegate engine operations.
    float shape_params[4];
    CPH_ParseBodySize(py_size, shape_params);

    JPH_Shape *shape             = nullptr;
    Py_BEGIN_ALLOW_THREADS shape = CPH_PhysicsRagdollFindOrCreateShape(
        self->world ? &self->world->physics : nullptr, shape_type, shape_params);
    Py_END_ALLOW_THREADS if (!shape) {
        return PyErr_Format(PyExc_ValueError, "Invalid shape configuration");
    }

    bool has_position  = false;
    JPH_RVec3 position = {0, 0, 0};
    if (py_pos && py_pos != Py_None) {
        CPH_PosStride p_stride = {};
        if (CPH_ParsePyVec3Pos(py_pos, &p_stride)) {
            position     = (JPH_RVec3){p_stride.x, p_stride.y, p_stride.z};
            has_position = true;
        }
    }

    const CPH_PhysicsRagdollPartRequest request = {
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
        CPH_PhysicsRagdollSettingsAddPart(self->settings, shape, &request, &joint_count);

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

#define CPH_RD_FASTCALL(fn, name) CPH_FEAT(fn, name, METH_FASTCALL | METH_KEYWORDS)
#define CPH_RD_NOARGS(fn, name) CPH_FEAT(fn, name, METH_NOARGS)

#define CPH_SKEL_FASTCALL(fn, name) CPH_FEAT(fn, name, METH_FASTCALL | METH_KEYWORDS)
#define CPH_SKEL_NOARGS(fn, name) CPH_FEAT(fn, name, METH_NOARGS)

#define CPH_RDS_FASTCALL(fn, name) CPH_FEAT(fn, name, METH_FASTCALL | METH_KEYWORDS)
#define CPH_RDS_NOARGS(fn, name) CPH_FEAT(fn, name, METH_NOARGS)

static PyMethodDef kRagdollMethods[] = {

                     CPH_RDS_FASTCALL(CPH_PyRagdollSettings_AddPart, add_part), CPH_RDS_NOARGS(CPH_PyRagdollSettings_Stabilize, stabilize), {nullptr, nullptr, 0, nullptr}};

PyType_Spec RagdollSettings_spec = {
    .name      = "culverin._culverin_c.RagdollSettings",
    .basicsize = sizeof(CPH_RagdollSettingsObject),
    .flags     = Py_TPFLAGS_DEFAULT,
    .slots =
        (PyType_Slot[]){

            {.slot = Py_tp_dealloc, .pfunc = CPH_PyRagdollSettings_Dealloc},
            {.slot = Py_tp_methods, .pfunc = kRagdollMethods },
            {},

        },
};

PyType_Spec Skeleton_spec = {
    .name      = "culverin._culverin_c.Skeleton",
    .basicsize = sizeof(CPH_SkeletonObject),
    .flags     = Py_TPFLAGS_DEFAULT,
    .slots =
        (PyType_Slot[]){

            {.slot = Py_tp_new, .pfunc = CPH_PySkeleton_New},
            {.slot = Py_tp_dealloc, .pfunc = CPH_PySkeleton_Dealloc},
            {.slot = Py_tp_methods,
             .pfunc =
                 (PyMethodDef[]){

                     CPH_SKEL_FASTCALL(CPH_PySkeleton_AddJoint, add_joint),
                     CPH_SKEL_FASTCALL(CPH_PySkeleton_GetJointIndex, get_joint_index),
                     CPH_SKEL_NOARGS(CPH_PySkeleton_Finalize, finalize),
                     {}

                 }},
            {},

        },
};

PyType_Spec Ragdoll_spec = {
    .name      = "culverin._culverin_c.Ragdoll",
    .basicsize = sizeof(CPH_RagdollObject),
    .flags     = Py_TPFLAGS_DEFAULT,
    .slots =
        (PyType_Slot[]){
            {.slot = Py_tp_dealloc, .pfunc = CPH_PyRagdoll_Dealloc},
            {.slot = Py_tp_methods,
             .pfunc =
                 (PyMethodDef[]){

                     CPH_RD_FASTCALL(CPH_PyRagdoll_DriveToPose, drive_to_pose),
                     CPH_RD_NOARGS(CPH_PyRagdoll_GetBodyHandles, get_body_handles),
                     CPH_RD_NOARGS(CPH_PyRagdoll_GetDebugInfo, get_debug_info),
                     {}

                 }},
            {},

        },
};