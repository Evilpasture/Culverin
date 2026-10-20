#include "ragdolls/Ragdoll.hpp"
#include "bindings/PythonCommon.hpp"
#include "bindings/FastBuild.hpp"
#include "bindings/PythonSupport.hpp"
#include "core/Compiler.hpp"
#include "physics/PhysicsSync.hpp"
#include "physics/Ragdoll.hpp"

namespace CPH::Python {

// Default mass
static constexpr float kRagdollDefaultPartMass = 10.0f;

// Ragdoll constraint angle limits (in radians)
static constexpr float kRagdollDefaultTwistMin = -0.1f;
static constexpr float kRagdollDefaultTwistMax = 0.1f;

[[nodiscard]] PyObject * SkeletonAddJoint(SkeletonObject *self,
                                                            PyObject *const *args, size_t nargsf,
                                                            PyObject *kwnames) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    // 1. Setup Targets
    PyObject *name_obj = nullptr;
    int parent_idx     = -1; // Default to root

    // 2. High Speed Parse
    const ParseResult parsed =
        self->parsers->AddJoint.Parse(args, nargs, kwnames, name_obj, parent_idx);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    // 3. Logic
    const char *name = PyUnicode_AsUTF8(name_obj);
    if (!name) {
        return nullptr;
    }

    const int idx = PhysicsRagdollSkeletonAddJoint(self->skeleton, name, parent_idx);
    return PyLong_FromLong(idx);
}

[[nodiscard]] PyObject * SkeletonGetJointIndex(SkeletonObject *self,
                                                                 PyObject *const *args,
                                                                 size_t nargsf, PyObject *kwnames) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    // 1. Setup Targets
    PyObject *name_obj = nullptr;

    // 2. High Speed Parse
    const ParseResult parsed =
        self->parsers->GetJointIdx.Parse(args, nargs, kwnames, name_obj);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    // 3. Logic
    const char *name = PyUnicode_AsUTF8(name_obj);
    if (!name) {
        return nullptr;
    }

    const int idx = PhysicsRagdollSkeletonGetJointIndex(self->skeleton, name);
    return PyLong_FromLong(idx);
}

[[nodiscard]] PyObject * SkeletonFinalize(SkeletonObject *self,
                                                            PyObject *args [[maybe_unused]]) {
    if (!PhysicsRagdollSkeletonFinalize(self->skeleton)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Skeleton joints are out of order (parent must be added before child)");
        return nullptr;
    }
    Py_RETURN_NONE;
}

[[nodiscard]] PyObject *
RagdollSettingsStabilize(RagdollSettingsObject *self, PyObject *args [[maybe_unused]]) {
    if (PhysicsRagdollSettingsStabilize(self->settings)) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

[[nodiscard]] PyObject * RagdollDriveToPose(RagdollObject *self,
                                                              PyObject *const *args, size_t nargsf,
                                                              PyObject *kwnames) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    // 1. FAST ARGUMENT PARSING
    PosStride root_p  = {};
    AuxStride root_q  = {.w = 1.0f};
    PyObject *py_matrices = nullptr;

    const ParseResult parsed =
        self->parsers->RagdollDrive.Parse(args, nargs, kwnames, root_p, root_q, py_matrices);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    if (CPH_UNLIKELY(!self->ragdoll || !self->world)) {
        Py_RETURN_NONE;
    }

    // 2. RESOURCE ACQUISITION
    const int joint_count = PhysicsRagdollGetJointCount(self->ragdoll);

    // Validate Buffer Size
    Py_buffer view;
    if (PyObject_GetBuffer(py_matrices, &view, PyBUF_SIMPLE) < 0) {
        return nullptr;
    }

    const size_t required_size = static_cast<size_t>(joint_count) * sizeof(JPH_Mat4);
    if (CPH_UNLIKELY(static_cast<size_t>(view.len) < required_size)) {
        PyBuffer_Release(&view);
        return PyErr_Format(PyExc_ValueError,
                            "Matrices buffer too small. Expected %zu bytes for %d joints, got %zd",
                            required_size, joint_count, view.len);
    }

    JPH_Mat4 *matrices            = static_cast<JPH_Mat4 *>(view.buf);
    const JPH_RVec3 root_position = {root_p.x, root_p.y, root_p.z};

    // 3. PHYSICS EXECUTION (Shadow Locked)
    (self->world->physics.shadow_lock).lock();
    BlockUntilNotStepping(self->world);
    const bool pose_applied =
        PhysicsRagdollDriveToPose(self->ragdoll, root_position,
                                      {matrices, static_cast<size_t>(joint_count)});
    (self->world->physics.shadow_lock).unlock();

    // 4. CLEANUP
    PyBuffer_Release(&view);
    if (!pose_applied) {
        return PyErr_NoMemory();
    }
    Py_RETURN_NONE;
}

[[nodiscard]] PyObject * RagdollGetBodyHandles(RagdollObject *self,
                                                                 PyObject *args [[maybe_unused]]) {
    if (self->body_count > static_cast<size_t>(PY_SSIZE_T_MAX) ||
        self->body_count > SIZE_MAX / sizeof(PhysicsRagdollBodyHandleInfo)) {
        return PyErr_NoMemory();
    }

    const Py_ssize_t body_count = static_cast<Py_ssize_t>(self->body_count);
    PyObject *list              = PyList_New(body_count);
    if (!list) {
        return nullptr;
    }

    PhysicsRagdollBodyHandleInfo *handles =
        body_count ? static_cast<PhysicsRagdollBodyHandleInfo *>(
                         Allocate(static_cast<size_t>(body_count) * sizeof(*handles)))
                   : nullptr;
    if (body_count && !handles) {
        Py_DECREF(list);
        return PyErr_NoMemory();
    }

    self->world->physics.CopyRagdollBodyHandles(
        {self->body_slots, static_cast<size_t>(body_count)},
        {handles, static_cast<size_t>(body_count)});
    for (Py_ssize_t i = 0; i < body_count; ++i) {
        if (handles[i].alive) {
            PyObject *handle =
                PyLong_FromUnsignedLongLong(static_cast<uint64_t>(handles[i].handle));
            if (!handle) {
                Deallocate(handles);
                Py_DECREF(list);
                return nullptr;
            }
            PyList_SET_ITEM(list, i, handle);
        } else {
            Py_INCREF(Py_None);
            PyList_SET_ITEM(list, i, Py_None);
        }
    }

    Deallocate(handles);
    return list;
}

[[nodiscard]] PyObject * RagdollGetDebugInfo(RagdollObject *self,
                                                               PyObject *ignored [[maybe_unused]]) {
    if (!self->ragdoll || !self->world) {
        Py_RETURN_NONE;
    }

    const int body_count = PhysicsRagdollGetBodyCount(self->ragdoll);
    if (body_count < 0 ||
        static_cast<size_t>(body_count) > SIZE_MAX / sizeof(PhysicsRagdollBodyDebugInfo)) {
        return PyErr_NoMemory();
    }

    PyObject *list = PyList_New(body_count);
    if (!list) {
        return nullptr;
    }

    PhysicsRagdollBodyDebugInfo *info =
        body_count ? static_cast<PhysicsRagdollBodyDebugInfo *>(
                         Allocate(static_cast<size_t>(body_count) * sizeof(*info)))
                   : nullptr;
    if (body_count && !info) {
        Py_DECREF(list);
        return PyErr_NoMemory();
    }
    if (!self->world->physics.CopyRagdollDebugInfo(
            *self->ragdoll, {info, static_cast<size_t>(body_count)})) {
        Deallocate(info);
        Py_DECREF(list);
        return PyErr_Format(PyExc_RuntimeError, "Unable to read ragdoll body state");
    }

    for (int i = 0; i < body_count; ++i) {
        const PhysicsRagdollBodyDebugInfo *part = &info[i];
        PyObject *dict                              = BuildDictionary(
            "index", i, "pos",
            BuildTuple(part->position.x, part->position.y, part->position.z), "vel",
            BuildTuple(part->linear_velocity.x, part->linear_velocity.y,
                                                            part->linear_velocity.z));

        if (!dict) {
            Deallocate(info);
            Py_DECREF(list);
            return nullptr;
        }

        PyList_SET_ITEM(list, i, dict);
    }

    Deallocate(info);
    return list;
}

void SkeletonDealloc(SkeletonObject *self) {
    if (self->skeleton) {
        PhysicsRagdollSkeletonDestroy(self->skeleton);
    }
    if (self->parsers) {

        delete self->parsers;
    }
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

[[nodiscard]] PyObject * SkeletonNew(PyTypeObject *type,
                                                       PyObject *args [[maybe_unused]],
                                                       PyObject *kwds [[maybe_unused]]) {
    auto self = reinterpret_cast<SkeletonObject *>(type->tp_alloc(type, 0));
    if (self) {
        self->skeleton = PhysicsRagdollSkeletonCreate();
        if (!self->skeleton) {
            Py_DECREF(self);
            return PyErr_NoMemory();
        }
        self->parsers = CPH::Python::NewParserGroup<CPH::Python::SkeletonParsers>();
        if (!self->parsers) {
            PhysicsRagdollSkeletonDestroy(self->skeleton);
            self->skeleton = nullptr;
            Py_DECREF(self);
            return PyErr_NoMemory();
        }
    }
    return reinterpret_cast<PyObject *>(self);
}

void RagdollSettingsDealloc(RagdollSettingsObject *self) {
    if (self->settings) {
        PhysicsRagdollSettingsDestroy(self->settings);
    }
    if (self->parsers) {

        delete self->parsers;
    }
    Py_XDECREF(self->world);
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

// --- Ragdoll Instance Implementation ---

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void RagdollDealloc(RagdollObject *self) {
    if (self->ragdoll) {
        PhysicsWorld *world = self->world ? &self->world->physics : nullptr;
        Py_BEGIN_ALLOW_THREADS PhysicsRagdollDeallocate(
            world, self->ragdoll, {self->body_slots, static_cast<size_t>(self->body_count)});
        Py_END_ALLOW_THREADS self->ragdoll = nullptr;
    }

    if (self->body_slots) {
        Deallocate(self->body_slots);
        self->body_slots = nullptr;
    }

    if (self->parsers) {

        delete self->parsers;
    }

    Py_XDECREF(self->world);
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

[[nodiscard]] PyObject * RagdollSettingsAddPart(RagdollSettingsObject *self,
                                                                  PyObject *const *args,
                                                                  size_t nargsf,
                                                                  PyObject *kwnames) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
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
    Vec3f axis   = {.x = 1.0f, .y = 0.0f, .z = 0.0f};
    Vec3f normal = {.x = 0.0f, .y = 1.0f, .z = 0.0f};

    // 2. High Speed Parse
    const ParseResult parsed =
        self->parsers->RagdollAddPart.Parse(args, nargs, kwnames, joint_idx, shape_type, py_size,
                                             mass, parent_idx, twist_min, twist_max, cone_angle,
                                             axis, normal, py_pos);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    // 3. Parse native shape and pose parameters, then delegate engine operations.
    float shape_params[4];
    ParseBodySize(py_size, shape_params);

    JPH::ShapeRefC shape;
    if (self->world) {
        Py_BEGIN_ALLOW_THREADS shape =
            self->world->physics.FindOrCreateShape(shape_type, shape_params);
        Py_END_ALLOW_THREADS
    }
    if (!shape) {
        return PyErr_Format(PyExc_ValueError, "Invalid shape configuration");
    }

    bool has_position  = false;
    JPH_RVec3 position = {0, 0, 0};
    if (py_pos && py_pos != Py_None) {
        PosStride p_stride = {};
        // Preserved quirk: a malformed `pos` is not fatal here — the position
        // is simply left unset and the call carries on, *with the exception
        // still pending* (CPython then reports the NULL/exception-set
        // mismatch). That is the pre-B3 behaviour verbatim; the pending error
        // has to be reinstated because ParseError owns it now.
        const ParseResult parsed = parse_vec3_into(py_pos, p_stride);
        if (parsed) {
            position     = JPH_RVec3{p_stride.x, p_stride.y, p_stride.z};
            has_position = true;
        } else {
            RaiseParseError(parsed.error());
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
    const std::expected<int, Error> status =
        PhysicsRagdollSettingsAddPart(self->settings, shape.GetPtr(), request);

    if (!status && status.error().Code().Is(RagdollError::InvalidShape)) {
        return PyErr_Format(PyExc_ValueError, "Invalid shape configuration");
    }
    if (!status && status.error().Code().Is(RagdollError::JointOutOfRange)) {
        return PyErr_Format(PyExc_IndexError, "Joint index %d out of bounds", joint_idx);
    }
    if (!status) {
        return PyErr_Format(PyExc_RuntimeError, "Invalid ragdoll settings state");
    }

    Py_RETURN_NONE;
}

static PyMethodDef kRagdollMethods[] = {

    CPH::Python::FastCallMethod("add_part", RagdollSettingsAddPart),
    CPH::Python::NoArgsMethod("stabilize", RagdollSettingsStabilize),
    {nullptr, nullptr, 0, nullptr}};

static PyType_Slot RagdollSettingsSlots[] = {
    {.slot = Py_tp_dealloc,
     .pfunc = reinterpret_cast<void *>(RagdollSettingsDealloc)},
    {.slot = Py_tp_methods, .pfunc = reinterpret_cast<void *>(kRagdollMethods)},
    {},
};

PyType_Spec RagdollSettingsSpec = {
    .name      = "culverin._culverin_c.RagdollSettings",
    .basicsize = sizeof(RagdollSettingsObject),
    .flags     = Py_TPFLAGS_DEFAULT,
    .slots     = RagdollSettingsSlots,
};

static PyMethodDef SkeletonMethods[] = {
    CPH::Python::FastCallMethod("add_joint", SkeletonAddJoint),
    CPH::Python::FastCallMethod("get_joint_index", SkeletonGetJointIndex),
    CPH::Python::NoArgsMethod("finalize", SkeletonFinalize),
    {},
};

static PyType_Slot SkeletonSlots[] = {
    {.slot = Py_tp_new, .pfunc = reinterpret_cast<void *>(SkeletonNew)},
    {.slot = Py_tp_dealloc, .pfunc = reinterpret_cast<void *>(SkeletonDealloc)},
    {.slot = Py_tp_methods, .pfunc = reinterpret_cast<void *>(SkeletonMethods)},
    {},
};

PyType_Spec SkeletonSpec = {
    .name      = "culverin._culverin_c.Skeleton",
    .basicsize = sizeof(SkeletonObject),
    .flags     = Py_TPFLAGS_DEFAULT,
    .slots     = SkeletonSlots,
};

static PyMethodDef RagdollMethods[] = {
    CPH::Python::FastCallMethod("drive_to_pose", RagdollDriveToPose),
    CPH::Python::NoArgsMethod("get_body_handles", RagdollGetBodyHandles),
    CPH::Python::NoArgsMethod("get_debug_info", RagdollGetDebugInfo),
    {},
};

static PyType_Slot RagdollSlots[] = {
    {.slot = Py_tp_dealloc, .pfunc = reinterpret_cast<void *>(RagdollDealloc)},
    {.slot = Py_tp_methods, .pfunc = reinterpret_cast<void *>(RagdollMethods)},
    {},
};

PyType_Spec RagdollSpec = {
    .name      = "culverin._culverin_c.Ragdoll",
    .basicsize = sizeof(RagdollObject),
    .flags     = Py_TPFLAGS_DEFAULT,
    .slots     = RagdollSlots,
};

} // namespace CPH::Python
