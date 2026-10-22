// Python adapter for skeletons, ragdoll settings and ragdoll instances.
//
// The skeleton bookkeeping, the part layout, the world-side slot teardown and
// the pose driving all live in src/ragdolls/Ragdoll.cpp. Everything below
// parses arguments, calls a native method and packs the result.

#include "ragdolls/RagdollBinding.hpp"
#include "bindings/PythonCommon.hpp"
#include "bindings/BufferView.hpp"
#include "bindings/ThreadState.hpp"
#include "bindings/FastBuild.hpp"
#include "bindings/PythonSupport.hpp"
#include "core/Compiler.hpp"
#include "physics/PhysicsSync.hpp"

#include <cstdint>

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
        self->parsers->addJoint.Parse(args, nargs, kwnames, name_obj, parent_idx);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    // 3. Logic
    const char *name = PyUnicode_AsUTF8(name_obj);
    if (!name) {
        return nullptr;
    }

    const int idx = self->skeleton->AddJoint(name, parent_idx);
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
        self->parsers->getJointIdx.Parse(args, nargs, kwnames, name_obj);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    // 3. Logic
    const char *name = PyUnicode_AsUTF8(name_obj);
    if (!name) {
        return nullptr;
    }

    const int idx = self->skeleton->GetJointIndex(name);
    return PyLong_FromLong(idx);
}

[[nodiscard]] PyObject * SkeletonFinalize(SkeletonObject *self,
                                                            PyObject *args [[maybe_unused]]) {
    if (!self->skeleton->Finalize()) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Skeleton joints are out of order (parent must be added before child)");
        return nullptr;
    }
    Py_RETURN_NONE;
}

[[nodiscard]] PyObject *
RagdollSettingsStabilize(RagdollSettingsObject *self, PyObject *args [[maybe_unused]]) {
    if (self->settings->Stabilize()) {
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
        self->parsers->ragdollDrive.Parse(args, nargs, kwnames, root_p, root_q, py_matrices);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    // TODO(smell): a ragdoll object cannot be built without both of these, so
    // this branch is unreachable unless construction half-failed -- and if it
    // ever is reached it reports success by returning None, which is the one
    // place in this API that treats a missing native object as a non-error.
    // get_body_handles and get_debug_info carry the same early return.
    if (!self->ragdoll || !self->world) [[unlikely]] {
        Py_RETURN_NONE;
    }

    // 2. RESOURCE ACQUISITION
    const int joint_count = self->ragdoll->JointCount();

    // Validate Buffer Size
    const auto matrices_view = BufferView::Acquire(py_matrices);
    if (!matrices_view) {
        return nullptr;
    }

    const size_t required_size = static_cast<size_t>(joint_count) * sizeof(JPH::Mat44);
    const Py_ssize_t matrices_bytes = static_cast<Py_ssize_t>(matrices_view->SizeBytes());
    if (!matrices_view->RequireAtLeast(
            required_size, "Matrices buffer too small. Expected %zu bytes for %d joints, got %zd",
            required_size, joint_count, matrices_bytes)) [[unlikely]] {
        return nullptr;
    }

    // The caller's float32 buffer is 4 column-major Vec4 per joint (64 bytes),
    // which is exactly JPH::Mat44's layout, so it is read in place.
    const std::span<const JPH::Mat44> matrices = matrices_view->As<JPH::Mat44>();
    const JPH::RVec3 root_position = JPH::RVec3(root_p.x, root_p.y, root_p.z);

    // 3. PHYSICS EXECUTION (the native ragdoll parks on the shadow lock)
    const bool pose_applied = self->ragdoll->DriveToPose(
        root_position, {matrices.data(), static_cast<size_t>(joint_count)});
    if (!pose_applied) {
        return PyErr_NoMemory();
    }
    Py_RETURN_NONE;
}

[[nodiscard]] PyObject * RagdollGetBodyHandles(RagdollObject *self,
                                                                 PyObject *args [[maybe_unused]]) {
    if (!self->ragdoll) {
        Py_RETURN_NONE;
    }

    const int body_count = self->ragdoll->BodyCount();
    if (body_count < 0 ||
        static_cast<size_t>(body_count) > static_cast<size_t>(PY_SSIZE_T_MAX) ||
        static_cast<size_t>(body_count) > SIZE_MAX / sizeof(PhysicsRagdollBodyHandleInfo)) {
        return PyErr_NoMemory();
    }

    const Py_ssize_t count = static_cast<Py_ssize_t>(body_count);
    PyObject *list         = PyList_New(count);
    if (!list) {
        return nullptr;
    }

    PhysicsRagdollBodyHandleInfo *handles =
        count ? static_cast<PhysicsRagdollBodyHandleInfo *>(
                    Allocate(static_cast<size_t>(count) * sizeof(*handles)))
              : nullptr;
    if (count && !handles) {
        Py_DECREF(list);
        return PyErr_NoMemory();
    }

    self->ragdoll->CopyBodyHandles({handles, static_cast<size_t>(count)});
    for (Py_ssize_t i = 0; i < count; ++i) {
        if (handles[i].alive) {
            PyObject *handle = PyLong_FromUnsignedLongLong(handles[i].handle.Raw());
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

    const int body_count = self->ragdoll->BodyCount();
    if (body_count < 0 ||
        static_cast<size_t>(body_count) > SIZE_MAX / sizeof(PhysicsRagdollBodyDebugInfo)) {
        return PyErr_NoMemory();
    }

    PyObject *list = PyList_New(body_count);
    if (!list) {
        return nullptr;
    }

    // PhysicsRagdollBodyDebugInfo holds Jolt's SIMD vector types, so alignof is
    // 32, while Allocate() (PyMem_RawMalloc) only promises alignof(max_align_t).
    // Over-allocate and hand-align, keeping the block base for Deallocate().
    // Stays on Allocate() rather than std::vector because the build is
    // -fno-exceptions: a vector resize failure would std::terminate instead of
    // returning PyErr_NoMemory().
    constexpr std::size_t info_align = alignof(PhysicsRagdollBodyDebugInfo);
    static_assert((info_align & (info_align - 1)) == 0, "alignment must be a power of two");

    void *info_block =
        Allocate(static_cast<size_t>(body_count) * sizeof(PhysicsRagdollBodyDebugInfo) +
                 info_align);
    if (body_count && !info_block) {
        Py_DECREF(list);
        return PyErr_NoMemory();
    }
    auto *info = reinterpret_cast<PhysicsRagdollBodyDebugInfo *>(
        (reinterpret_cast<std::uintptr_t>(info_block) + (info_align - 1)) & ~(info_align - 1));

    if (!self->ragdoll->CopyDebugInfo({info, static_cast<size_t>(body_count)})) {
        Deallocate(info_block);
        Py_DECREF(list);
        return PyErr_Format(PyExc_RuntimeError, "Unable to read ragdoll body state");
    }

    for (int i = 0; i < body_count; ++i) {
        const PhysicsRagdollBodyDebugInfo *part = &info[i];
        // TODO(smell): `part->rotation` is fetched from Jolt on every call but
        // never surfaced in the returned dict -- only "pos" and "vel" are.
        // Pre-existing; left as-is to keep the Python payload unchanged.
        PyObject *dict = BuildDictionary(
            "index", i, "pos",
            BuildTuple(part->position.GetX(), part->position.GetY(), part->position.GetZ()), "vel",
            BuildTuple(part->linearVelocity.GetX(), part->linearVelocity.GetY(),
                       part->linearVelocity.GetZ()));

        if (!dict) {
            Deallocate(info_block);
            Py_DECREF(list);
            return nullptr;
        }

        PyList_SET_ITEM(list, i, dict);
    }

    Deallocate(info_block);
    return list;
}

void SkeletonDealloc(SkeletonObject *self) {
    delete self->skeleton;
    self->skeleton = nullptr;

    delete self->parsers;
    self->parsers = nullptr;

    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

[[nodiscard]] PyObject * SkeletonNew(PyTypeObject *type,
                                                       PyObject *args [[maybe_unused]],
                                                       PyObject *kwds [[maybe_unused]]) {
    auto self = reinterpret_cast<SkeletonObject *>(type->tp_alloc(type, 0));
    if (self) {
        self->skeleton = nullptr;
        self->parsers  = nullptr;

        std::unique_ptr<CPH::Skeleton> skeleton = CPH::Skeleton::Create();
        if (!skeleton) {
            Py_DECREF(self);
            return PyErr_NoMemory();
        }
        self->parsers = CPH::Python::NewParserGroup<CPH::Python::SkeletonParsers>();
        if (!self->parsers) {
            Py_DECREF(self);
            return PyErr_NoMemory();
        }
        self->skeleton = skeleton.release();
    }
    return reinterpret_cast<PyObject *>(self);
}

void RagdollSettingsDealloc(RagdollSettingsObject *self) {
    delete self->settings;
    self->settings = nullptr;

    delete self->parsers;
    self->parsers = nullptr;

    Py_XDECREF(self->world);
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

// --- Ragdoll Instance Implementation ---

void RagdollDealloc(RagdollObject *self) {
    // Destroying the native ragdoll takes it out of the physics system and
    // releases its body slots; it has to happen before the world reference goes.
    delete self->ragdoll;
    self->ragdoll = nullptr;

    delete self->parsers;
    self->parsers = nullptr;

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
    JPH::Float3 axis{1.0f, 0.0f, 0.0f};
    JPH::Float3 normal{0.0f, 1.0f, 0.0f};

    // 2. High Speed Parse
    const ParseResult parsed =
        self->parsers->ragdollAddPart.Parse(args, nargs, kwnames, joint_idx, shape_type, py_size,
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
        {
            const AllowThreads allow;
            shape =
            self->world->physics.FindOrCreateShape(shape_type, shape_params);
        }
    }
    if (!shape) {
        return PyErr_Format(PyExc_ValueError, "Invalid shape configuration");
    }

    bool has_position = false;
    JPH::RVec3 position = JPH::RVec3::sZero();
    if (py_pos && py_pos != Py_None) {
        PosStride p_stride = {};
        // TODO(bug): a malformed `pos` is not fatal here -- the position is
        // simply left unset and the call carries on *with the exception still
        // pending*, so CPython reports the NULL/exception-set mismatch at the
        // frame boundary rather than this call returning NULL. That is the
        // pre-B3 behaviour verbatim; the pending error has to be reinstated
        // because ParseError owns it now. Returning nullptr here is the fix,
        // and it is a behaviour change: the part would no longer be added.
        const ParseResult parsed_pos = ParseVec3Into(py_pos, p_stride);
        if (parsed_pos) {
            position     = JPH::RVec3(p_stride.x, p_stride.y, p_stride.z);
            has_position = true;
        } else {
            RaiseParseError(parsed_pos.error());
        }
    }

    const PhysicsRagdollPartRequest request = {
        .jointIndex  = joint_idx,
        .parentIndex = parent_idx,
        .mass         = mass,
        .twistMin    = twist_min,
        .twistMax    = twist_max,
        .coneAngle   = cone_angle,
        .axis         = JPH::Vec3(axis.x, axis.y, axis.z),
        .normal       = JPH::Vec3(normal.x, normal.y, normal.z),
        .hasPosition = has_position,
        .position     = position,
    };
    const std::expected<int, Error> status = self->settings->AddPart(shape.GetPtr(), request);

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
