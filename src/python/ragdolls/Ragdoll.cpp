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
#include "bindings/Result.hpp"

namespace CPH::Python {

// Default mass
static constexpr float kRagdollDefaultPartMass = 10.0f;

// Ragdoll constraint angle limits (in radians)
static constexpr float kRagdollDefaultTwistMin = -0.1f;
static constexpr float kRagdollDefaultTwistMax = 0.1f;

[[nodiscard]] BoundResult SkeletonAddJoint(SkeletonObject *self,
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
        return FailCurrent();
    }

    // 3. Logic
    const char *name = PyUnicode_AsUTF8(name_obj);
    if (!name) {
        return FailCurrent();
    }

    const int idx = self->skeleton->AddJoint(name, parent_idx);
    return Adopt(PyLong_FromLong(idx));
}

[[nodiscard]] BoundResult SkeletonGetJointIndex(SkeletonObject *self,
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
        return FailCurrent();
    }

    // 3. Logic
    const char *name = PyUnicode_AsUTF8(name_obj);
    if (!name) {
        return FailCurrent();
    }

    const int idx = self->skeleton->GetJointIndex(name);
    return Adopt(PyLong_FromLong(idx));
}

[[nodiscard]] BoundResult SkeletonFinalize(SkeletonObject *self,
                                                            PyObject *args [[maybe_unused]]) {
    if (!self->skeleton->Finalize()) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Skeleton joints are out of order (parent must be added before child)");
        return FailCurrent();
    }
    return BorrowedValue(Py_None);
}

[[nodiscard]] BoundResult
RagdollSettingsStabilize(RagdollSettingsObject *self, PyObject *args [[maybe_unused]]) {
    if (self->settings->Stabilize()) {
        return BorrowedValue(Py_True);
    }
    return BorrowedValue(Py_False);
}

// One message for every access-after-destroy. Kept separate from the
// construction-failure wording: a caller that never called destroy() and still
// sees this has a construction bug, and a caller that did call it should be
// told exactly that instead of being pointed at construction.
[[nodiscard]] PyObject *RagdollNotLive() {
    PyErr_SetString(PyExc_RuntimeError,
                    "Ragdoll has been destroyed or was never attached to a world");
    return nullptr;
}

[[nodiscard]] BoundResult RagdollDriveToPose(RagdollObject *self,
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
        return FailCurrent();
    }

    // `self->ragdoll == nullptr` means destroy() already ran (or construction
    // failed). It raises rather than returning None: reporting success for a
    // call that did nothing is indistinguishable from "the pose was applied"
    // at the Python boundary. `self->world == nullptr` is the cleared-by-GC
    // case; the native ragdoll is still alive, so the call is still refused
    // because it needs the world's shadow lock.
    if (!self->ragdoll || !self->world) [[unlikely]] {
        return Adopt(RagdollNotLive());
    }

    // 2. RESOURCE ACQUISITION
    const int joint_count = self->ragdoll->JointCount();

    // Validate Buffer Size
    const auto matrices_view = BufferView::Acquire(py_matrices);
    if (!matrices_view) {
        return FailCurrent();
    }

    const size_t required_size = static_cast<size_t>(joint_count) * sizeof(JPH::Mat44);
    const Py_ssize_t matrices_bytes = static_cast<Py_ssize_t>(matrices_view->SizeBytes());
    if (!matrices_view->RequireAtLeast(
            required_size, "Matrices buffer too small. Expected %zu bytes for %d joints, got %zd",
            required_size, joint_count, matrices_bytes)) [[unlikely]] {
        return FailCurrent();
    }

    // The caller's float32 buffer is 4 column-major Vec4 per joint (64 bytes),
    // which is exactly JPH::Mat44's layout, so it is read in place.
    const std::span<const JPH::Mat44> matrices = matrices_view->As<JPH::Mat44>();
    const JPH::RVec3 root_position = JPH::RVec3(root_p.x, root_p.y, root_p.z);

    // 3. PHYSICS EXECUTION (the native ragdoll parks on the shadow lock)
    const bool pose_applied = self->ragdoll->DriveToPose(
        root_position, {matrices.data(), static_cast<size_t>(joint_count)});
    if (!pose_applied) {
        return Adopt(PyErr_NoMemory());
    }
    return BorrowedValue(Py_None);
}

[[nodiscard]] BoundResult RagdollGetBodyHandles(RagdollObject *self,
                                                                 PyObject *args [[maybe_unused]]) {
    // Same contract as drive_to_pose and get_debug_info: a missing native
    // ragdoll is an error, not an empty answer. After destroy() there are no
    // part handles left to report.
    if (!self->ragdoll) [[unlikely]] {
        return Adopt(RagdollNotLive());
    }

    const int body_count = self->ragdoll->BodyCount();
    if (body_count < 0 ||
        static_cast<size_t>(body_count) > static_cast<size_t>(PY_SSIZE_T_MAX) ||
        static_cast<size_t>(body_count) > SIZE_MAX / sizeof(PhysicsRagdollBodyHandleInfo)) {
        return Adopt(PyErr_NoMemory());
    }

    const Py_ssize_t count = static_cast<Py_ssize_t>(body_count);
    PyObject *list         = PyList_New(count);
    if (!list) {
        return FailCurrent();
    }

    PhysicsRagdollBodyHandleInfo *handles =
        count ? static_cast<PhysicsRagdollBodyHandleInfo *>(
                    Allocate(static_cast<size_t>(count) * sizeof(*handles)))
              : nullptr;
    if (count && !handles) {
        Py_DECREF(list);
        return Adopt(PyErr_NoMemory());
    }

    self->ragdoll->CopyBodyHandles({handles, static_cast<size_t>(count)});
    for (Py_ssize_t i = 0; i < count; ++i) {
        if (handles[i].alive) {
            PyObject *handle = PyLong_FromUnsignedLongLong(handles[i].handle.Raw());
            if (!handle) {
                Deallocate(handles);
                Py_DECREF(list);
                return FailCurrent();
            }
            PyList_SET_ITEM(list, i, handle);
        } else {
            Py_INCREF(Py_None);
            PyList_SET_ITEM(list, i, Py_None);
        }
    }

    Deallocate(handles);
    return Adopt(list);
}

// Manual lifetime control, matching Vehicle.destroy().
//
// Releases the native ragdoll immediately -- out of the physics system, body
// slots returned, part handles invalidated -- instead of waiting for the last
// Python reference to go. The Python object stays valid but inert: every
// accessor raises instead of reading freed memory, and calling destroy() again
// is a no-op, so teardown paths can call it without tracking state.
[[nodiscard]] BoundResult RagdollDestroy(RagdollObject *self,
                                       PyObject *ignored [[maybe_unused]]) {
    // Same hook the collector uses, so destroy() and collection cannot drift.
    self->BeforeClear();

    // Release the world reference held for the ragdoll's lifetime now rather
    // than at dealloc: this is the point at which the ragdoll stops needing
    // the world, and holding it longer is what made a destroyed ragdoll keep a
    // whole PhysicsWorld alive.
    Py_CLEAR(self->world);

    return BorrowedValue(Py_None);
}

[[nodiscard]] BoundResult RagdollGetDebugInfo(RagdollObject *self,
                                                               PyObject *ignored [[maybe_unused]]) {
    if (!self->ragdoll || !self->world) [[unlikely]] {
        return Adopt(RagdollNotLive());
    }

    const int body_count = self->ragdoll->BodyCount();
    if (body_count < 0 ||
        static_cast<size_t>(body_count) > SIZE_MAX / sizeof(PhysicsRagdollBodyDebugInfo)) {
        return Adopt(PyErr_NoMemory());
    }

    PyObject *list = PyList_New(body_count);
    if (!list) {
        return FailCurrent();
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
        return Adopt(PyErr_NoMemory());
    }
    auto *info = reinterpret_cast<PhysicsRagdollBodyDebugInfo *>(
        (reinterpret_cast<std::uintptr_t>(info_block) + (info_align - 1)) & ~(info_align - 1));

    if (!self->ragdoll->CopyDebugInfo({info, static_cast<size_t>(body_count)})) {
        Deallocate(info_block);
        Py_DECREF(list);
        return Adopt(PyErr_Format(PyExc_RuntimeError, "Unable to read ragdoll body state"));
    }

    for (int i = 0; i < body_count; ++i) {
        const PhysicsRagdollBodyDebugInfo *part = &info[i];
        // "rot" is reported alongside "pos" and "vel". CopyDebugInfo has
        // always fetched the body's rotation, but this dict dropped it, so the
        // only consumer of a rotation it went to the trouble of reading was
        // nothing. The key is additive: "index", "pos" and "vel" keep their
        // names, types and order, so existing readers are unaffected.
        PyObject *dict = BuildDictionary(
            "index", i, "pos",
            BuildTuple(part->position.GetX(), part->position.GetY(), part->position.GetZ()), "rot",
            BuildTuple(part->rotation.GetX(), part->rotation.GetY(), part->rotation.GetZ(),
                       part->rotation.GetW()),
            "vel",
            BuildTuple(part->linearVelocity.GetX(), part->linearVelocity.GetY(),
                       part->linearVelocity.GetZ()));

        if (!dict) {
            Deallocate(info_block);
            Py_DECREF(list);
            return FailCurrent();
        }

        PyList_SET_ITEM(list, i, dict);
    }

    Deallocate(info_block);
    return Adopt(list);
}

void SkeletonDealloc(SkeletonObject *self) {
    delete self->skeleton;
    self->skeleton = nullptr;

    delete self->parsers;
    self->parsers = nullptr;

    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

[[nodiscard]] BoundResult SkeletonNew(PyTypeObject *type, PyObject *args [[maybe_unused]],
                                        PyObject *kwds [[maybe_unused]]) {
    auto self = reinterpret_cast<SkeletonObject *>(type->tp_alloc(type, 0));
    if (self) {
        self->skeleton = nullptr;
        self->parsers  = nullptr;

        std::unique_ptr<CPH::Skeleton> skeleton = CPH::Skeleton::Create();
        if (!skeleton) {
            Py_DECREF(self);
            return Adopt(PyErr_NoMemory());
        }
        self->parsers = CPH::Python::NewParserGroup<CPH::Python::SkeletonParsers>();
        if (!self->parsers) {
            Py_DECREF(self);
            return Adopt(PyErr_NoMemory());
        }
        self->skeleton = skeleton.release();
    }
    return Adopt(reinterpret_cast<PyObject *>(self));
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

// --- Object lifetime ---------------------------------------------------------
//
// dealloc, traverse and clear all come from PyGcObject. RagdollBinding.hpp
// declares what they act on: BeforeClear() releases the native ragdoll before
// any reference is dropped (the ordering ~Ragdoll requires), VisitRefs() names
// the world reference, and BeforeFree() releases the parsers.

[[nodiscard]] BoundResult RagdollSettingsAddPart(RagdollSettingsObject *self,
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
        return FailCurrent();
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
        return Adopt(PyErr_Format(PyExc_ValueError, "Invalid shape configuration"));
    }

    bool has_position = false;
    JPH::RVec3 position = JPH::RVec3::sZero();
    if (py_pos && py_pos != Py_None) {
        PosStride p_stride = {};
        // A malformed `pos` is fatal. This used to raise the parse error and
        // then carry on anyway, leaving the exception pending while the part
        // went on to be added: the caller got a part it was told had failed,
        // and CPython's frame boundary reported the mismatch as a SystemError
        // rather than the ValueError the user caused. Returning nullptr here
        // means the part is not added and the user sees their own error.
        const ParseResult parsed_pos = ParseVec3Into(py_pos, p_stride);
        if (!parsed_pos) [[unlikely]] {
            RaiseParseError(parsed_pos.error());
            return FailCurrent();
        }
        position     = JPH::RVec3(p_stride.x, p_stride.y, p_stride.z);
        has_position = true;
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
        return Adopt(PyErr_Format(PyExc_ValueError, "Invalid shape configuration"));
    }
    if (!status && status.error().Code().Is(RagdollError::JointOutOfRange)) {
        return Adopt(PyErr_Format(PyExc_IndexError, "Joint index %d out of bounds", joint_idx));
    }
    if (!status) {
        return Adopt(PyErr_Format(PyExc_RuntimeError, "Invalid ragdoll settings state"));
    }

    return BorrowedValue(Py_None);
}

static PyMethodDef kRagdollMethods[] = {

    CPH::Python::Method<&RagdollSettingsAddPart>("add_part"),
    CPH::Python::NoArgsMethod<&RagdollSettingsStabilize>("stabilize"),
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
    CPH::Python::Method<&SkeletonAddJoint>("add_joint"),
    CPH::Python::Method<&SkeletonGetJointIndex>("get_joint_index"),
    CPH::Python::NoArgsMethod<&SkeletonFinalize>("finalize"),
    {},
};

static PyType_Slot SkeletonSlots[] = {
    CPH::Python::NewSlot<&SkeletonNew>(),
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
    CPH::Python::Method<&RagdollDriveToPose>("drive_to_pose"),
    CPH::Python::NoArgsMethod<&RagdollGetBodyHandles>("get_body_handles"),
    CPH::Python::NoArgsMethod<&RagdollGetDebugInfo>("get_debug_info"),
    CPH::Python::NoArgsMethod<&RagdollDestroy>("destroy"),
    {},
};

static PyType_Slot RagdollSlots[] = {
    CPH::Python::GcDeallocSlot<RagdollObject>(),
    CPH::Python::GcTraverseSlot<RagdollObject>(),
    CPH::Python::GcClearSlot<RagdollObject>(),
    {.slot = Py_tp_methods, .pfunc = reinterpret_cast<void *>(RagdollMethods)},
    {},
};

PyType_Spec RagdollSpec = CPH::Python::GcTypeSpec("culverin._culverin_c.Ragdoll",
                                                   sizeof(RagdollObject), RagdollSlots);

} // namespace CPH::Python
