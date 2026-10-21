#pragma once

#include "bindings/PythonCommon.hpp"
#include "physics/PhysicsWorldParsers.hpp"
#include "physics/PhysicsWorldState.hpp"
#include <Python.h>

#include <cstddef>
#include <optional>

namespace CPH::Python {

// --- CPython entry-point object ---
// Python-only buffer-protocol metadata and parser state live outside PhysicsWorld.
struct PhysicsWorldObject {
    // No weakreflist member: the type opts into Py_TPFLAGS_MANAGED_WEAKREF,
    // so CPython stores the weakref list itself. That is what makes the type
    // layout independent of offsetof -- see the flags in WorldSpec.
    PyObject_HEAD
    PhysicsWorld physics;
    Py_ssize_t view_shape[2];
    Py_ssize_t view_strides[2];
    CPH::Python::WorldParsers *parsers;
};

static_assert(alignof(PhysicsWorldObject) <= alignof(std::max_align_t),
              "CPython's allocator cannot provide over-aligned PhysicsWorldObject storage");

// Python/GIL-aware adapter around the native command processor.
void SyncAndFlushInternal(PhysicsWorldObject *self);

// Validate a raw handle against the shadow state. On failure the shadow lock
// is released and ValueError is raised; the caller must return nullptr.
//
// TODO(smell): the helper releases a lock it never took, so every caller has
// to hold shadow_lock before calling it and nothing verifies that. A caller
// that validates without the lock unlocks an unowned mutex (UB). The fix is
// for the caller's scoped lock to own the release, not for the validator to
// reach into it.
[[nodiscard]] static inline std::optional<std::uint32_t> ValidateHandle(PhysicsWorldObject *self,
                                                                        uint64_t h_raw) {
    const std::optional<std::uint32_t> slot =
        self->physics.Resolve(static_cast<BodyHandle>(h_raw));
    if (!slot) [[unlikely]] {
        (self->physics.shadow_lock).unlock();
        PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
        return std::nullopt;
    }
    return slot;
}

} // namespace CPH::Python
