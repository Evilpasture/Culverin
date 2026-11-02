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
    Py_ssize_t viewShape[2];
    Py_ssize_t viewStrides[2];
    CPH::Python::WorldParsers *parsers;
};

static_assert(alignof(PhysicsWorldObject) <= alignof(std::max_align_t),
              "CPython's allocator cannot provide over-aligned PhysicsWorldObject storage");

// Python/GIL-aware adapter around the native command processor.
void SyncAndFlushInternal(PhysicsWorldObject *self);

// Validate a raw handle against the shadow state. Raises ValueError and returns
// nullopt when the handle is stale; the caller must return nullptr.
//
// This function takes and releases no lock. Resolution goes through
// BodyRegistry, whose generation check is a lock-free acquire load, so
// validation does not need the shadow lock to be correct -- it only observes
// the registry.
//
// It used to unlock `shadowLock` on the failure path, which meant every caller
// had to hold that lock before calling it and nothing verified that a caller
// did. A caller that validated without the lock unlocked an unowned mutex,
// which is undefined behaviour, and the requirement was documented in prose
// only. Lock ownership now belongs entirely to the caller's scoped lock: this
// helper can no longer unlock anything, so it cannot unlock something it does
// not own.
//
// On the success path the caller still holds the shadow lock it took before
// calling (see WorldGetSoftBodyVertices), and the slot stays stable because
// retiring a slot requires that same lock.
[[nodiscard]] static inline std::optional<std::uint32_t> ValidateHandle(PhysicsWorldObject *self,
                                                                        uint64_t h_raw) noexcept {
    const std::optional<std::uint32_t> slot =
        self->physics.Resolve(static_cast<BodyHandle>(h_raw));
    if (!slot) [[unlikely]] {
        PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
        return std::nullopt;
    }
    return slot;
}

} // namespace CPH::Python
