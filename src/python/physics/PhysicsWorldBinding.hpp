#pragma once

#include "bindings/PythonCommon.hpp"
#include "physics/PhysicsWorldParsers.hpp"
#include "physics/PhysicsWorldState.h"
#include <Python.h>

#include <cstddef>

namespace CPH::Python {

// --- CPython entry-point object ---
// Python-only buffer-protocol metadata and parser state live outside PhysicsWorld.
struct PhysicsWorldObject {
    PyObject_HEAD PyObject *weakreflist;
    CPH_PhysicsWorld physics;
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
[[nodiscard]] static inline bool ValidateHandle(PhysicsWorldObject *self, uint64_t h_raw,
                                                    uint32_t *out_slot) {
    if (CPH_UNLIKELY(!unpack_handle(&self->physics, static_cast<BodyHandle>(h_raw), out_slot))) {
        CPH_SHADOW_UNLOCK(&self->physics.shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
        return false;
    }
    return true;
}

} // namespace CPH::Python
