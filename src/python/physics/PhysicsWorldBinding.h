#pragma once

#include "bindings/BindingCommon.h"
#include "physics/PhysicsWorldParsers.hpp"
#include "physics/PhysicsWorldState.h"
#include <Python.h>

// --- CPython entry-point object ---
// Python-only buffer-protocol metadata and parser state live outside PhysicsWorld.
typedef struct CPH_PhysicsWorldObject {
    PyObject_HEAD PyObject *weakreflist;
    CPH_PhysicsWorld physics;
    Py_ssize_t view_shape[2];
    Py_ssize_t view_strides[2];
    CPH::Python::WorldParsers *parsers;
} CPH_PhysicsWorldObject;

// Python/GIL-aware adapter around the native command processor.
void CPH_SyncAndFlushInternal(CPH_PhysicsWorldObject *self);

// Validate a raw handle against the shadow state. On failure the shadow lock
// is released and ValueError is raised; the caller must return nullptr.
[[nodiscard]] static inline bool CPH_ValidateHandle(CPH_PhysicsWorldObject *self, uint64_t h_raw,
                                                    uint32_t *out_slot) {
    if (CPH_UNLIKELY(!unpack_handle(&self->physics, static_cast<BodyHandle>(h_raw), out_slot))) {
        CPH_SHADOW_UNLOCK(&self->physics.shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
        return false;
    }
    return true;
}
