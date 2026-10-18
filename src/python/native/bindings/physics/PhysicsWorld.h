#pragma once

#include "bindings/BindingCommon.h"
#include "bindings/parsing/ArgIndices.h"
#include "physics/PhysicsWorldState.h"
#include <Python.h>

// --- CPython entry-point object ---
// Python-only buffer-protocol metadata and parser state live outside PhysicsWorld.
typedef struct CPH_PhysicsWorldObject {
    PyObject_HEAD PyObject *weakreflist;
    CPH_PhysicsWorld physics;
    Py_ssize_t view_shape[2];
    Py_ssize_t view_strides[2];
    CPH_WorldParsers *parsers;
} CPH_PhysicsWorldObject;

// Python/GIL-aware adapter around the native command processor.
void CPH_SyncAndFlushInternal(CPH_PhysicsWorldObject *self);

// Translate native handle validation into Python exceptions only at the binding boundary.
#define CPH_CHECK_HANDLE(h_raw, slot_out)                                                              \
    do {                                                                                           \
        static_assert(sizeof(CPH_TYPE_OF(h_raw)) == sizeof(uint64_t));                            \
        static_assert(sizeof(CPH_TYPE_OF(slot_out)) == sizeof(uint32_t));                         \
        if (CPH_UNLIKELY(!unpack_handle(&self->physics, (BodyHandle)(h_raw), &(slot_out)))) {          \
            CPH_SHADOW_UNLOCK(&self->physics.shadow_lock);                                            \
            CPH_RAISE_STALE_HANDLE();                                                                  \
        }                                                                                          \
    } while (false)

#define CPH_CHECK_STATE(state_val, slot_mask)                                                          \
    do {                                                                                           \
        static_assert(sizeof(state_val) == sizeof(slot_mask));                                     \
        if (CPH_UNLIKELY(!is_state_valid((state_val), (slot_mask)))) {                                 \
            CPH_SHADOW_UNLOCK(&self->physics.shadow_lock);                                            \
            CPH_RAISE_STALE_HANDLE();                                                                  \
        }                                                                                          \
    } while (false)
