#pragma once

#include "culverin.h"
#include "culverin_arg_indices.h"
#include "culverin_physics_world_state.h"
#include <Python.h>

// --- CPython entry-point object ---
// Python-only buffer-protocol metadata and parser state live outside PhysicsWorld.
typedef struct PhysicsWorldObject {
    PyObject_HEAD PyObject *weakreflist;
    PhysicsWorld physics;
    Py_ssize_t view_shape[2];
    Py_ssize_t view_strides[2];
    WorldParsers *parsers;
} PhysicsWorldObject;

// Python/GIL-aware adapter around the native command processor.
void sync_and_flush_internal(PhysicsWorldObject *self);

// Translate native handle validation into Python exceptions only at the binding boundary.
#define CHECK_HANDLE(h_raw, slot_out)                                                              \
    do {                                                                                           \
        static_assert(sizeof(CULV_TYPE_OF(h_raw)) == sizeof(uint64_t));                            \
        static_assert(sizeof(CULV_TYPE_OF(slot_out)) == sizeof(uint32_t));                         \
        if (UNLIKELY(!unpack_handle(&self->physics, (BodyHandle)(h_raw), &(slot_out)))) {          \
            SHADOW_UNLOCK(&self->physics.shadow_lock);                                            \
            RAISE_STALE_HANDLE();                                                                  \
        }                                                                                          \
    } while (false)

#define CHECK_STATE(state_val, slot_mask)                                                          \
    do {                                                                                           \
        static_assert(sizeof(state_val) == sizeof(slot_mask));                                     \
        if (UNLIKELY(!is_state_valid((state_val), (slot_mask)))) {                                 \
            SHADOW_UNLOCK(&self->physics.shadow_lock);                                            \
            RAISE_STALE_HANDLE();                                                                  \
        }                                                                                          \
    } while (false)
