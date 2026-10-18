#include "ecs/Ecs.h"
#include "bindings/BindingCommon.h"
#include "bindings/Getters.h"
#include "bindings/ModuleState.h"
#include "bindings/PythonSupport.h"
#include "bindings/parsing/ParserRegistry.hpp"
#include "bindings/physics/PhysicsWorld.h"
#include <stddef.h>

// --- INTERNAL HELPERS ---
static constexpr uint32_t kInvalidDenseIndex    = 0xFFFFFFFF;
static constexpr auto kInitialEntityCapacity    = 1024;
static constexpr auto kInitialSparseCapacity    = 1024;
static constexpr auto kInitialComponentCapacity = 16;

static PyObject *make_ecs_proxy(CPH_RegistryObject *self, uint32_t comp_id, bool entities) {
    CPH_ModuleState *st = CPH_GetModuleState(PyType_GetModule(Py_TYPE(self)));
    CPH_SparseSet *set  = &self->components[comp_id];

    CPH_BufferProxyObject *proxy = PyObject_GC_New(
        CPH_BufferProxyObject, reinterpret_cast<PyTypeObject *>(st->BufferProxyType));
    if (!proxy) {
        return nullptr;
    }

    proxy->owner = reinterpret_cast<PyObject *>(self);
    Py_INCREF(self);

    if (entities) {
        proxy->buf_type    = PROXY_ECS_ENTITIES;
        proxy->dynamic_ptr = set->dense;
        proxy->format      = "Q"; // uint64
        proxy->itemsize    = sizeof(uint64_t);
        proxy->shape[0]    = set->count;
    } else {
        proxy->buf_type    = PROXY_ECS_DATA;
        proxy->dynamic_ptr = set->data;
        proxy->format      = "B"; // Generic bytes, user casts in NumPy
        proxy->itemsize    = 1;
        proxy->shape[0]    = set->count * set->element_size;
    }

    proxy->stride = 1;
    atomic_fetch_add_explicit(&self->view_export_count, 1, memory_order_relaxed);

    PyObject_GC_Track(proxy);
    return reinterpret_cast<PyObject *>(proxy);
}

static void SparseSet_Init(CPH_SparseSet *set, uint32_t element_size) {
    set->sparse          = nullptr;
    set->dense           = nullptr;
    set->data            = nullptr;
    set->element_size    = element_size;
    set->count           = 0;
    set->dense_capacity  = 0;
    set->sparse_capacity = 0;
}

static void SparseSet_Destroy(CPH_SparseSet *set) {
    if (set->sparse) {
        CPH_PyFree(set->sparse);
    }
    if (set->dense) {
        CPH_PyFree(set->dense);
    }
    if (set->data) {
        CPH_PyFree(set->data);
    }
}

static bool SparseSet_EnsureSparseCapacity(CPH_SparseSet *set, uint32_t required_capacity) {
    if (set->sparse_capacity >= required_capacity) {
        return true;
    }

    uint32_t new_cap =
        set->sparse_capacity == 0 ? kInitialSparseCapacity : set->sparse_capacity * 2;
    while (new_cap < required_capacity) {
        new_cap *= 2;
    }

    uint32_t *new_sparse =
        static_cast<uint32_t *>(CPH_PyRealloc(set->sparse, new_cap * sizeof(uint32_t)));
    if (!new_sparse) {
        PyErr_NoMemory(); // Set error here
        return false;
    }

    for (uint32_t i = set->sparse_capacity; i < new_cap; i++) {
        new_sparse[i] = kInvalidDenseIndex;
    }

    set->sparse          = new_sparse;
    set->sparse_capacity = new_cap;
    return true;
}

static bool SparseSet_EnsureDenseCapacity(CPH_RegistryObject *reg, CPH_SparseSet *set) {
    if (set->count < set->dense_capacity) {
        return true;
    }

    // 1. PROTECTION CHECK: Must happen before any reallocation logic
    if (atomic_load_explicit(&reg->view_export_count, memory_order_acquire) > 0) {
        PyErr_SetString(PyExc_BufferError,
                        "Cannot resize ECS component while a memoryview is held. "
                        "Delete the array or view before adding more entities.");
        return false;
    }

    constexpr auto FIRST_DENSE_CAPACITY = 64;
    uint32_t new_cap = set->dense_capacity == 0 ? FIRST_DENSE_CAPACITY : set->dense_capacity * 2;

    CPH_Entity *new_dense =
        static_cast<CPH_Entity *>(CPH_PyRealloc(set->dense, new_cap * sizeof(CPH_Entity)));
    if (!new_dense) {
        PyErr_NoMemory(); // Set error here
        return false;
    }

    uint8_t *new_data = static_cast<uint8_t *>(
        CPH_PyRealloc(set->data, static_cast<size_t>(new_cap) * set->element_size));
    if (!new_data) {
        // Fallback: we failed data, but dense was already realloc'd.
        // In a real engine we'd roll back, but here we just error.
        PyErr_NoMemory();
        return false;
    }

    set->dense          = new_dense;
    set->data           = new_data;
    set->dense_capacity = new_cap;
    return true;
}

// --- LIFECYCLE ---
PyType_DeclareSlot_StatusFromModule CPH_PyRegistry_Init(CPH_RegistryObject *self,
                                                        [[maybe_unused]] PyObject *args,
                                                        [[maybe_unused]] PyObject *kwds) {
    self->entity_capacity = kInitialEntityCapacity;
    self->active_entities = 0;
    self->generations =
        static_cast<uint32_t *>(CPH_PyCalloc(self->entity_capacity, sizeof(uint32_t)));
    self->free_indices =
        static_cast<uint32_t *>(CPH_PyAlloc(self->entity_capacity * sizeof(uint32_t)));

    if (!self->generations || !self->free_indices) {
        return -1;
    }

    for (uint32_t i = 0; i < self->entity_capacity; i++) {
        self->free_indices[i] = (self->entity_capacity - 1) - i;
        self->generations[i]  = 1;
    }
    self->free_count = self->entity_capacity;

    self->component_capacity = kInitialComponentCapacity;
    self->component_count    = 0;
    self->components =
        static_cast<CPH_SparseSet *>(CPH_PyAlloc(self->component_capacity * sizeof(CPH_SparseSet)));

    if (!self->components) {
        return -1;
    }
    CPH_INIT_LOCK(self->ecs_lock);
    self->parsers = CPH::NewParserGroup<CPH_ECSParsers>();
    if (!self->parsers) {
        return -1;
    }
    self->parsers->Init();
    return 0;
}
PyType_DeclareSlot_VoidFromModule CPH_PyRegistry_Dealloc(CPH_RegistryObject *self) {
    if (self->generations) {
        CPH_PyFree(self->generations);
    }
    if (self->free_indices) {
        CPH_PyFree(self->free_indices);
    }
    if (self->components) {
        for (uint32_t i = 0; i < self->component_count; i++) {
            SparseSet_Destroy(&self->components[i]);
        }
        CPH_PyFree(self->components);
    }
    if (self->parsers) {

        delete self->parsers;
    }
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

// --- ENTITY MANAGEMENT ---

// Signature updated for METH_NOARGS
PyCFunction_DeclareMethodFromModule CPH_PyRegistry_Create(CPH_RegistryObject *self,
                                                          [[maybe_unused]] PyObject *args) {
    CPH_SHADOW_LOCK(&self->ecs_lock);
    if (self->free_count == 0) {
        uint32_t new_cap = self->entity_capacity * 2;
        uint32_t *new_gens =
            static_cast<uint32_t *>(CPH_PyRealloc(self->generations, new_cap * sizeof(uint32_t)));
        uint32_t *new_free =
            static_cast<uint32_t *>(CPH_PyRealloc(self->free_indices, new_cap * sizeof(uint32_t)));
        if (!new_gens || !new_free) {
            CPH_SHADOW_UNLOCK(&self->ecs_lock);
            return PyErr_NoMemory();
        }

        for (uint32_t i = self->entity_capacity; i < new_cap; i++) {
            new_gens[i]                  = 1;
            new_free[self->free_count++] = (new_cap - 1) - (i - self->entity_capacity);
        }
        self->generations     = new_gens;
        self->free_indices    = new_free;
        self->entity_capacity = new_cap;
    }

    uint32_t index    = self->free_indices[--self->free_count];
    uint32_t gen      = self->generations[index];
    CPH_Entity handle = (static_cast<uint64_t>(gen) << kHandleIndexBits) | index;
    self->active_entities++;

    CPH_SHADOW_UNLOCK(&self->ecs_lock);

    return PyLong_FromUnsignedLongLong(handle);
}

PyCFunction_DeclareMethodFromModule CPH_PyRegistry_Destroy(CPH_RegistryObject *self,
                                                           PyObject *const *args, size_t nargsf,
                                                           PyObject *kwnames) {
    uint64_t handle;

    if (!self->parsers->RegEntityOnly.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, handle)) {
        return nullptr;
    }

    CPH_SHADOW_LOCK(&self->ecs_lock);

    uint32_t index = static_cast<uint32_t>(handle & kHandleIndexMask);
    uint32_t gen   = static_cast<uint32_t>(handle >> kHandleIndexBits);

    if (index >= self->entity_capacity || self->generations[index] != gen) {
        CPH_SHADOW_UNLOCK(&self->ecs_lock);
        Py_RETURN_NONE; // Already dead
    }

    // 1. Remove entity from all components
    for (uint32_t i = 0; i < self->component_count; i++) {
        CPH_SparseSet *set = &self->components[i];
        if (index < set->sparse_capacity && set->sparse[index] != kInvalidDenseIndex) {
            uint32_t dense_idx = set->sparse[index];
            uint32_t last_idx  = set->count - 1;

            if (dense_idx != last_idx) {
                // Swap and pop
                CPH_Entity last_entity   = set->dense[last_idx];
                uint32_t last_entity_idx = static_cast<uint32_t>(last_entity & kHandleIndexMask);

                set->dense[dense_idx]        = last_entity;
                set->sparse[last_entity_idx] = dense_idx;
                memcpy(&set->data[static_cast<size_t>(dense_idx) * set->element_size],
                       &set->data[static_cast<size_t>(last_idx) * set->element_size],
                       set->element_size);
            }
            set->sparse[index] = kInvalidDenseIndex;
            set->count--;
        }
    }

    // 2. Kill entity handle
    self->generations[index]++;
    self->free_indices[self->free_count++] = index;
    self->active_entities--;

    CPH_SHADOW_UNLOCK(&self->ecs_lock);

    Py_RETURN_NONE;
}

PyCFunction_DeclareMethodFromModule CPH_PyRegistry_IsAlive(CPH_RegistryObject *self,
                                                           PyObject *const *args, size_t nargsf,
                                                           PyObject *kwnames) {
    uint64_t handle;

    if (!self->parsers->RegEntityOnly.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, handle)) {
        return nullptr;
    }

    CPH_SHADOW_LOCK(&self->ecs_lock);

    uint32_t index = static_cast<uint32_t>(handle & kHandleIndexMask);
    uint32_t gen   = static_cast<uint32_t>(handle >> kHandleIndexBits);

    bool alive = (index < self->entity_capacity && self->generations[index] == gen) != 0;

    CPH_SHADOW_UNLOCK(&self->ecs_lock);

    if (alive) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

// --- COMPONENT MANAGEMENT ---

PyCFunction_DeclareMethodFromModule CPH_PyRegistry_RegisterComponent(CPH_RegistryObject *self,
                                                                     PyObject *const *args,
                                                                     size_t nargsf,
                                                                     PyObject *kwnames) {
    uint32_t size;

    if (!self->parsers->RegRegComp.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, size)) {
        return nullptr;
    }

    CPH_SHADOW_LOCK(&self->ecs_lock);

    if (self->component_count >= self->component_capacity) {
        // Manual check for register_component too!
        if (atomic_load_explicit(&self->view_export_count, memory_order_acquire) > 0) {
            PyErr_SetString(PyExc_BufferError,
                            "Cannot register new components while a view is held.");
            CPH_SHADOW_UNLOCK(&self->ecs_lock);
            return nullptr;
        }

        uint32_t new_cap         = self->component_capacity * 2;
        CPH_SparseSet *new_comps = static_cast<CPH_SparseSet *>(
            CPH_PyRealloc(self->components, new_cap * sizeof(CPH_SparseSet)));
        if (!new_comps) {
            CPH_SHADOW_UNLOCK(&self->ecs_lock);
            return PyErr_NoMemory();
        }
        self->components         = new_comps;
        self->component_capacity = new_cap;
    }

    uint32_t comp_id = self->component_count++;
    SparseSet_Init(&self->components[comp_id], size);

    CPH_SHADOW_UNLOCK(&self->ecs_lock);

    return PyLong_FromUnsignedLong(comp_id);
}

PyCFunction_DeclareMethodFromModule CPH_PyRegistry_Add(CPH_RegistryObject *self,
                                                       PyObject *const *args, size_t nargsf,
                                                       PyObject *kwnames) {
    uint64_t handle;
    uint32_t comp_id;
    PyObject *data_obj = nullptr;

    if (!self->parsers->RegAdd.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, handle, comp_id,
                                     data_obj)) {
        return nullptr;
    }

    uint32_t index = static_cast<uint32_t>(handle & kHandleIndexMask);
    uint32_t gen   = static_cast<uint32_t>(handle >> kHandleIndexBits);

    if (index >= self->entity_capacity || self->generations[index] != gen) {
        return PyErr_Format(PyExc_ValueError, "Invalid or stale entity handle");
    }

    if (comp_id >= self->component_count) {
        return PyErr_Format(PyExc_ValueError, "Invalid component ID");
    }

    CPH_SHADOW_LOCK(&self->ecs_lock);

    CPH_SparseSet *set = &self->components[comp_id];

    Py_buffer view;
    if (data_obj && data_obj != Py_None) {
        if (PyObject_GetBuffer(data_obj, &view, PyBUF_SIMPLE) != 0) {
            CPH_SHADOW_UNLOCK(&self->ecs_lock);
            return nullptr;
        }
        if (view.len != set->element_size) {
            CPH_SHADOW_UNLOCK(&self->ecs_lock);
            PyBuffer_Release(&view);
            return PyErr_Format(PyExc_ValueError, "Data size mismatch: expected %u, got %zd",
                                set->element_size, view.len);
        }
    }

    uint32_t dense_idx = kInvalidDenseIndex;
    if (!SparseSet_EnsureSparseCapacity(set, index + 1)) {
        goto fail_quiet;
    }

    dense_idx = set->sparse[index];
    if (dense_idx == kInvalidDenseIndex) {
        // This helper now sets its own BufferError or MemoryError
        if (!SparseSet_EnsureDenseCapacity(self, set)) {
            goto fail_quiet;
        }
        dense_idx             = set->count++;
        set->dense[dense_idx] = handle;
        set->sparse[index]    = dense_idx;
    }

    // Write data
    if (data_obj && data_obj != Py_None) {
        memcpy(&set->data[static_cast<size_t>(dense_idx) * set->element_size], view.buf,
               set->element_size);
        PyBuffer_Release(&view);
    } else {
        memset(&set->data[static_cast<size_t>(dense_idx) * set->element_size], 0,
               set->element_size);
    }

    CPH_SHADOW_UNLOCK(&self->ecs_lock);

    Py_RETURN_NONE;

fail_quiet:
    CPH_SHADOW_UNLOCK(&self->ecs_lock);
    if (data_obj && data_obj != Py_None) {
        PyBuffer_Release(&view);
    }
    return nullptr; // Return the error already set in the helper
}

PyCFunction_DeclareMethodFromModule CPH_PyRegistry_Remove(CPH_RegistryObject *self,
                                                          PyObject *const *args, size_t nargsf,
                                                          PyObject *kwnames) {
    uint64_t handle;
    uint32_t comp_id;

    if (!self->parsers->RegEntComp.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, handle,
                                         comp_id)) {
        return nullptr;
    }

    uint32_t index = static_cast<uint32_t>(handle & kHandleIndexMask);
    if (comp_id >= self->component_count) {
        return PyErr_Format(PyExc_ValueError, "Invalid component ID");
    }

    CPH_SHADOW_LOCK(&self->ecs_lock);

    CPH_SparseSet *set = &self->components[comp_id];

    if (index < set->sparse_capacity && set->sparse[index] != kInvalidDenseIndex) {
        uint32_t dense_idx = set->sparse[index];
        uint32_t last_idx  = set->count - 1;

        if (dense_idx != last_idx) {
            // Swap and pop
            CPH_Entity last_entity   = set->dense[last_idx];
            uint32_t last_entity_idx = static_cast<uint32_t>(last_entity & kHandleIndexMask);

            set->dense[dense_idx]        = last_entity;
            set->sparse[last_entity_idx] = dense_idx;
            memcpy(&set->data[static_cast<size_t>(dense_idx) * set->element_size],
                   &set->data[static_cast<size_t>(last_idx) * set->element_size],
                   set->element_size);
        }
        set->sparse[index] = kInvalidDenseIndex;
        set->count--;
    }

    CPH_SHADOW_UNLOCK(&self->ecs_lock);

    Py_RETURN_NONE;
}

PyCFunction_DeclareMethodFromModule CPH_PyRegistry_Has(CPH_RegistryObject *self,
                                                       PyObject *const *args, size_t nargsf,
                                                       PyObject *kwnames) {
    uint64_t handle;
    uint32_t comp_id;

    if (!self->parsers->RegEntComp.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, handle,
                                         comp_id)) {
        return nullptr;
    }

    uint32_t index = static_cast<uint32_t>(handle & kHandleIndexMask);
    if (comp_id >= self->component_count) {
        return PyErr_Format(PyExc_ValueError, "Invalid component ID");
    }
    CPH_SHADOW_LOCK(&self->ecs_lock);
    CPH_SparseSet *set = &self->components[comp_id];
    bool has_comp = (index < set->sparse_capacity && set->sparse[index] != kInvalidDenseIndex) != 0;
    CPH_SHADOW_UNLOCK(&self->ecs_lock);
    if (has_comp) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

PyCFunction_DeclareMethodFromModule CPH_PyRegistry_Get(CPH_RegistryObject *self,
                                                       PyObject *const *args, size_t nargsf,
                                                       PyObject *kwnames) {
    uint64_t handle;
    uint32_t comp_id;

    if (!self->parsers->RegEntComp.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, handle,
                                         comp_id)) {
        return nullptr;
    }

    uint32_t index = static_cast<uint32_t>(handle & kHandleIndexMask);
    if (comp_id >= self->component_count) {
        return PyErr_Format(PyExc_ValueError, "Invalid component ID");
    }

    CPH_SHADOW_LOCK(&self->ecs_lock);
    CPH_SparseSet *set = &self->components[comp_id];

    if (index >= set->sparse_capacity || set->sparse[index] == kInvalidDenseIndex) {
        CPH_SHADOW_UNLOCK(&self->ecs_lock);
        Py_RETURN_NONE;
    }

    uint32_t dense_idx = set->sparse[index];
    PyObject *result   = PyBytes_FromStringAndSize(
        reinterpret_cast<char *>(&set->data[static_cast<size_t>(dense_idx) * set->element_size]),
        set->element_size);
    CPH_SHADOW_UNLOCK(&self->ecs_lock);

    return result;
}

PyCFunction_DeclareMethodFromModule CPH_PyRegistry_GetView(CPH_RegistryObject *self,
                                                           PyObject *const *args, size_t nargsf,
                                                           PyObject *kwnames) {
    uint32_t comp_id;

    if (!self->parsers->RegCompOnly.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, comp_id)) {
        return nullptr;
    }

    if (comp_id >= self->component_count) {
        return PyErr_Format(PyExc_ValueError, "Invalid component ID");
    }

    CPH_SHADOW_LOCK(&self->ecs_lock);
    PyObject *proxy = make_ecs_proxy(self, comp_id, false);
    CPH_SHADOW_UNLOCK(&self->ecs_lock);
    return proxy;
}

PyCFunction_DeclareMethodFromModule CPH_PyRegistry_GetEntities(CPH_RegistryObject *self,
                                                               PyObject *const *args, size_t nargsf,
                                                               PyObject *kwnames) {
    uint32_t comp_id;

    if (!self->parsers->RegCompOnly.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, comp_id)) {
        return nullptr;
    }

    if (comp_id >= self->component_count) {
        return PyErr_Format(PyExc_ValueError, "Invalid component ID");
    }

    CPH_SHADOW_LOCK(&self->ecs_lock);
    // make_ecs_proxy(self, comp_id, true) handles the PROXY_ECS_ENTITIES case
    PyObject *proxy = make_ecs_proxy(self, comp_id, true);
    CPH_SHADOW_UNLOCK(&self->ecs_lock);

    return proxy;
}

PyCFunction_DeclareMethodFromModule CPH_PyRegistry_SyncFromWorld(CPH_RegistryObject *self,
                                                                 PyObject *const *args,
                                                                 size_t nargsf, PyObject *kwnames) {
    PyObject *world_obj = nullptr;
    uint32_t h_comp_id;
    int p_comp_id = -1;
    int r_comp_id = -1; // -1 means ignore

    if (!self->parsers->RegSyncPhys.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, world_obj,
                                          h_comp_id, p_comp_id, r_comp_id)) {
        return nullptr;
    }

    CPH_PhysicsWorldObject *world = reinterpret_cast<CPH_PhysicsWorldObject *>(world_obj);

    if (h_comp_id >= self->component_count) {
        return PyErr_Format(PyExc_ValueError, "Invalid handle component ID");
    }

    CPH_SparseSet *h_set = &self->components[h_comp_id];
    CPH_SparseSet *p_set =
        (p_comp_id >= 0 && static_cast<uint32_t>(p_comp_id) < self->component_count)
            ? &self->components[p_comp_id]
            : nullptr;
    CPH_SparseSet *r_set =
        (r_comp_id >= 0 && static_cast<uint32_t>(r_comp_id) < self->component_count)
            ? &self->components[r_comp_id]
            : nullptr;

    if (h_set->element_size != sizeof(uint64_t)) {
        return PyErr_Format(PyExc_TypeError, "Handle component must be 8 bytes (uint64)");
    }
    if (p_set && p_set->element_size != sizeof(float) * 3) {
        return PyErr_Format(PyExc_TypeError, "Position component must be 12 bytes (3x float32)");
    }
    if (r_set && r_set->element_size != sizeof(float) * 4) {
        return PyErr_Format(PyExc_TypeError, "Rotation component must be 16 bytes (4x float32)");
    }

    // Lock both Physics and ECS
    CPH_SHADOW_LOCK(&world->physics.shadow_lock);
    CPH_SHADOW_LOCK(&self->ecs_lock);

    for (uint32_t i = 0; i < h_set->count; i++) {
        CPH_Entity ent   = h_set->dense[i];
        uint32_t ent_idx = static_cast<uint32_t>(ent & kHandleIndexMask);

        uint64_t handle;
        memcpy(&handle, &h_set->data[static_cast<size_t>(i) * sizeof(uint64_t)], sizeof(uint64_t));

        uint32_t slot;
        if (unpack_handle(&world->physics, handle, &slot)) {
            uint32_t phys_dense = world->physics.slot_to_dense[slot];

            // 1. Sync Position (Using your PositionVector / PosStride logic)
            if (p_set && ent_idx < p_set->sparse_capacity &&
                p_set->sparse[ent_idx] != kInvalidDenseIndex) {
                uint32_t p_dense = p_set->sparse[ent_idx];

                // Assuming world->physics.positions stores 1 PositionVector (or 4 Reals) per entity
                // depending on your stride. Using PosStride as defined in your types header.
                CPH_PosStride *p =
                    &(reinterpret_cast<CPH_PosStride *>(world->physics.positions))[phys_dense];

                float *out = reinterpret_cast<float *>(
                    &p_set->data[static_cast<size_t>(p_dense) * sizeof(float) * 3]);
                out[0] = static_cast<float>(p->x);
                out[1] = static_cast<float>(p->y);
                out[2] = static_cast<float>(p->z);
            }

            // 2. Sync Rotation
            // world->physics.rotations is a flat float array (4 floats per quaternion)
            if (r_set && ent_idx < r_set->sparse_capacity &&
                r_set->sparse[ent_idx] != kInvalidDenseIndex) {
                uint32_t r_dense = r_set->sparse[ent_idx];

                // Index directly into the float array
                float *phys_rot = &world->physics.rotations[static_cast<size_t>(phys_dense) * 4];
                float *out      = reinterpret_cast<float *>(
                    &r_set->data[static_cast<size_t>(r_dense) * sizeof(float) * 4]);

                // Fast direct float copy (16 bytes)
                memcpy(out, phys_rot, sizeof(float) * 4);
            }
        }
    }

    CPH_SHADOW_UNLOCK(&self->ecs_lock);
    CPH_SHADOW_UNLOCK(&world->physics.shadow_lock);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethodFromModule CPH_PyRegistry_Clear(CPH_RegistryObject *self,
                                                         [[maybe_unused]] PyObject *args) {
    CPH_SHADOW_LOCK(&self->ecs_lock);

    if (atomic_load_explicit(&self->view_export_count, memory_order_acquire) > 0) {
        CPH_SHADOW_UNLOCK(&self->ecs_lock);
        return PyErr_Format(PyExc_BufferError,
                            "Cannot clear registry while memoryviews are exported.");
    }

    // Clear all components
    for (uint32_t i = 0; i < self->component_count; i++) {
        self->components[i].count = 0;
        if (self->components[i].sparse) {
            // INVALID_DENSE_INDEX is usually 0xFFFFFFFF, so memset 0xFF is valid here
            memset(self->components[i].sparse, 0xFF,
                   self->components[i].sparse_capacity * sizeof(uint32_t));
        }
    }

    // Invalidate entities
    for (uint32_t i = 0; i < self->entity_capacity; i++) {
        self->free_indices[i] = (self->entity_capacity - 1) - i;
        self->generations[i]++; // Kills all active handles instantly
    }

    self->free_count      = self->entity_capacity;
    self->active_entities = 0;

    CPH_SHADOW_UNLOCK(&self->ecs_lock);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethodFromModule CPH_PyRegistry_GetActiveCount(CPH_RegistryObject *self,
                                                                  [[maybe_unused]] PyObject *args) {
    return PyLong_FromUnsignedLong(self->active_entities);
}

PyCFunction_DeclareMethodFromModule CPH_PyRegistry_GetComponentCount(CPH_RegistryObject *self,
                                                                     PyObject *const *args,
                                                                     size_t nargsf,
                                                                     PyObject *kwnames) {
    uint32_t comp_id;

    if (!self->parsers->RegCompOnly.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, comp_id)) {
        return nullptr;
    }

    if (comp_id >= self->component_count) {
        return PyLong_FromLong(0);
    }

    return PyLong_FromSsize_t(self->components[comp_id].count);
}

static PyMethodDef kRegistryMethods[] = {

    CPH::Python::NoArgsMethod("create", CPH_PyRegistry_Create),
    CPH::Python::FastCallMethod("destroy", CPH_PyRegistry_Destroy),
    CPH::Python::FastCallMethod("is_alive", CPH_PyRegistry_IsAlive),
    CPH::Python::NoArgsMethod("clear", CPH_PyRegistry_Clear), // Wipes the registry
    CPH::Python::FastCallMethod("register_component", CPH_PyRegistry_RegisterComponent),
    CPH::Python::FastCallMethod("add", CPH_PyRegistry_Add),
    CPH::Python::FastCallMethod("remove", CPH_PyRegistry_Remove),
    CPH::Python::FastCallMethod("has", CPH_PyRegistry_Has),
    CPH::Python::FastCallMethod("get", CPH_PyRegistry_Get), // Single entity data access
    CPH::Python::FastCallMethod("get_view", CPH_PyRegistry_GetView),
    CPH::Python::FastCallMethod("get_entities", CPH_PyRegistry_GetEntities),
    CPH::Python::FastCallMethod("sync_from_world", CPH_PyRegistry_SyncFromWorld),
    CPH::Python::NoArgsMethod("get_active_count", CPH_PyRegistry_GetActiveCount), // ECS Statistics
    CPH::Python::FastCallMethod("get_component_count",
                                CPH_PyRegistry_GetComponentCount), // ECS Statistics
    {nullptr, nullptr, 0, nullptr}};

PyType_Spec Registry_spec = {
    .name      = "culverin._culverin_c.Registry",
    .basicsize = sizeof(CPH_RegistryObject),
    .flags     = Py_TPFLAGS_DEFAULT,
    .slots =
        (PyType_Slot[]){

            {.slot = Py_tp_new, .pfunc = reinterpret_cast<void *>(PyType_GenericNew)},
            {.slot = Py_tp_init, .pfunc = reinterpret_cast<void *>(CPH_PyRegistry_Init)},
            {.slot = Py_tp_dealloc, .pfunc = reinterpret_cast<void *>(CPH_PyRegistry_Dealloc)},
            {.slot = Py_tp_methods, .pfunc = reinterpret_cast<void *>(kRegistryMethods)},
            {},

        },
};