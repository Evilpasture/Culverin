#include "culverin_ecs.h"
#include "culverin.h"
#include "culverin_arg_indices.h"
#include "culverin_getters.h"
#include "culverin_module.h"
#include "culverin_physics_world.h"
#include "culverin_python.h"
#include <stddef.h>

// --- INTERNAL HELPERS ---
static constexpr uint32_t INVALID_DENSE_INDEX    = 0xFFFFFFFF;
static constexpr auto INITIAL_ENTITY_CAPACITY    = 1024;
static constexpr auto INITIAL_SPARSE_CAPACITY    = 1024;
static constexpr auto INITIAL_COMPONENT_CAPACITY = 16;

static PyObject *make_ecs_proxy(RegistryObject *self, uint32_t comp_id, bool entities) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    SparseSet *set    = &self->components[comp_id];

    BufferProxyObject *proxy =
        PyObject_GC_New(BufferProxyObject, (PyTypeObject *)st->BufferProxyType);
    if (!proxy) {
        return nullptr;
    }

    proxy->owner = (PyObject *)self;
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
    return (PyObject *)proxy;
}

static void SparseSet_Init(SparseSet *set, uint32_t element_size) {
    set->sparse          = nullptr;
    set->dense           = nullptr;
    set->data            = nullptr;
    set->element_size    = element_size;
    set->count           = 0;
    set->dense_capacity  = 0;
    set->sparse_capacity = 0;
}

static void SparseSet_Destroy(SparseSet *set) {
    if (set->sparse) {
        CULV_RAW_FREE(set->sparse);
    }
    if (set->dense) {
        CULV_RAW_FREE(set->dense);
    }
    if (set->data) {
        CULV_RAW_FREE(set->data);
    }
}

static bool SparseSet_EnsureSparseCapacity(SparseSet *set, uint32_t required_capacity) {
    if (set->sparse_capacity >= required_capacity) {
        return true;
    }

    uint32_t new_cap =
        set->sparse_capacity == 0 ? INITIAL_SPARSE_CAPACITY : set->sparse_capacity * 2;
    while (new_cap < required_capacity) {
        new_cap *= 2;
    }

    uint32_t *new_sparse = (uint32_t *)CULV_RAW_REALLOC(set->sparse, new_cap * sizeof(uint32_t));
    if (!new_sparse) {
        PyErr_NoMemory(); // Set error here
        return false;
    }

    for (uint32_t i = set->sparse_capacity; i < new_cap; i++) {
        new_sparse[i] = INVALID_DENSE_INDEX;
    }

    set->sparse          = new_sparse;
    set->sparse_capacity = new_cap;
    return true;
}

static bool SparseSet_EnsureDenseCapacity(RegistryObject *reg, SparseSet *set) {
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

    CulvEntity *new_dense =
        (CulvEntity *)CULV_RAW_REALLOC(set->dense, new_cap * sizeof(CulvEntity));
    if (!new_dense) {
        PyErr_NoMemory(); // Set error here
        return false;
    }

    uint8_t *new_data = (uint8_t *)CULV_RAW_REALLOC(set->data, (size_t)new_cap * set->element_size);
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
void culverin_init_ecs_parsers(ECSParsers *ep);
PyType_DeclareSlot_StatusFromModule Registry_init(RegistryObject *self,
                                                  CULV_MAYBE_UNUSED PyObject *args,
                                                  CULV_MAYBE_UNUSED PyObject *kwds) {
    self->entity_capacity = INITIAL_ENTITY_CAPACITY;
    self->active_entities = 0;
    self->generations     = (uint32_t *)CULV_RAW_CALLOC(self->entity_capacity, sizeof(uint32_t));
    self->free_indices    = (uint32_t *)CULV_RAW_MALLOC(self->entity_capacity * sizeof(uint32_t));

    if (!self->generations || !self->free_indices) {
        return -1;
    }

    for (uint32_t i = 0; i < self->entity_capacity; i++) {
        self->free_indices[i] = (self->entity_capacity - 1) - i;
        self->generations[i]  = 1;
    }
    self->free_count = self->entity_capacity;

    self->component_capacity = INITIAL_COMPONENT_CAPACITY;
    self->component_count    = 0;
    self->components = (SparseSet *)CULV_RAW_MALLOC(self->component_capacity * sizeof(SparseSet));

    if (!self->components) {
        return -1;
    }
    INIT_LOCK(self->ecs_lock);
    self->parsers = (ECSParsers *)PyMem_Malloc(sizeof(ECSParsers));
    if (!self->parsers) {
        return -1;
    }
    culverin_init_ecs_parsers(self->parsers);
    return 0;
}
void culverin_free_ecs_parsers(ECSParsers *ep);
PyType_DeclareSlot_VoidFromModule Registry_dealloc(RegistryObject *self) {
    if (self->generations) {
        CULV_RAW_FREE(self->generations);
    }
    if (self->free_indices) {
        CULV_RAW_FREE(self->free_indices);
    }
    if (self->components) {
        for (uint32_t i = 0; i < self->component_count; i++) {
            SparseSet_Destroy(&self->components[i]);
        }
        CULV_RAW_FREE(self->components);
    }
    if (self->parsers) {
        culverin_free_ecs_parsers(self->parsers);
        PyMem_Free(self->parsers);
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

// --- ENTITY MANAGEMENT ---

// Signature updated for METH_NOARGS
PyCFunction_DeclareMethodFromModule Registry_create(RegistryObject *self,
                                                    CULV_MAYBE_UNUSED PyObject *args) {
    SHADOW_LOCK(&self->ecs_lock);
    if (self->free_count == 0) {
        uint32_t new_cap = self->entity_capacity * 2;
        uint32_t *new_gens =
            (uint32_t *)CULV_RAW_REALLOC(self->generations, new_cap * sizeof(uint32_t));
        uint32_t *new_free =
            (uint32_t *)CULV_RAW_REALLOC(self->free_indices, new_cap * sizeof(uint32_t));
        if (!new_gens || !new_free) {
            SHADOW_UNLOCK(&self->ecs_lock);
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
    CulvEntity handle = ((uint64_t)gen << HANDLE_INDEX_BITS) | index;
    self->active_entities++;

    SHADOW_UNLOCK(&self->ecs_lock);

    return PyLong_FromUnsignedLongLong(handle);
}

PyCFunction_DeclareMethodFromModule Registry_destroy(RegistryObject *self, PyObject *const *args,
                                                     size_t nargsf, PyObject *kwnames) {
    uint64_t handle;
    void *targets[RegEntityOnly_COUNT] = {[IDX_REO_ENT] = &handle};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->RegEntityOnlyParser, targets)) {
        return nullptr;
    }

    SHADOW_LOCK(&self->ecs_lock);

    uint32_t index = (uint32_t)(handle & HANDLE_INDEX_MASK);
    uint32_t gen   = (uint32_t)(handle >> HANDLE_INDEX_BITS);

    if (index >= self->entity_capacity || self->generations[index] != gen) {
        SHADOW_UNLOCK(&self->ecs_lock);
        Py_RETURN_NONE; // Already dead
    }

    // 1. Remove entity from all components
    for (uint32_t i = 0; i < self->component_count; i++) {
        SparseSet *set = &self->components[i];
        if (index < set->sparse_capacity && set->sparse[index] != INVALID_DENSE_INDEX) {
            uint32_t dense_idx = set->sparse[index];
            uint32_t last_idx  = set->count - 1;

            if (dense_idx != last_idx) {
                // Swap and pop
                CulvEntity last_entity   = set->dense[last_idx];
                uint32_t last_entity_idx = (uint32_t)(last_entity & HANDLE_INDEX_MASK);

                set->dense[dense_idx]        = last_entity;
                set->sparse[last_entity_idx] = dense_idx;
                memcpy(&set->data[(size_t)dense_idx * set->element_size],
                       &set->data[(size_t)last_idx * set->element_size], set->element_size);
            }
            set->sparse[index] = INVALID_DENSE_INDEX;
            set->count--;
        }
    }

    // 2. Kill entity handle
    self->generations[index]++;
    self->free_indices[self->free_count++] = index;
    self->active_entities--;

    SHADOW_UNLOCK(&self->ecs_lock);

    Py_RETURN_NONE;
}

PyCFunction_DeclareMethodFromModule Registry_is_alive(RegistryObject *self, PyObject *const *args,
                                                      size_t nargsf, PyObject *kwnames) {
    uint64_t handle;
    void *targets[RegEntityOnly_COUNT] = {[IDX_REO_ENT] = &handle};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->RegEntityOnlyParser, targets)) {
        return nullptr;
    }

    SHADOW_LOCK(&self->ecs_lock);

    uint32_t index = (uint32_t)(handle & HANDLE_INDEX_MASK);
    uint32_t gen   = (uint32_t)(handle >> HANDLE_INDEX_BITS);

    bool alive = (index < self->entity_capacity && self->generations[index] == gen) != 0;

    SHADOW_UNLOCK(&self->ecs_lock);

    if (alive) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

// --- COMPONENT MANAGEMENT ---

PyCFunction_DeclareMethodFromModule Registry_register_component(RegistryObject *self,
                                                                PyObject *const *args,
                                                                size_t nargsf, PyObject *kwnames) {
    uint32_t size;
    void *targets[RegRegComp_COUNT] = {[IDX_RRC_SIZE] = &size};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->RegRegCompParser, targets)) {
        return nullptr;
    }

    SHADOW_LOCK(&self->ecs_lock);

    if (self->component_count >= self->component_capacity) {
        // Manual check for register_component too!
        if (atomic_load_explicit(&self->view_export_count, memory_order_acquire) > 0) {
            PyErr_SetString(PyExc_BufferError,
                            "Cannot register new components while a view is held.");
            SHADOW_UNLOCK(&self->ecs_lock);
            return nullptr;
        }

        uint32_t new_cap = self->component_capacity * 2;
        SparseSet *new_comps =
            (SparseSet *)CULV_RAW_REALLOC(self->components, new_cap * sizeof(SparseSet));
        if (!new_comps) {
            SHADOW_UNLOCK(&self->ecs_lock);
            return PyErr_NoMemory();
        }
        self->components         = new_comps;
        self->component_capacity = new_cap;
    }

    uint32_t comp_id = self->component_count++;
    SparseSet_Init(&self->components[comp_id], size);

    SHADOW_UNLOCK(&self->ecs_lock);

    return PyLong_FromUnsignedLong(comp_id);
}

PyCFunction_DeclareMethodFromModule Registry_add(RegistryObject *self, PyObject *const *args,
                                                 size_t nargsf, PyObject *kwnames) {
    uint64_t handle;
    uint32_t comp_id;
    PyObject *data_obj = nullptr;

    void *targets[RegAdd_COUNT] = {[IDX_RA_ENT]  = (void *)&handle,
                                   [IDX_RA_COMP] = (void *)&comp_id,
                                   [IDX_RA_DATA] = (void *)&data_obj};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &self->parsers->RegAddParser,
                           targets)) {
        return nullptr;
    }

    uint32_t index = (uint32_t)(handle & HANDLE_INDEX_MASK);
    uint32_t gen   = (uint32_t)(handle >> HANDLE_INDEX_BITS);

    if (index >= self->entity_capacity || self->generations[index] != gen) {
        return PyErr_Format(PyExc_ValueError, "Invalid or stale entity handle");
    }

    if (comp_id >= self->component_count) {
        return PyErr_Format(PyExc_ValueError, "Invalid component ID");
    }

    SHADOW_LOCK(&self->ecs_lock);

    SparseSet *set = &self->components[comp_id];

    Py_buffer view;
    if (data_obj && data_obj != Py_None) {
        if (PyObject_GetBuffer(data_obj, &view, PyBUF_SIMPLE) != 0) {
            SHADOW_UNLOCK(&self->ecs_lock);
            return nullptr;
        }
        if (view.len != set->element_size) {
            SHADOW_UNLOCK(&self->ecs_lock);
            PyBuffer_Release(&view);
            return PyErr_Format(PyExc_ValueError, "Data size mismatch: expected %u, got %zd",
                                set->element_size, view.len);
        }
    }

    if (!SparseSet_EnsureSparseCapacity(set, index + 1)) {
        goto fail_quiet;
    }

    uint32_t dense_idx = set->sparse[index];
    if (dense_idx == INVALID_DENSE_INDEX) {
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
        memcpy(&set->data[(size_t)dense_idx * set->element_size], view.buf, set->element_size);
        PyBuffer_Release(&view);
    } else {
        memset(&set->data[(size_t)dense_idx * set->element_size], 0, set->element_size);
    }

    SHADOW_UNLOCK(&self->ecs_lock);

    Py_RETURN_NONE;

fail_quiet:
    SHADOW_UNLOCK(&self->ecs_lock);
    if (data_obj && data_obj != Py_None) {
        PyBuffer_Release(&view);
    }
    return nullptr; // Return the error already set in the helper
}

PyCFunction_DeclareMethodFromModule Registry_remove(RegistryObject *self, PyObject *const *args,
                                                    size_t nargsf, PyObject *kwnames) {
    uint64_t handle;
    uint32_t comp_id;

    void *targets[RegEntComp_COUNT] = {[IDX_REC_ENT] = &handle, [IDX_REC_COMP] = &comp_id};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->RegEntCompParser, targets)) {
        return nullptr;
    }

    uint32_t index = (uint32_t)(handle & HANDLE_INDEX_MASK);
    if (comp_id >= self->component_count) {
        return PyErr_Format(PyExc_ValueError, "Invalid component ID");
    }

    SHADOW_LOCK(&self->ecs_lock);

    SparseSet *set = &self->components[comp_id];

    if (index < set->sparse_capacity && set->sparse[index] != INVALID_DENSE_INDEX) {
        uint32_t dense_idx = set->sparse[index];
        uint32_t last_idx  = set->count - 1;

        if (dense_idx != last_idx) {
            // Swap and pop
            CulvEntity last_entity   = set->dense[last_idx];
            uint32_t last_entity_idx = (uint32_t)(last_entity & HANDLE_INDEX_MASK);

            set->dense[dense_idx]        = last_entity;
            set->sparse[last_entity_idx] = dense_idx;
            memcpy(&set->data[(size_t)dense_idx * set->element_size],
                   &set->data[(size_t)last_idx * set->element_size], set->element_size);
        }
        set->sparse[index] = INVALID_DENSE_INDEX;
        set->count--;
    }

    SHADOW_UNLOCK(&self->ecs_lock);

    Py_RETURN_NONE;
}

PyCFunction_DeclareMethodFromModule Registry_has(RegistryObject *self, PyObject *const *args,
                                                 size_t nargsf, PyObject *kwnames) {
    uint64_t handle;
    uint32_t comp_id;

    void *targets[RegEntComp_COUNT] = {[IDX_REC_ENT] = &handle, [IDX_REC_COMP] = &comp_id};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->RegEntCompParser, targets)) {
        return nullptr;
    }

    uint32_t index = (uint32_t)(handle & HANDLE_INDEX_MASK);
    if (comp_id >= self->component_count) {
        return PyErr_Format(PyExc_ValueError, "Invalid component ID");
    }
    SHADOW_LOCK(&self->ecs_lock);
    SparseSet *set = &self->components[comp_id];
    bool has_comp =
        (index < set->sparse_capacity && set->sparse[index] != INVALID_DENSE_INDEX) != 0;
    SHADOW_UNLOCK(&self->ecs_lock);
    if (has_comp) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

PyCFunction_DeclareMethodFromModule Registry_get(RegistryObject *self, PyObject *const *args,
                                                 size_t nargsf, PyObject *kwnames) {
    uint64_t handle;
    uint32_t comp_id;
    void *targets[RegEntComp_COUNT] = {[IDX_REC_ENT] = &handle, [IDX_REC_COMP] = &comp_id};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->RegEntCompParser, targets)) {
        return nullptr;
    }

    uint32_t index = (uint32_t)(handle & HANDLE_INDEX_MASK);
    if (comp_id >= self->component_count) {
        return PyErr_Format(PyExc_ValueError, "Invalid component ID");
    }

    SHADOW_LOCK(&self->ecs_lock);
    SparseSet *set = &self->components[comp_id];

    if (index >= set->sparse_capacity || set->sparse[index] == INVALID_DENSE_INDEX) {
        SHADOW_UNLOCK(&self->ecs_lock);
        Py_RETURN_NONE;
    }

    uint32_t dense_idx = set->sparse[index];
    PyObject *result   = PyBytes_FromStringAndSize(
        (char *)&set->data[(size_t)dense_idx * set->element_size], set->element_size);
    SHADOW_UNLOCK(&self->ecs_lock);

    return result;
}

PyCFunction_DeclareMethodFromModule Registry_get_view(RegistryObject *self, PyObject *const *args,
                                                      size_t nargsf, PyObject *kwnames) {
    uint32_t comp_id;
    void *targets[RegCompOnly_COUNT] = {[IDX_RCO_COMP] = &comp_id};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->RegCompOnlyParser, targets)) {
        return nullptr;
    }

    if (comp_id >= self->component_count) {
        return PyErr_Format(PyExc_ValueError, "Invalid component ID");
    }

    SHADOW_LOCK(&self->ecs_lock);
    PyObject *proxy = make_ecs_proxy(self, comp_id, false);
    SHADOW_UNLOCK(&self->ecs_lock);
    return proxy;
}

PyCFunction_DeclareMethodFromModule Registry_get_entities(RegistryObject *self,
                                                          PyObject *const *args, size_t nargsf,
                                                          PyObject *kwnames) {
    uint32_t comp_id;
    void *targets[RegCompOnly_COUNT] = {[IDX_RCO_COMP] = &comp_id};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->RegCompOnlyParser, targets)) {
        return nullptr;
    }

    if (comp_id >= self->component_count) {
        return PyErr_Format(PyExc_ValueError, "Invalid component ID");
    }

    SHADOW_LOCK(&self->ecs_lock);
    // make_ecs_proxy(self, comp_id, true) handles the PROXY_ECS_ENTITIES case
    PyObject *proxy = make_ecs_proxy(self, comp_id, true);
    SHADOW_UNLOCK(&self->ecs_lock);

    return proxy;
}

PyCFunction_DeclareMethodFromModule Registry_sync_from_world(RegistryObject *self,
                                                             PyObject *const *args, size_t nargsf,
                                                             PyObject *kwnames) {
    PyObject *world_obj = nullptr;
    uint32_t h_comp_id;
    int p_comp_id = -1;
    int r_comp_id = -1; // -1 means ignore

    void *targets[RegSyncPhys_COUNT] = {[IDX_RSP_WORLD]  = (void *)&world_obj,
                                        [IDX_RSP_H_COMP] = (void *)&h_comp_id,
                                        [IDX_RSP_T_COMP] = (void *)&p_comp_id,
                                        [IDX_RSP_R_COMP] = (void *)&r_comp_id};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->RegSyncPhysParser, targets)) {
        return nullptr;
    }

    PhysicsWorldObject *world = (PhysicsWorldObject *)world_obj;

    if (h_comp_id >= self->component_count) {
        return PyErr_Format(PyExc_ValueError, "Invalid handle component ID");
    }

    SparseSet *h_set = &self->components[h_comp_id];
    SparseSet *p_set = (p_comp_id >= 0 && (uint32_t)p_comp_id < self->component_count)
                           ? &self->components[p_comp_id]
                           : nullptr;
    SparseSet *r_set = (r_comp_id >= 0 && (uint32_t)r_comp_id < self->component_count)
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
    SHADOW_LOCK(&world->shadow_lock);
    SHADOW_LOCK(&self->ecs_lock);

    for (uint32_t i = 0; i < h_set->count; i++) {
        CulvEntity ent   = h_set->dense[i];
        uint32_t ent_idx = (uint32_t)(ent & HANDLE_INDEX_MASK);

        uint64_t handle;
        memcpy(&handle, &h_set->data[(size_t)i * sizeof(uint64_t)], sizeof(uint64_t));

        uint32_t slot;
        if (unpack_handle(world, handle, &slot)) {
            uint32_t phys_dense = world->slot_to_dense[slot];

            // 1. Sync Position (Using your PositionVector / PosStride logic)
            if (p_set && ent_idx < p_set->sparse_capacity &&
                p_set->sparse[ent_idx] != INVALID_DENSE_INDEX) {
                uint32_t p_dense = p_set->sparse[ent_idx];

                // Assuming world->positions stores 1 PositionVector (or 4 Reals) per entity
                // depending on your stride. Using PosStride as defined in your types header.
                PosStride *p = &((PosStride *)world->positions)[phys_dense];

                float *out = (float *)&p_set->data[(size_t)p_dense * sizeof(float) * 3];
                out[0]     = (float)p->x;
                out[1]     = (float)p->y;
                out[2]     = (float)p->z;
            }

            // 2. Sync Rotation
            // world->rotations is a flat float array (4 floats per quaternion)
            if (r_set && ent_idx < r_set->sparse_capacity &&
                r_set->sparse[ent_idx] != INVALID_DENSE_INDEX) {
                uint32_t r_dense = r_set->sparse[ent_idx];

                // Index directly into the float array
                float *phys_rot = &world->rotations[(size_t)phys_dense * 4];
                float *out      = (float *)&r_set->data[(size_t)r_dense * sizeof(float) * 4];

                // Fast direct float copy (16 bytes)
                memcpy(out, phys_rot, sizeof(float) * 4);
            }
        }
    }

    SHADOW_UNLOCK(&self->ecs_lock);
    SHADOW_UNLOCK(&world->shadow_lock);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethodFromModule Registry_clear(RegistryObject *self,
                                                   CULV_MAYBE_UNUSED PyObject *args) {
    SHADOW_LOCK(&self->ecs_lock);

    if (atomic_load_explicit(&self->view_export_count, memory_order_acquire) > 0) {
        SHADOW_UNLOCK(&self->ecs_lock);
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

    SHADOW_UNLOCK(&self->ecs_lock);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethodFromModule Registry_get_active_count(RegistryObject *self,
                                                              CULV_MAYBE_UNUSED PyObject *args) {
    return PyLong_FromUnsignedLong(self->active_entities);
}

PyCFunction_DeclareMethodFromModule Registry_get_component_count(RegistryObject *self,
                                                                 PyObject *const *args,
                                                                 size_t nargsf, PyObject *kwnames) {
    uint32_t comp_id;
    void *targets[RegCompOnly_COUNT] = {[IDX_RCO_COMP] = &comp_id};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->RegCompOnlyParser, targets)) {
        return nullptr;
    }

    if (comp_id >= self->component_count) {
        return PyLong_FromLong(0);
    }

    return PyLong_FromSsize_t(self->components[comp_id].count);
}

#define REG_FASTCALL(name) CULV_FEAT(Registry, name, METH_FASTCALL | METH_KEYWORDS)
#define REG_NOARGS(name) CULV_FEAT(Registry, name, METH_NOARGS)

PyType_Spec Registry_spec = {
    .name      = "culverin._culverin_c.Registry",
    .basicsize = sizeof(RegistryObject),
    .flags     = Py_TPFLAGS_DEFAULT,
    .slots =
        (PyType_Slot[]){

            {.slot = Py_tp_new, .pfunc = PyType_GenericNew},
            {.slot = Py_tp_init, .pfunc = Registry_init},
            {.slot = Py_tp_dealloc, .pfunc = Registry_dealloc},
            {.slot = Py_tp_methods,
             .pfunc =
                 (PyMethodDef[]){

                     REG_NOARGS(create),
                     REG_FASTCALL(destroy),
                     REG_FASTCALL(is_alive),
                     REG_NOARGS(clear), // Wipes the registry
                     REG_FASTCALL(register_component),
                     REG_FASTCALL(add),
                     REG_FASTCALL(remove),
                     REG_FASTCALL(has),
                     REG_FASTCALL(get), // Single entity data access
                     REG_FASTCALL(get_view),
                     REG_FASTCALL(get_entities),
                     REG_FASTCALL(sync_from_world),
                     REG_NOARGS(get_active_count),      // ECS Statistics
                     REG_FASTCALL(get_component_count), // ECS Statistics
                     {}

                 }},
            {},

        },
};