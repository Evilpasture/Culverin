#include "culverin_ecs.h"

#include "culverin.h"
#include "culverin_arg_indices.h"
#include "culverin_getters.h"
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
        return false;
    }

    // Initialize new sparse indices to INVALID
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

    // CRITICAL SAFETY CHECK
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
        return false;
    }

    uint8_t *new_data = (uint8_t *)CULV_RAW_REALLOC(set->data, (size_t)new_cap * set->element_size);
    if (!new_data) {
        CULV_RAW_FREE(new_dense);
        return false;
    }

    set->dense          = new_dense;
    set->data           = new_data;
    set->dense_capacity = new_cap;
    return true;
}

// --- LIFECYCLE ---

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
    return 0;
}

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
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    uint64_t handle;
    void *targets[RegEntityOnly_COUNT] = {[IDX_REO_ENT] = &handle};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &st->parsers.RegEntityOnlyParser, targets)) {
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
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    uint64_t handle;
    void *targets[RegEntityOnly_COUNT] = {[IDX_REO_ENT] = &handle};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &st->parsers.RegEntityOnlyParser, targets)) {
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
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    uint32_t size;
    void *targets[RegRegComp_COUNT] = {[IDX_RRC_SIZE] = &size};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &st->parsers.RegRegCompParser,
                           targets)) {
        return nullptr;
    }

    SHADOW_LOCK(&self->ecs_lock);

    if (self->component_count >= self->component_capacity) {
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
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    uint64_t handle;
    uint32_t comp_id;
    PyObject *data_obj = nullptr;

    void *targets[RegAdd_COUNT] = {[IDX_RA_ENT]  = (void *)&handle,
                                   [IDX_RA_COMP] = (void *)&comp_id,
                                   [IDX_RA_DATA] = (void *)&data_obj};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &st->parsers.RegAddParser,
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
        goto fail;
    }

    uint32_t dense_idx = set->sparse[index];
    if (dense_idx == INVALID_DENSE_INDEX) {
        // Adding new component
        if (!SparseSet_EnsureDenseCapacity(self, set)) {
            goto fail;
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

fail:
    SHADOW_UNLOCK(&self->ecs_lock);
    if (data_obj && data_obj != Py_None) {
        PyBuffer_Release(&view);
    }
    return PyErr_NoMemory();
}

PyCFunction_DeclareMethodFromModule Registry_remove(RegistryObject *self, PyObject *const *args,
                                                    size_t nargsf, PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    uint64_t handle;
    uint32_t comp_id;

    void *targets[RegEntComp_COUNT] = {[IDX_REC_ENT] = &handle, [IDX_REC_COMP] = &comp_id};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &st->parsers.RegEntCompParser,
                           targets)) {
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
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    uint64_t handle;
    uint32_t comp_id;

    void *targets[RegEntComp_COUNT] = {[IDX_REC_ENT] = &handle, [IDX_REC_COMP] = &comp_id};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &st->parsers.RegEntCompParser,
                           targets)) {
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

PyCFunction_DeclareMethodFromModule Registry_get_view(RegistryObject *self, PyObject *const *args,
                                                      size_t nargsf, PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    uint32_t comp_id;
    void *targets[RegCompOnly_COUNT] = {[IDX_RCO_COMP] = &comp_id};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &st->parsers.RegCompOnlyParser, targets)) {
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
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    uint32_t comp_id;
    void *targets[RegCompOnly_COUNT] = {[IDX_RCO_COMP] = &comp_id};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &st->parsers.RegCompOnlyParser, targets)) {
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
    CulverinState *st   = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    PyObject *world_obj = nullptr;
    uint32_t h_comp_id;
    uint32_t t_comp_id;

    void *targets[RegSyncPhys_COUNT] = {[IDX_RSP_WORLD]  = (void *)&world_obj,
                                        [IDX_RSP_H_COMP] = (void *)&h_comp_id,
                                        [IDX_RSP_T_COMP] = (void *)&t_comp_id};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &st->parsers.RegSyncPhysParser, targets)) {
        return nullptr;
    }

    PhysicsWorldObject *world = (PhysicsWorldObject *)world_obj;
    SparseSet *h_set          = &self->components[h_comp_id];
    SparseSet *t_set          = &self->components[t_comp_id];

    // Verification
    if (h_comp_id >= self->component_count || t_comp_id >= self->component_count) {
        return PyErr_Format(PyExc_ValueError, "Invalid component ID");
    }
    if (h_set->element_size != sizeof(uint64_t)) {
        return PyErr_Format(PyExc_TypeError, "Handle component must be 8 bytes (uint64)");
    }
    if (t_set->element_size != sizeof(float) * 3) {
        return PyErr_Format(PyExc_TypeError, "Transform component must be 12 bytes (3x float32)");
    }

    SHADOW_LOCK(&world->shadow_lock);
    SHADOW_LOCK(&self->ecs_lock);

    // We iterate over the entities in the Transform set
    for (uint32_t i = 0; i < t_set->count; i++) {
        CulvEntity ent   = t_set->dense[i];
        uint32_t ent_idx = JPH_ID_TO_INDEX((uint32_t)ent);

        // Check if this entity also has a physics handle
        if (ent_idx < h_set->sparse_capacity && h_set->sparse[ent_idx] != INVALID_DENSE_INDEX) {
            uint32_t h_dense = h_set->sparse[ent_idx];
            uint64_t handle;
            memcpy(&handle, &h_set->data[(size_t)h_dense * sizeof(uint64_t)], sizeof(uint64_t));

            uint32_t slot;
            if (unpack_handle(world, handle, &slot)) {
                uint32_t phys_dense = world->slot_to_dense[slot];
                PosStride *p        = &((PosStride *)world->positions)[phys_dense];
                float *out          = (float *)&t_set->data[(size_t)i * sizeof(float) * 3];

                // Direct Copy: Double (Physics) -> Float (ECS)
                out[0] = (float)p->x;
                out[1] = (float)p->y;
                out[2] = (float)p->z;
            }
        }
    }

    SHADOW_UNLOCK(&self->ecs_lock);
    SHADOW_UNLOCK(&world->shadow_lock);
    Py_RETURN_NONE;
}