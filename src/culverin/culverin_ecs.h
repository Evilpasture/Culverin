#pragma once
#include <Python.h>
#include "culverin_compiler_specifics.h"


// A 64-bit Entity Handle: [32-bit Generation | 32-bit Index]
typedef uint64_t CulvEntity;

// The Sparse Set: Maps sparse Entity IDs to dense, contiguous data arrays.
typedef struct {
    uint32_t *sparse;      // Index matches Entity ID. Value is Dense Index.
    CulvEntity *dense;     // Index matches Dense Index. Value is Entity Handle.
    uint8_t *data;         // Contiguous byte array of component data
    uint32_t element_size; // Size of one component in bytes
    Py_ssize_t count;      // Active components
    uint32_t dense_capacity;
    uint32_t sparse_capacity;
} SparseSet;

// The ECS Registry
typedef struct {
    PyObject_HEAD

        // Entity Management
        uint32_t *generations;
    uint32_t *free_indices;
    uint32_t free_count;
    uint32_t entity_capacity;
    uint32_t active_entities;

    // Component Management
    SparseSet *components;
    uint32_t component_count;
    uint32_t component_capacity;

} RegistryObject;

// Lifecycle
PyType_DeclareSlot_StatusFromModule Registry_init(RegistryObject *self, PyObject *args,
                                                  PyObject *kwds);
PyType_DeclareSlot_VoidFromModule Registry_dealloc(RegistryObject *self);

// Entity Methods
PyCFunction_DeclareMethodFromModule Registry_create(RegistryObject *self,
                                                    CULV_MAYBE_UNUSED PyObject *args);
PyCFunction_DeclareMethodFromModule Registry_destroy(RegistryObject *self, PyObject *const *args,
                                                     size_t nargsf, PyObject *kwnames);
PyCFunction_DeclareMethodFromModule Registry_is_alive(RegistryObject *self, PyObject *const *args,
                                                      size_t nargsf, PyObject *kwnames);

// Component Methods
PyCFunction_DeclareMethodFromModule Registry_register_component(RegistryObject *self,
                                                                PyObject *const *args,
                                                                size_t nargsf, PyObject *kwnames);
PyCFunction_DeclareMethodFromModule Registry_add(RegistryObject *self, PyObject *const *args,
                                                 size_t nargsf, PyObject *kwnames);
PyCFunction_DeclareMethodFromModule Registry_remove(RegistryObject *self, PyObject *const *args,
                                                    size_t nargsf, PyObject *kwnames);
PyCFunction_DeclareMethodFromModule Registry_has(RegistryObject *self, PyObject *const *args,
                                                 size_t nargsf, PyObject *kwnames);

// Data Access
PyCFunction_DeclareMethodFromModule Registry_get_view(RegistryObject *self, PyObject *const *args,
                                                      size_t nargsf, PyObject *kwnames);
PyCFunction_DeclareMethodFromModule Registry_get_entities(RegistryObject *self,
                                                          PyObject *const *args, size_t nargsf,
                                                          PyObject *kwnames);