#pragma once
#include "bindings/parsing/ArgIndices.h"
#include "core/CompilerSpecifics.h"
#include "core/Threading.h"
#include <Python.h>
#include "bindings/PythonDeclarations.h"

// A 64-bit Entity Handle: [32-bit Generation | 32-bit Index]
typedef uint64_t CPH_Entity;

// The Sparse Set: Maps sparse Entity IDs to dense, contiguous data arrays.
typedef struct {
    uint32_t *sparse;      // Index matches Entity ID. Value is Dense Index.
    CPH_Entity *dense;     // Index matches Dense Index. Value is Entity Handle.
    uint8_t *data;         // Contiguous byte array of component data
    uint32_t element_size; // Size of one component in bytes
    Py_ssize_t count;      // Active components
    uint32_t dense_capacity;
    uint32_t sparse_capacity;
} CPH_SparseSet;

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
    CPH_SparseSet *components;
    uint32_t component_count;
    uint32_t component_capacity;
    atomic_int view_export_count; // Track how many memoryviews are active
    ShadowMutex ecs_lock;

    // Parsers
    CPH_ECSParsers *parsers;
} CPH_RegistryObject;

// Lifecycle
PyType_DeclareSlot_StatusFromModule CPH_PyRegistry_Init(CPH_RegistryObject *self, PyObject *args,
                                                  PyObject *kwds);
PyType_DeclareSlot_VoidFromModule CPH_PyRegistry_Dealloc(CPH_RegistryObject *self);

// Entity Methods
PyCFunction_DeclareMethodFromModule CPH_PyRegistry_Create(CPH_RegistryObject *self,
                                                    [[maybe_unused]] PyObject *args);
PyCFunction_DeclareMethodFromModule CPH_PyRegistry_Destroy(CPH_RegistryObject *self, PyObject *const *args,
                                                     size_t nargsf, PyObject *kwnames);
PyCFunction_DeclareMethodFromModule CPH_PyRegistry_IsAlive(CPH_RegistryObject *self, PyObject *const *args,
                                                      size_t nargsf, PyObject *kwnames);
PyCFunction_DeclareMethodFromModule CPH_PyRegistry_Clear(CPH_RegistryObject *self,
                                                   [[maybe_unused]] PyObject *args);

// Component Methods
PyCFunction_DeclareMethodFromModule CPH_PyRegistry_RegisterComponent(CPH_RegistryObject *self,
                                                                PyObject *const *args,
                                                                size_t nargsf, PyObject *kwnames);
PyCFunction_DeclareMethodFromModule CPH_PyRegistry_Add(CPH_RegistryObject *self, PyObject *const *args,
                                                 size_t nargsf, PyObject *kwnames);
PyCFunction_DeclareMethodFromModule CPH_PyRegistry_Remove(CPH_RegistryObject *self, PyObject *const *args,
                                                    size_t nargsf, PyObject *kwnames);
PyCFunction_DeclareMethodFromModule CPH_PyRegistry_Has(CPH_RegistryObject *self, PyObject *const *args,
                                                 size_t nargsf, PyObject *kwnames);

// Data Access
PyCFunction_DeclareMethodFromModule CPH_PyRegistry_Get(CPH_RegistryObject *self, PyObject *const *args,
                                                 size_t nargsf, PyObject *kwnames);
PyCFunction_DeclareMethodFromModule CPH_PyRegistry_GetView(CPH_RegistryObject *self, PyObject *const *args,
                                                      size_t nargsf, PyObject *kwnames);
PyCFunction_DeclareMethodFromModule CPH_PyRegistry_GetEntities(CPH_RegistryObject *self,
                                                          PyObject *const *args, size_t nargsf,
                                                          PyObject *kwnames);

// Synchronization
PyCFunction_DeclareMethodFromModule CPH_PyRegistry_SyncFromWorld(CPH_RegistryObject *self,
                                                             PyObject *const *args, size_t nargsf,
                                                             PyObject *kwnames);

// Stats
PyCFunction_DeclareMethodFromModule CPH_PyRegistry_GetActiveCount(CPH_RegistryObject *self,
                                                              [[maybe_unused]] PyObject *args);
PyCFunction_DeclareMethodFromModule CPH_PyRegistry_GetComponentCount(CPH_RegistryObject *self,
                                                                 PyObject *const *args,
                                                                 size_t nargsf, PyObject *kwnames);