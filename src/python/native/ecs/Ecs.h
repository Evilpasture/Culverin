#pragma once
#include "bindings/parsing/ArgIndices.h"
#include "core/CompilerSpecifics.h"
#include "core/Threading.h"
#include <Python.h>
#include "bindings/PythonDeclarations.h"

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
PyType_DeclareSlot_StatusFromModule CPH_PyRegistryInit(CPH_RegistryObject *self, PyObject *args,
                                                  PyObject *kwds);
PyType_DeclareSlot_VoidFromModule CPH_PyRegistryDealloc(CPH_RegistryObject *self);

// Entity Methods
PyCFunction_DeclareMethodFromModule CPH_PyRegistryCreate(CPH_RegistryObject *self,
                                                    [[maybe_unused]] PyObject *args);
PyCFunction_DeclareMethodFromModule CPH_PyRegistryDestroy(CPH_RegistryObject *self, PyObject *const *args,
                                                     size_t nargsf, PyObject *kwnames);
PyCFunction_DeclareMethodFromModule CPH_PyRegistryIsAlive(CPH_RegistryObject *self, PyObject *const *args,
                                                      size_t nargsf, PyObject *kwnames);
PyCFunction_DeclareMethodFromModule CPH_PyRegistryClear(CPH_RegistryObject *self,
                                                   [[maybe_unused]] PyObject *args);

// Component Methods
PyCFunction_DeclareMethodFromModule CPH_PyRegistryRegisterComponent(CPH_RegistryObject *self,
                                                                PyObject *const *args,
                                                                size_t nargsf, PyObject *kwnames);
PyCFunction_DeclareMethodFromModule CPH_PyRegistryAdd(CPH_RegistryObject *self, PyObject *const *args,
                                                 size_t nargsf, PyObject *kwnames);
PyCFunction_DeclareMethodFromModule CPH_PyRegistryRemove(CPH_RegistryObject *self, PyObject *const *args,
                                                    size_t nargsf, PyObject *kwnames);
PyCFunction_DeclareMethodFromModule CPH_PyRegistryHas(CPH_RegistryObject *self, PyObject *const *args,
                                                 size_t nargsf, PyObject *kwnames);

// Data Access
PyCFunction_DeclareMethodFromModule CPH_PyRegistryGet(CPH_RegistryObject *self, PyObject *const *args,
                                                 size_t nargsf, PyObject *kwnames);
PyCFunction_DeclareMethodFromModule CPH_PyRegistryGetView(CPH_RegistryObject *self, PyObject *const *args,
                                                      size_t nargsf, PyObject *kwnames);
PyCFunction_DeclareMethodFromModule CPH_PyRegistryGetEntities(CPH_RegistryObject *self,
                                                          PyObject *const *args, size_t nargsf,
                                                          PyObject *kwnames);

// Synchronization
PyCFunction_DeclareMethodFromModule CPH_PyRegistrySyncFromWorld(CPH_RegistryObject *self,
                                                             PyObject *const *args, size_t nargsf,
                                                             PyObject *kwnames);

// Stats
PyCFunction_DeclareMethodFromModule CPH_PyRegistryGetActiveCount(CPH_RegistryObject *self,
                                                              [[maybe_unused]] PyObject *args);
PyCFunction_DeclareMethodFromModule CPH_PyRegistryGetComponentCount(CPH_RegistryObject *self,
                                                                 PyObject *const *args,
                                                                 size_t nargsf, PyObject *kwnames);