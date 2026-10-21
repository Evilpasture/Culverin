#pragma once
#include "core/Compiler.hpp"
#include "core/Threading.hpp"
#include "ecs/EcsParsers.hpp"
#include <Python.h>

namespace CPH::Python {

// A 64-bit entity handle: [32-bit generation | 32-bit index]. Same layout and
// same class template as BodyHandle/ConstraintHandle (core/Handle.hpp), so the
// packing helpers are shared rather than re-derived here.
using Entity = CPH::GenerationalHandle<CPH::EntityTag>;

// The Sparse Set: Maps sparse Entity IDs to dense, contiguous data arrays.
struct SparseSet {
    uint32_t *sparse;      // Index matches Entity ID. Value is Dense Index.
    Entity *dense;     // Index matches Dense Index. Value is Entity Handle.
    uint8_t *data;         // Contiguous byte array of component data
    uint32_t element_size; // Size of one component in bytes
    Py_ssize_t count;      // Active components
    uint32_t dense_capacity;
    uint32_t sparse_capacity;
};

// The ECS Registry
struct RegistryObject {
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
    std::atomic<int> view_export_count; // Track how many memoryviews are active
    std::mutex ecs_lock;

    // Parsers
    CPH::Python::EcsParsers *parsers;
};

// Lifecycle
[[nodiscard]] int RegistryInit(RegistryObject *self, PyObject *args,
                                                        PyObject *kwds);
void RegistryDealloc(RegistryObject *self);

// Entity Methods
[[nodiscard]] PyObject * RegistryCreate(RegistryObject *self,
                                                          [[maybe_unused]] PyObject *args);
[[nodiscard]] PyObject * RegistryDestroy(RegistryObject *self,
                                                           PyObject *const *args, size_t nargsf,
                                                           PyObject *kwnames);
[[nodiscard]] PyObject * RegistryIsAlive(RegistryObject *self,
                                                           PyObject *const *args, size_t nargsf,
                                                           PyObject *kwnames);
[[nodiscard]] PyObject * RegistryClear(RegistryObject *self,
                                                         [[maybe_unused]] PyObject *args);

// Component Methods
[[nodiscard]] PyObject * RegistryRegisterComponent(RegistryObject *self,
                                                                     PyObject *const *args,
                                                                     size_t nargsf,
                                                                     PyObject *kwnames);
[[nodiscard]] PyObject * RegistryAdd(RegistryObject *self,
                                                       PyObject *const *args, size_t nargsf,
                                                       PyObject *kwnames);
[[nodiscard]] PyObject * RegistryRemove(RegistryObject *self,
                                                          PyObject *const *args, size_t nargsf,
                                                          PyObject *kwnames);
[[nodiscard]] PyObject * RegistryHas(RegistryObject *self,
                                                       PyObject *const *args, size_t nargsf,
                                                       PyObject *kwnames);

// Data Access
[[nodiscard]] PyObject * RegistryGet(RegistryObject *self,
                                                       PyObject *const *args, size_t nargsf,
                                                       PyObject *kwnames);
[[nodiscard]] PyObject * RegistryGetView(RegistryObject *self,
                                                           PyObject *const *args, size_t nargsf,
                                                           PyObject *kwnames);
[[nodiscard]] PyObject * RegistryGetEntities(RegistryObject *self,
                                                               PyObject *const *args, size_t nargsf,
                                                               PyObject *kwnames);

// Synchronization
[[nodiscard]] PyObject * RegistrySyncFromWorld(RegistryObject *self,
                                                                 PyObject *const *args,
                                                                 size_t nargsf, PyObject *kwnames);

// Stats
[[nodiscard]] PyObject * RegistryGetActiveCount(RegistryObject *self,
                                                                  [[maybe_unused]] PyObject *args);
[[nodiscard]] PyObject * RegistryGetComponentCount(RegistryObject *self,
                                                                     PyObject *const *args,
                                                                     size_t nargsf,
                                                                     PyObject *kwnames);

} // namespace CPH::Python
