#pragma once

// Thin adapter: the Registry the user sees is CPH::Registry (src/ecs/), this
// file only carries it plus the CPython object header and the argument
// parsers. No ECS algorithm lives here.

#include "core/Compiler.hpp"
#include "ecs/EcsParsers.hpp"
#include "ecs/Registry.hpp"
#include <Python.h>
#include <memory>

namespace CPH::Python {

struct RegistryObject {
    PyObject_HEAD
    std::unique_ptr<CPH::Registry> registry;
    CPH::Python::EcsParsers       *parsers;
};

// Lifecycle
[[nodiscard]] int  RegistryInit(RegistryObject *self, PyObject *args, PyObject *kwds);
void               RegistryDealloc(RegistryObject *self);

// Entity Methods
[[nodiscard]] PyObject *RegistryCreate(RegistryObject *self, [[maybe_unused]] PyObject *args);
[[nodiscard]] PyObject *RegistryDestroy(RegistryObject *self, PyObject *const *args, size_t nargsf,
                                        PyObject *kwnames);
[[nodiscard]] PyObject *RegistryIsAlive(RegistryObject *self, PyObject *const *args, size_t nargsf,
                                        PyObject *kwnames);
[[nodiscard]] PyObject *RegistryClear(RegistryObject *self, [[maybe_unused]] PyObject *args);

// Component Methods
[[nodiscard]] PyObject *RegistryRegisterComponent(RegistryObject *self, PyObject *const *args,
                                                  size_t nargsf, PyObject *kwnames);
[[nodiscard]] PyObject *RegistryAdd(RegistryObject *self, PyObject *const *args, size_t nargsf,
                                    PyObject *kwnames);
[[nodiscard]] PyObject *RegistryRemove(RegistryObject *self, PyObject *const *args, size_t nargsf,
                                       PyObject *kwnames);
[[nodiscard]] PyObject *RegistryHas(RegistryObject *self, PyObject *const *args, size_t nargsf,
                                    PyObject *kwnames);

// Data Access
[[nodiscard]] PyObject *RegistryGet(RegistryObject *self, PyObject *const *args, size_t nargsf,
                                    PyObject *kwnames);
[[nodiscard]] PyObject *RegistryGetView(RegistryObject *self, PyObject *const *args,
                                        size_t nargsf, PyObject *kwnames);
[[nodiscard]] PyObject *RegistryGetEntities(RegistryObject *self, PyObject *const *args,
                                            size_t nargsf, PyObject *kwnames);

// Synchronization
[[nodiscard]] PyObject *RegistrySyncFromWorld(RegistryObject *self, PyObject *const *args,
                                              size_t nargsf, PyObject *kwnames);

// Stats
[[nodiscard]] PyObject *RegistryGetActiveCount(RegistryObject *self,
                                               [[maybe_unused]] PyObject *args);
[[nodiscard]] PyObject *RegistryGetComponentCount(RegistryObject *self, PyObject *const *args,
                                                  size_t nargsf, PyObject *kwnames);

} // namespace CPH::Python
