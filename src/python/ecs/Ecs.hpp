#pragma once

// Thin adapter: the Registry the user sees is CPH::Registry (src/ecs/), this
// file only carries it plus the CPython object header and the argument
// parsers. No ECS algorithm lives here.

#include "core/Compiler.hpp"
#include "ecs/EcsParsers.hpp"
#include "ecs/Registry.hpp"
#include <Python.h>
#include <memory>
#include "bindings/Result.hpp"

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
[[nodiscard]] BoundResult RegistryCreate(RegistryObject *self, [[maybe_unused]] PyObject *args);
[[nodiscard]] BoundResult RegistryDestroy(RegistryObject *self, PyObject *const *args, size_t nargsf,
                                        PyObject *kwnames);
[[nodiscard]] BoundResult RegistryIsAlive(RegistryObject *self, PyObject *const *args, size_t nargsf,
                                        PyObject *kwnames);
[[nodiscard]] BoundResult RegistryClear(RegistryObject *self, [[maybe_unused]] PyObject *args);

// Component Methods
[[nodiscard]] BoundResult RegistryRegisterComponent(RegistryObject *self, PyObject *const *args,
                                                  size_t nargsf, PyObject *kwnames);
[[nodiscard]] BoundResult RegistryAdd(RegistryObject *self, PyObject *const *args, size_t nargsf,
                                    PyObject *kwnames);
[[nodiscard]] BoundResult RegistryRemove(RegistryObject *self, PyObject *const *args, size_t nargsf,
                                       PyObject *kwnames);
[[nodiscard]] BoundResult RegistryHas(RegistryObject *self, PyObject *const *args, size_t nargsf,
                                    PyObject *kwnames);

// Data Access
[[nodiscard]] BoundResult RegistryGet(RegistryObject *self, PyObject *const *args, size_t nargsf,
                                    PyObject *kwnames);
[[nodiscard]] BoundResult RegistryGetView(RegistryObject *self, PyObject *const *args,
                                        size_t nargsf, PyObject *kwnames);
[[nodiscard]] BoundResult RegistryGetEntities(RegistryObject *self, PyObject *const *args,
                                            size_t nargsf, PyObject *kwnames);

// Synchronization
[[nodiscard]] BoundResult RegistrySyncFromWorld(RegistryObject *self, PyObject *const *args,
                                              size_t nargsf, PyObject *kwnames);

// Stats
[[nodiscard]] BoundResult RegistryGetActiveCount(RegistryObject *self,
                                               [[maybe_unused]] PyObject *args);
[[nodiscard]] BoundResult RegistryGetComponentCount(RegistryObject *self, PyObject *const *args,
                                                  size_t nargsf, PyObject *kwnames);

} // namespace CPH::Python
