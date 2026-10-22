#pragma once

#include <Python.h>

#include <memory>

#include "bindings/ModuleParsers.hpp"

namespace CPH::Python {

// C++ state lives on the heap: CPython allocates m_size storage as raw bytes,
// so non-trivial members must not be placed directly in that storage.
struct ModuleState {
    PyObject *helper{}; // Reference to culverin._culverin module
    PyObject *physicsWorldType{};
    PyObject *characterType{};
    PyObject *vehicleType{};
    PyObject *shipType{};
    PyObject *skeletonType{};
    PyObject *ragdollSettingsType{};
    PyObject *softBodySharedSettingsType{};
    PyObject *ragdollType{};
    PyObject *bufferProxyType{};
    PyObject *registryType{};
    PyObject *mathServiceType{};

    // RAII-owned parser group for top-level module functions.
    std::unique_ptr<CPH::Python::ModuleParsers> parsers{};
};

// Trivial pointer slot stored in CPython's zero-initialized per-module state.
struct ModuleStateStorage {
    ModuleState *state;
};

[[maybe_unused, nodiscard]] static inline ModuleStateStorage *
GetModuleStateStorage(PyObject *module) noexcept {
    return static_cast<ModuleStateStorage *>(PyModule_GetState(module));
}

[[maybe_unused, nodiscard]] static inline ModuleState *
GetModuleState(PyObject *module) noexcept {
    const auto *storage = GetModuleStateStorage(module);
    return storage == nullptr ? nullptr : storage->state;
}

} // namespace CPH::Python
