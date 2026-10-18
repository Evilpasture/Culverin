#pragma once

#include <Python.h>

#include <memory>

#include "bindings/parsing/ParserRegistry.hpp"

// C++ state lives on the heap: CPython allocates m_size storage as raw bytes,
// so non-trivial members must not be placed directly in that storage.
struct CPH_ModuleState {
    PyObject *helper{}; // Reference to culverin._culverin module
    PyObject *PhysicsWorldType{};
    PyObject *CharacterType{};
    PyObject *VehicleType{};
    PyObject *ShipType{};
    PyObject *SkeletonType{};
    PyObject *RagdollSettingsType{};
    PyObject *SoftBodySharedSettingsType{};
    PyObject *RagdollType{};
    PyObject *BufferProxyType{};
    PyObject *RegistryType{};
    PyObject *MathServiceType{};

    // RAII-owned parser group for top-level module functions.
    std::unique_ptr<CPH_ModuleParsers> parsers{};
};

// Trivial pointer slot stored in CPython's zero-initialized per-module state.
struct CPH_ModuleStateStorage {
    CPH_ModuleState *state;
};

[[maybe_unused, nodiscard]] static inline CPH_ModuleStateStorage *
CPH_GetModuleStateStorage(PyObject *module) noexcept {
    return static_cast<CPH_ModuleStateStorage *>(PyModule_GetState(module));
}

[[maybe_unused, nodiscard]] static inline CPH_ModuleState *
CPH_GetModuleState(PyObject *module) noexcept {
    const auto *storage = CPH_GetModuleStateStorage(module);
    return storage == nullptr ? nullptr : storage->state;
}
