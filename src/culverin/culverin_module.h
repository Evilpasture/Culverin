#pragma once

// --- Module State (PEP 489) ---
#include <Python.h>
typedef struct {
    PyObject *helper;           // Reference to culverin._culverin module
    PyObject *PhysicsWorldType; // Reference to the class
    PyObject *CharacterType;    // Reference to the character class
    PyObject *VehicleType;      // Reference to the vehicle class
    PyObject *ShipType;
    PyObject *SkeletonType;
    PyObject *RagdollSettingsType;
    PyObject *SoftBodySharedSettingsType;
    PyObject *RagdollType;
    PyObject *BufferProxyType;
    PyObject *RegistryType;
    PyObject *MathServiceType;
} CulverinState;

// Helper to retrieve state from the module object
[[maybe_unused, nodiscard]]
static inline CulverinState *get_culverin_state(PyObject *module) {
    return (CulverinState *)PyModule_GetState(module);
}