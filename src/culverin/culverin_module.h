#pragma once

// --- Module State (PEP 489) ---
#include "culverin_arg_indices.h"
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
    CulverinParsers parsers;
} CulverinState;

// Helper to retrieve state from the module object
CULV_NODISCARD
CULV_MAYBE_UNUSED
static inline CulverinState *get_culverin_state(PyObject *module) {
    return (CulverinState *)PyModule_GetState(module);
}