#ifndef PYJOLT_H
#define PYJOLT_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "joltc.h"

// --- Threading Primitives (Python 3.14t support) ---
#if PY_VERSION_HEX >= 0x030D0000
    typedef PyMutex ShadowMutex;
    #define SHADOW_LOCK(m) PyMutex_Lock(m)
    #define SHADOW_UNLOCK(m) PyMutex_Unlock(m)
#else
    typedef PyThread_type_lock ShadowMutex;
    #define SHADOW_LOCK(m) PyThread_acquire_lock(m, 1)
    #define SHADOW_UNLOCK(m) PyThread_release_lock(m)
#endif

// --- The Object Struct ---
typedef struct {
    PyObject_HEAD
    
    // Jolt Handles
    JPH_PhysicsSystem* system;
    JPH_BodyInterface* body_interface;
    JPH_JobSystem* job_system;
    
    // Filters
    JPH_BroadPhaseLayerInterface* bp_interface;
    JPH_ObjectLayerPairFilter* pair_filter;
    JPH_ObjectVsBroadPhaseLayerFilter* bp_filter;

    // Shadow Buffers
    float* positions;
    float* rotations;
    float* linear_velocities;
    float* angular_velocities;
    JPH_BodyID* body_ids;
    
    size_t count;
    size_t capacity;
    double time;

    ShadowMutex shadow_lock;

    // View Metadata
    Py_ssize_t view_shape[2];
    Py_ssize_t view_strides[2];
} PhysicsWorldObject;

// --- Module State (PEP 489) ---
typedef struct {
    PyObject *helper;           // Reference to culverin._culverin module
    PyObject *PhysicsWorldType; // Reference to the class
} CulverinState;

// Helper to retrieve state from the module object
static inline CulverinState* get_culverin_state(PyObject *module) {
    return (CulverinState*)PyModule_GetState(module);
}

// Sync function (defined in shadow_sync.c)
void culverin_sync_shadow_buffers(PhysicsWorldObject* self);

#endif