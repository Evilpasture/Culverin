#ifndef PYJOLT_H
#define PYJOLT_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "joltc.h"

#ifndef JPH_INVALID_BODY_ID
#define JPH_INVALID_BODY_ID 0xFFFFFFFF
#endif

// Allocate 'Type' on the stack with guaranteed 32-byte alignment.
// USAGE: JPH_STACK_ALLOC(JPH_RVec3, my_vec);
// This macro satisfies how Jolt Physics expects aligned data structures.
#define JPH_STACK_ALLOC(Type, Name) \
    char _mem_##Name[sizeof(Type) + 32]; \
    Type* Name = (Type*)((uintptr_t)(_mem_##Name + 31) & ~31)

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

// --- Shape Caching ---
typedef struct {
    uint32_t type;  // 0=Box, 1=Sphere, 2=Capsule
    float p1, p2, p3; // Box: x,y,z | Sphere: r,0,0 | Capsule: h,r,0
} ShapeKey;

typedef struct {
    ShapeKey key;
    JPH_Shape* shape;
} ShapeEntry;

// Minimal Handle Helper
// Python handles will be 64-bit integers: (Generation << 32) | SlotIndex
typedef uint64_t BodyHandle;

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

    // --- Indirection System ---
    uint32_t* generations;     // [Slot] -> Generation
    uint32_t* slot_to_dense;   // [Slot] -> Dense Index
    uint32_t* dense_to_slot;   // [Dense Index] -> Slot
    
    uint32_t* free_slots;      // Stack of available slots
    size_t free_count;
    size_t slot_capacity;      // Size of the mapping arrays

    ShapeEntry* shape_cache;
    size_t shape_cache_count;
    size_t shape_cache_capacity;
    
    size_t count;
    size_t capacity;
    double time;

    ShadowMutex shadow_lock;

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