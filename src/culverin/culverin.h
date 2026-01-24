#ifndef PYJOLT_H
#define PYJOLT_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>
#include <math.h>
#include "joltc.h"

#ifndef JPH_INVALID_BODY_ID
#define JPH_INVALID_BODY_ID 0xFFFFFFFF
#endif

// Allocate 'Type' on the stack with guaranteed 32-byte alignment.
// USAGE: JPH_STACK_ALLOC(JPH_RVec3, my_vec);
// This macro satisfies how Jolt Physics expects aligned data structures.
// I know. This is a hack.
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
    uint32_t type;     // 0=Box, 1=Sphere, 2=Capsule, 3=Cylinder, 4=Plane
    float p1, p2, p3, p4; 
} ShapeKey;

typedef struct {
    ShapeKey key;
    JPH_Shape* shape;
} ShapeEntry;

// Minimal Handle Helper
// Python handles will be 64-bit integers: (Generation << 32) | SlotIndex
typedef uint64_t BodyHandle;

// --- Slot State Machine ---
typedef enum {
    SLOT_EMPTY = 0,
    SLOT_PENDING_CREATE = 1,
    SLOT_ALIVE = 2,
    SLOT_PENDING_DESTROY = 3
} SlotState;

// --- Command Buffer ---
typedef enum {
    CMD_CREATE_BODY,
    CMD_DESTROY_BODY,
    CMD_SET_POS,
    CMD_SET_ROT,
    CMD_SET_TRNS, // Position + Rotation
    CMD_SET_LINVEL,
    CMD_SET_ANGVEL,
    CMD_SET_MOTION,
    CMD_ACTIVATE,
    CMD_DEACTIVATE,
    CMD_SET_USER_DATA
} CommandType;

typedef struct {
    CommandType type;
    uint32_t slot; // Every command targets a slot
    union {
        // CMD_CREATE_BODY
        struct {
            JPH_BodyCreationSettings* settings;
            uint64_t user_data;
        } create;

        // CMD_SET_POS, CMD_SET_ROT, CMD_SET_LINVEL, CMD_SET_ANGVEL
        struct {
            float x, y, z, w; // w used for rotation
        } vec;

        // CMD_SET_TRNS (Combined Pos + Rot)
        struct {
            float px, py, pz;
            float rx, ry, rz, rw;
        } transform;

        // CMD_SET_MOTION
        int motion_type;

        // CMD_SET_USER_DATA
        uint64_t user_data_val;

        // Note: CMD_DESTROY_BODY, CMD_ACTIVATE, CMD_DEACTIVATE 
        // only need the 'slot' member defined above.
    } data;
} PhysicsCommand;

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
    uint64_t* user_data;

    // --- Indirection System ---
    uint32_t* generations;     // [Slot] -> Generation
    uint32_t* slot_to_dense;   // [Slot] -> Dense Index
    uint32_t* dense_to_slot;   // [Dense Index] -> Slot
    
    uint32_t* free_slots;      // Stack of available slots
    uint8_t* slot_states;
    size_t free_count;
    size_t slot_capacity;      // Size of the mapping arrays

    PhysicsCommand* command_queue;
    size_t command_count;
    size_t command_capacity;

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

// --- Character Object ---
typedef struct {
    PyObject_HEAD
    JPH_CharacterVirtual* character;
    PhysicsWorldObject* world; // Keep a reference to keep the world alive
    
    // We need filters for the character's movement query
    JPH_BodyFilter* body_filter;
    JPH_ShapeFilter* shape_filter;
    JPH_BroadPhaseLayerFilter* bp_filter;
    JPH_ObjectLayerFilter* obj_filter;

    JPH_CharacterContactListener* listener;
} CharacterObject;

extern PyType_Spec Character_spec;

typedef struct {
    PhysicsWorldObject* world;
    PyObject* result_list; // Python List to append handles to
} QueryContext;

// Helper for Overlap Callbacks
// Context struct to pass Python List into the C callback
typedef struct {
    PhysicsWorldObject* world;
    PyObject* result_list;
} OverlapContext;

// --- Module State (PEP 489) ---
typedef struct {
    PyObject *helper;           // Reference to culverin._culverin module
    PyObject *PhysicsWorldType; // Reference to the class
    PyObject *CharacterType;    // Reference to the character class
} CulverinState;

// Helper to retrieve state from the module object
static inline CulverinState* get_culverin_state(PyObject *module) {
    return (CulverinState*)PyModule_GetState(module);
}

// Sync function (defined in shadow_sync.c)
void culverin_sync_shadow_buffers(PhysicsWorldObject* self);

#endif