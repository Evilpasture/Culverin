#pragma once

#ifndef JPH_DOUBLE_PRECISION
#    define JPH_DOUBLE_PRECISION 1
#endif

#define PY_SSIZE_T_CLEAN
#include "joltc.h" // Amer Koleci's JoltC binder.
#include <Python.h>

// =========================================================================
// ASAN COMPATIBILITY ALLOCATORS
// =========================================================================
#ifdef ENABLE_SANITIZER
// Bypass mimalloc entirely so ASan can catch buffer overflows
#    define CULV_RAW_MALLOC(sz) malloc(sz)
#    define CULV_RAW_CALLOC(n, sz) calloc(n, sz)
#    define CULV_RAW_REALLOC(ptr, sz) realloc(ptr, sz)
#    define CULV_RAW_FREE(ptr) free(ptr)
#else
// Use Python's ultra-fast mimalloc for Release builds
#    define CULV_RAW_MALLOC(sz) PyMem_RawMalloc(sz)
#    define CULV_RAW_CALLOC(n, sz) PyMem_RawCalloc(n, sz)
#    define CULV_RAW_REALLOC(ptr, sz) PyMem_RawRealloc(ptr, sz)
#    define CULV_RAW_FREE(ptr) PyMem_RawFree(ptr)
#endif
// =========================================================================

#include "culverin_command_buffer.h"
#include "culverin_compiler_specifics.h"
#include "culverin_debug_render.h"
#include "culverin_default_config.h"
#include "culverin_internal_query.h"
#include "culverin_threading.h"
#include "culverin_tracked_vehicle.h"
#include "culverin_types.h"
#include <float.h>
#include <math.h>
#include <stdatomic.h>
#include <stddef.h>
#include <string.h>

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    include <windows.h>
#elif defined(__linux__) || defined(__apple__)
#    include <sched.h>
#    include <unistd.h>
#endif

#ifndef JPH_INVALID_BODY_ID
#    define JPH_INVALID_BODY_ID 0xFFFFFFFF
#endif

// Jolt BodyID layout: [8 bits sequence | 24 bits index]
#ifndef JPH_BODY_ID_INDEX_MASK
#    define JPH_BODY_ID_INDEX_MASK 0x00FFFFFF
#endif

// Mask for the raw array index (Stripping the 24th bit used for Static flags)
#define JPH_ID_TO_INDEX(id) ((id) & 0x7FFFFF)

// Allocate 'Type' on the stack with guaranteed 32-byte alignment.
// USAGE: JPH_STACK_ALLOC(JPH_RVec3, my_vec);
#if defined(__clang__) || defined(__GNUC__)
// Clang/LLVM alignment logic (highly robust)
#    define JPH_ALIGNED_STORAGE(Type, Name, Align) Type Name __attribute__((aligned(Align)))
#elif defined(_MSC_VER)
#    define JPH_ALIGNED_STORAGE(Type, Name, Align) __declspec(align(Align)) Type Name
#else
#    include <stdalign.h>
#    define JPH_ALIGNED_STORAGE(Type, Name, Align) alignas(Align) Type Name
#endif

#define JPH_STACK_ALLOC(Type, Name)                                                                \
    JPH_ALIGNED_STORAGE(Type, Name##_storage, 32);                                                 \
    Type *Name = &Name##_storage

#ifdef CULVERIN_DEBUG
#    define DEBUG_LOG(fmt, ...) fprintf(stderr, "[Culverin] " fmt "\n", ##__VA_ARGS__)
#else
#    define DEBUG_LOG(fmt, ...)
#endif

// --- Callback Logic ---
// Old ContactEvent for compatibility
typedef struct ContactEvent {
    BodyHandle body1;
    BodyHandle body2;
    float px, py, pz;
    float nx, ny, nz;
    float impulse;
    float sliding_speed_sq; // Scratching speed squared(tangential)
    uint32_t mat1;
    uint32_t mat2;
    uint32_t type;
    uint32_t _pad;
} ContactEvent;

_Static_assert(sizeof(ContactEvent) == MEMORY_ALIGNMENT_SIZE,
               "ContactEvent must be 64 bytes for performance");

CULV_MAYBE_UNUSED static constexpr int CONTACT_MAX_CAPACITY = 64 * 8 << 5;

// --- Raycast Batch Result (Aligned to 16-bytes, Total 48-bytes) ---
#ifdef _MSC_VER
#    pragma pack(push, 1)
#endif
typedef struct
#ifndef _MSC_VER
    __attribute__((packed))
#endif
{
    uint64_t handle;      // 8 bytes
    float fraction;       // 4 bytes
    float nx, ny, nz;     // 12 bytes
    float px, py, pz;     // 12 bytes
    uint32_t subShapeID;  // 4 bytes
    uint32_t material_id; // 4 bytes
    uint32_t _pad;
} RayCastBatchResult;
#ifdef _MSC_VER
#    pragma pack(pop)
#endif

_Static_assert(sizeof(RayCastBatchResult) == 48, "RayCastBatchResult size mismatch");

// --- Material Registry ---
typedef struct {
    uint32_t id;
    float friction;
    float restitution;
    // Padding/Alignment isn't critical here as this is a lookup array, not a
    // stream
} MaterialData;

typedef struct {
    int max_bodies;
    int max_pairs;
} WorldLimits;

typedef struct {
    float gx;
    float gy;
    float gz;
} GravityVector;

typedef struct {
    JPH_Real px;
    JPH_Real py;
    JPH_Real pz;
} PositionVector;

typedef struct {
    float mass;
    float friction;
    float restitution;
    int is_sensor;
    int use_ccd;
} BodyCreationProps;

typedef struct {
    float friction;
    float restitution;
} MaterialSettings;

typedef struct {
    float mass;
    float friction;
    float restitution;
    int is_sensor;
    int use_ccd;
    int motion_type;
} BodyConfig;

// Struct to hold parsed Python data safely in C
typedef struct {
    JPH_Vec3 local_p;
    JPH_Quat local_q;
    float params[4];
    int type;
} CompoundPart;

typedef struct {
    uint32_t tri_count;
    uint32_t vertex_count;
} MeshBounds;

// Temporary container for resize
typedef struct {
    JPH_Real *pos, *ppos;
    float *rot, *prot, *lvel, *avel;
    JPH_BodyID *bids;
    uint64_t *udat;
    uint32_t *gens, *s2d, *d2s, *free, *cats, *masks, *mats;
    uint8_t *stat;
} NewBuffers;

// --- The Object Struct ---
typedef struct PhysicsWorldObject {
    PyObject_HEAD // 16 bytes
        PyObject *weakreflist;

    // --- BUCKET 1: Pointers & 8-byte types (Zero Padding) ---
    JPH_PhysicsSystem *system;
    JPH_CharacterVsCharacterCollision *char_vs_char_manager;
    JPH_BodyInterface *body_interface;
    JPH_JobSystem *job_system;
    JPH_BroadPhaseLayerInterface *bp_interface;
    JPH_ObjectLayerPairFilter *pair_filter;
    JPH_ObjectVsBroadPhaseLayerFilter *bp_filter;
    JPH_ContactListener *contact_listener;

    // --- HOT SYNC BLOCK: Kept together for L1d Locality ---
    JPH_Real *positions;
    JPH_Real *prev_positions;
    float *rotations;
    float *prev_rotations;
    float *linear_velocities;
    float *angular_velocities;
    JPH_BodyID *body_ids;
    uint64_t *user_data;
    uint32_t *material_ids;

    // --- Data Buffers ---
    ContactEvent *contact_events;
    ContactEvent *contact_buffer;
    MaterialData *materials;
    PhysicsCommand *command_queue;
    PhysicsCommand *command_queue_spare;
    ShapeEntry *shape_cache;
    BodyHandle *id_to_handle_map;
    JPH_Constraint **constraints;
    uint32_t *categories;
    uint32_t *masks;
    uint32_t *generations;
    uint32_t *slot_to_dense;
    uint32_t *dense_to_slot;
    uint32_t *free_slots;
    uint32_t *constraint_generations;
    uint32_t *free_constraint_slots;

    // --- Counters (8-byte) ---
    size_t contact_count;
    size_t contact_capacity;
    size_t contact_max_capacity;
    atomic_size_t contact_atomic_idx;
    size_t material_count;
    size_t material_capacity;
    size_t free_count;
    size_t slot_capacity;
    size_t command_count;
    size_t command_capacity;
    size_t spare_capacity;
    size_t shape_cache_count;
    size_t shape_cache_capacity;
    size_t count;
    size_t capacity;
    size_t constraint_count;
    size_t constraint_capacity;
    size_t free_constraint_count;
    double time;

    // --- DEFERRED GARBAGE COLLECTION ---
    NewBuffers *trash_buffers;
    size_t trash_count;
    size_t trash_capacity;

    // --- BUCKET 2: 4-byte types (Packed 2-per-slot) ---
    // These three now share 12 bytes total + 4 bytes padding at the end
    // instead of creating holes between every pointer.
    uint32_t max_jolt_bodies;
    atomic_int active_queries;
    int view_export_count;

    // --- BUCKET 3: Structs & Complex Types ---
    ShadowSync step_sync;    // 16 bytes (Internal 8-byte alignment)
    ShadowMutex shadow_lock; // PyMutex (usually 1-4 bytes)

    // --- BUCKET 4: Small types (Packed at the tail) ---
    uint8_t *slot_states;
    uint8_t *constraint_states;
    atomic_bool step_requested;
    atomic_bool is_stepping;
    bool needs_optimization;

    // --- Large Tail Arrays ---
    Py_ssize_t view_shape[2];
    Py_ssize_t view_strides[2];

    // --- Debug Renderer ---
    JPH_DebugRenderer *debug_renderer;
    DebugBuffer debug_lines;
    DebugBuffer debug_triangles;
} PhysicsWorldObject;

typedef enum {
    CULV_SHAPE_BOX         = 0,
    CULV_SHAPE_SPHERE      = 1,
    CULV_SHAPE_CAPSULE     = 2,
    CULV_SHAPE_CYLINDER    = 3,
    CULV_SHAPE_PLANE       = 4,
    CULV_SHAPE_MESH        = 5,
    CULV_SHAPE_HEIGHTFIELD = 6,
    CULV_SHAPE_CONVEX_HULL = 7
} CulvShapeType;

// --- Module State (PEP 489) ---
typedef struct {
    PyObject *helper;           // Reference to culverin._culverin module
    PyObject *PhysicsWorldType; // Reference to the class
    PyObject *CharacterType;    // Reference to the character class
    PyObject *VehicleType;      // Reference to the vehicle class
    PyObject *SkeletonType;
    PyObject *RagdollSettingsType;
    PyObject *RagdollType;
} CulverinState;

// Helper to retrieve state from the module object
CULV_NODISCARD
CULV_MAYBE_UNUSED
static inline CulverinState *get_culverin_state(PyObject *module) {
    return (CulverinState *)PyModule_GetState(module);
}

// --- Handle Helper ---
CULV_NODISCARD
CULV_MAYBE_UNUSED
static inline BodyHandle make_handle(uint32_t slot, uint32_t gen) {
    return ((uint64_t)gen << 32) | (uint64_t)slot;
}
CULV_NODISCARD
CULV_MAYBE_UNUSED
static inline bool unpack_handle(PhysicsWorldObject *self, BodyHandle h, uint32_t *slot) {
    *slot        = (uint32_t)(h & 0xFFFFFFFF);
    uint32_t gen = (uint32_t)(h >> 32);

    if (*slot >= self->slot_capacity || self->slot_capacity == 0) {
        return false;
    }
    return self->generations[*slot] == gen;
}

// 32-bit Float Exponent Mask
// Sign: 0 | Exponent: 11111111 | Mantissa: 000...
CULV_MAYBE_UNUSED
static constexpr uint32_t IEEE754_FLOAT_NONFINITE_MASK = 0x7F800000U;

// 64-bit Double Exponent Mask
// Sign: 0 | Exponent: 11111111111 | Mantissa: 000...
CULV_MAYBE_UNUSED
static constexpr uint64_t IEEE754_DOUBLE_NONFINITE_MASK = 0x7FF0000000000000ULL;

// --- Bit-Level Numerical Guards (Optimizer-Proof) ---
CULV_MAYBE_UNUSED CULV_NODISCARD static inline bool culv_is_finite_f(float f) {
    uint32_t i;
    // Use volatile to prevent the compiler from "seeing through" the cast
    memcpy(&i, &f, sizeof(uint32_t));
    volatile uint32_t vi = i;
    return (vi & IEEE754_FLOAT_NONFINITE_MASK) != IEEE754_FLOAT_NONFINITE_MASK;
}

CULV_MAYBE_UNUSED CULV_NODISCARD static inline bool culv_is_finite_d(double d) {
    uint64_t i;
    memcpy(&i, &d, sizeof(uint64_t));
    volatile uint64_t vi = i;
    return (vi & IEEE754_DOUBLE_NONFINITE_MASK) != IEEE754_DOUBLE_NONFINITE_MASK;
}

// C-Type Generic Dispatcher
#define CULV_IS_FINITE(val)                                                                        \
    _Generic((val), float: culv_is_finite_f(val), double: culv_is_finite_d(val))

#define VALIDATE_FINITE_FLOAT(val, name)                                                           \
    if (UNLIKELY(!CULV_IS_FINITE(val))) {                                                          \
        PyErr_Format(PyExc_ValueError, "Numerical Error: '%s' must be finite", name);              \
        return NULL;                                                                               \
    }

#define VALIDATE_FINITE_VEC3(x, y, z, msg)                                                         \
    if (UNLIKELY(!CULV_IS_FINITE(x) || !CULV_IS_FINITE(y) || !CULV_IS_FINITE(z))) {                \
        char buf[256];                                                                             \
        PyOS_snprintf(buf, sizeof(buf), "Numerical Error: %s must be finite (got [%f, %f, %f])",   \
                      msg, (double)(x), (double)(y), (double)(z));                                 \
        PyErr_SetString(PyExc_ValueError, buf);                                                    \
        return NULL;                                                                               \
    }

#define VALIDATE_FINITE_QUAT(x, y, z, w, msg)                                                      \
    if (UNLIKELY(!CULV_IS_FINITE(x) || !CULV_IS_FINITE(y) || !CULV_IS_FINITE(z) ||                 \
                 !CULV_IS_FINITE(w))) {                                                            \
        char buf[256];                                                                             \
        PyOS_snprintf(buf, sizeof(buf),                                                            \
                      "Numerical Error: %s must be finite (got [%f, %f, %f, %f])", msg,            \
                      (double)(x), (double)(y), (double)(z), (double)(w));                         \
        PyErr_SetString(PyExc_ValueError, buf);                                                    \
        return NULL;                                                                               \
    }

#define VALIDATE_FINITE_VEC4(x, y, z, w, msg)                                                      \
    if (UNLIKELY(!CULV_IS_FINITE(x) || !CULV_IS_FINITE(y) || !CULV_IS_FINITE(z) ||                 \
                 !CULV_IS_FINITE(w))) {                                                            \
        char buf[256];                                                                             \
        PyOS_snprintf(buf, sizeof(buf),                                                            \
                      "Numerical Error: %s components must be finite (got [%f, %f, %f, %f])", msg, \
                      (double)(x), (double)(y), (double)(z), (double)(w));                         \
        PyErr_SetString(PyExc_ValueError, buf);                                                    \
        return NULL;                                                                               \
    }
