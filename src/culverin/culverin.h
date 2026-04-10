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
#include "culverin_internal_query.h"
#include "culverin_threading.h"
#include "culverin_tracked_vehicle.h"
#include "culverin_types.h"
#ifdef __cplusplus
#    include <atomic>
#else
#    include <stdatomic.h>
#endif
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
CULV_MAYBE_UNUSED static constexpr size_t JPH_INVALID_BODY_ID = 0xFFFFFFFF;
#endif

// Jolt BodyID layout: [8 bits sequence | 24 bits index]
#ifndef JPH_BODY_ID_INDEX_MASK
CULV_MAYBE_UNUSED static constexpr size_t JPH_BODY_ID_INDEX_MASK = 0x00FFFFFF;
#endif

CULV_NODISCARD [[gnu::const]]
static inline uint32_t JPH_ID_TO_INDEX(uint32_t id) {
    // Mask for the raw array index (Stripping the 24th bit used for Static flags)
    static constexpr unsigned _BitInt(24) ID_TO_INDEX_MASK = 0x7FFFFF;
    static_assert(ID_TO_INDEX_MASK == 0x7FFFFF);
    return id & ID_TO_INDEX_MASK;
}

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
    /*NOLINTNEXTLINE(bugprone-macro-parentheses)*/                                                 \
    Type *Name = &Name##_storage

#ifdef CULVERIN_DEBUG
#    define DEBUG_LOG(fmt, ...) fprintf(stderr, "[Culverin] " fmt "\n", ##__VA_ARGS__)
#else
#    define DEBUG_LOG(fmt, ...)
#endif

#if defined(__cplusplus)
using namespace std;
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

static_assert(sizeof(ContactEvent) == MEMORY_ALIGNMENT_SIZE);

CULV_MAYBE_UNUSED static constexpr int CONTACT_MAX_CAPACITY = sizeof(ContactEvent) * 8 << 5;

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

static constexpr size_t RAYCAST_RESULT_SIZE = 48;

static_assert(sizeof(RayCastBatchResult) == RAYCAST_RESULT_SIZE);

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
    _Atomic uint32_t *gens; // Updated to Atomic
    uint32_t *s2d, *d2s, *free, *cats, *masks, *mats;
    _Atomic uint8_t *stat; // Updated to Atomic
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
    _Atomic uint32_t *generations;
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
    atomic_size_t free_count;
    size_t slot_capacity;
    size_t command_count;
    size_t command_capacity;
    size_t spare_capacity;
    size_t shape_cache_count;
    size_t shape_cache_capacity;
    atomic_size_t count;
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
    #if !defined(Py_GIL_DISABLED)
    atomic_int waiting_threads;
    #endif

    // --- BUCKET 3: Structs & Complex Types ---
    ShadowSync step_sync;    // 16 bytes (Internal 8-byte alignment)
    ShadowMutex shadow_lock; // PyMutex (usually 1-4 bytes)

    // --- BUCKET 4: Small types (Packed at the tail) ---
    _Atomic uint8_t *slot_states;
    uint8_t *constraint_states;
    atomic_bool step_requested;
    atomic_bool is_stepping;
    bool needs_optimization;
    atomic_bool is_resizing;
    atomic_bool is_deallocating;

    // --- Large Tail Arrays ---
    Py_ssize_t view_shape[2];
    Py_ssize_t view_strides[2];

    // --- Debug Renderer ---
    JPH_DebugRenderer *debug_renderer;
    DebugBuffer debug_lines;
    DebugBuffer debug_triangles;
} PhysicsWorldObject;

typedef enum : uint8_t {
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
#include "culverin_arg_indices.h"
typedef struct {
    PyObject *helper;           // Reference to culverin._culverin module
    PyObject *PhysicsWorldType; // Reference to the class
    PyObject *CharacterType;    // Reference to the character class
    PyObject *VehicleType;      // Reference to the vehicle class
    PyObject *SkeletonType;
    PyObject *RagdollSettingsType;
    PyObject *RagdollType;
    CulverinParsers parsers;
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
    uint64_t val = ((uint64_t)gen << HANDLE_INDEX_BITS) | (uint64_t)slot;
    BodyHandle h;
#ifdef __cplusplus
    reinterpret_cast<std::atomic<uint64_t> *>(&h)->store(val, std::memory_order_relaxed);
#else
    atomic_init(&h, val);
#endif
    return h;
}

CULV_NODISCARD
CULV_MAYBE_UNUSED
static inline bool unpack_handle(PhysicsWorldObject *self, BodyHandle h, uint32_t *slot) {
#ifdef __cplusplus
    uint64_t h_val = reinterpret_cast<std::atomic<uint64_t> *>(&h)->load(std::memory_order_relaxed);
#else
    uint64_t h_val = atomic_load_explicit(&h, memory_order_relaxed);
#endif

    *slot        = (uint32_t)(h_val & HANDLE_INDEX_MASK);
    uint32_t gen = (uint32_t)(h_val >> HANDLE_INDEX_BITS);

    if (UNLIKELY(*slot >= self->slot_capacity)) {
        return false;
    }

#ifdef __cplusplus
    uint32_t current_gen = __atomic_load_n((uint32_t *)&self->generations[*slot], __ATOMIC_ACQUIRE);
#else
    // C23 handles _Atomic pointers natively with atomic_load_explicit
    uint32_t current_gen = atomic_load_explicit(&self->generations[*slot], memory_order_acquire);
#endif

    return (bool)(current_gen == gen);
}

// --- Hardened Checkers (No Casts) ---
CULV_NODISCARD
static inline bool culv_is_finite_f(float f) {
    static constexpr uint32_t MASK_F32 = 0x7F800000U;
    static_assert(sizeof(MASK_F32 + 0) == sizeof(uint32_t));
    uint32_t i;
    memcpy(&i, &f, sizeof(float));
    volatile uint32_t vi = i;
    static_assert((sizeof(CULV_TYPE_OF(vi)) == sizeof(uint32_t) &&
                   sizeof(CULV_TYPE_OF(vi)) == sizeof(float)) != 0);
    return (vi & MASK_F32) != MASK_F32;
}

CULV_NODISCARD
static inline bool culv_is_finite_d(double d) {
    static constexpr uint64_t MASK_F64 = 0x7FF0000000000000ULL;
    static_assert(sizeof(MASK_F64 + 0) == sizeof(uint64_t));
    uint64_t i;
    memcpy(&i, &d, sizeof(double));
    volatile uint64_t vi = i;
    static_assert((sizeof(CULV_TYPE_OF(vi)) == sizeof(uint64_t) &&
                   sizeof(CULV_TYPE_OF(vi)) == sizeof(double)) != 0);
    return (vi & MASK_F64) != MASK_F64;
}

// --- The Generic Dispatcher (Preserves Types) ---
#define IS_FINITE(x)                                                                               \
    _Generic((x),                                                                                  \
        float: culv_is_finite_f(x),                                                                \
        double: culv_is_finite_d(x),                                                               \
        default: culv_is_finite_d((double)(x)))

// --- Error Reporting (One instance, no bloat) ---
CULV_MAYBE_UNUSED
static PyObject *culv_raise_finite_err(const char *msg) {
    PyErr_Format(PyExc_ValueError, "Numerical Error: '%s' must be finite", msg);
    return nullptr;
}

// --- The Macro Engine (Variadic Expansion) ---
// This expands into individual checks without creating arrays or casting types.
#define VALIDATE_FINITE_CORE(msg, ...)                                                             \
    CULV_EXPAND(CULV_CONCAT(CULV_VAL_ARGS_, CULV_NARGS(__VA_ARGS__))(msg, __VA_ARGS__))

#define CULV_VAL_ARGS_1(m, x)                                                                      \
    do {                                                                                           \
        if (UNLIKELY(!IS_FINITE(x)))                                                               \
            return culv_raise_finite_err(m);                                                       \
    } while (0)
#define CULV_VAL_ARGS_2(m, x, ...)                                                                 \
    CULV_VAL_ARGS_1(m, x);                                                                         \
    CULV_VAL_ARGS_1(m, __VA_ARGS__)
#define CULV_VAL_ARGS_3(m, x, ...)                                                                 \
    CULV_VAL_ARGS_1(m, x);                                                                         \
    CULV_VAL_ARGS_2(m, __VA_ARGS__)
#define CULV_VAL_ARGS_4(m, x, ...)                                                                 \
    CULV_VAL_ARGS_1(m, x);                                                                         \
    CULV_VAL_ARGS_3(m, __VA_ARGS__)

#define CULV_NARGS(...) CULV_NARGS_IMP(__VA_ARGS__, 4, 3, 2, 1)
#define CULV_NARGS_IMP(_1, _2, _3, _4, N, ...) N
#define CULV_GLUE(a, b) a##b
#define CULV_CONCAT(a, b) CULV_GLUE(a, b)
#define CULV_EXPAND(x) x

// --- User API (Zero cost wrappers) ---
#define VALIDATE_FINITE_FLOAT(val, name) VALIDATE_FINITE_CORE(name, val)
#define VALIDATE_FINITE_VEC3(x, y, z, name) VALIDATE_FINITE_CORE(name, x, y, z)
#define VALIDATE_FINITE_QUAT(x, y, z, w, name) VALIDATE_FINITE_CORE(name, x, y, z, w)
#define VALIDATE_FINITE_VEC4(x, y, z, w, name) VALIDATE_FINITE_CORE(name, x, y, z, w)

// DOESN'T DO THE SHADOW UNLOCKING FOR YOU. REMEMBER!
// DEFINE STRICT_HANDLE_ENABLED IN THE BUILD IF YOU WANT IT TO FAIL FAST
#if defined(STRICT_HANDLE_ENABLED)
#    define RAISE_STALE_HANDLE()                                                                   \
        do {                                                                                       \
            PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");                          \
            return nullptr;                                                                        \
        } while (false)
#else
#    define RAISE_STALE_HANDLE()                                                                   \
        do {                                                                                       \
            Py_RETURN_NONE;                                                                        \
        } while (false)
#endif

CULV_NODISCARD [[gnu::const]]
static inline bool is_state_valid(uint8_t state, uint8_t mask) {
    return (bool)((1U << state) & mask);
}

#define CHECK_HANDLE(h_raw, slot_out)                                                              \
    do {                                                                                           \
        static_assert(sizeof(CULV_TYPE_OF(h_raw)) == sizeof(uint64_t));                            \
        static_assert(sizeof(CULV_TYPE_OF(slot_out)) == sizeof(uint32_t));                         \
        if (UNLIKELY(!unpack_handle(self, (BodyHandle)(h_raw), &(slot_out)))) {                    \
            SHADOW_UNLOCK(&self->shadow_lock);                                                     \
            RAISE_STALE_HANDLE();                                                                  \
        }                                                                                          \
    } while (false)

#define CHECK_STATE(state_val, slot_mask)                                                          \
    do {                                                                                           \
        static_assert(sizeof(state_val) == sizeof(slot_mask));                                     \
        if (UNLIKELY(!is_state_valid((state_val), (slot_mask)))) {                                 \
            SHADOW_UNLOCK(&self->shadow_lock);                                                     \
            RAISE_STALE_HANDLE();                                                                  \
        }                                                                                          \
    } while (false)

typedef struct {
    unsigned _BitInt(1) is_immediate : 1;
    unsigned _BitInt(1) is_deferred : 1;
    unsigned _BitInt(1) is_executable : 1;
    unsigned _BitInt(5) _unused : 5;
} SlotPredicate;

static_assert(sizeof(SlotPredicate) == sizeof(uint8_t));

// Standard masks for reuse
static constexpr uint32_t MASK_IMM_STANDARD = (1u << SLOT_ALIVE) | (1u << SLOT_CHARACTER);
static constexpr uint32_t MASK_IMM_STRICT   = (1u << SLOT_ALIVE);
static constexpr uint32_t MASK_DEFERRED     = (1u << SLOT_PENDING_CREATE);
// Define the mask for states that can be destroyed
static constexpr uint32_t MASK_DESTRUCTIBLE =
    (1u << SLOT_ALIVE) | (1u << SLOT_PENDING_CREATE) | (1u << SLOT_CHARACTER);

[[gnu::const]] CULV_NODISCARD static CULV_FORCE_INLINE SlotPredicate
get_slot_predicate(uint8_t state, uint32_t imm_mask) {
    const uint32_t state_bit = 1u << (state & 7);

    auto imm = (unsigned _BitInt(1)) !!(bool)(state_bit & imm_mask);
    auto def = (unsigned _BitInt(1)) !!(bool)(state_bit & MASK_DEFERRED);

    return (SlotPredicate){
        .is_immediate = imm, .is_deferred = def, .is_executable = imm | def, ._unused = {}};
}