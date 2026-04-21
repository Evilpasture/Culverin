#pragma once

#ifndef JPH_DOUBLE_PRECISION
#    define JPH_DOUBLE_PRECISION 1
#endif

#define PY_SSIZE_T_CLEAN
#include "culverin_compiler_specifics.h"
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
#include "culverin_internal_query.h"
#include "culverin_physics_world.h"
#include "culverin_tracked_vehicle.h"
#include "culverin_types.h"
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
    static constexpr culv_u23 ID_TO_INDEX_MASK = 0x7FFFFF;
    constexpr auto stfu                        = 0x7FFFFF;
    static_assert(ID_TO_INDEX_MASK == stfu);
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
    CULV_ATOMIC(BodyHandle) body1;
    CULV_ATOMIC(BodyHandle) body2;
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

// --- Hardened Checkers (No Casts) ---
CULV_NODISCARD
static inline bool culv_is_finite_f(float f) {
    static constexpr uint32_t MASK_F32 = 0x7F800000U;
    static_assert(sizeof(MASK_F32 + 0) == sizeof(uint32_t));
    uint32_t i;
    memcpy(&i, &f, sizeof(float));
    volatile uint32_t vi = i;
    static_assert((int)(sizeof(CULV_TYPE_OF(vi)) == sizeof(uint32_t) &&
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
    static_assert((int)(sizeof(CULV_TYPE_OF(vi)) == sizeof(uint64_t) &&
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

typedef struct {
    culv_u1 is_immediate : 1;
    culv_u1 is_deferred : 1;
    culv_u1 is_executable : 1;
    culv_u5 _unused : 5;
} SlotPredicate;

static_assert(sizeof(SlotPredicate) == sizeof(uint8_t));

// Standard masks for reuse
static constexpr uint32_t MASK_IMM_STANDARD =
    (1u << SLOT_ALIVE) | (1u << SLOT_CHARACTER) | (1u << SLOT_SOFT_BODY);
static constexpr uint32_t MASK_IMM_STRICT = (1u << SLOT_ALIVE) | (1u << SLOT_SOFT_BODY);
static constexpr uint32_t MASK_DEFERRED   = (1u << SLOT_PENDING_CREATE);
// Define the mask for states that can be destroyed
static constexpr uint32_t MASK_DESTRUCTIBLE = (1u << SLOT_ALIVE) | (1u << SLOT_PENDING_CREATE) |
                                              (1u << SLOT_CHARACTER) | (1u << SLOT_SOFT_BODY);

[[gnu::const]] CULV_NODISCARD static CULV_FORCE_INLINE SlotPredicate
get_slot_predicate(uint8_t state, uint32_t imm_mask) {
    const uint32_t state_bit = 1u << (state & 7);

    culv_u1 imm = (culv_u1) !!(bool)(state_bit & imm_mask);
    culv_u1 def = (culv_u1) !!(bool)(state_bit & MASK_DEFERRED);

    return (SlotPredicate){.is_immediate  = imm,
                           .is_deferred   = def,
                           .is_executable = (culv_u1)(imm | def),
                           ._unused       = 0};
}