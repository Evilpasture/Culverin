#pragma once

#define PY_SSIZE_T_CLEAN
#include "culverin_jolt_config.h"
#include "culverin_compiler_specifics.h"
#include "joltc.h" // Amer Koleci's JoltC binder.
#include "culverin_jolt_utils.h"
#include <Python.h>
#include "culverin_python_declare.h"

// =========================================================================
// ASAN COMPATIBILITY ALLOCATORS
// =========================================================================
[[clang::ownership_returns(culv_mem)]] [[gnu::always_inline]]
static inline void *CULV_RAW_MALLOC(size_t sz) {
#ifdef ENABLE_SANITIZER
    return malloc(sz);
#else
    return PyMem_RawMalloc(sz);
#endif
}
[[clang::ownership_returns(culv_mem)]] [[gnu::always_inline]]
static inline void *CULV_RAW_CALLOC(size_t n, size_t sz) {
#ifdef ENABLE_SANITIZER
    return calloc(n, sz);
#else
    return PyMem_RawCalloc(n, sz);
#endif
}
[[clang::ownership_takes(culv_mem,
                         1)]] [[clang::ownership_returns(culv_mem)]] [[gnu::always_inline]]
static inline void *CULV_RAW_REALLOC(void *ptr, size_t sz) {
#ifdef ENABLE_SANITIZER
    return realloc(ptr, sz);
#else
    return PyMem_RawRealloc(ptr, sz);
#endif
}
[[clang::ownership_takes(culv_mem, 1)]] [[gnu::always_inline]]
static inline void CULV_RAW_FREE(void *ptr) {
#ifdef ENABLE_SANITIZER
    free(ptr);
#else
    PyMem_RawFree(ptr);
#endif
}

static inline void PyObject_cleanup(void *ptr) {
    PyObject **obj = (PyObject **)ptr;
    Py_CLEAR(*obj);
}
#define AUTO_DECREF [[gnu::cleanup(PyObject_cleanup)]]
static inline PyObject *internal_move_obj(PyObject **obj) {
    auto tmp = *obj;
    *obj     = nullptr;
    return tmp;
}
#define RETURN_RAII_OBJECT(obj) return internal_move_obj(&(obj))

/* How to use:
static inline PyObject *test_function() {
    AUTO_DECREF PyObject *result = PyLong_FromLong(50);
    RETURN_RAII_OBJECT(result);
}
*/

// =========================================================================
#include "culverin_command_buffer.h"
#include "culverin_slot_state.h"
#include "culverin_types.h"
#include <stddef.h>
#include <string.h>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    define WIN32_LEAN_AND_MEAN
#    include <windows.h>
#elif defined(__linux__) || defined(__apple__)
#    include <sched.h>
#    include <unistd.h>
#endif

#ifdef CULVERIN_DEBUG
#    define DEBUG_LOG(fmt, ...) fprintf(stderr, "[Culverin] " fmt "\n", ##__VA_ARGS__)
#else
#    define DEBUG_LOG(fmt, ...)
#endif

#if defined(__cplusplus)
using namespace std;
#endif

typedef struct {
    JPH_Real px;
    JPH_Real py;
    JPH_Real pz;
} PositionVector;

typedef struct {
    float friction;
    float restitution;
} MaterialSettings;

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
