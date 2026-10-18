#pragma once

#define PY_SSIZE_T_CLEAN
#include "jolt/JoltConfig.h"
#include "core/CompilerSpecifics.h"
#include "joltc.h" // Amer Koleci's JoltC binder.
#include "jolt/JoltUtils.h"
#include <Python.h>
#include "bindings/PythonDeclarations.h"

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
#define CPH_AUTO_DECREF [[gnu::cleanup(PyObject_cleanup)]]
static inline PyObject *internal_move_obj(PyObject **obj) {
    auto tmp = *obj;
    *obj     = nullptr;
    return tmp;
}
#define CPH_RETURN_RAII_OBJECT(obj) return internal_move_obj(&(obj))

/* How to use:
static inline PyObject *test_function() {
    AUTO_DECREF PyObject *result = PyLong_FromLong(50);
    RETURN_RAII_OBJECT(result);
}
*/

// =========================================================================
#include "physics/CommandBuffer.h"
#include "physics/SlotState.h"
#include "core/Types.h"
#include <stddef.h>
#include <string.h>

#ifdef _WIN32
#    ifndef CPH_NOMINMAX
#        define CPH_NOMINMAX
#    endif
#    define CPH_WIN32_LEAN_AND_MEAN
#    include <windows.h>
#elif defined(__linux__) || defined(__apple__)
#    include <sched.h>
#    include <unistd.h>
#endif

#ifdef CULVERIN_DEBUG
#    define CPH_DEBUG_LOG(fmt, ...) fprintf(stderr, "[Culverin] " fmt "\n", ##__VA_ARGS__)
#else
#    define CPH_DEBUG_LOG(fmt, ...)
#endif

#if defined(__cplusplus)
using namespace std;
#endif

typedef struct {
    JPH_Real px;
    JPH_Real py;
    JPH_Real pz;
} CPH_PositionVector;

typedef struct {
    float friction;
    float restitution;
} CPH_MaterialSettings;

// Struct to hold parsed Python data safely in C
typedef struct {
    JPH_Vec3 local_p;
    JPH_Quat local_q;
    float params[4];
    int type;
} CPH_CompoundPart;

typedef struct {
    uint32_t tri_count;
    uint32_t vertex_count;
} CPH_MeshBounds;

// --- Hardened Checkers (No Casts) ---
[[nodiscard]]
static inline bool culv_is_finite_f(float f) {
    static constexpr uint32_t MASK_F32 = 0x7F800000U;
    static_assert(sizeof(MASK_F32 + 0) == sizeof(uint32_t));
    uint32_t i;
    memcpy(&i, &f, sizeof(float));
    volatile uint32_t vi = i;
    static_assert((sizeof(CPH_TYPE_OF(vi)) == sizeof(uint32_t) &&
                   sizeof(CPH_TYPE_OF(vi)) == sizeof(float)) != 0);
    return (vi & MASK_F32) != MASK_F32;
}

[[nodiscard]]
static inline bool culv_is_finite_d(double d) {
    static constexpr uint64_t MASK_F64 = 0x7FF0000000000000ULL;
    static_assert(sizeof(MASK_F64 + 0) == sizeof(uint64_t));
    uint64_t i;
    memcpy(&i, &d, sizeof(double));
    volatile uint64_t vi = i;
    static_assert((sizeof(CPH_TYPE_OF(vi)) == sizeof(uint64_t) &&
                   sizeof(CPH_TYPE_OF(vi)) == sizeof(double)) != 0);
    return (vi & MASK_F64) != MASK_F64;
}

// --- The Generic Dispatcher (Preserves Types) ---
#define CPH_IS_FINITE(x)                                                                               \
    _Generic((x),                                                                                  \
        float: culv_is_finite_f(x),                                                                \
        double: culv_is_finite_d(x),                                                               \
        default: culv_is_finite_d((double)(x)))

// --- Error Reporting (One instance, no bloat) ---
[[maybe_unused]]
static PyObject *culv_raise_finite_err(const char *msg) {
    PyErr_Format(PyExc_ValueError, "Numerical Error: '%s' must be finite", msg);
    return nullptr;
}

// --- The Macro Engine (Variadic Expansion) ---
// This expands into individual checks without creating arrays or casting types.
#define CPH_VALIDATE_FINITE_CORE(msg, ...)                                                             \
    CPH_EXPAND(CPH_CONCAT(CPH_VAL_ARGS_, CPH_NARGS(__VA_ARGS__))(msg, __VA_ARGS__))

#define CPH_VAL_ARGS_1(m, x)                                                                      \
    do {                                                                                           \
        if (CPH_UNLIKELY(!CPH_IS_FINITE(x)))                                                               \
            return culv_raise_finite_err(m);                                                       \
    } while (0)
#define CPH_VAL_ARGS_2(m, x, ...)                                                                 \
    CPH_VAL_ARGS_1(m, x);                                                                         \
    CPH_VAL_ARGS_1(m, __VA_ARGS__)
#define CPH_VAL_ARGS_3(m, x, ...)                                                                 \
    CPH_VAL_ARGS_1(m, x);                                                                         \
    CPH_VAL_ARGS_2(m, __VA_ARGS__)
#define CPH_VAL_ARGS_4(m, x, ...)                                                                 \
    CPH_VAL_ARGS_1(m, x);                                                                         \
    CPH_VAL_ARGS_3(m, __VA_ARGS__)

#define CPH_NARGS(...) CPH_NARGS_IMP(__VA_ARGS__, 4, 3, 2, 1)
#define CPH_NARGS_IMP(_1, _2, _3, _4, N, ...) N
#define CPH_EXPAND(x) x

// --- User API (Zero cost wrappers) ---
#define CPH_VALIDATE_FINITE_FLOAT(val, name) CPH_VALIDATE_FINITE_CORE(name, val)
#define CPH_VALIDATE_FINITE_VEC3(x, y, z, name) CPH_VALIDATE_FINITE_CORE(name, x, y, z)
#define CPH_VALIDATE_FINITE_QUAT(x, y, z, w, name) CPH_VALIDATE_FINITE_CORE(name, x, y, z, w)
#define CPH_VALIDATE_FINITE_VEC4(x, y, z, w, name) CPH_VALIDATE_FINITE_CORE(name, x, y, z, w)

// DOESN'T DO THE SHADOW UNLOCKING FOR YOU. REMEMBER!
// DEFINE STRICT_HANDLE_ENABLED IN THE BUILD IF YOU WANT IT TO FAIL FAST
#if defined(STRICT_HANDLE_ENABLED)
#    define CPH_RAISE_STALE_HANDLE()                                                                   \
        do {                                                                                       \
            PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");                          \
            return nullptr;                                                                        \
        } while (false)
#else
#    define CPH_RAISE_STALE_HANDLE()                                                                   \
        do {                                                                                       \
            Py_RETURN_NONE;                                                                        \
        } while (false)
#endif
