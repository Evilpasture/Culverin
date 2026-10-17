#pragma once

#include <Python.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#    define CULV_FAST_PARSE_LIKELY(x) __builtin_expect(!!(x), 1)
#    define CULV_FAST_PARSE_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#    define CULV_FAST_PARSE_LIKELY(x) (x)
#    define CULV_FAST_PARSE_UNLIKELY(x) (x)
#endif

#if defined(__cplusplus)
#    define CULV_FAST_PARSE_RESTRICT __restrict
#else
#    define CULV_FAST_PARSE_RESTRICT restrict
#endif

#ifdef __cplusplus
// C++: Strip qualifiers from the brackets entirely.
// The "1" stays just for documentation/size.
#    define CULV_FAST_PARSE_STATIC_SIZE(qualifiers_and_size) 1
#else
// C: Keep the beautiful C99 optimization hints.
#    define CULV_FAST_PARSE_STATIC_SIZE(qualifiers_and_size) static qualifiers_and_size
#endif

#ifdef __cplusplus
#    define CULV_FAST_PARSE_ARR_QUALIFIERS(qualifiers)
#else
// C: Keep the beautiful C99 optimization hints.
#    define CULV_FAST_PARSE_ARR_QUALIFIERS(qualifiers) qualifiers
#endif

/**
 * ============================================================================
 * FAST PARSE ENGINE
 * ============================================================================
 * A high-performance, zero-allocation argument parsing system for Python C
 * extensions. Optimized for PEP 703 (Free-threading) and PEP 489 (Multi-phase
 * initialization).
 * ============================================================================
 */

static constexpr int CULV_FAST_PARSE_EMPTY_SLOT = 0xFFFF;
/** --- 1. TYPES & STRUCTS --- **/

typedef struct {
    PyObject *interned;                        // For keyword matching
    PyTypeObject *type_guard;                  // For type checking
    bool (*convert)(PyObject *, const void *); // The workhorse
} CulvFastArgSpecHot;                              // 24 bytes

typedef struct {
    const char *name;
    const char *type_name;
} CulvFastArgSpecCold; // 16 bytes

typedef struct CulvFastParser CulvFastParser;

typedef bool (*CulvFastParseFunc)(
    PyObject *const args[CULV_FAST_PARSE_STATIC_SIZE(const CULV_FAST_PARSE_RESTRICT 1)], Py_ssize_t nargs,
    PyObject kwnames[CULV_FAST_PARSE_ARR_QUALIFIERS(const CULV_FAST_PARSE_RESTRICT)],
    const CulvFastParser *const CULV_FAST_PARSE_RESTRICT fastparser,
    const void *const CULV_FAST_PARSE_RESTRICT targets[CULV_FAST_PARSE_STATIC_SIZE(const CULV_FAST_PARSE_RESTRICT 1)]);

static constexpr auto CULV_FAST_PARSE_ALIGNMENT_SIZE = 64;
typedef struct {
    const char *name;
    const char *type_name;
    PyTypeObject *type_guard;
    bool (*convert)(PyObject *, const void *);
    bool required;
} CulvFastArgDef;

struct CulvFastParser {
    alignas(CULV_FAST_PARSE_ALIGNMENT_SIZE) const char *parser_name;
    CulvFastArgSpecHot *hot_specs;
    CulvFastArgSpecCold *cold_specs;
    uint16_t *lookup_table;
    CulvFastParseFunc hot_path;
    size_t count;
    size_t table_mask;
    uint64_t required_mask;
    uint64_t type_guard_mask;
    uint64_t warned_mask;
};

static_assert(sizeof(struct CulvFastParser) == (size_t)CULV_FAST_PARSE_ALIGNMENT_SIZE * 2);
static_assert(alignof(struct CulvFastParser) == CULV_FAST_PARSE_ALIGNMENT_SIZE);

// Forward declaration of the generic fallback
[[nodiscard]] static inline bool
culv_fast_parse_vector(PyObject *const args[CULV_FAST_PARSE_STATIC_SIZE(const CULV_FAST_PARSE_RESTRICT 1)], Py_ssize_t nargs,
                PyObject kwnames[CULV_FAST_PARSE_ARR_QUALIFIERS(const CULV_FAST_PARSE_RESTRICT)],
                const CulvFastParser fastparser[CULV_FAST_PARSE_STATIC_SIZE(const CULV_FAST_PARSE_RESTRICT 1)],
                const void *const CULV_FAST_PARSE_RESTRICT targets[CULV_FAST_PARSE_STATIC_SIZE(const CULV_FAST_PARSE_RESTRICT 1)]);

/** --- 2. EXTERN DECLARATIONS --- **/
[[gnu::noinline, nodiscard]]
extern bool culv_fast_parse_report_type_error(const CulvFastParser *fastparser, size_t index, PyObject *val);
[[gnu::noinline, nodiscard]]
extern bool culv_fast_parse_report_missing(const CulvFastParser *fastparser, uint64_t provided_mask);
[[gnu::noinline, nodiscard]]
extern bool culv_fast_parse_report_multiple(const CulvFastParser *fastparser, size_t index);
[[gnu::noinline, nodiscard]]
extern bool culv_fast_parse_report_too_many(const CulvFastParser *fastparser, Py_ssize_t nargs);
[[gnu::noinline, nodiscard]]
extern bool culv_fast_parse_report_unknown_keyword(const CulvFastParser *fastparser, PyObject *key);
[[gnu::noinline, nodiscard]]
extern bool culv_fast_parse_report_conversion_error(const CulvFastParser *fastparser, size_t index, PyObject *val);
extern void culv_fast_parse_init_impl(CulvFastParser *fastparser, const CulvFastArgDef *defs, size_t count);
extern void culv_fast_parse_deinit(CulvFastParser *fastparser);
extern bool
culv_fast_parse_legacy(PyObject args[CULV_FAST_PARSE_STATIC_SIZE(const CULV_FAST_PARSE_RESTRICT 1)],
                PyObject kwargs[CULV_FAST_PARSE_ARR_QUALIFIERS(const CULV_FAST_PARSE_RESTRICT)],
                [[maybe_unused]] PyObject *unused,
                const CulvFastParser fastparser[CULV_FAST_PARSE_STATIC_SIZE(const CULV_FAST_PARSE_RESTRICT 1)],
                const void *const CULV_FAST_PARSE_RESTRICT targets[CULV_FAST_PARSE_STATIC_SIZE(const CULV_FAST_PARSE_RESTRICT 1)]);

/** --- 3. SPECULATIVE STUBS (Monomorphic fast-paths) --- **/

enum CulvFastParseIndex : uint8_t {
    CULV_FAST_PARSE_IDX_0,
    CULV_FAST_PARSE_IDX_1,
    CULV_FAST_PARSE_IDX_2,
    CULV_FAST_PARSE_IDX_3,
    CULV_FAST_PARSE_IDX_4,
    CULV_FAST_PARSE_IDX_5,
    CULV_FAST_PARSE_IDX_6,
    CULV_FAST_PARSE_IDX_7,
    CULV_FAST_PARSE_IDX_8
};
[[gnu::always_inline]]
static inline bool culv_fast_parse_dispatch_conv(bool (*conv)(PyObject *, const void *), PyObject *obj,
                                    const void *target);

[[nodiscard, gnu::always_inline]]
static inline bool
culv_fast_parse_spec_check_and_conv(const CulvFastArgSpecHot spec[CULV_FAST_PARSE_STATIC_SIZE(const CULV_FAST_PARSE_RESTRICT 1)], size_t idx,
                       PyObject *val, const void *target,
                       const CulvFastParser fastparser[CULV_FAST_PARSE_STATIC_SIZE(const CULV_FAST_PARSE_RESTRICT 1)]) {
    // 1. Type Guard (Combined pointer + inheritance check)
    if (spec->type_guard) {
        if (CULV_FAST_PARSE_LIKELY(Py_TYPE(val) != spec->type_guard)) {
            if (CULV_FAST_PARSE_UNLIKELY(!PyObject_TypeCheck(val, spec->type_guard))) {
                return culv_fast_parse_report_type_error(fastparser, idx, val);
            }
        }
    }

    // 2. Conversion via Static Dispatcher
    return culv_fast_parse_dispatch_conv(spec->convert, val, target);
}

// Individual conversion call using the Enum and the safety helper
#define CULV_FAST_PARSE_CALL_CONV(n)                                                                            \
    culv_fast_parse_spec_check_and_conv(&fastparser->hot_specs[CULV_FAST_PARSE_IDX_##n], CULV_FAST_PARSE_IDX_##n, args[CULV_FAST_PARSE_IDX_##n],       \
                           targets[CULV_FAST_PARSE_IDX_##n], fastparser)

/**
 * DIRECT CONVERSION CALL
 * Bypasses type-guard inheritance checks and elides devirtualization branches.
 * This is a raw indirect call through the function pointer stored in hot_specs.
 *
 * Logic:
 * - No check for spec->type_guard.
 * - No check for interned identity.
 * - Direct call to .convert().
 */
#define CULV_FAST_PARSE_CALL_CONV_DIRECT(n)                                                                     \
    fastparser->hot_specs[CULV_FAST_PARSE_IDX_##n].convert(args[CULV_FAST_PARSE_IDX_##n], targets[CULV_FAST_PARSE_IDX_##n])

// Stub Function Boilerplate
#define CULV_FAST_PARSE_GEN_STUB(N, ...)                                                                        \
    [[gnu::always_inline]] [[nodiscard]] static inline bool culv_fast_parse_speculate_p##N##_naked(             \
        PyObject *const args[CULV_FAST_PARSE_STATIC_SIZE(const CULV_FAST_PARSE_RESTRICT 1)], Py_ssize_t nargs,               \
        PyObject kwnames[CULV_FAST_PARSE_ARR_QUALIFIERS(const CULV_FAST_PARSE_RESTRICT)],                                    \
        const CulvFastParser fastparser[CULV_FAST_PARSE_STATIC_SIZE(const CULV_FAST_PARSE_RESTRICT 1)],                          \
        const void *const CULV_FAST_PARSE_RESTRICT targets[CULV_FAST_PARSE_STATIC_SIZE(const CULV_FAST_PARSE_RESTRICT 1)]) {              \
        if (CULV_FAST_PARSE_LIKELY(nargs == N && kwnames == nullptr)) {                                         \
            return (__VA_ARGS__);                                                                  \
        }                                                                                          \
        [[clang::noinline]] return culv_fast_parse_vector(args, nargs, kwnames, fastparser, targets);     \
    }

// 0 Args (Special case)
[[nodiscard]] static inline bool
culv_fast_parse_speculate_p0(PyObject *const args[CULV_FAST_PARSE_STATIC_SIZE(const CULV_FAST_PARSE_RESTRICT 1)], Py_ssize_t nargs,
                PyObject kwnames[CULV_FAST_PARSE_ARR_QUALIFIERS(const CULV_FAST_PARSE_RESTRICT)],
                const CulvFastParser fastparser[CULV_FAST_PARSE_STATIC_SIZE(const CULV_FAST_PARSE_RESTRICT 1)],
                const void *const CULV_FAST_PARSE_RESTRICT targets[CULV_FAST_PARSE_STATIC_SIZE(const CULV_FAST_PARSE_RESTRICT 1)]) {
    if (CULV_FAST_PARSE_LIKELY(nargs == 0 && kwnames == nullptr)) {
        return true;
    }
    [[clang::noinline]] return culv_fast_parse_vector(args, nargs, kwnames, fastparser, targets);
}

// Generate stubs 1 through 8
CULV_FAST_PARSE_GEN_STUB(1, CULV_FAST_PARSE_CALL_CONV(0))

CULV_FAST_PARSE_GEN_STUB(2, CULV_FAST_PARSE_CALL_CONV(0) && CULV_FAST_PARSE_CALL_CONV(1))

CULV_FAST_PARSE_GEN_STUB(3, CULV_FAST_PARSE_CALL_CONV(0) && CULV_FAST_PARSE_CALL_CONV(1) && CULV_FAST_PARSE_CALL_CONV(2))

CULV_FAST_PARSE_GEN_STUB(4, CULV_FAST_PARSE_CALL_CONV(0) && CULV_FAST_PARSE_CALL_CONV(1) && CULV_FAST_PARSE_CALL_CONV(2) && CULV_FAST_PARSE_CALL_CONV(3))

CULV_FAST_PARSE_GEN_STUB(5, CULV_FAST_PARSE_CALL_CONV(0) && CULV_FAST_PARSE_CALL_CONV(1) && CULV_FAST_PARSE_CALL_CONV(2) && CULV_FAST_PARSE_CALL_CONV(3) &&
                   CULV_FAST_PARSE_CALL_CONV(4))

CULV_FAST_PARSE_GEN_STUB(6, CULV_FAST_PARSE_CALL_CONV(0) && CULV_FAST_PARSE_CALL_CONV(1) && CULV_FAST_PARSE_CALL_CONV(2) && CULV_FAST_PARSE_CALL_CONV(3) &&
                   CULV_FAST_PARSE_CALL_CONV(4) && CULV_FAST_PARSE_CALL_CONV(5))

CULV_FAST_PARSE_GEN_STUB(7, CULV_FAST_PARSE_CALL_CONV(0) && CULV_FAST_PARSE_CALL_CONV(1) && CULV_FAST_PARSE_CALL_CONV(2) && CULV_FAST_PARSE_CALL_CONV(3) &&
                   CULV_FAST_PARSE_CALL_CONV(4) && CULV_FAST_PARSE_CALL_CONV(5) && CULV_FAST_PARSE_CALL_CONV(6))

CULV_FAST_PARSE_GEN_STUB(8, CULV_FAST_PARSE_CALL_CONV(0) && CULV_FAST_PARSE_CALL_CONV(1) && CULV_FAST_PARSE_CALL_CONV(2) && CULV_FAST_PARSE_CALL_CONV(3) &&
                   CULV_FAST_PARSE_CALL_CONV(4) && CULV_FAST_PARSE_CALL_CONV(5) && CULV_FAST_PARSE_CALL_CONV(6) && CULV_FAST_PARSE_CALL_CONV(7))

/** --- 3. CONVERTER DISPATCH --- **/

[[nodiscard, gnu::always_inline]] static inline bool culv_fast_parse_conv_float(PyObject *obj,
                                                                   const void *target) {
    // HOT PATH: Exact float match (Pointer comparison)
    if (PyFloat_CheckExact(obj)) {
        *(float *)target = (float)((PyFloatObject *)obj)->ob_fval;
        return true;
    }

    // COLD PATH: Handle ints or float subclasses
    double val = PyFloat_AsDouble(obj);
    if (CULV_FAST_PARSE_UNLIKELY(val == -1.0 && PyErr_Occurred())) {
        return false;
    }
    *(float *)target = (float)val;
    return true;
}

[[nodiscard, gnu::always_inline]] static inline bool culv_fast_parse_conv_double(PyObject *obj,
                                                                    const void *target) {
    // HOT PATH: Exact float match
    // Bypasses the overhead of PyFloat_AsDouble (magic method checks, int conversion logic)
    if (PyFloat_CheckExact(obj)) {
        *(double *)target = ((PyFloatObject *)obj)->ob_fval;
        return true;
    }

    // COLD PATH: Handle ints, None, or float subclasses
    if (CULV_FAST_PARSE_UNLIKELY(obj == Py_None)) {
        PyErr_SetString(PyExc_TypeError, "double argument cannot be None");
        return false;
    }

    double val = PyFloat_AsDouble(obj);
    if (CULV_FAST_PARSE_UNLIKELY(val == -1.0 && PyErr_Occurred())) {
        return false;
    }
    *(double *)target = val;
    return true;
}

[[nodiscard, gnu::always_inline]] static inline bool culv_fast_parse_conv_int(PyObject *obj,
                                                                 const void *target) {
    // 1. Singletons (Pointer identity is the fastest possible check)
    if (obj == Py_True) {
        *(int *)target = 1;
        return true;
    }
    if (obj == Py_False) {
        *(int *)target = 0;
        return true;
    }

    // 2. None Check (Cold path, but keeps PyLong_AsLong clean)
    if (CULV_FAST_PARSE_UNLIKELY(obj == Py_None)) {
        PyErr_SetString(PyExc_TypeError, "int argument cannot be None");
        return false;
    }

    // 3. Single Conversion Call
    long val = PyLong_AsLong(obj);

    // 4. Hot Path: 99.9% of integers in physics/logic are not -1
    if (CULV_FAST_PARSE_LIKELY(val != -1)) {
        *(int *)target = (int)val;
        return true;
    }

    // 5. Cold Path: Was it actually -1 or a conversion error?
    if (PyErr_Occurred()) {
        return false;
    }

    *(int *)target = -1;
    return true;
}

[[nodiscard, gnu::always_inline]] static inline bool culv_fast_parse_conv_u32(PyObject *obj,
                                                                 const void *target) {
    if (obj == Py_True) {
        *(uint32_t *)target = 1;
        return true;
    }
    if (obj == Py_False) {
        *(uint32_t *)target = 0;
        return true;
    }
    if (CULV_FAST_PARSE_UNLIKELY(obj == Py_None)) {
        PyErr_SetString(PyExc_TypeError, "uint32 argument cannot be None");
        return false;
    }

    unsigned long val = PyLong_AsUnsignedLong(obj);

    if (CULV_FAST_PARSE_LIKELY(val != (unsigned long)-1)) {
        *(uint32_t *)target = (uint32_t)val;
        return true;
    }

    if (PyErr_Occurred()) {
        return false;
    }

    *(uint32_t *)target = (uint32_t)-1;
    return true;
}

[[nodiscard, gnu::always_inline]] static inline bool culv_fast_parse_conv_u64(PyObject *obj,
                                                                 const void *target) {
    if (obj == Py_True) {
        *(uint64_t *)target = 1;
        return true;
    }
    if (obj == Py_False) {
        *(uint64_t *)target = 0;
        return true;
    }
    // u64 often used for handles, which might be None
    if (obj == Py_None) {
        *(uint64_t *)target = 0;
        return true;
    }

    unsigned long long val = PyLong_AsUnsignedLongLong(obj);

    if (CULV_FAST_PARSE_LIKELY(val != (unsigned long long)-1)) {
        *(uint64_t *)target = (uint64_t)val;
        return true;
    }

    if (PyErr_Occurred()) {
        return false;
    }

    *(uint64_t *)target = (uint64_t)-1;
    return true;
}

[[nodiscard, gnu::always_inline]] static inline bool culv_fast_parse_conv_bool(PyObject *obj,
                                                                  const void *target) {
    // Exact pointer check for the singletons
    if (obj == Py_True) {
        *(bool *)target = true;
        return true;
    }
    if (obj == Py_False) {
        *(bool *)target = false;
        return true;
    }

    // Fallback for truthiness of lists, strings, etc.
    int val = PyObject_IsTrue(obj);
    if (CULV_FAST_PARSE_UNLIKELY(val == -1)) {
        return false;
    }
    *(bool *)target = (bool)val;
    return true;
}

[[nodiscard, gnu::always_inline]] static inline bool culv_fast_parse_conv_pyobj(PyObject *obj,
                                                                   const void *target) {
    *(PyObject **)target = obj;
    return true;
}

[[nodiscard, gnu::always_inline]] static inline bool culv_fast_parse_conv_str(PyObject *obj,
                                                                 const void *target) {
    if (CULV_FAST_PARSE_UNLIKELY(!PyUnicode_Check(obj))) {
        PyErr_Format(PyExc_TypeError, "expected str, got %s", Py_TYPE(obj)->tp_name);
        return false;
    }
    const char *str = PyUnicode_AsUTF8(obj);
    if (CULV_FAST_PARSE_UNLIKELY(str == nullptr)) {
        return false;
    }
    *(const char **)target = str;
    return true;
}

[[nodiscard, gnu::always_inline]] static inline bool culv_fast_parse_conv_ssize(PyObject *obj,
                                                                   const void *target) {
    // HOT PATH: Exact integer match
    if (CULV_FAST_PARSE_LIKELY(PyLong_CheckExact(obj))) {
        Py_ssize_t val = PyLong_AsSsize_t(obj);
        // Optimization: Values like buffer sizes and counts are rarely exactly -1.
        if (CULV_FAST_PARSE_LIKELY(val != -1)) {
            *(Py_ssize_t *)target = val;
            return true;
        }
    }

    // COLD PATH: Handle None, subclasses, or the literal value -1
    if (obj == Py_None) {
        PyErr_SetString(PyExc_TypeError, "size/index argument cannot be None");
        return false;
    }

    Py_ssize_t val = PyLong_AsSsize_t(obj);
    if (val == -1 && PyErr_Occurred()) {
        return false;
    }
    *(Py_ssize_t *)target = val;
    return true;
}

[[gnu::always_inline, nodiscard]]
static inline bool culv_fast_parse_dispatch_conv(bool (*conv)(PyObject *, const void *), PyObject *obj,
                                    const void *target) {
    // These comparisons are just address checks (extremely fast)
    // If a match is found, the compiler inlines the specific converter.
    if (conv == culv_fast_parse_conv_float) {
        return culv_fast_parse_conv_float(obj, target);
    }
    if (conv == culv_fast_parse_conv_int) {
        return culv_fast_parse_conv_int(obj, target);
    }
    if (conv == culv_fast_parse_conv_u64) {
        return culv_fast_parse_conv_u64(obj, target);
    }
    if (conv == culv_fast_parse_conv_ssize) {
        return culv_fast_parse_conv_ssize(obj, target);
    }
    if (conv == culv_fast_parse_conv_bool) {
        return culv_fast_parse_conv_bool(obj, target);
    }
    if (conv == culv_fast_parse_conv_pyobj) {
        return culv_fast_parse_conv_pyobj(obj, target);
    }

    // Fallback for custom or less common converters
    return conv(obj, target);
}

// Allow host projects to inject custom types
#ifndef CULV_FAST_PARSE_CUSTOM_CONVERTERS
#    define CULV_FAST_PARSE_CUSTOM_CONVERTERS /* empty */
#endif

extern void culv_fast_parse_unsupported_type(void);

#define CULV_FAST_PARSE_GET_CONVERTER(T)                                                                        \
    _Generic((T),                                                                                  \
        float: culv_fast_parse_conv_float,                                                                      \
        double: culv_fast_parse_conv_double,                                                                    \
        int: culv_fast_parse_conv_int,                                                                          \
        uint32_t: culv_fast_parse_conv_u32,                                                                     \
        uint64_t: culv_fast_parse_conv_u64,                                                                     \
        Py_ssize_t: culv_fast_parse_conv_ssize,                                                                 \
        bool: culv_fast_parse_conv_bool,                                                                        \
        const char *: culv_fast_parse_conv_str,                                                                 \
        PyObject *: culv_fast_parse_conv_pyobj CULV_FAST_PARSE_CUSTOM_CONVERTERS, /* INJECTION POINT */                      \
        default: culv_fast_parse_unsupported_type)

#ifndef CULV_FAST_PARSE_CUSTOM_TYPE_NAMES
#    define CULV_FAST_PARSE_CUSTOM_TYPE_NAMES /* empty */
#endif

#define CULV_FAST_PARSE_GET_TYPE_NAME(T)                                                                        \
    _Generic((T),                                                                                  \
        float: "float",                                                                            \
        double: "float",                                                                           \
        int: "int",                                                                                \
        uint32_t: "int",                                                                           \
        uint64_t: "int",                                                                           \
        Py_ssize_t: "int",                                                                         \
        bool: "bool",                                                                              \
        const char *: "str",                                                                       \
        PyObject *: "object" CULV_FAST_PARSE_CUSTOM_TYPE_NAMES, /* INJECTION POINT */                           \
        default: "any")

#define CULV_FAST_PARSE_ARG(name_str, var)                                                                      \
    (CulvFastArgDef) {                                                                                 \
        .name = (name_str), .type_name = CULV_FAST_PARSE_GET_TYPE_NAME((typeof_unqual(var)){}),                 \
        .type_guard = NULL, .convert = CULV_FAST_PARSE_GET_CONVERTER((typeof_unqual(var)){}), .required = false \
    }

#define CULV_FAST_PARSE_REQ_ARG(name_str, var)                                                                  \
    (CulvFastArgDef) {                                                                                 \
        .name = (name_str), .type_name = CULV_FAST_PARSE_GET_TYPE_NAME((typeof_unqual(var)){}),                 \
        .type_guard = NULL, .convert = CULV_FAST_PARSE_GET_CONVERTER((typeof_unqual(var)){}), .required = true  \
    }

#define CULV_FAST_PARSE_ARG_GUARD(name_str, guard_type_ptr, var)                                                \
    (CulvFastArgDef) {                                                                                 \
        .name = (name_str), .type_name = (guard_type_ptr)->tp_name,                                \
        .type_guard = (guard_type_ptr), .convert = CULV_FAST_PARSE_GET_CONVERTER((typeof_unqual(var)){}),       \
        .required = false                                                                          \
    }

#define CULV_FAST_PARSE_REQ_ARG_GUARD(name_str, guard_type_ptr, var)                                            \
    (CulvFastArgDef) {                                                                                 \
        .name = (name_str), .type_name = (guard_type_ptr)->tp_name,                                \
        .type_guard = (guard_type_ptr), .convert = CULV_FAST_PARSE_GET_CONVERTER((typeof_unqual(var)){}),       \
        .required = true                                                                           \
    }

#define CULV_FAST_PARSE_ARG_CUSTOM(name_str, conv_func)                                                         \
    (CulvFastArgDef) {                                                                                 \
        .name = (name_str), .type_name = "object", .type_guard = NULL, .convert = (conv_func),     \
        .required = false                                                                          \
    }

/** --- 4. THE HOT PATH --- **/
[[gnu::always_inline, gnu::const, maybe_unused, nodiscard]]
static inline size_t culv_fast_parse_hash_ptr(PyObject *ptr, size_t mask) {
    auto val = (uintptr_t)ptr;
    // Golden ratio multiplier spreads pointer bits effectively
    constexpr auto gratio     = 11400714819323198485ULL;
    constexpr auto shift_bits = 32ULL;
    return ((val * gratio) >> shift_bits) & mask;
}
[[gnu::always_inline, nodiscard]]
static inline bool
culv_fast_parse_check_type_guard(const CulvFastArgSpecHot spec[CULV_FAST_PARSE_STATIC_SIZE(const CULV_FAST_PARSE_RESTRICT 1)],
                    PyObject val[CULV_FAST_PARSE_STATIC_SIZE(const CULV_FAST_PARSE_RESTRICT 1)]) {
    PyTypeObject *guard = spec->type_guard;

    // If no guard is defined (NULL), this branch is perfectly predicted as "not taken"
    if (CULV_FAST_PARSE_LIKELY(guard == nullptr)) {
        return true;
    }

    // HOT PATH: Exact pointer match (The most common case in Python)
    if (CULV_FAST_PARSE_LIKELY(Py_IS_TYPE(val, guard))) {
        return true;
    }

    // COLD PATH: Inheritance check (Subclasses)
    return PyObject_TypeCheck(val, guard);
}
[[gnu::noinline, gnu::cold]]
extern bool culv_fast_parse_warn_uninterned_slow(PyObject key[], const char pname[]);

/**
 * COLD FALLBACK: culv_fast_parse_cmp_slow
 * Separated to keep the instruction cache lean for interned keys.
 */
[[gnu::noinline, gnu::cold]]
extern size_t culv_fast_parse_cmp_slow(PyObject key[CULV_FAST_PARSE_STATIC_SIZE(const CULV_FAST_PARSE_RESTRICT 1)],
                          const CulvFastParser fastparse[CULV_FAST_PARSE_STATIC_SIZE(const CULV_FAST_PARSE_RESTRICT 1)]);

[[gnu::always_inline, nodiscard]]
static inline size_t
culv_fast_parse_find_keyword_index(PyObject key[CULV_FAST_PARSE_STATIC_SIZE(const CULV_FAST_PARSE_RESTRICT 1)],
                      const CulvFastParser fastparse[CULV_FAST_PARSE_STATIC_SIZE(const CULV_FAST_PARSE_RESTRICT 1)]) {
    const CulvFastArgSpecHot *specs = fastparse->hot_specs;
    const uint16_t *ltable      = fastparse->lookup_table;

    if (CULV_FAST_PARSE_LIKELY(ltable != nullptr)) {
        size_t hash = culv_fast_parse_hash_ptr(key, fastparse->table_mask);

        // THE FIX: Probe until we hit an empty slot (table is 50% empty, so this is fast)
        // We use table_mask as a safe upper bound to prevent infinite loops.
        for (size_t i = 0; i <= fastparse->table_mask; i++) {
            size_t slot = ltable[(hash + i) & fastparse->table_mask];
            if (slot == CULV_FAST_PARSE_EMPTY_SLOT) {
                break;
            }
            if (specs[slot].interned == key) {
                return slot;
            }
        }
    } else {
        const size_t count = fastparse->count;
        for (size_t j = 0; j < count; ++j) {
            if (specs[j].interned == key) {
                return j;
            }
        }
    }

    return culv_fast_parse_cmp_slow(key, fastparse);
}

[[gnu::always_inline]]
static inline bool
culv_fast_parse_process_pos(const CulvFastParser fastparse[CULV_FAST_PARSE_STATIC_SIZE(const CULV_FAST_PARSE_RESTRICT 1)],
               PyObject *const args[CULV_FAST_PARSE_STATIC_SIZE(const CULV_FAST_PARSE_RESTRICT 1)], Py_ssize_t nargs,
               const void *const CULV_FAST_PARSE_RESTRICT targets[CULV_FAST_PARSE_STATIC_SIZE(const CULV_FAST_PARSE_RESTRICT 1)]) {
#pragma unroll 4
    for (Py_ssize_t i = 0; i < nargs; ++i) {
        // spec is 24 bytes, targets[i] is 8 bytes.
        // We are touching very little memory here.
        const CulvFastArgSpecHot *spec = &fastparse->hot_specs[i];

        // 1. Safety check (Type Guard)
        // We use the helper to keep the "Happy Path" linear.
        if (CULV_FAST_PARSE_UNLIKELY(!culv_fast_parse_check_type_guard(spec, args[i]))) {
            return culv_fast_parse_report_type_error(fastparse, (size_t)i, args[i]);
        }

        // 2. Conversion
        if (CULV_FAST_PARSE_UNLIKELY(!spec->convert(args[i], targets[i]))) {
            return culv_fast_parse_report_conversion_error(fastparse, (size_t)i, args[i]);
        }
    }
    return true;
}
[[gnu::always_inline, nodiscard]]
static inline bool
culv_fast_parse_process_kw(const CulvFastParser fastparse[CULV_FAST_PARSE_STATIC_SIZE(const CULV_FAST_PARSE_RESTRICT 1)],
              PyObject *const args[CULV_FAST_PARSE_STATIC_SIZE(const CULV_FAST_PARSE_RESTRICT 1)], Py_ssize_t nargs,
              PyObject kwnames[CULV_FAST_PARSE_ARR_QUALIFIERS(const CULV_FAST_PARSE_RESTRICT)],
              uint64_t mask[CULV_FAST_PARSE_STATIC_SIZE(const CULV_FAST_PARSE_RESTRICT 1)],
              const void *const CULV_FAST_PARSE_RESTRICT targets[CULV_FAST_PARSE_STATIC_SIZE(const CULV_FAST_PARSE_RESTRICT 1)]) {
    if (!kwnames) {
        return true;
    }

    const Py_ssize_t nkw     = PyTuple_GET_SIZE(kwnames);
    PyObject *const *kw_vals = args + nargs;

#pragma unroll 4
    for (Py_ssize_t i = 0; i < nkw; ++i) {
        PyObject *key = PyTuple_GET_ITEM(kwnames, i);
        size_t idx    = culv_fast_parse_find_keyword_index(key, fastparse);

        if (CULV_FAST_PARSE_UNLIKELY(idx == CULV_FAST_PARSE_EMPTY_SLOT)) {
            return culv_fast_parse_report_unknown_keyword(fastparse, key);
        }

        if (CULV_FAST_PARSE_UNLIKELY(*mask & (1ULL << idx))) {
            return culv_fast_parse_report_multiple(fastparse, idx);
        }

        // UPDATED: Now only takes 2 arguments
        if (!culv_fast_parse_check_type_guard(&fastparse->hot_specs[idx], kw_vals[i])) {
            return culv_fast_parse_report_type_error(fastparse, idx, kw_vals[i]);
        }

        if (CULV_FAST_PARSE_UNLIKELY(!fastparse->hot_specs[idx].convert(kw_vals[i], targets[idx]))) {
            return culv_fast_parse_report_conversion_error(fastparse, idx, kw_vals[i]);
        }
        *mask |= (1ULL << idx);
    }
    return true;
}

[[gnu::always_inline, gnu::const, nodiscard]]
static inline uint64_t culv_fast_parse_make_mask(size_t n) {
    if (n == 0) {
        return 0;
    }
    static constexpr uint64_t FULL_BITS = 64;
    if (n >= FULL_BITS) {
        return ~(uint64_t)0;
    }
    return (n >= 64) ? ~(uint64_t)0 : (1ULL << n) - 1;
}

// unadulterated speed!!!
[[nodiscard, gnu::always_inline, gnu::hot, gnu::no_stack_protector,
  gnu::nonnull(1, 4, 5)]] static inline bool
culv_fast_parse_vector(PyObject *const args[CULV_FAST_PARSE_STATIC_SIZE(const CULV_FAST_PARSE_RESTRICT 1)], Py_ssize_t nargs,
                PyObject kwnames[CULV_FAST_PARSE_ARR_QUALIFIERS(const CULV_FAST_PARSE_RESTRICT)],
                const CulvFastParser fastparse[CULV_FAST_PARSE_STATIC_SIZE(const CULV_FAST_PARSE_RESTRICT 1)],
                const void *const CULV_FAST_PARSE_RESTRICT targets[CULV_FAST_PARSE_STATIC_SIZE(const CULV_FAST_PARSE_RESTRICT 1)]) {
    // 1. Pre-flight Check
    if (CULV_FAST_PARSE_UNLIKELY(nargs > (Py_ssize_t)fastparse->count)) {
        return culv_fast_parse_report_too_many(fastparse, nargs);
    }

    // 2. Handle Positionals
    if (CULV_FAST_PARSE_UNLIKELY(!culv_fast_parse_process_pos(fastparse, args, nargs, targets))) {
        return false;
    }

    // 3. Setup Mask & Handle Keywords

    uint64_t mask = culv_fast_parse_make_mask(nargs);
    if (CULV_FAST_PARSE_UNLIKELY(!culv_fast_parse_process_kw(fastparse, args, nargs, kwnames, &mask, targets))) {
        return false;
    }

    // 4. Final Verification
    if (CULV_FAST_PARSE_UNLIKELY((mask & fastparse->required_mask) != fastparse->required_mask)) {
        return culv_fast_parse_report_missing(fastparse, mask);
    }

    return true;
}

/** --- 5. PUBLIC MACROS --- **/

// Triggered if argument 1 is of an unsupported type
extern void culv_fast_parse_invalid_argument_vector(void);
// NOLINTNEXTLINE(readability-identifier-naming)
#define CULV_FAST_PARSE_UNIFIED(arg1, arg2, arg3, arg4, arg5)                                            \
    _Generic((arg1),                                                                               \
        PyObject *const *: (arg4)->hot_path,                                                       \
        PyObject **: (arg4)->hot_path,                                                             \
        PyObject *: culv_fast_parse_legacy,                                                               \
        default: culv_fast_parse_invalid_argument_vector)(                \
        (arg1), (arg2), (arg3), (arg4), (arg5))
// NOLINTNEXTLINE(readability-identifier-naming)
#define CULV_FAST_PARSE_INIT(fastparser, specs, count)                                                   \
    do {                                                                                           \
        static_assert((count) <= 64, "Culverin parser supports at most 64 arguments");                \
        culv_fast_parse_init_impl(fastparser, specs, count);                                                    \
    } while (false)