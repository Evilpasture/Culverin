#pragma once

#include "culverin_compiler_specifics.h"
#include <Python.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/**
 * ============================================================================
 * CULVERIN FAST BUILD ENGINE (UNIFIED C23 EDITION)
 * ============================================================================
 * Replaces Py_BuildValue. Uses _Generic and Macro Mapping to completely
 * eliminate format-string parsing. Builds Tuples, Lists, and Dicts at
 * register-speed with zero memory leaks.
 *
 * Usage Examples:
 *    FastBuild_Value(x)               -> Py_BuildValue("i", x)
 *    FastBuild_Tuple(x, y, z)         -> Py_BuildValue("(fff)", x, y, z)
 *    FastBuild_List(x, y, z)          -> Py_BuildValue("[fff]", x, y, z)
 *    FastBuild_Dict("k", x, "v", y)   -> Py_BuildValue("{s:i, s:i}", "k", x, "v", y)
 *
 * O(1) Dictionary Keys:
 *    Use FastKey() to fetch pre-interned strings from your FastParsers!
 *    FastBuild_Dict(FastKey(&Parser, IDX_X), x)
 * ============================================================================
 */

/* --- 1. TYPE CONSTRUCTORS (Inlined) --- */

CULV_NODISCARD CULV_FORCE_INLINE static PyObject *fb_from_float(float v) {
    return PyFloat_FromDouble((double)v);
}
CULV_NODISCARD CULV_FORCE_INLINE static PyObject *fb_from_int(int v) {
    return PyLong_FromLong((long)v);
}
CULV_NODISCARD CULV_FORCE_INLINE static PyObject *fb_from_u32(uint32_t v) {
    return PyLong_FromUnsignedLong((unsigned long)v);
}
CULV_NODISCARD CULV_FORCE_INLINE static PyObject *fb_from_u64(uint64_t v) {
    return PyLong_FromUnsignedLongLong((unsigned long long)v);
}
CULV_NODISCARD CULV_FORCE_INLINE static PyObject *fb_from_str(const char *v) {
    return PyUnicode_FromString(v);
}

CULV_NODISCARD CULV_FORCE_INLINE static PyObject *fb_from_bool(bool v) {
    PyObject *res = (int)v ? Py_True : Py_False;
    Py_INCREF(res); // Must return a new reference
    return res;
}

// Pass-through for nested objects (Tuples inside Dicts, FastKeys, etc.)
CULV_NODISCARD CULV_FORCE_INLINE static PyObject *fb_incref(PyObject *v) {
    Py_XINCREF(v);
    return v;
}

/**
 * C23 nullptr_t handler.
 * Returns a new reference to Py_None.
 */
CULV_NODISCARD CULV_FORCE_INLINE static PyObject *fb_from_none(nullptr_t v) {
    (void)v;
    Py_RETURN_NONE;
}

/* --- 2. THE C23 COMPILE-TIME ROUTER --- */

// Deliberately undefined function to trigger a clear error message
extern PyObject *CULVERIN_UNSUPPORTED_TYPE_PASSED_TO_FASTBUILD(void);

#define FB_VAL(x)                                                                                  \
    _Generic((x),                                                                                  \
        float: fb_from_float,                                                                      \
        double: PyFloat_FromDouble,                                                                \
        int: fb_from_int,                                                                          \
        long: PyLong_FromLong,                                                                     \
        long long: PyLong_FromLongLong,                                                            \
        unsigned int: fb_from_u32,                                                                 \
        unsigned long: PyLong_FromUnsignedLong,                                                    \
        unsigned long long: fb_from_u64,                                                           \
        bool: fb_from_bool,                                                                        \
        char *: fb_from_str,                                                                       \
        const char *: fb_from_str,                                                                 \
        nullptr_t: fb_from_none,                                                                   \
        PyObject *: fb_incref,                                                                     \
        default: CULVERIN_UNSUPPORTED_TYPE_PASSED_TO_FASTBUILD)(x)

/* --- 3. THE CONTAINER PACKERS (Unrolled by Compiler) --- */

CULV_NODISCARD CULV_FORCE_INLINE static PyObject *fb_pack_tuple(size_t n, PyObject **arr) {
    for (size_t i = 0; i < n; i++) {
        if (UNLIKELY(!arr[i])) {
            goto error;
        }
    }

    PyObject *t = PyTuple_New((Py_ssize_t)n);
    if (UNLIKELY(!t)) {
        goto error;
    }

    for (size_t i = 0; i < n; i++) {
        PyTuple_SET_ITEM(t, i, arr[i]); // SET_ITEM safely steals the reference
    }
    return t;

error:
    for (size_t i = 0; i < n; i++) {
        Py_XDECREF(arr[i]);
    }
    return nullptr;
}

CULV_NODISCARD CULV_FORCE_INLINE static PyObject *fb_pack_list(size_t n, PyObject **arr) {
    for (size_t i = 0; i < n; i++) {
        if (UNLIKELY(!arr[i])) {
            goto error;
        }
    }

    PyObject *l = PyList_New((Py_ssize_t)n);
    if (UNLIKELY(!l)) {
        goto error;
    }

    for (size_t i = 0; i < n; i++) {
        PyList_SET_ITEM(l, i, arr[i]); // SET_ITEM safely steals the reference
    }
    return l;

error:
    for (size_t i = 0; i < n; i++) {
        Py_XDECREF(arr[i]);
    }
    return nullptr;
}

CULV_NODISCARD CULV_FORCE_INLINE static PyObject *fb_pack_dict(size_t n, PyObject **arr) {
    // 1. Pre-flight check: Did any argument fail to allocate?
    for (size_t i = 0; i < n; i++) {
        if (UNLIKELY(!arr[i])) {
            goto error;
        }
    }

    // 2. Dicts must have an even number of arguments (Key-Value pairs)
    if (UNLIKELY(n % 2 != 0)) {
        goto error;
    }

    PyObject *d = PyDict_New();
    if (UNLIKELY(!d)) {
        goto error;
    }

    // 3. Populate Dictionary
    for (size_t i = 0; i < n; i += 2) {
        // PyDict_SetItem DOES NOT steal references (it INCREFs internally)
        if (UNLIKELY(PyDict_SetItem(d, arr[i], arr[i + 1]) < 0)) {
            Py_DECREF(d);
            goto error;
        }
    }

    // 4. Clean up our local references since PyDict_SetItem INCREF'd them
    for (size_t i = 0; i < n; i++) {
        Py_DECREF(arr[i]);
    }
    return d;

error:
    // Safe sweep: Py_XDECREF ignores NULLs
    for (size_t i = 0; i < n; i++) {
        Py_XDECREF(arr[i]);
    }
    return nullptr;
}

/* --- 4. PREPROCESSOR MAPPING (Up to 16 Args / 8 KV Pairs) --- */

#define FB_EXPAND(x) x
// Shifts arguments so that 0 args returns 0, 1 arg returns 1, etc.
#define FB_NARGS_IMPL(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16,   \
                      N, ...)                                                                      \
    N
#define FB_NARGS(...)                                                                              \
    FB_NARGS_IMPL(0 __VA_OPT__(, ) __VA_ARGS__, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3,   \
                  2, 1, 0)

#define FB_MAP_1(x) FB_VAL(x)
#define FB_MAP_2(x, ...) FB_VAL(x), FB_EXPAND(FB_MAP_1(__VA_ARGS__))
#define FB_MAP_3(x, ...) FB_VAL(x), FB_EXPAND(FB_MAP_2(__VA_ARGS__))
#define FB_MAP_4(x, ...) FB_VAL(x), FB_EXPAND(FB_MAP_3(__VA_ARGS__))
#define FB_MAP_5(x, ...) FB_VAL(x), FB_EXPAND(FB_MAP_4(__VA_ARGS__))
#define FB_MAP_6(x, ...) FB_VAL(x), FB_EXPAND(FB_MAP_5(__VA_ARGS__))
#define FB_MAP_7(x, ...) FB_VAL(x), FB_EXPAND(FB_MAP_6(__VA_ARGS__))
#define FB_MAP_8(x, ...) FB_VAL(x), FB_EXPAND(FB_MAP_7(__VA_ARGS__))
#define FB_MAP_9(x, ...) FB_VAL(x), FB_EXPAND(FB_MAP_8(__VA_ARGS__))
#define FB_MAP_10(x, ...) FB_VAL(x), FB_EXPAND(FB_MAP_9(__VA_ARGS__))
#define FB_MAP_11(x, ...) FB_VAL(x), FB_EXPAND(FB_MAP_10(__VA_ARGS__))
#define FB_MAP_12(x, ...) FB_VAL(x), FB_EXPAND(FB_MAP_11(__VA_ARGS__))
#define FB_MAP_13(x, ...) FB_VAL(x), FB_EXPAND(FB_MAP_12(__VA_ARGS__))
#define FB_MAP_14(x, ...) FB_VAL(x), FB_EXPAND(FB_MAP_13(__VA_ARGS__))
#define FB_MAP_15(x, ...) FB_VAL(x), FB_EXPAND(FB_MAP_14(__VA_ARGS__))
#define FB_MAP_16(x, ...) FB_VAL(x), FB_EXPAND(FB_MAP_15(__VA_ARGS__))

#define FB_CONCAT_IMPL(a, b) a##b
#define FB_CONCAT(a, b) FB_CONCAT_IMPL(a, b)
#define FB_MAP(...) FB_EXPAND(FB_CONCAT(FB_MAP_, FB_NARGS(__VA_ARGS__))(__VA_ARGS__))

/* --- 5. THE PUBLIC API --- */

/**
 * FastKey(parser_ptr, index)
 * Fetches an interned PyObject* string in O(1) time. Use for Dict Keys.
 * Example: FastKey(&ForceParser, IDX_V3_X)
 */
#define FastKey(parser_ptr, idx) ((parser_ptr)->specs[(idx)].interned)

/** Returns a single Python primitive. */
#define FastBuild_Value(x) FB_VAL(x)

/** Builds a Python Tuple from C variables. */
#define FastBuild_Tuple(...)                                                                       \
    fb_pack_tuple(FB_NARGS(__VA_ARGS__), FB_NARGS(__VA_ARGS__)                                     \
                                             ? (PyObject *[]){__VA_OPT__(FB_MAP(__VA_ARGS__))}     \
                                             : nullptr)

/** Builds a Python List from C variables. */
#define FastBuild_List(...)                                                                        \
    fb_pack_list(FB_NARGS(__VA_ARGS__), FB_NARGS(__VA_ARGS__)                                      \
                                            ? (PyObject *[]){__VA_OPT__(FB_MAP(__VA_ARGS__))}      \
                                            : nullptr)

/**
 * Builds a Python Dictionary. Must be passed in key-value pairs.
 * Example: FastBuild_Dict(FastKey(&MyParser, IDX_NAME), name_val)
 */
#define FastBuild_Dict(...) fb_pack_dict(FB_NARGS(__VA_ARGS__), (PyObject *[]){FB_MAP(__VA_ARGS__)})