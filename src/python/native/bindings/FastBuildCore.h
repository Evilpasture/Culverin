#pragma once

#include <Python.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* --- 0. CONFIGURATION & COMPILER COMPATIBILITY --- */

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L
#    ifndef _NULLPTR_T
#        define _NULLPTR_T
typedef typeof(nullptr) nullptr_t;
#    endif
#endif

#ifndef CPH_PY_BUILD_LIKELY
#    if defined(__GNUC__) || defined(__clang__)
#        define CPH_PY_BUILD_LIKELY(x) __builtin_expect(!!(x), 1)
#        define CPH_PY_BUILD_UNLIKELY(x) __builtin_expect(!!(x), 0)
#    else
#        define CPH_PY_BUILD_LIKELY(x) (x)
#        define CPH_PY_BUILD_UNLIKELY(x) (x)
#    endif
#endif

#ifndef CPH_PY_BUILD_FORCE_INLINE
#    define CPH_PY_BUILD_FORCE_INLINE inline
#endif

/* --- 1. TYPE CONSTRUCTORS (Inlined) --- */

[[nodiscard]] CPH_PY_BUILD_FORCE_INLINE static PyObject *culv_py_build_from_float(float v) {
    return PyFloat_FromDouble((double)v);
}
[[nodiscard]] CPH_PY_BUILD_FORCE_INLINE static PyObject *culv_py_build_from_double(double v) {
    return PyFloat_FromDouble(v);
}
[[nodiscard]] CPH_PY_BUILD_FORCE_INLINE static PyObject *culv_py_build_from_int(int v) {
    return PyLong_FromLong((long)v);
}
[[nodiscard]] CPH_PY_BUILD_FORCE_INLINE static PyObject *culv_py_build_from_long(long v) {
    return PyLong_FromLong(v);
}
[[nodiscard]] CPH_PY_BUILD_FORCE_INLINE static PyObject *culv_py_build_from_longlong(long long v) {
    return PyLong_FromLongLong(v);
}
[[nodiscard]] CPH_PY_BUILD_FORCE_INLINE static PyObject *culv_py_build_from_u32(uint32_t v) {
    return PyLong_FromUnsignedLong((unsigned long)v);
}
[[nodiscard]] CPH_PY_BUILD_FORCE_INLINE static PyObject *culv_py_build_from_u64(uint64_t v) {
    return PyLong_FromUnsignedLongLong((unsigned long long)v);
}
[[nodiscard]] CPH_PY_BUILD_FORCE_INLINE static PyObject *culv_py_build_from_str(const char *v) {
    return PyUnicode_FromString(v);
}
[[nodiscard]] CPH_PY_BUILD_FORCE_INLINE static PyObject *culv_py_build_from_bool(bool v) {
    PyObject *res = (int)v ? Py_True : Py_False;
    Py_INCREF(res);
    return res;
}
[[nodiscard]] CPH_PY_BUILD_FORCE_INLINE static PyObject *culv_py_build_incref(PyObject *v) {
    Py_XINCREF(v);
    return v;
}
[[nodiscard]] CPH_PY_BUILD_FORCE_INLINE static PyObject *
culv_py_build_from_none([[maybe_unused]] nullptr_t v) {
    Py_RETURN_NONE;
}

/* --- 2. THE C23 COMPILE-TIME ROUTER --- */

extern PyObject *culv_py_build_unsupported_type(void);

// Allow host projects to inject custom builders
#ifndef CPH_PY_BUILD_CUSTOM_CONVERTERS
#    define CPH_PY_BUILD_CUSTOM_CONVERTERS /* empty */
#endif

#define CPH_PY_BUILD_VAL(x)                                                                                  \
    _Generic((x),                                                                                  \
        float: culv_py_build_from_float,                                                                      \
        double: culv_py_build_from_double,                                                                    \
        int: culv_py_build_from_int,                                                                          \
        long: culv_py_build_from_long,                                                                        \
        long long: culv_py_build_from_longlong,                                                               \
        unsigned int: culv_py_build_from_u32,                                                                 \
        unsigned long: PyLong_FromUnsignedLong,                                                    \
        unsigned long long: culv_py_build_from_u64,                                                           \
        bool: culv_py_build_from_bool,                                                                        \
        char *: culv_py_build_from_str,                                                                       \
        const char *: culv_py_build_from_str,                                                                 \
        nullptr_t: culv_py_build_from_none,                                                                   \
        PyObject *: culv_py_build_incref CPH_PY_BUILD_CUSTOM_CONVERTERS,                                                \
        default: culv_py_build_unsupported_type)(x)

/* --- 3. THE CONTAINER PACKERS --- */

[[nodiscard]] [[gnu::always_inline]]
static inline PyObject *culv_py_build_pack_tuple(size_t n, PyObject **arr) {
    if (CPH_PY_BUILD_UNLIKELY(n == 0)) {
        return PyTuple_New(0);
    }

    PyObject *t = PyTuple_New((Py_ssize_t)n);
    if (CPH_PY_BUILD_UNLIKELY(!t)) {
        for (size_t i = 0; i < n; i++) {
            Py_XDECREF(arr[i]);
        }
        return nullptr;
    }

    for (size_t i = 0; i < n; i++) {
        PyObject *obj = arr[i];

        if (CPH_PY_BUILD_UNLIKELY(!obj)) {
            // Builder failure cleanup:
            Py_DECREF(t);

            for (size_t j = i + 1; j < n; j++) {
                Py_XDECREF(arr[j]);
            }
            return nullptr;
        }

        // Steals the reference from the builder array
        PyTuple_SET_ITEM(t, i, obj);
    }
    return t;
}

[[nodiscard]] CPH_PY_BUILD_FORCE_INLINE static PyObject *culv_py_build_pack_list(size_t n,
                                                                                  PyObject **arr) {
    if (CPH_PY_BUILD_UNLIKELY(n == 0)) {
        return PyList_New(0);
    }
#pragma unroll 4
    for (size_t i = 0; i < n; i++) {
        if (CPH_PY_BUILD_UNLIKELY(!arr[i])) {
            goto error;
        }
    }

    PyObject *l = PyList_New((Py_ssize_t)n);
    if (CPH_PY_BUILD_UNLIKELY(!l)) {
        goto error;
    }
#pragma unroll 4
    for (size_t i = 0; i < n; i++) {
        PyList_SET_ITEM(l, i, arr[i]); // Steals reference
    }
    return l;

error:
#pragma unroll 2
    for (size_t i = 0; i < n; i++) {
        Py_XDECREF(arr[i]);
    }
    return nullptr;
}

[[nodiscard]] CPH_PY_BUILD_FORCE_INLINE static PyObject *culv_py_build_pack_dict(size_t n,
                                                                                  PyObject **arr) {
    if (CPH_PY_BUILD_UNLIKELY(n == 0)) {
        return PyDict_New();
    }
#pragma unroll 4
    for (size_t i = 0; i < n; i++) {
        if (CPH_PY_BUILD_UNLIKELY(!arr[i])) {
            goto error;
        }
    }

    PyObject *d = PyDict_New();
    if (CPH_PY_BUILD_UNLIKELY(!d)) {
        goto error;
    }
#pragma unroll 4
    for (size_t i = 0; i < n; i += 2) {
        if (CPH_PY_BUILD_UNLIKELY(PyDict_SetItem(d, arr[i], arr[i + 1]) < 0)) {
            Py_DECREF(d);
            goto error;
        }
    }
#pragma unroll 4
    for (size_t i = 0; i < n; i++) {
        Py_DECREF(arr[i]);
    }
    return d;

error:
#pragma unroll 2
    for (size_t i = 0; i < n; i++) {
        Py_XDECREF(arr[i]);
    }
    return nullptr;
}

/* --- 4. PREPROCESSOR MAPPING (Up to 16 Args / 8 KV Pairs) --- */

#define CPH_PY_BUILD_EXPAND(x) x

#define CPH_PY_BUILD_NARGS_IMPL(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17,  \
                      _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32,   \
                      N, ...)                                                                      \
    N

#define CPH_PY_BUILD_NARGS(...)                                                                              \
    CPH_PY_BUILD_NARGS_IMPL(__VA_ARGS__, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, \
                  15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)

#define CPH_PY_BUILD_MAP_1(x) CPH_PY_BUILD_VAL(x)
#define CPH_PY_BUILD_MAP_2(x, ...) CPH_PY_BUILD_VAL(x), CPH_PY_BUILD_EXPAND(CPH_PY_BUILD_MAP_1(__VA_ARGS__))
#define CPH_PY_BUILD_MAP_3(x, ...) CPH_PY_BUILD_VAL(x), CPH_PY_BUILD_EXPAND(CPH_PY_BUILD_MAP_2(__VA_ARGS__))
#define CPH_PY_BUILD_MAP_4(x, ...) CPH_PY_BUILD_VAL(x), CPH_PY_BUILD_EXPAND(CPH_PY_BUILD_MAP_3(__VA_ARGS__))
#define CPH_PY_BUILD_MAP_5(x, ...) CPH_PY_BUILD_VAL(x), CPH_PY_BUILD_EXPAND(CPH_PY_BUILD_MAP_4(__VA_ARGS__))
#define CPH_PY_BUILD_MAP_6(x, ...) CPH_PY_BUILD_VAL(x), CPH_PY_BUILD_EXPAND(CPH_PY_BUILD_MAP_5(__VA_ARGS__))
#define CPH_PY_BUILD_MAP_7(x, ...) CPH_PY_BUILD_VAL(x), CPH_PY_BUILD_EXPAND(CPH_PY_BUILD_MAP_6(__VA_ARGS__))
#define CPH_PY_BUILD_MAP_8(x, ...) CPH_PY_BUILD_VAL(x), CPH_PY_BUILD_EXPAND(CPH_PY_BUILD_MAP_7(__VA_ARGS__))
#define CPH_PY_BUILD_MAP_9(x, ...) CPH_PY_BUILD_VAL(x), CPH_PY_BUILD_EXPAND(CPH_PY_BUILD_MAP_8(__VA_ARGS__))
#define CPH_PY_BUILD_MAP_10(x, ...) CPH_PY_BUILD_VAL(x), CPH_PY_BUILD_EXPAND(CPH_PY_BUILD_MAP_9(__VA_ARGS__))
#define CPH_PY_BUILD_MAP_11(x, ...) CPH_PY_BUILD_VAL(x), CPH_PY_BUILD_EXPAND(CPH_PY_BUILD_MAP_10(__VA_ARGS__))
#define CPH_PY_BUILD_MAP_12(x, ...) CPH_PY_BUILD_VAL(x), CPH_PY_BUILD_EXPAND(CPH_PY_BUILD_MAP_11(__VA_ARGS__))
#define CPH_PY_BUILD_MAP_13(x, ...) CPH_PY_BUILD_VAL(x), CPH_PY_BUILD_EXPAND(CPH_PY_BUILD_MAP_12(__VA_ARGS__))
#define CPH_PY_BUILD_MAP_14(x, ...) CPH_PY_BUILD_VAL(x), CPH_PY_BUILD_EXPAND(CPH_PY_BUILD_MAP_13(__VA_ARGS__))
#define CPH_PY_BUILD_MAP_15(x, ...) CPH_PY_BUILD_VAL(x), CPH_PY_BUILD_EXPAND(CPH_PY_BUILD_MAP_14(__VA_ARGS__))
#define CPH_PY_BUILD_MAP_16(x, ...) CPH_PY_BUILD_VAL(x), CPH_PY_BUILD_EXPAND(CPH_PY_BUILD_MAP_15(__VA_ARGS__))
#define CPH_PY_BUILD_MAP_17(x, ...) CPH_PY_BUILD_VAL(x), CPH_PY_BUILD_EXPAND(CPH_PY_BUILD_MAP_16(__VA_ARGS__))
#define CPH_PY_BUILD_MAP_18(x, ...) CPH_PY_BUILD_VAL(x), CPH_PY_BUILD_EXPAND(CPH_PY_BUILD_MAP_17(__VA_ARGS__))
#define CPH_PY_BUILD_MAP_19(x, ...) CPH_PY_BUILD_VAL(x), CPH_PY_BUILD_EXPAND(CPH_PY_BUILD_MAP_18(__VA_ARGS__))
#define CPH_PY_BUILD_MAP_20(x, ...) CPH_PY_BUILD_VAL(x), CPH_PY_BUILD_EXPAND(CPH_PY_BUILD_MAP_19(__VA_ARGS__))
#define CPH_PY_BUILD_MAP_21(x, ...) CPH_PY_BUILD_VAL(x), CPH_PY_BUILD_EXPAND(CPH_PY_BUILD_MAP_20(__VA_ARGS__))
#define CPH_PY_BUILD_MAP_22(x, ...) CPH_PY_BUILD_VAL(x), CPH_PY_BUILD_EXPAND(CPH_PY_BUILD_MAP_21(__VA_ARGS__))
#define CPH_PY_BUILD_MAP_23(x, ...) CPH_PY_BUILD_VAL(x), CPH_PY_BUILD_EXPAND(CPH_PY_BUILD_MAP_22(__VA_ARGS__))
#define CPH_PY_BUILD_MAP_24(x, ...) CPH_PY_BUILD_VAL(x), CPH_PY_BUILD_EXPAND(CPH_PY_BUILD_MAP_23(__VA_ARGS__))
#define CPH_PY_BUILD_MAP_25(x, ...) CPH_PY_BUILD_VAL(x), CPH_PY_BUILD_EXPAND(CPH_PY_BUILD_MAP_24(__VA_ARGS__))
#define CPH_PY_BUILD_MAP_26(x, ...) CPH_PY_BUILD_VAL(x), CPH_PY_BUILD_EXPAND(CPH_PY_BUILD_MAP_25(__VA_ARGS__))
#define CPH_PY_BUILD_MAP_27(x, ...) CPH_PY_BUILD_VAL(x), CPH_PY_BUILD_EXPAND(CPH_PY_BUILD_MAP_26(__VA_ARGS__))
#define CPH_PY_BUILD_MAP_28(x, ...) CPH_PY_BUILD_VAL(x), CPH_PY_BUILD_EXPAND(CPH_PY_BUILD_MAP_27(__VA_ARGS__))
#define CPH_PY_BUILD_MAP_29(x, ...) CPH_PY_BUILD_VAL(x), CPH_PY_BUILD_EXPAND(CPH_PY_BUILD_MAP_28(__VA_ARGS__))
#define CPH_PY_BUILD_MAP_30(x, ...) CPH_PY_BUILD_VAL(x), CPH_PY_BUILD_EXPAND(CPH_PY_BUILD_MAP_29(__VA_ARGS__))
#define CPH_PY_BUILD_MAP_31(x, ...) CPH_PY_BUILD_VAL(x), CPH_PY_BUILD_EXPAND(CPH_PY_BUILD_MAP_30(__VA_ARGS__))
#define CPH_PY_BUILD_MAP_32(x, ...) CPH_PY_BUILD_VAL(x), CPH_PY_BUILD_EXPAND(CPH_PY_BUILD_MAP_31(__VA_ARGS__))

#define CPH_PY_BUILD_CONCAT_IMPL(a, b) a##b
#define CPH_PY_BUILD_CONCAT(a, b) CPH_PY_BUILD_CONCAT_IMPL(a, b)
#define CPH_PY_BUILD_MAP(...) CPH_PY_BUILD_EXPAND(CPH_PY_BUILD_CONCAT(CPH_PY_BUILD_MAP_, CPH_PY_BUILD_NARGS(__VA_ARGS__))(__VA_ARGS__))

/* --- 5. THE PUBLIC API --- */

#define CPH_PY_BUILD_KEY(parser_ptr, idx) ((parser_ptr)->hot_specs[(idx)].interned)

#define CPH_PY_BUILD_VALUE(x) CPH_PY_BUILD_VAL(x)

#define CPH_PY_BUILD_TUPLE(...)                                                                       \
    culv_py_build_pack_tuple(CPH_PY_BUILD_NARGS(__VA_ARGS__), CPH_PY_BUILD_NARGS(__VA_ARGS__)                                     \
                                             ? (PyObject *[]){__VA_OPT__(CPH_PY_BUILD_MAP(__VA_ARGS__))}     \
                                             : nullptr)

#define CPH_PY_BUILD_LIST(...)                                                                        \
    culv_py_build_pack_list(CPH_PY_BUILD_NARGS(__VA_ARGS__), CPH_PY_BUILD_NARGS(__VA_ARGS__)                                      \
                                            ? (PyObject *[]){__VA_OPT__(CPH_PY_BUILD_MAP(__VA_ARGS__))}      \
                                            : nullptr)

#define CPH_PY_BUILD_DICT(...)                                                                        \
    (CPH_PY_BUILD_NARGS(__VA_ARGS__) % 2 == 0                                                                \
         ? culv_py_build_pack_dict(CPH_PY_BUILD_NARGS(__VA_ARGS__),                                                     \
                        CPH_PY_BUILD_NARGS(__VA_ARGS__) ? (PyObject *[]){__VA_OPT__(CPH_PY_BUILD_MAP(__VA_ARGS__))}    \
                                              : nullptr)                                           \
         : culv_py_build_unsupported_type()) // This will fail to link or compile
