#pragma once

#include <Python.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

/* --- 0. CONFIGURATION & COMPILER COMPATIBILITY --- */

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

[[nodiscard]] CPH_PY_BUILD_FORCE_INLINE static PyObject *CPH_PyBuildFromFloat(float v) {
    return PyFloat_FromDouble((double)v);
}
[[nodiscard]] CPH_PY_BUILD_FORCE_INLINE static PyObject *CPH_PyBuildFromDouble(double v) {
    return PyFloat_FromDouble(v);
}
[[nodiscard]] CPH_PY_BUILD_FORCE_INLINE static PyObject *CPH_PyBuildFromInt(int v) {
    return PyLong_FromLong((long)v);
}
[[nodiscard]] CPH_PY_BUILD_FORCE_INLINE static PyObject *CPH_PyBuildFromLong(long v) {
    return PyLong_FromLong(v);
}
[[nodiscard]] CPH_PY_BUILD_FORCE_INLINE static PyObject *CPH_PyBuildFromLonglong(long long v) {
    return PyLong_FromLongLong(v);
}
[[nodiscard]] CPH_PY_BUILD_FORCE_INLINE static PyObject *CPH_PyBuildFromU32(uint32_t v) {
    return PyLong_FromUnsignedLong((unsigned long)v);
}
[[nodiscard]] CPH_PY_BUILD_FORCE_INLINE static PyObject *CPH_PyBuildFromU64(uint64_t v) {
    return PyLong_FromUnsignedLongLong((unsigned long long)v);
}
[[nodiscard]] CPH_PY_BUILD_FORCE_INLINE static PyObject *CPH_PyBuildFromStr(const char *v) {
    return PyUnicode_FromString(v);
}
[[nodiscard]] CPH_PY_BUILD_FORCE_INLINE static PyObject *CPH_PyBuildFromBool(bool v) {
    PyObject *res = (int)v ? Py_True : Py_False;
    Py_INCREF(res);
    return res;
}
[[nodiscard]] CPH_PY_BUILD_FORCE_INLINE static PyObject *CPH_PyBuildIncref(PyObject *v) {
    Py_XINCREF(v);
    return v;
}
[[nodiscard]] CPH_PY_BUILD_FORCE_INLINE static PyObject *
CPH_PyBuildFromNone([[maybe_unused]] std::nullptr_t v) {
    Py_RETURN_NONE;
}

/* --- 2. C++23 VALUE CONVERSION --- */

[[nodiscard]] CPH_PY_BUILD_FORCE_INLINE static PyObject *CPH_PyBuildUnsupportedType() {
    PyErr_SetString(PyExc_TypeError, "Unsupported value type in CPH Python builder");
    return nullptr;
}

template <typename T>
[[nodiscard]] CPH_PY_BUILD_FORCE_INLINE static PyObject *CPH_PyBuildValue(T &&value) {
    using Value = std::remove_cvref_t<T>;

    if constexpr (std::is_same_v<Value, float>) {
        return CPH_PyBuildFromFloat(value);
    } else if constexpr (std::is_same_v<Value, double>) {
        return CPH_PyBuildFromDouble(value);
    } else if constexpr (std::is_same_v<Value, int>) {
        return CPH_PyBuildFromInt(value);
    } else if constexpr (std::is_same_v<Value, long>) {
        return CPH_PyBuildFromLong(value);
    } else if constexpr (std::is_same_v<Value, long long>) {
        return CPH_PyBuildFromLonglong(value);
    } else if constexpr (std::is_same_v<Value, unsigned int>) {
        return CPH_PyBuildFromU32(value);
    } else if constexpr (std::is_same_v<Value, unsigned long>) {
        return PyLong_FromUnsignedLong(value);
    } else if constexpr (std::is_same_v<Value, unsigned long long>) {
        return CPH_PyBuildFromU64(value);
    } else if constexpr (std::is_same_v<Value, bool>) {
        return CPH_PyBuildFromBool(value);
    } else if constexpr (std::is_same_v<Value, std::nullptr_t>) {
        return CPH_PyBuildFromNone(value);
    } else if constexpr (std::is_same_v<Value, PyObject *>) {
        return CPH_PyBuildIncref(value);
    } else if constexpr (std::is_convertible_v<T &&, const char *>) {
        return CPH_PyBuildFromStr(value);
    } else {
        return CPH_PyBuildUnsupportedType();
    }
}

#define CPH_PY_BUILD_VAL(x) CPH_PyBuildValue((x))

/* --- 3. THE CONTAINER PACKERS --- */

[[nodiscard]] [[gnu::always_inline]]
static inline PyObject *CPH_PyBuildPackTuple(size_t n, PyObject **arr) {
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

[[nodiscard]] CPH_PY_BUILD_FORCE_INLINE static PyObject *CPH_PyBuildPackList(size_t n,
                                                                                  PyObject **arr) {
    if (CPH_PY_BUILD_UNLIKELY(n == 0)) {
        return PyList_New(0);
    }
    PyObject *l = nullptr;
#pragma unroll 4
    for (size_t i = 0; i < n; i++) {
        if (CPH_PY_BUILD_UNLIKELY(!arr[i])) {
            goto error;
        }
    }

    l = PyList_New((Py_ssize_t)n);
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

[[nodiscard]] CPH_PY_BUILD_FORCE_INLINE static PyObject *CPH_PyBuildPackDict(size_t n,
                                                                                  PyObject **arr) {
    if (CPH_PY_BUILD_UNLIKELY(n == 0)) {
        return PyDict_New();
    }
    PyObject *d = nullptr;
#pragma unroll 4
    for (size_t i = 0; i < n; i++) {
        if (CPH_PY_BUILD_UNLIKELY(!arr[i])) {
            goto error;
        }
    }

    d = PyDict_New();
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

/* --- 4. C++23 CONTAINER BUILDERS --- */

template <typename... Ts>
[[nodiscard]] CPH_PY_BUILD_FORCE_INLINE static PyObject *CPH_PyBuildPackTupleValues(Ts &&...values) {
    if constexpr (sizeof...(Ts) == 0) {
        return CPH_PyBuildPackTuple(0, nullptr);
    } else {
        std::array<PyObject *, sizeof...(Ts)> converted{
            CPH_PyBuildValue(std::forward<Ts>(values))...,
        };
        return CPH_PyBuildPackTuple(converted.size(), converted.data());
    }
}

template <typename... Ts>
[[nodiscard]] CPH_PY_BUILD_FORCE_INLINE static PyObject *CPH_PyBuildPackListValues(Ts &&...values) {
    if constexpr (sizeof...(Ts) == 0) {
        return CPH_PyBuildPackList(0, nullptr);
    } else {
        std::array<PyObject *, sizeof...(Ts)> converted{
            CPH_PyBuildValue(std::forward<Ts>(values))...,
        };
        return CPH_PyBuildPackList(converted.size(), converted.data());
    }
}

template <typename... Ts>
[[nodiscard]] CPH_PY_BUILD_FORCE_INLINE static PyObject *CPH_PyBuildPackDictValues(Ts &&...values) {
    static_assert(sizeof...(Ts) % 2 == 0, "CPH_PY_BUILD_DICT requires key/value pairs");
    if constexpr (sizeof...(Ts) == 0) {
        return CPH_PyBuildPackDict(0, nullptr);
    } else {
        std::array<PyObject *, sizeof...(Ts)> converted{
            CPH_PyBuildValue(std::forward<Ts>(values))...,
        };
        return CPH_PyBuildPackDict(converted.size(), converted.data());
    }
}

/* --- 5. THE PUBLIC API --- */

#define CPH_PY_BUILD_KEY(parser_ptr, idx) ((parser_ptr)->hot_specs[(idx)].interned)
#define CPH_PY_BUILD_VALUE(x) CPH_PY_BUILD_VAL(x)
#define CPH_PY_BUILD_TUPLE(...) CPH_PyBuildPackTupleValues(__VA_ARGS__)
#define CPH_PY_BUILD_LIST(...) CPH_PyBuildPackListValues(__VA_ARGS__)
#define CPH_PY_BUILD_DICT(...) CPH_PyBuildPackDictValues(__VA_ARGS__)
