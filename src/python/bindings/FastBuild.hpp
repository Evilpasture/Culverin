#pragma once

#include "bindings/PythonCommon.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace CPH::Python {

[[nodiscard]] inline PyObject *BuildFromFloat(float value) noexcept {
    return PyFloat_FromDouble(static_cast<double>(value));
}

[[nodiscard]] inline PyObject *BuildFromDouble(double value) noexcept {
    return PyFloat_FromDouble(value);
}

[[nodiscard]] inline PyObject *BuildFromInt(int value) noexcept {
    return PyLong_FromLong(static_cast<long>(value));
}

[[nodiscard]] inline PyObject *BuildFromLong(long value) noexcept {
    return PyLong_FromLong(value);
}

[[nodiscard]] inline PyObject *BuildFromLongLong(long long value) noexcept {
    return PyLong_FromLongLong(value);
}

[[nodiscard]] inline PyObject *BuildFromUnsignedInt(unsigned int value) noexcept {
    return PyLong_FromUnsignedLong(static_cast<unsigned long>(value));
}

[[nodiscard]] inline PyObject *BuildFromUnsignedLong(unsigned long value) noexcept {
    return PyLong_FromUnsignedLong(value);
}

[[nodiscard]] inline PyObject *BuildFromUnsignedLongLong(unsigned long long value) noexcept {
    return PyLong_FromUnsignedLongLong(value);
}

[[nodiscard]] inline PyObject *BuildFromString(const char *value) noexcept {
    return PyUnicode_FromString(value);
}

[[nodiscard]] inline PyObject *BuildFromBool(bool value) noexcept {
    PyObject *result = value ? Py_True : Py_False;
    Py_INCREF(result);
    return result;
}

[[nodiscard]] inline PyObject *BuildFromOwnedObject(PyObject *value) noexcept {
    Py_XINCREF(value);
    return value;
}

[[nodiscard]] inline PyObject *BuildFromNone([[maybe_unused]] std::nullptr_t value) noexcept {
    Py_RETURN_NONE;
}

[[nodiscard]] inline PyObject *BuildUnsupportedValue() noexcept {
    PyErr_SetString(PyExc_TypeError, "Unsupported value type in CPH Python builder");
    return nullptr;
}

template <typename T>
[[nodiscard]] inline PyObject *BuildValue(T &&value) noexcept {
    using Value = std::remove_cvref_t<T>;

    if constexpr (std::is_same_v<Value, float>) {
        return BuildFromFloat(value);
    } else if constexpr (std::is_same_v<Value, double>) {
        return BuildFromDouble(value);
    } else if constexpr (std::is_same_v<Value, int>) {
        return BuildFromInt(value);
    } else if constexpr (std::is_same_v<Value, long>) {
        return BuildFromLong(value);
    } else if constexpr (std::is_same_v<Value, long long>) {
        return BuildFromLongLong(value);
    } else if constexpr (std::is_same_v<Value, unsigned int>) {
        return BuildFromUnsignedInt(value);
    } else if constexpr (std::is_same_v<Value, unsigned long>) {
        return BuildFromUnsignedLong(value);
    } else if constexpr (std::is_same_v<Value, unsigned long long>) {
        return BuildFromUnsignedLongLong(value);
    } else if constexpr (std::is_same_v<Value, bool>) {
        return BuildFromBool(value);
    } else if constexpr (std::is_same_v<Value, std::nullptr_t>) {
        return BuildFromNone(value);
    } else if constexpr (std::is_same_v<Value, PyObject *>) {
        return BuildFromOwnedObject(value);
    } else if constexpr (std::is_convertible_v<T &&, const char *>) {
        return BuildFromString(value);
    } else {
        return BuildUnsupportedValue();
    }
}

[[nodiscard]] inline PyObject *PackTuple(std::size_t count, PyObject **items) noexcept {
    if (count == 0) {
        return PyTuple_New(0);
    }

    PyObject *tuple = PyTuple_New(static_cast<Py_ssize_t>(count));
    if (tuple == nullptr) {
        for (std::size_t i = 0; i < count; ++i) {
            Py_XDECREF(items[i]);
        }
        return nullptr;
    }

    for (std::size_t i = 0; i < count; ++i) {
        PyObject *item = items[i];
        if (item == nullptr) {
            Py_DECREF(tuple);
            for (std::size_t j = i + 1; j < count; ++j) {
                Py_XDECREF(items[j]);
            }
            return nullptr;
        }
        PyTuple_SET_ITEM(tuple, static_cast<Py_ssize_t>(i), item);
    }
    return tuple;
}

[[nodiscard]] inline PyObject *PackList(std::size_t count, PyObject **items) noexcept {
    if (count == 0) {
        return PyList_New(0);
    }

    for (std::size_t i = 0; i < count; ++i) {
        if (items[i] == nullptr) {
            for (std::size_t j = 0; j < count; ++j) {
                Py_XDECREF(items[j]);
            }
            return nullptr;
        }
    }

    PyObject *list = PyList_New(static_cast<Py_ssize_t>(count));
    if (list == nullptr) {
        for (std::size_t i = 0; i < count; ++i) {
            Py_XDECREF(items[i]);
        }
        return nullptr;
    }

    for (std::size_t i = 0; i < count; ++i) {
        PyList_SET_ITEM(list, static_cast<Py_ssize_t>(i), items[i]);
    }
    return list;
}

[[nodiscard]] inline PyObject *PackDictionary(std::size_t count, PyObject **items) noexcept {
    if (count % 2 != 0) {
        PyErr_SetString(PyExc_SystemError, "dictionary builder requires key/value pairs");
        for (std::size_t i = 0; i < count; ++i) {
            Py_XDECREF(items[i]);
        }
        return nullptr;
    }
    if (count == 0) {
        return PyDict_New();
    }

    for (std::size_t i = 0; i < count; ++i) {
        if (items[i] == nullptr) {
            for (std::size_t j = 0; j < count; ++j) {
                Py_XDECREF(items[j]);
            }
            return nullptr;
        }
    }

    PyObject *dictionary = PyDict_New();
    if (dictionary == nullptr) {
        for (std::size_t i = 0; i < count; ++i) {
            Py_XDECREF(items[i]);
        }
        return nullptr;
    }

    for (std::size_t i = 0; i < count; i += 2) {
        if (PyDict_SetItem(dictionary, items[i], items[i + 1]) < 0) {
            Py_DECREF(dictionary);
            for (std::size_t j = 0; j < count; ++j) {
                Py_XDECREF(items[j]);
            }
            return nullptr;
        }
    }
    for (std::size_t i = 0; i < count; ++i) {
        Py_DECREF(items[i]);
    }
    return dictionary;
}

template <typename... Values>
[[nodiscard]] inline PyObject *BuildTuple(Values &&...values) noexcept {
    if constexpr (sizeof...(Values) == 0) {
        return PackTuple(0, nullptr);
    } else {
        std::array<PyObject *, sizeof...(Values)> items{
            BuildValue(std::forward<Values>(values))...,
        };
        return PackTuple(items.size(), items.data());
    }
}

template <typename... Values>
[[nodiscard]] inline PyObject *BuildList(Values &&...values) noexcept {
    if constexpr (sizeof...(Values) == 0) {
        return PackList(0, nullptr);
    } else {
        std::array<PyObject *, sizeof...(Values)> items{
            BuildValue(std::forward<Values>(values))...,
        };
        return PackList(items.size(), items.data());
    }
}

template <typename... Values>
[[nodiscard]] inline PyObject *BuildDictionary(Values &&...values) noexcept {
    static_assert(sizeof...(Values) % 2 == 0, "BuildDictionary requires key/value pairs");
    if constexpr (sizeof...(Values) == 0) {
        return PackDictionary(0, nullptr);
    } else {
        std::array<PyObject *, sizeof...(Values)> items{
            BuildValue(std::forward<Values>(values))...,
        };
        return PackDictionary(items.size(), items.data());
    }
}

} // namespace CPH::Python
