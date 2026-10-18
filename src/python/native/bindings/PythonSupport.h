#pragma once
#include <Python.h>

#include <type_traits>

static inline float py_to_float(PyObject *obj) { return static_cast<float>(PyFloat_AsDouble(obj)); }
static inline double py_to_double(PyObject *obj) { return PyFloat_AsDouble(obj); }
static inline int py_to_int(PyObject *obj) { return static_cast<int>(PyLong_AsLong(obj)); }
static inline long py_to_long(PyObject *obj) { return PyLong_AsLong(obj); }
static inline const char *py_to_str(PyObject *obj) { return PyUnicode_AsUTF8(obj); }

template <typename T>
[[nodiscard]] static inline T py_to_value(PyObject *obj) {
    if constexpr (std::is_same_v<T, float>) {
        return py_to_float(obj);
    } else if constexpr (std::is_same_v<T, double>) {
        return py_to_double(obj);
    } else if constexpr (std::is_same_v<T, int>) {
        return py_to_int(obj);
    } else if constexpr (std::is_same_v<T, long>) {
        return py_to_long(obj);
    } else if constexpr (std::is_same_v<T, const char *>) {
        return py_to_str(obj);
    } else {
        static_assert(std::is_same_v<T, void>, "Unsupported Python default-value type");
    }
}

template <typename T>
[[nodiscard]] static inline T CPH_get_py_attr(PyObject *obj, const char *name, T default_value) {
    if (obj == nullptr || obj == Py_None) {
        return default_value;
    }

    T result = default_value;
    PyObject *attr = PyObject_GetAttrString(obj, name);
    if (attr != nullptr) {
        const T value = py_to_value<T>(attr);
        if (!PyErr_Occurred()) {
            result = value;
        }
        Py_DECREF(attr);
    }
    PyErr_Clear();
    return result;
}

template <typename T>
[[nodiscard]] static inline T CPH_get_py_dict(PyObject *dict, const char *key, T default_value) {
    if (dict == nullptr || !PyDict_Check(dict)) {
        return default_value;
    }

    PyObject *item = PyDict_GetItemString(dict, key);
    if (item != nullptr) {
        const T value = py_to_value<T>(item);
        if (!PyErr_Occurred()) {
            return value;
        }
    }
    PyErr_Clear();
    return default_value;
}

#define CPH_CAST(m) (PyCFunction)(void (*)(void))(m)

#define CPH_FEAT(prefix, name, method_type)                                                       \
    {.ml_name  = #name,                                                                            \
     .ml_meth  = CPH_CAST(prefix),                                                       \
     .ml_flags = (method_type),                                                                    \
     .ml_doc   = nullptr} // Initialized to nullptr to be filled by stitcher

#define CPH_FEAT_INTERNAL(prefix, name, method_type)                                              \
    {.ml_name  = "_" #name,                                                                        \
     .ml_meth  = CPH_CAST(prefix),                                                     \
     .ml_flags = (method_type),                                                                    \
     .ml_doc   = nullptr}

// Getter/Property macro - concise initialization
#define CPH_GETSET(name_str, getter_func)                                                              \
    {.name    = (name_str),                                                                        \
     .get     = (getter)(getter_func),                                                             \
     .set     = nullptr,                                                                           \
     .doc     = nullptr,                                                                           \
     .closure = nullptr}

extern PyType_Spec CPH_PyWorld_Spec;
extern PyType_Spec Vehicle_spec;
extern PyType_Spec RagdollSettings_spec;
extern PyType_Spec Ragdoll_spec;
extern PyType_Spec Skeleton_spec;
extern PyType_Spec SoftBodySharedSettings_spec;
extern PyType_Spec Registry_spec;
extern PyType_Spec Ship_spec;