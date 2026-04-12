#pragma once
#include <Python.h>

static inline float py_to_float(PyObject *i) { return (float)PyFloat_AsDouble(i); }
static inline double py_to_double(PyObject *i) { return PyFloat_AsDouble(i); }
static inline int py_to_int(PyObject *i) { return (int)PyLong_AsLong(i); }
static inline long py_to_long(PyObject *i) { return PyLong_AsLong(i); }
static inline const char* py_to_str(PyObject *i) { return PyUnicode_AsUTF8(i); }

#define DEFINE_ATTR_GETTER(suffix, type, conv)                                 \
[[nodiscard]] static inline type get_py_attr_##suffix(PyObject *obj, const char *n, type def) { \
    if (obj == nullptr || obj == Py_None) return def;                          \
    type result = def;                                                         \
    PyObject *attr = PyObject_GetAttrString(obj, n);                           \
    if (attr) {                                                                \
        type val = conv(attr);                                                 \
        if (!PyErr_Occurred()) result = val;                                   \
        Py_DECREF(attr);                                                       \
    }                                                                          \
    PyErr_Clear(); return result;                                              \
}

#define DEFINE_DICT_GETTER(suffix, type, conv)                                 \
[[nodiscard]] static inline type get_py_dict_##suffix(PyObject *dict, const char *k, type def) { \
    if (dict == nullptr || !PyDict_Check(dict)) return def;                    \
    PyObject *item = PyDict_GetItemString(dict, k);                            \
    if (item) {                                                                \
        type val = conv(item);                                                 \
        if (!PyErr_Occurred()) return val;                                     \
    }                                                                          \
    PyErr_Clear(); return def;                                                 \
}

DEFINE_ATTR_GETTER(f, float, py_to_float)
DEFINE_ATTR_GETTER(d, double, py_to_double)
DEFINE_ATTR_GETTER(i, int, py_to_int)
DEFINE_ATTR_GETTER(l, long, py_to_long)
DEFINE_ATTR_GETTER(s, const char*, py_to_str)

DEFINE_DICT_GETTER(f, float, py_to_float)
DEFINE_DICT_GETTER(d, double, py_to_double)
DEFINE_DICT_GETTER(i, int, py_to_int)
DEFINE_DICT_GETTER(l, long, py_to_long)
DEFINE_DICT_GETTER(s, const char*, py_to_str)

#define get_py_attr(obj, name, def) _Generic((def), \
    float: get_py_attr_f, double: get_py_attr_d,    \
    int: get_py_attr_i, long: get_py_attr_l,        \
    char*: get_py_attr_s, const char*: get_py_attr_s)(obj, name, def)

#define get_py_dict(dict, key, def) _Generic((def), \
    float: get_py_dict_f, double: get_py_dict_d,    \
    int: get_py_dict_i, long: get_py_dict_l,        \
    char*: get_py_dict_s, const char*: get_py_dict_s)(dict, key, def)