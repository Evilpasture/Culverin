#pragma once
#include <Python.h>

static inline float py_to_float(PyObject *i) { return (float)PyFloat_AsDouble(i); }
static inline double py_to_double(PyObject *i) { return PyFloat_AsDouble(i); }
static inline int py_to_int(PyObject *i) { return (int)PyLong_AsLong(i); }
static inline long py_to_long(PyObject *i) { return PyLong_AsLong(i); }
static inline const char *py_to_str(PyObject *i) { return PyUnicode_AsUTF8(i); }

#define DEFINE_ATTR_GETTER(suffix, type, conv)                                                     \
    [[nodiscard]] static inline type get_py_attr_##suffix(PyObject *obj, const char *n,            \
                                                          type def) {                              \
        if (obj == nullptr || obj == Py_None)                                                      \
            return def;                                                                            \
        type result    = def;                                                                      \
        PyObject *attr = PyObject_GetAttrString(obj, n);                                           \
        if (attr) {                                                                                \
            type val = conv(attr);                                                                 \
            if (!PyErr_Occurred())                                                                 \
                result = val;                                                                      \
            Py_DECREF(attr);                                                                       \
        }                                                                                          \
        PyErr_Clear();                                                                             \
        return result;                                                                             \
    }

#define DEFINE_DICT_GETTER(suffix, type, conv)                                                     \
    [[nodiscard]] static inline type get_py_dict_##suffix(PyObject *dict, const char *k,           \
                                                          type def) {                              \
        if (dict == nullptr || !PyDict_Check(dict))                                                \
            return def;                                                                            \
        PyObject *item = PyDict_GetItemString(dict, k);                                            \
        if (item) {                                                                                \
            type val = conv(item);                                                                 \
            if (!PyErr_Occurred())                                                                 \
                return val;                                                                        \
        }                                                                                          \
        PyErr_Clear();                                                                             \
        return def;                                                                                \
    }

DEFINE_ATTR_GETTER(f, float, py_to_float)
DEFINE_ATTR_GETTER(d, double, py_to_double)
DEFINE_ATTR_GETTER(i, int, py_to_int)
DEFINE_ATTR_GETTER(l, long, py_to_long)
DEFINE_ATTR_GETTER(s, const char *, py_to_str)

DEFINE_DICT_GETTER(f, float, py_to_float)
DEFINE_DICT_GETTER(d, double, py_to_double)
DEFINE_DICT_GETTER(i, int, py_to_int)
DEFINE_DICT_GETTER(l, long, py_to_long)
DEFINE_DICT_GETTER(s, const char *, py_to_str)

#define get_py_attr(obj, name, def)                                                                \
    _Generic((def),                                                                                \
        float: get_py_attr_f,                                                                      \
        double: get_py_attr_d,                                                                     \
        int: get_py_attr_i,                                                                        \
        long: get_py_attr_l,                                                                       \
        char *: get_py_attr_s,                                                                     \
        const char *: get_py_attr_s)(obj, name, def)

#define get_py_dict(dict, key, def)                                                                \
    _Generic((def),                                                                                \
        float: get_py_dict_f,                                                                      \
        double: get_py_dict_d,                                                                     \
        int: get_py_dict_i,                                                                        \
        long: get_py_dict_l,                                                                       \
        char *: get_py_dict_s,                                                                     \
        const char *: get_py_dict_s)(dict, key, def)

#define CULV_CAST(m) (PyCFunction)(void (*)(void))(m)

#define CULV_FEAT(prefix, name, method_type)                                                       \
    {.ml_name  = #name,                                                                            \
     .ml_meth  = CULV_CAST(prefix##_##name),                                                       \
     .ml_flags = (method_type),                                                                    \
     .ml_doc   = nullptr} // Initialized to nullptr to be filled by stitcher

#define CULV_FEAT_INTERNAL(prefix, name, method_type)                                              \
    {.ml_name  = "_" #name,                                                                        \
     .ml_meth  = (PyCFunction)prefix##_##name,                                                     \
     .ml_flags = (method_type),                                                                    \
     .ml_doc   = nullptr}

// Getter/Property macro - concise initialization
#define GETSET(name_str, getter_func)                                                              \
    {.name    = (name_str),                                                                        \
     .get     = (getter)(getter_func),                                                             \
     .set     = nullptr,                                                                           \
     .doc     = nullptr,                                                                           \
     .closure = nullptr}

extern PyType_Spec PhysicsWorld_spec;
extern PyType_Spec Vehicle_spec;
extern PyType_Spec RagdollSettings_spec;
extern PyType_Spec Ragdoll_spec;
extern PyType_Spec Skeleton_spec;
extern PyType_Spec SoftBodySharedSettings_spec;
extern PyType_Spec Registry_spec;
extern PyType_Spec Ship_spec;