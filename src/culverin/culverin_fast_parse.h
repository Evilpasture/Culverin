#pragma once

#include <Python.h>
#include <stdbool.h>
#include <stdint.h>

/** --- 1. THE BASICS --- **/
static inline void _fp_conv_float(PyObject *o, void *t)  { *(float *)t = (float)PyFloat_AsDouble(o); }
static inline void _fp_conv_double(PyObject *o, void *t) { *(double *)t = PyFloat_AsDouble(o); }
static inline void _fp_conv_int(PyObject *o, void *t)    { *(int *)t = (int)PyLong_AsLong(o); }
static inline void _fp_conv_u32(PyObject *o, void *t)    { *(uint32_t *)t = (uint32_t)PyLong_AsUnsignedLongMask(o); }
static inline void _fp_conv_u64(PyObject *o, void *t)    { *(uint64_t *)t = PyLong_AsUnsignedLongLong(o); }
static inline void _fp_conv_bool(PyObject *o, void *t)   { *(bool *)t = (bool)PyObject_IsTrue(o); }
static inline void _fp_conv_pyobj(PyObject *o, void *t)  { *(PyObject **)t = o; }

#define FP_GET_CONVERTER(T) _Generic((T), \
    float: _fp_conv_float, double: _fp_conv_double, int: _fp_conv_int, \
    uint32_t: _fp_conv_u32, uint64_t: _fp_conv_u64, bool: _fp_conv_bool, \
    PyObject *: _fp_conv_pyobj)

typedef struct {
    const char *name;
    PyObject *interned;
    void (*convert)(PyObject *, void *);
} FastArgSpec;

#define FP_ARG(name_str, var) { .name = name_str, .interned = nullptr, \
    .convert = FP_GET_CONVERTER((typeof_unqual(var)){0}) }

/** --- 2. VECTORCALL ENGINE (Modern) --- **/
[[nodiscard]]
static inline bool _FastParse_Vector(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames, 
                                    const FastArgSpec *specs, void **targets, size_t count) {
    size_t pos_limit = ((size_t)nargs < count) ? (size_t)nargs : count;
    for (size_t i = 0; i < pos_limit; ++i) {
        if (args[i] != Py_None) specs[i].convert(args[i], targets[i]);
    }
    if (kwnames) {
        Py_ssize_t nkw = PyTuple_GET_SIZE(kwnames);
        PyObject *const *kw_vals = args + nargs;
        for (Py_ssize_t i = 0; i < nkw; ++i) {
            PyObject *key = PyTuple_GET_ITEM(kwnames, i);
            for (size_t j = 0; j < count; ++j) {
                if (key == specs[j].interned) {
                    if (kw_vals[i] != Py_None) specs[j].convert(kw_vals[i], targets[j]);
                    goto next_kw;
                }
            }
            PyErr_Format(PyExc_TypeError, "unexpected keyword argument '%U'", key);
            return false;
            next_kw:;
        }
    }
    return !PyErr_Occurred();
}

/** --- 3. LEGACY ENGINE (With Keyword Validation) --- **/
[[nodiscard]]
static inline bool _FastParse_Legacy(PyObject *args, PyObject *kwargs, const FastArgSpec *specs, 
                                    void **targets, size_t count) {
    if (args) {
        Py_ssize_t nargs = PyTuple_GET_SIZE(args);
        size_t pos_limit = ((size_t)nargs < count) ? (size_t)nargs : count;
        for (size_t i = 0; i < pos_limit; ++i) {
            PyObject *item = PyTuple_GET_ITEM(args, i);
            if (item != Py_None) specs[i].convert(item, targets[i]);
        }
    }
    
    if (kwargs) {
        Py_ssize_t dict_size = PyDict_Size(kwargs);
        if (dict_size > 0) {
            Py_ssize_t matched = 0;
            for (size_t i = 0; i < count; ++i) {
                // Use interned string for O(1) hash lookup
                PyObject *val = PyDict_GetItemWithError(kwargs, specs[i].interned);
                if (val) {
                    matched++;
                    if (val != Py_None) specs[i].convert(val, targets[i]);
                } else if (PyErr_Occurred()) return false;
            }

            // The "Keyword Trap" Check:
            if (matched < dict_size) {
                PyObject *key, *value;
                Py_ssize_t pos = 0;
                while (PyDict_Next(kwargs, &pos, &key, &value)) {
                    bool found = false;
                    for (size_t j = 0; j < count; ++j) {
                        if (key == specs[j].interned) { found = true; break; }
                        // Fallback for non-interned keys in kwargs
                        if (PyUnicode_Compare(key, specs[j].interned) == 0) { found = true; break; }
                    }
                    if (!found) {
                        PyErr_Format(PyExc_TypeError, "unexpected keyword argument '%U'", key);
                        return false;
                    }
                }
            }
        }
    }
    return !PyErr_Occurred();
}

/** --- 4. THE UNIFIED MACRO --- 
 * We use a specialized Dispatch struct to handle the different argument counts 
 * safely without relying on variadic macro "guessing".
**/

#define FastParse_Unified(arg1, arg2, ...) _Generic((arg1), \
    PyObject *const * : _FastParse_Vector, \
    PyObject *        : _FastParse_Legacy  \
)(arg1, arg2, __VA_ARGS__)

static inline void FastParse_Init(FastArgSpec *specs, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (specs[i].name) specs[i].interned = PyUnicode_InternFromString(specs[i].name);
    }
}
