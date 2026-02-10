#pragma once
#include <Python.h>
#include "culverin.h"

typedef struct {
    float x;
    float y;
    float z;
} Vec3f; // General Vec3f

float get_py_float_attr(PyObject *obj, const char *name,
                               float default_val);
int parse_py_vec3(PyObject *obj, Vec3f *out);