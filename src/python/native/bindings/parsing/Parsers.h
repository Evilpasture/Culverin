#pragma once
#include "physics/PhysicsParams.h"
#include <Python.h>

// Direct variable extraction (Fast path)
int CPH_ParseVec3F32(PyObject *obj, float *x, float *y, float *z);
int CPH_ParseVec3R64(PyObject *obj, double *x, double *y, double *z);
int CPH_ParseQuatF32(PyObject *obj, float *x, float *y, float *z, float *w);
int CPH_ParseQuatR64(PyObject *obj, double *x, double *y, double *z, double *w);

// Type-safe dispatchers
#define CPH_parse_vec3_direct(obj, x, y, z)                                                            \
    _Generic((x), float *: CPH_ParseVec3F32, double *: CPH_ParseVec3R64)(obj, x, y, z)

#define CPH_parse_quat_direct(obj, x, y, z, w)                                                         \
    _Generic((x), float *: CPH_ParseQuatF32, double *: CPH_ParseQuatR64)(obj, x, y, z, w)

int CPH_ParsePyVec3f(PyObject *obj, CPH_Vec3f *out);
int CPH_ParsePyVec3Pos(PyObject *obj, CPH_PosStride *out);
int CPH_ParsePyVec3Aux(PyObject *obj, CPH_AuxStride *out);

#define CPH_parse_py_vec3(obj, out)                                                                    \
    _Generic((out),                                                                                \
        CPH_PosStride *: CPH_ParsePyVec3Pos,                                                            \
        const CPH_PosStride *: CPH_ParsePyVec3Pos,                                                      \
        CPH_AuxStride *: CPH_ParsePyVec3Aux,                                                            \
        const CPH_AuxStride *: CPH_ParsePyVec3Aux,                                                      \
        CPH_Vec3f *: CPH_ParsePyVec3f,                                                                   \
        const CPH_Vec3f *: CPH_ParsePyVec3f)(obj, out)

void CPH_ParseMotorConfig(PyObject *motor_dict, CPH_ConstraintParams *p);
int CPH_ParsePointParams(PyObject *args, CPH_ConstraintParams *p);
int CPH_ParseHingeParams(PyObject *args, CPH_ConstraintParams *p);
int CPH_ParseSliderParams(PyObject *args, CPH_ConstraintParams *p);
int CPH_ParseConeParams(PyObject *args, CPH_ConstraintParams *p);
int CPH_ParseDistanceParams(PyObject *args, CPH_ConstraintParams *p);
void CPH_ParseBodySize(PyObject *py_size, float s[4]);

[[nodiscard]] int CPH_ParseTracksToC(PyObject *py_tracks, CPH_TrackData *out_data, int *num_out);