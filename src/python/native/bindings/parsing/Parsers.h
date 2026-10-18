#pragma once

#include "physics/PhysicsParams.h"
#include <Python.h>

int CPH_ParseVec3F32(PyObject *obj, float *x, float *y, float *z);
int CPH_ParseVec3R64(PyObject *obj, double *x, double *y, double *z);
int CPH_ParseQuatF32(PyObject *obj, float *x, float *y, float *z, float *w);
int CPH_ParseQuatR64(PyObject *obj, double *x, double *y, double *z, double *w);

[[nodiscard]] static inline int CPH_parse_vec3_direct(PyObject *obj, float *x, float *y, float *z) {
    return CPH_ParseVec3F32(obj, x, y, z);
}
[[nodiscard]] static inline int CPH_parse_vec3_direct(PyObject *obj, double *x, double *y,
                                                      double *z) {
    return CPH_ParseVec3R64(obj, x, y, z);
}

[[nodiscard]] static inline int CPH_parse_quat_direct(PyObject *obj, float *x, float *y, float *z,
                                                      float *w) {
    return CPH_ParseQuatF32(obj, x, y, z, w);
}
[[nodiscard]] static inline int CPH_parse_quat_direct(PyObject *obj, double *x, double *y,
                                                      double *z, double *w) {
    return CPH_ParseQuatR64(obj, x, y, z, w);
}

int CPH_ParsePyVec3f(PyObject *obj, CPH_Vec3f *out);
int CPH_ParsePyVec3Pos(PyObject *obj, CPH_PosStride *out);
int CPH_ParsePyVec3Aux(PyObject *obj, CPH_AuxStride *out);

[[nodiscard]] static inline int CPH_parse_py_vec3(PyObject *obj, CPH_Vec3f *out) {
    return CPH_ParsePyVec3f(obj, out);
}
[[nodiscard]] static inline int CPH_parse_py_vec3(PyObject *obj, CPH_PosStride *out) {
    return CPH_ParsePyVec3Pos(obj, out);
}
[[nodiscard]] static inline int CPH_parse_py_vec3(PyObject *obj, CPH_AuxStride *out) {
    return CPH_ParsePyVec3Aux(obj, out);
}

void CPH_ParseMotorConfig(PyObject *motor_dict, CPH_ConstraintParams *p);
int CPH_ParsePointParams(PyObject *args, CPH_ConstraintParams *p);
int CPH_ParseHingeParams(PyObject *args, CPH_ConstraintParams *p);
int CPH_ParseSliderParams(PyObject *args, CPH_ConstraintParams *p);
int CPH_ParseConeParams(PyObject *args, CPH_ConstraintParams *p);
int CPH_ParseDistanceParams(PyObject *args, CPH_ConstraintParams *p);
void CPH_ParseBodySize(PyObject *py_size, float s[4]);

[[nodiscard]] int CPH_ParseTracksToC(PyObject *py_tracks, CPH_TrackData *out_data, int *num_out);
