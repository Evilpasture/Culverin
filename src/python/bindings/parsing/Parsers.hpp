#pragma once

#include "physics/PhysicsParams.hpp"
#include <Python.h>

namespace CPH::Python {

int ParseVec3F32(PyObject *obj, float *x, float *y, float *z);
int ParseVec3R64(PyObject *obj, double *x, double *y, double *z);
int ParseQuatF32(PyObject *obj, float *x, float *y, float *z, float *w);
int ParseQuatR64(PyObject *obj, double *x, double *y, double *z, double *w);

[[nodiscard]] static inline int parse_vec3_direct(PyObject *obj, float *x, float *y, float *z) {
    return ParseVec3F32(obj, x, y, z);
}
[[nodiscard]] static inline int parse_vec3_direct(PyObject *obj, double *x, double *y,
                                                      double *z) {
    return ParseVec3R64(obj, x, y, z);
}

[[nodiscard]] static inline int parse_quat_direct(PyObject *obj, float *x, float *y, float *z,
                                                      float *w) {
    return ParseQuatF32(obj, x, y, z, w);
}
[[nodiscard]] static inline int parse_quat_direct(PyObject *obj, double *x, double *y,
                                                      double *z, double *w) {
    return ParseQuatR64(obj, x, y, z, w);
}

int ParsePyVec3f(PyObject *obj, CPH_Vec3f *out);
int ParsePyVec3Pos(PyObject *obj, CPH_PosStride *out);
int ParsePyVec3Aux(PyObject *obj, CPH_AuxStride *out);

[[nodiscard]] static inline int parse_py_vec3(PyObject *obj, CPH_Vec3f *out) {
    return ParsePyVec3f(obj, out);
}
[[nodiscard]] static inline int parse_py_vec3(PyObject *obj, CPH_PosStride *out) {
    return ParsePyVec3Pos(obj, out);
}
[[nodiscard]] static inline int parse_py_vec3(PyObject *obj, CPH_AuxStride *out) {
    return ParsePyVec3Aux(obj, out);
}

void ParseMotorConfig(PyObject *motor_dict, CPH_ConstraintParams *p);
int ParsePointParams(PyObject *args, CPH_ConstraintParams *p);
int ParseHingeParams(PyObject *args, CPH_ConstraintParams *p);
int ParseSliderParams(PyObject *args, CPH_ConstraintParams *p);
int ParseConeParams(PyObject *args, CPH_ConstraintParams *p);
int ParseDistanceParams(PyObject *args, CPH_ConstraintParams *p);
void ParseBodySize(PyObject *py_size, float s[4]);

[[nodiscard]] int ParseTracksToC(PyObject *py_tracks, CPH_TrackData *out_data, int *num_out);

} // namespace CPH::Python
