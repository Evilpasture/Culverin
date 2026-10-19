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

int ParsePyVec3f(PyObject *obj, Vec3f *out);
int ParsePyVec3Pos(PyObject *obj, PosStride *out);
int ParsePyVec3Aux(PyObject *obj, AuxStride *out);

[[nodiscard]] static inline int parse_py_vec3(PyObject *obj, Vec3f *out) {
    return ParsePyVec3f(obj, out);
}
[[nodiscard]] static inline int parse_py_vec3(PyObject *obj, PosStride *out) {
    return ParsePyVec3Pos(obj, out);
}
[[nodiscard]] static inline int parse_py_vec3(PyObject *obj, AuxStride *out) {
    return ParsePyVec3Aux(obj, out);
}

void ParseMotorConfig(PyObject *motor_dict, ConstraintParams *p);
int ParsePointParams(PyObject *args, ConstraintParams *p);
int ParseHingeParams(PyObject *args, ConstraintParams *p);
int ParseSliderParams(PyObject *args, ConstraintParams *p);
int ParseConeParams(PyObject *args, ConstraintParams *p);
int ParseDistanceParams(PyObject *args, ConstraintParams *p);
void ParseBodySize(PyObject *py_size, float s[4]);

[[nodiscard]] int ParseTracksToC(PyObject *py_tracks, TrackData *out_data, int *num_out);

} // namespace CPH::Python
