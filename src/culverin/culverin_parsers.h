#pragma once
#include "culverin_types.h"
#include <Python.h>

typedef struct {
    float p[4];
} ShapeParams;

typedef struct {
    uint32_t *indices;
    uint32_t count;
    uint32_t driven_idx;
} TrackData;

typedef struct {
    float px, py, pz, ax, ay, az;
    float limit_min, limit_max, half_cone_angle;
    bool has_motor;
    int motor_type;
    float motor_target, max_torque, frequency, damping, max_force;
} ConstraintParams;

// Direct variable extraction (Fast path)
int parse_vec3_f32(PyObject *obj, float *x, float *y, float *z);
int parse_vec3_r64(PyObject *obj, double *x, double *y, double *z);
int parse_quat_f32(PyObject *obj, float *x, float *y, float *z, float *w);
int parse_quat_r64(PyObject *obj, double *x, double *y, double *z, double *w);

// Type-safe dispatchers
#define parse_vec3_direct(obj, x, y, z)                                                            \
    _Generic((x), float *: parse_vec3_f32, double *: parse_vec3_r64)(obj, x, y, z)

#define parse_quat_direct(obj, x, y, z, w)                                                         \
    _Generic((x), float *: parse_quat_f32, double *: parse_quat_r64)(obj, x, y, z, w)

int parse_py_vec3f(PyObject *obj, Vec3f *out);
int parse_py_vec3_pos(PyObject *obj, PosStride *out);
int parse_py_vec3_aux(PyObject *obj, AuxStride *out);

#define parse_py_vec3(obj, out)                                                                    \
    _Generic((out),                                                                                \
        PosStride *: parse_py_vec3_pos,                                                            \
        const PosStride *: parse_py_vec3_pos,                                                      \
        AuxStride *: parse_py_vec3_aux,                                                            \
        const AuxStride *: parse_py_vec3_aux,                                                      \
        Vec3f *: parse_py_vec3f,                                                                   \
        const Vec3f *: parse_py_vec3f)(obj, out)

void parse_motor_config(PyObject *motor_dict, ConstraintParams *p);
int parse_point_params(PyObject *args, ConstraintParams *p);
int parse_hinge_params(PyObject *args, ConstraintParams *p);
int parse_slider_params(PyObject *args, ConstraintParams *p);
int parse_cone_params(PyObject *args, ConstraintParams *p);
int parse_distance_params(PyObject *args, ConstraintParams *p);
void parse_body_size(PyObject *py_size, float s[4]);

[[nodiscard]] int parse_tracks_to_c(PyObject *py_tracks, TrackData *out_data, int *num_out);