#pragma once
#include "culverin_tracked_vehicle.h"
#include <Python.h>
#include <stdbool.h>

typedef struct {
    float x;
    float y;
    float z;
} Vec3f; // General Vec3f

typedef struct {
    float p[4];
} ShapeParams; // 4 floats to prevent over-read

// --- Unified Parameter Struct ---
typedef struct {
    float px, py, pz; // Pivot
    float ax, ay, az; // Axis
    float limit_min;  // Limits
    float limit_max;
    float half_cone_angle;

    // --- Motor Settings ---
    bool has_motor;
    int motor_type;     // 0=Off, 1=Velocity, 2=Position
    float motor_target; // Target Velocity or Target Position
    float max_torque;   // Max Force/Torque
    float frequency;    // Spring stiffness (0 = stiff)
    float damping;      // Spring damping
} ConstraintParams;

// Direct variable extraction (Fast path)
// Internal specialized versions
int parse_vec3_f32(PyObject *obj, float *x, float *y, float *z);
int parse_vec3_r64(PyObject *obj, double *x, double *y, double *z);

// Type-safe dispatcher
#define parse_vec3_direct(obj, x, y, z)                                                            \
    _Generic((x), float *: parse_vec3_f32, double *: parse_vec3_r64)(obj, x, y, z)
int parse_quat_f32(PyObject *obj, float *x, float *y, float *z, float *w);
#define parse_quat_direct(obj, x, y, z, w) parse_quat_f32(obj, x, y, z, w)

// The global parser keys
void init_body_parser(void);

float get_py_float_attr(PyObject *obj, const char *name, float default_val);
int parse_py_vec3f(PyObject *obj, Vec3f *out);
int parse_py_vec3_pos(PyObject *obj, PosStride *out);
int parse_py_vec3_aux(PyObject *obj, AuxStride *out);
// NOLINTNEXTLINE(readability-identifier-naming)
#define parse_py_vec3(obj, out)                                                                    \
    _Generic((out),                                                                                \
        PosStride *: parse_py_vec3_pos,                                                            \
        const PosStride *: parse_py_vec3_pos,                                                      \
        AuxStride *: parse_py_vec3_aux,                                                            \
        const AuxStride *: parse_py_vec3_aux,                                                      \
        Vec3f *: parse_py_vec3f,                                                                   \
        const Vec3f *: parse_py_vec3f)(obj, out)

void parse_shape_params(PyObject *py_size, float s[4]);

float get_py_dict_float(PyObject *dict, const char *key, float default_val);

void parse_motor_config(PyObject *motor_dict, ConstraintParams *p);

int parse_point_params(PyObject *args, ConstraintParams *p);

int parse_hinge_params(PyObject *args, ConstraintParams *p);

int parse_slider_params(PyObject *args, ConstraintParams *p);

int parse_cone_params(PyObject *args, ConstraintParams *p);

int parse_distance_params(PyObject *args, ConstraintParams *p);

void parse_body_size(PyObject *py_size, float s[4]);
CULV_NODISCARD
int parse_tracks_to_c(PyObject *py_tracks, TrackData *out_data, int *num_out);

static inline PyObject *find_arg(Py_ssize_t pos_idx, PyObject *target_key, PyObject *const *args,
                                 Py_ssize_t nargs, PyObject *kwnames) {
    if (pos_idx < nargs) {
        return args[pos_idx];
    }
    if (kwnames) {
        Py_ssize_t nkw = PyTuple_GET_SIZE(kwnames);
        for (Py_ssize_t i = 0; i < nkw; i++) {
            if (PyTuple_GET_ITEM(kwnames, i) == target_key) {
                return args[nargs + i];
            }
        }
    }
    return NULL;
}
