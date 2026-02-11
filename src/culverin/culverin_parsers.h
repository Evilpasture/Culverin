#pragma once
#include <Python.h>
#include <stdbool.h>

typedef struct {
  float x;
  float y;
  float z;
} Vec3f; // General Vec3f

// --- Unified Parameter Struct ---
typedef struct {
  float px, py, pz; // Pivot
  float ax, ay, az; // Axis
  float limit_min;  // Limits
  float limit_max;
  float half_cone_angle;

  // --- NEW: Motor Settings ---
  bool has_motor;
  int motor_type;     // 0=Off, 1=Velocity, 2=Position
  float motor_target; // Target Velocity or Target Position
  float max_torque;   // Max Force/Torque
  float frequency;    // Spring stiffness (0 = stiff)
  float damping;      // Spring damping
} ConstraintParams;

float get_py_float_attr(PyObject *obj, const char *name, float default_val);
int parse_py_vec3(PyObject *obj, Vec3f *out);

void parse_shape_params(PyObject *py_size, float s[4]);

float get_py_dict_float(PyObject *dict, const char *key, float default_val);

void parse_motor_config(PyObject *motor_dict, ConstraintParams *p);

int parse_point_params(PyObject *args, ConstraintParams *p);

int parse_hinge_params(PyObject *args, ConstraintParams *p);

int parse_slider_params(PyObject *args, ConstraintParams *p);

int parse_cone_params(PyObject *args, ConstraintParams *p);

int parse_distance_params(PyObject *args, ConstraintParams *p);

void parse_body_size(PyObject *py_size, float s[4]);