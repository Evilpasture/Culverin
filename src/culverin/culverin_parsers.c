#include "culverin_parsers.h"
#include "culverin.h"

// --- Internal Helpers to reduce complexity ---

// --- Low-complexity helper to fetch attributes with a fallback ---
float get_py_float_attr(PyObject *obj, const char *name, float default_val) {
  if (!obj || obj == Py_None) {
    return default_val;
  }

  float result = default_val;
  PyObject *attr = PyObject_GetAttrString(obj, name);

  if (attr) {
    double v = PyFloat_AsDouble(attr);
    if (!PyErr_Occurred()) {
      result = (float)v;
    }
    Py_DECREF(attr);
  }

  // Clear any errors (like AttributeError) to allow fallback to default
  PyErr_Clear();
  return result;
}

// --- Reusable helper for Vec3 parsing (Complexity: 2) ---
int parse_py_vec3(PyObject *obj, Vec3f *out) {
  // 1. Initial validation
  if (!obj || !PySequence_Check(obj) || PySequence_Size(obj) != 3) {
    return 0;
  }

  float results[3];
  for (int i = 0; i < 3; i++) {
    PyObject *item = PySequence_GetItem(obj, i);
    if (!item) {
      return 0;
    }

    results[i] = (float)PyFloat_AsDouble(item);
    Py_DECREF(item);

    if (UNLIKELY(PyErr_Occurred())) {
      return 0;
    }
  }

  // 3. Assignment to struct members
  out->x = results[0];
  out->y = results[1];
  out->z = results[2];

  return 1;
}

// Helper: Parse shape parameters from Python tuple or float
void parse_shape_params(PyObject *py_size, float s[4]) {
  memset(s, 0, sizeof(float) * 4);
  if (!py_size || py_size == Py_None) {
    return;
  }

  if (PyTuple_Check(py_size)) {
    Py_ssize_t sz_len = PyTuple_Size(py_size);
    for (Py_ssize_t i = 0; i < sz_len && i < 4; i++) {
      PyObject *item = PyTuple_GetItem(py_size, i);
      if (PyNumber_Check(item)) {
        s[i] = (float)PyFloat_AsDouble(item);
      }
    }
  } else if (PyNumber_Check(py_size)) {
    s[0] = (float)PyFloat_AsDouble(py_size);
  }
}

// --- More Python Parsers ---

float get_py_dict_float(PyObject *dict, const char *key, float default_val) {
  if (!dict || !PyDict_Check(dict)) {
    return default_val;
  }
  PyObject *item =
      PyDict_GetItemString(dict, key); // Returns borrowed reference
  if (item) {
    double v = PyFloat_AsDouble(item);
    if (!PyErr_Occurred()) {
      return (float)v;
    }
    PyErr_Clear();
  }
  return default_val;
}

void parse_motor_config(PyObject *motor_dict, ConstraintParams *p) {
  if (!motor_dict || motor_dict == Py_None) {
    return;
  }

  p->has_motor = true;

  // Parse Mode string
  PyObject *type_obj = PyDict_GetItemString(motor_dict, "mode");
  if (type_obj) {
    const char *s = PyUnicode_AsUTF8(type_obj);
    if (s) {
      if (strcmp(s, "velocity") == 0) {
        p->motor_type = 1;
      } else if (strcmp(s, "position") == 0) {
        p->motor_type = 2;
      }
    }
  }

  // Parse Floats using DICT helper
  p->motor_target = get_py_dict_float(motor_dict, "target", 0.0f);
  p->max_torque = get_py_dict_float(motor_dict, "max_force", 1000.0f);
  p->frequency = get_py_dict_float(motor_dict, "stiffness", 0.0f);
  p->damping = get_py_dict_float(motor_dict, "damping", 1.0f);
}

int parse_point_params(PyObject *args, ConstraintParams *p) {
  if (!args || args == Py_None) {
    return 1; // Use defaults (0,0,0)
  }
  return PyArg_ParseTuple(args, "fff", &p->px, &p->py, &p->pz);
}

int parse_hinge_params(PyObject *args, ConstraintParams *p) {
  p->limit_min = -JPH_M_PI;
  p->limit_max = JPH_M_PI; // Hinge defaults
  if (!args) {
    return 1;
  }
  // (Pivot), (Axis), [Min, Max]
  return PyArg_ParseTuple(args, "(fff)(fff)|ff", &p->px, &p->py, &p->pz, &p->ax,
                          &p->ay, &p->az, &p->limit_min, &p->limit_max);
}

int parse_slider_params(PyObject *args, ConstraintParams *p) {
  // Slider axis defaults to X usually, but Y is fine. Limits default to free.
  if (!args) {
    return 1;
  }
  return PyArg_ParseTuple(args, "(fff)(fff)|ff", &p->px, &p->py, &p->pz, &p->ax,
                          &p->ay, &p->az, &p->limit_min, &p->limit_max);
}

int parse_cone_params(PyObject *args, ConstraintParams *p) {
  if (!args) {
    return 1;
  }
  // (Pivot), (TwistAxis), HalfAngle
  return PyArg_ParseTuple(args, "(fff)(fff)f", &p->px, &p->py, &p->pz, &p->ax,
                          &p->ay, &p->az, &p->half_cone_angle);
}

int parse_distance_params(PyObject *args, ConstraintParams *p) {
  p->limit_min = 0.0f;
  p->limit_max = 10.0f;
  if (!args) {
    return 1;
  }
  // Min, Max
  return PyArg_ParseTuple(args, "ff", &p->limit_min, &p->limit_max);
}

// Helper 2: Parse the size object (tuple or float) into a 4-float array
void parse_body_size(PyObject *py_size, float s[4]) {
  s[0] = 1.0f;
  s[1] = 1.0f;
  s[2] = 1.0f;
  s[3] = 0.0f; // Defaults
  if (!py_size || py_size == Py_None) {
    return;
  }
  if (PyTuple_Check(py_size)) {
    Py_ssize_t sz_len = PyTuple_Size(py_size);
    for (Py_ssize_t i = 0; i < sz_len && i < 4; i++) {
      PyObject *item = PyTuple_GetItem(py_size, i);
      if (PyNumber_Check(item)) {
        s[i] = (float)PyFloat_AsDouble(item);
      }
    }
  } else if (PyNumber_Check(py_size)) {
    s[0] = (float)PyFloat_AsDouble(py_size);
  }
}

void parse_tracks_to_c(PyObject *py_tracks, TrackData *out_data, int *num_out) {
    Py_ssize_t num = PyList_Size(py_tracks);
    if (num > 2) num = 2;
    *num_out = (int)num;

    for (int t = 0; t < *num_out; t++) {
        PyObject *dict = PyList_GetItem(py_tracks, t);
        PyObject *py_idxs = PyDict_GetItemString(dict, "indices");
        
        out_data[t].count = 0;
        out_data[t].indices = NULL;
        out_data[t].driven_idx = 0;

        if (py_idxs && PyList_Check(py_idxs)) {
            out_data[t].count = (uint32_t)PyList_Size(py_idxs);
            out_data[t].indices = PyMem_RawMalloc(out_data[t].count * sizeof(uint32_t));
            for (uint32_t k = 0; k < out_data[t].count; k++) {
                out_data[t].indices[k] = (uint32_t)PyLong_AsLong(PyList_GetItem(py_idxs, k));
            }
        }

        PyObject *py_driven = PyDict_GetItemString(dict, "driven_wheel");
        if (py_driven) {
            out_data[t].driven_idx = (uint32_t)PyLong_AsUnsignedLong(py_driven);
        }
    }
}