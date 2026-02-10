#include "culverin_parsers.h"

// --- Internal Helpers to reduce complexity ---

// --- Low-complexity helper to fetch attributes with a fallback ---
float get_py_float_attr(PyObject *obj, const char *name,
                               float default_val) {
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
