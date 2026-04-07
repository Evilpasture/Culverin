#include "culverin_parsers.h"
#include "culverin.h"
#include "culverin_compiler_specifics.h"

/**
 * INTERNAL HELPER: parse_sequence_to_floats
 * Extracts N floats/doubles from a Python sequence.
 */
static int parse_sequence_internal(PyObject *obj, void *out, int count, bool is_double) {
    if (UNLIKELY(!obj || obj == Py_None)) {
        PyErr_Format(PyExc_TypeError, "Expected a sequence of %d numbers, got NoneType", count);
        return 0;
    }

    PyObject *seq = PySequence_Fast(obj, "Expected a sequence (list/tuple) of numbers");
    if (!seq) {
        return 0;
    }

    if (UNLIKELY(PySequence_Fast_GET_SIZE(seq) != count)) {
        PyErr_Format(PyExc_TypeError, "Sequence must have exactly %d components (got %zd)", count,
                     PySequence_Fast_GET_SIZE(seq));
        Py_DECREF(seq);
        return 0;
    }

    for (int i = 0; i < count; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        double val     = PyFloat_AsDouble(item);

        if (UNLIKELY(PyErr_Occurred())) {
            Py_DECREF(seq);
            return 0;
        }

        if (is_double) {
            ((double *)out)[i] = val;
        } else {
            ((float *)out)[i] = (float)val;
        }
    }

    Py_DECREF(seq);
    return 1;
}

// --- Specific Implementations ---

int parse_vec3_f32(PyObject *obj, float *x, float *y, float *z) {
    float res[3];
    if (parse_sequence_internal(obj, res, 3, false)) {
        *x = res[0];
        *y = res[1];
        *z = res[2];
        return 1;
    }
    return 0;
}

int parse_vec3_r64(PyObject *obj, double *x, double *y, double *z) {
    double res[3];
    if (parse_sequence_internal(obj, res, 3, true)) {
        *x = res[0];
        *y = res[1];
        *z = res[2];
        return 1;
    }
    return 0;
}

int parse_quat_f32(PyObject *obj, float *x, float *y, float *z, float *w) {
    float res[4];
    if (parse_sequence_internal(obj, res, 4, false)) {
        *x = res[0];
        *y = res[1];
        *z = res[2];
        *w = res[3];
        return 1;
    }
    return 0;
}

// --- Low-complexity helper to fetch attributes with a fallback ---
float get_py_float_attr(PyObject *obj, const char *name, float default_val) {
    if (!obj || obj == Py_None) {
        return default_val;
    }

    float result   = default_val;
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

// --- Struct-based Wrappers ---

int parse_py_vec3f(PyObject *obj, Vec3f *out) {
    return parse_sequence_internal(obj, out, 3, false);
}

int parse_py_vec3_pos(PyObject *obj, PosStride *out) {
    // JPH_Real is handled by checking if it's double or float at compile time
    return parse_sequence_internal(obj, out, 3, (sizeof(JPH_Real) == 8));
}

int parse_py_vec3_aux(PyObject *obj, AuxStride *out) {
    return parse_sequence_internal(obj, out, 4, false);
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
    PyObject *item = PyDict_GetItemString(dict, key); // Returns borrowed reference
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
    p->max_torque   = get_py_dict_float(motor_dict, "max_force", 1000.0f);
    p->frequency    = get_py_dict_float(motor_dict, "stiffness", 0.0f);
    p->damping      = get_py_dict_float(motor_dict, "damping", 1.0f);
}

int parse_point_params(PyObject *args, ConstraintParams *p) {
    if (!args || args == Py_None) {
        return 1;
    }
    // Expecting a single vector in a tuple: params=((x, y, z),)
    return PyArg_ParseTuple(args, "(fff)", &p->px, &p->py, &p->pz);
}

int parse_hinge_params(PyObject *args, ConstraintParams *p) {
    p->limit_min = -JPH_M_PI;
    p->limit_max = JPH_M_PI; // Hinge defaults
    if (!args) {
        return 1;
    }
    // (Pivot), (Axis), [Min, Max]
    return PyArg_ParseTuple(args, "(fff)(fff)|ff", &p->px, &p->py, &p->pz, &p->ax, &p->ay, &p->az,
                            &p->limit_min, &p->limit_max);
}

int parse_slider_params(PyObject *args, ConstraintParams *p) {
    // Slider axis defaults to X usually, but Y is fine. Limits default to free.
    if (!args) {
        return 1;
    }
    return PyArg_ParseTuple(args, "(fff)(fff)|ff", &p->px, &p->py, &p->pz, &p->ax, &p->ay, &p->az,
                            &p->limit_min, &p->limit_max);
}

int parse_cone_params(PyObject *args, ConstraintParams *p) {
    if (!args) {
        return 1;
    }
    // (Pivot), (TwistAxis), HalfAngle
    return PyArg_ParseTuple(args, "(fff)(fff)f", &p->px, &p->py, &p->pz, &p->ax, &p->ay, &p->az,
                            &p->half_cone_angle);
}

int parse_distance_params(PyObject *args, ConstraintParams *p) {
    // Default: -1.0 can be used as a sentinel in your create_distance
    // logic to say "calculate distance from current positions"
    p->limit_min = -1.0f;
    p->limit_max = -1.0f;

    if (!args || args == Py_None) {
        return 1;
    }

    // Pattern: (Pivot1_xyz), (Pivot2_xyz), [optional min_dist, optional max_dist]
    // This now matches the format used by Hinge and Slider.
    return PyArg_ParseTuple(args, "(fff)(fff)|ff", &p->px, &p->py, &p->pz, &p->ax, &p->ay, &p->az,
                            &p->limit_min, &p->limit_max);
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

/**
 * parse_tracks_to_c
 * Parses a list of track dictionaries into C structures.
 * Returns 1 on success, 0 on failure (sets Python exception).
 */
CULV_NODISCARD
int parse_tracks_to_c(PyObject *py_tracks, TrackData *out_data, int *num_out) {
    // 1. Validate input is a list
    if (UNLIKELY(!py_tracks || !PyList_Check(py_tracks))) {
        PyErr_SetString(PyExc_TypeError, "tracks must be a list of dictionaries");
        return 0;
    }

    Py_ssize_t num = PyList_Size(py_tracks);
    if (num > 2) {
        num = 2; // Jolt limitation or architectural choice
    }
    *num_out = (int)num;

    // Reset data to prevent freeing garbage on early exit
    for (int i = 0; i < 2; i++) {
        out_data[i].count      = 0;
        out_data[i].indices    = nullptr;
        out_data[i].driven_idx = 0;
    }

    for (int t = 0; t < *num_out; t++) {
        PyObject *dict = PyList_GetItem(py_tracks, t); // Borrowed
        if (UNLIKELY(!PyDict_Check(dict))) {
            PyErr_Format(PyExc_TypeError, "Track entry %d must be a dictionary", t);
            goto fail;
        }

        // --- Handle 'indices' key ---
        PyObject *py_idxs = PyDict_GetItemString(dict, "indices"); // Borrowed
        if (py_idxs && PyList_Check(py_idxs)) {
            uint32_t count    = (uint32_t)PyList_Size(py_idxs);
            out_data[t].count = count;

            // Allocation
            out_data[t].indices = CULV_RAW_MALLOC(count * sizeof(uint32_t));
            if (!out_data[t].indices) {
                PyErr_NoMemory();
                goto fail;
            }

            for (uint32_t k = 0; k < count; k++) {
                PyObject *item = PyList_GetItem(py_idxs, k);
                long val       = PyLong_AsLong(item);
                if (val == -1 && PyErr_Occurred()) {
                    goto fail; // Element wasn't an integer
                }
                out_data[t].indices[k] = (uint32_t)val;
            }
        } else {
            PyErr_Format(PyExc_TypeError, "Track entry %d missing 'indices' list", t);
            goto fail;
        }

        // --- Handle 'driven_wheel' key ---
        PyObject *py_driven = PyDict_GetItemString(dict, "driven_wheel");
        if (py_driven) {
            unsigned long driven = PyLong_AsUnsignedLong(py_driven);
            if (PyErr_Occurred()) {
                goto fail;
            }
            out_data[t].driven_idx = (uint32_t)driven;
        }
    }

    return 1; // Success

fail:
    // Cleanup allocated memory for indices on failure
    for (int i = 0; i < *num_out; i++) {
        if (out_data[i].indices) {
            CULV_RAW_FREE(out_data[i].indices);
            out_data[i].indices = nullptr;
        }
    }
    return 0; // Failure
}
