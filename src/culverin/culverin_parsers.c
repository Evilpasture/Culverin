#include "culverin_parsers.h"
#include "culverin.h"
#include "culverin_python.h"

static int parse_sequence_internal(PyObject *obj, void *restrict out, int count, bool is_double) {
    if (!obj || obj == Py_None) {
        return 0;
    }

    // 1. Optimized Buffer Path (NumPy, array.array, bytes)
    // Use PyObject_CheckBuffer to avoid triggering a TypeError on lists/tuples
    if (PyObject_CheckBuffer(obj)) {
        Py_buffer view;
        if (PyObject_GetBuffer(obj, &view, PyBUF_SIMPLE) == 0) {
            size_t expected_size = (size_t)count * (is_double ? sizeof(double) : sizeof(float));
            if ((size_t)view.len == expected_size) {
                memcpy(out, view.buf, expected_size);
                PyBuffer_Release(&view);
                return 1;
            }
            PyBuffer_Release(&view);
            // If size didn't match, we don't clear error (none was set)
            // and fall through to sequence logic which will likely fail/return 0.
        } else {
            // This clears the "a bytes-like object is required" error
            // so we can proceed to try the sequence path.
            PyErr_Clear();
        }
    }

    // 2. Standard Sequence Path (list, tuple, etc.)
    PyObject *seq = PySequence_Fast(obj, "Expected sequence");
    if (!seq) {
        // If PySequence_Fast fails, it sets its own error; we leave it for Python.
        return 0;
    }

    if (PySequence_Fast_GET_SIZE(seq) != count) {
        Py_DECREF(seq);
        // Important: PySequence_Fast doesn't set an error for size mismatch,
        // so we set one here to be helpful.
        PyErr_Format(PyExc_ValueError, "Expected sequence of length %d, got %zd", count,
                     PySequence_Fast_GET_SIZE(seq));
        return 0;
    }

    PyObject **items = PySequence_Fast_ITEMS(seq);

    // Hoist the type check outside the loop
    if (is_double) {
        double *restrict d_out = (double *)out;
        for (int i = 0; i < count; i++) {
            double val = PyFloat_AsDouble(items[i]);
            if (val == -1.0 && PyErr_Occurred()) {
                Py_DECREF(seq);
                return 0;
            }
            d_out[i] = val;
        }
    } else {
        float *restrict f_out = (float *)out;
        for (int i = 0; i < count; i++) {
            double val = PyFloat_AsDouble(items[i]);
            if (val == -1.0 && PyErr_Occurred()) {
                Py_DECREF(seq);
                return 0;
            }
            f_out[i] = (float)val;
        }
    }

    Py_DECREF(seq);
    return 1;
}

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

int parse_quat_r64(PyObject *obj, double *x, double *y, double *z, double *w) {
    double res[4];
    if (parse_sequence_internal(obj, res, 4, true)) {
        *x = res[0];
        *y = res[1];
        *z = res[2];
        *w = res[3];
        return 1;
    }
    return 0;
}

int parse_py_vec3f(PyObject *obj, Vec3f *out) {
    return parse_sequence_internal(obj, out, 3, false);
}
int parse_py_vec3_pos(PyObject *obj, PosStride *out) {
    return parse_sequence_internal(obj, out, 3, (sizeof(JPH_Real) == 8));
}
int parse_py_vec3_aux(PyObject *obj, AuxStride *out) {
    return parse_sequence_internal(obj, out, 4, false);
}

void parse_motor_config(PyObject *motor_dict, ConstraintParams *p) {
    if (!motor_dict || motor_dict == Py_None) {
        return;
    }

    p->has_motor = true;

    // Use Generic to fetch string
    const char *mode = get_py_dict(motor_dict, "mode", "");
    if (strcmp(mode, "velocity") == 0) {
        p->motor_type = 1;
    } else if (strcmp(mode, "position") == 0) {
        p->motor_type = 2;
    }

    // Use Generic to fetch floats
    p->motor_target = get_py_dict(motor_dict, "target", 0.0f);
    p->max_torque   = get_py_dict(motor_dict, "max_force", 1000.0f);
    p->frequency    = get_py_dict(motor_dict, "stiffness", 0.0f);
    p->damping      = get_py_dict(motor_dict, "damping", 1.0f);
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
    s[3] = 0.0f;
    if (!py_size || py_size == Py_None) {
        return;
    }

    if (PyTuple_Check(py_size)) {
        auto sz_len = PyTuple_Size(py_size);
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

int parse_tracks_to_c(PyObject *py_tracks, TrackData *out_data, int *num_out) {
    if (!py_tracks || !PyList_Check(py_tracks)) {
        return 0;
    }

    int num = (int)PyList_Size(py_tracks);
    if (num > 2) {
        num = 2;
    }
    *num_out = num;

    for (int t = 0; t < num; t++) {
        PyObject *dict    = PyList_GetItem(py_tracks, t);
        PyObject *py_idxs = PyDict_GetItemString(dict, "indices");

        if (py_idxs && PyList_Check(py_idxs)) {
            uint32_t count      = (uint32_t)PyList_Size(py_idxs);
            out_data[t].count   = count;
            out_data[t].indices = CULV_RAW_MALLOC(count * sizeof(uint32_t));

            for (uint32_t k = 0; k < count; k++) {
                out_data[t].indices[k] = (uint32_t)PyLong_AsLong(PyList_GetItem(py_idxs, k));
            }
        }
        // Generic used here for driven wheel
        out_data[t].driven_idx = (uint32_t)get_py_dict(dict, "driven_wheel", 0L);
    }
    return 1;
}