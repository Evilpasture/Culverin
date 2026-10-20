#include "bindings/parsing/Parsers.hpp"

#include "bindings/PythonCommon.hpp"
#include "bindings/PythonSupport.hpp"

#include <cstring>

namespace CPH::Python {

namespace {

// Shared implementation for every fixed-length numeric sequence argument.
// `out` receives `count` values of `element_size` bytes.
[[nodiscard]] ParseResult parse_sequence_into(Borrowed obj, void *out, int count,
                                              bool is_double) {
    if (!obj || obj.is_none()) {
        // Legacy contract: None is a *failure* here, and one that reports no
        // exception. Callers either reject None before calling or let the
        // boundary translate the NULL return, exactly as before B3.
        return SilentFailure();
    }

    // 1. Optimized Buffer Path (NumPy, array.array, bytes)
    // Use PyObject_CheckBuffer to avoid triggering a TypeError on lists/tuples
    if (PyObject_CheckBuffer(obj)) {
        Py_buffer view;
        if (PyObject_GetBuffer(obj, &view, PyBUF_SIMPLE) == 0) {
            const size_t expected_size =
                static_cast<size_t>(count) * (is_double ? sizeof(double) : sizeof(float));
            if (static_cast<size_t>(view.len) == expected_size) {
                std::memcpy(out, view.buf, expected_size);
                PyBuffer_Release(&view);
                return {};
            }
            PyBuffer_Release(&view);
            // If size didn't match, we don't clear error (none was set)
            // and fall through to sequence logic which will likely fail.
        } else {
            // This clears the "a bytes-like object is required" error
            // so we can proceed to try the sequence path.
            PyErr_Clear();
        }
    }

    // 2. Standard Sequence Path (list, tuple, etc.)
    Ref seq{PySequence_Fast(obj, "Expected sequence")};
    if (!seq) {
        // If PySequence_Fast fails, it sets its own error; capture it as-is.
        return CaptureFailure();
    }

    const Py_ssize_t sequence_size = PySequence_Fast_GET_SIZE(seq.get());
    if (sequence_size != count) {
        // PySequence_Fast doesn't set an error for size mismatch, so we set
        // one here to be helpful. Note this happens before the Ref releases.
        PyErr_Format(PyExc_ValueError, "Expected sequence of length %d, got %zd", count,
                     sequence_size);
        return CaptureFailure();
    }

    PyObject **items = PySequence_Fast_ITEMS(seq.get());

    // Hoist the type check outside the loop
    if (is_double) {
        auto *d_out = static_cast<double *>(out);
        for (int i = 0; i < count; i++) {
            const double val = PyFloat_AsDouble(items[i]);
            if (val == -1.0 && PyErr_Occurred()) {
                return CaptureFailure();
            }
            d_out[i] = val;
        }
    } else {
        auto *f_out = static_cast<float *>(out);
        for (int i = 0; i < count; i++) {
            const double val = PyFloat_AsDouble(items[i]);
            if (val == -1.0 && PyErr_Occurred()) {
                return CaptureFailure();
            }
            f_out[i] = static_cast<float>(val);
        }
    }

    return {};
}

} // namespace

ParseResult parse_vec3_into(Borrowed obj, float &x, float &y, float &z) {
    float res[3];
    if (ParseResult parsed = parse_sequence_into(obj, res, 3, false); !parsed) {
        return parsed;
    }
    x = res[0];
    y = res[1];
    z = res[2];
    return {};
}

ParseResult parse_vec3_into(Borrowed obj, double &x, double &y, double &z) {
    double res[3];
    if (ParseResult parsed = parse_sequence_into(obj, res, 3, true); !parsed) {
        return parsed;
    }
    x = res[0];
    y = res[1];
    z = res[2];
    return {};
}

ParseResult parse_quat_into(Borrowed obj, float &x, float &y, float &z, float &w) {
    float res[4];
    if (ParseResult parsed = parse_sequence_into(obj, res, 4, false); !parsed) {
        return parsed;
    }
    x = res[0];
    y = res[1];
    z = res[2];
    w = res[3];
    return {};
}

ParseResult parse_quat_into(Borrowed obj, double &x, double &y, double &z, double &w) {
    double res[4];
    if (ParseResult parsed = parse_sequence_into(obj, res, 4, true); !parsed) {
        return parsed;
    }
    x = res[0];
    y = res[1];
    z = res[2];
    w = res[3];
    return {};
}

ParseResult parse_vec3_into(Borrowed obj, Vec3f &out) {
    return parse_vec3_into(obj, out.x, out.y, out.z);
}

ParseResult parse_vec3_into(Borrowed obj, PosStride &out) {
    return parse_vec3_into(obj, out.x, out.y, out.z);
}

ParseResult parse_quat_into(Borrowed obj, AuxStride &out) {
    return parse_quat_into(obj, out.x, out.y, out.z, out.w);
}

std::expected<Vec3f, ParseError> parse_vec3(Borrowed obj) {
    Vec3f value{};
    if (ParseResult parsed = parse_vec3_into(obj, value); !parsed) {
        return std::unexpected(std::move(parsed.error()));
    }
    return value;
}

std::expected<PosStride, ParseError> parse_position(Borrowed obj) {
    PosStride value{};
    if (ParseResult parsed = parse_vec3_into(obj, value); !parsed) {
        return std::unexpected(std::move(parsed.error()));
    }
    return value;
}

std::expected<AuxStride, ParseError> parse_rotation(Borrowed obj) {
    AuxStride value{};
    if (ParseResult parsed = parse_quat_into(obj, value); !parsed) {
        return std::unexpected(std::move(parsed.error()));
    }
    return value;
}

void ParseMotorConfig(Borrowed motor_dict, ConstraintParams &params) {
    if (!motor_dict || motor_dict.is_none()) {
        return;
    }

    params.has_motor = true;

    // Use Generic to fetch string
    const char *mode = GetDictionaryItem(motor_dict, "mode", "");
    if (std::strcmp(mode, "velocity") == 0) {
        params.motor_type = 1;
    } else if (std::strcmp(mode, "position") == 0) {
        params.motor_type = 2;
    }

    // Use Generic to fetch floats
    params.motor_target = GetDictionaryItem(motor_dict, "target", 0.0f);
    params.max_torque   = GetDictionaryItem(motor_dict, "max_force", 1000.0f);
    params.frequency    = GetDictionaryItem(motor_dict, "stiffness", 0.0f);
    params.damping      = GetDictionaryItem(motor_dict, "damping", 1.0f);
}

ParseResult ParsePointParams(Borrowed args, ConstraintParams &params) {
    if (!args || args.is_none()) {
        return {};
    }
    // Expecting a single vector in a tuple: params=((x, y, z),)
    if (PyArg_ParseTuple(args, "(fff)", &params.px, &params.py, &params.pz) == 0) {
        return CaptureFailure();
    }
    return {};
}

ParseResult ParseHingeParams(Borrowed args, ConstraintParams &params) {
    params.limit_min = -JPH_M_PI;
    params.limit_max = JPH_M_PI; // Hinge defaults
    if (!args) {
        return {};
    }
    // (Pivot), (Axis), [Min, Max]
    if (PyArg_ParseTuple(args, "(fff)(fff)|ff", &params.px, &params.py, &params.pz, &params.ax,
                         &params.ay, &params.az, &params.limit_min, &params.limit_max) == 0) {
        return CaptureFailure();
    }
    return {};
}

ParseResult ParseSliderParams(Borrowed args, ConstraintParams &params) {
    // Slider axis defaults to X usually, but Y is fine. Limits default to free.
    if (!args) {
        return {};
    }
    if (PyArg_ParseTuple(args, "(fff)(fff)|ff", &params.px, &params.py, &params.pz, &params.ax,
                         &params.ay, &params.az, &params.limit_min, &params.limit_max) == 0) {
        return CaptureFailure();
    }
    return {};
}

ParseResult ParseConeParams(Borrowed args, ConstraintParams &params) {
    if (!args) {
        return {};
    }
    // (Pivot), (TwistAxis), HalfAngle
    if (PyArg_ParseTuple(args, "(fff)(fff)f", &params.px, &params.py, &params.pz, &params.ax,
                         &params.ay, &params.az, &params.half_cone_angle) == 0) {
        return CaptureFailure();
    }
    return {};
}

ParseResult ParseDistanceParams(Borrowed args, ConstraintParams &params) {
    // Default: -1.0 can be used as a sentinel in create_distance
    // logic to say "calculate distance from current positions"
    params.limit_min = -1.0f;
    params.limit_max = -1.0f;

    if (!args || args.is_none()) {
        return {};
    }

    // Pattern: (Pivot1_xyz), (Pivot2_xyz), [optional min_dist, optional max_dist]
    // This now matches the format used by Hinge and Slider.
    if (PyArg_ParseTuple(args, "(fff)(fff)|ff", &params.px, &params.py, &params.pz, &params.ax,
                         &params.ay, &params.az, &params.limit_min, &params.limit_max) == 0) {
        return CaptureFailure();
    }
    return {};
}

// Helper 2: Parse the size object (tuple or float) into a 4-float array
void ParseBodySize(Borrowed py_size, float size[4]) {
    size[0] = 1.0f;
    size[1] = 1.0f;
    size[2] = 1.0f;
    size[3] = 0.0f;
    if (!py_size || py_size.is_none()) {
        return;
    }

    if (PyTuple_Check(py_size)) {
        const auto size_length = PyTuple_Size(py_size);
        for (Py_ssize_t i = 0; i < size_length && i < 4; i++) {
            PyObject *item = PyTuple_GetItem(py_size, i);
            if (PyNumber_Check(item)) {
                size[i] = static_cast<float>(PyFloat_AsDouble(item));
            }
        }
    } else if (PyNumber_Check(py_size)) {
        size[0] = static_cast<float>(PyFloat_AsDouble(py_size));
    }
}

std::expected<ParsedTracks, ParseError> ParseTracksToC(Borrowed py_tracks, TrackData *out_data) {
    if (!py_tracks || !PyList_Check(py_tracks)) {
        // Legacy returned 0 without setting an exception; the caller's
        // `python_fail` label raises "Tracked vehicle creation failed".
        return SilentFailure<ParsedTracks>();
    }

    int num = static_cast<int>(PyList_Size(py_tracks));
    if (num > 2) {
        num = 2;
    }

    for (int t = 0; t < num; t++) {
        PyObject *dict    = PyList_GetItem(py_tracks, t);
        PyObject *py_idxs = PyDict_GetItemString(dict, "indices");

        if (py_idxs && PyList_Check(py_idxs)) {
            const auto count       = static_cast<uint32_t>(PyList_Size(py_idxs));
            out_data[t].count      = count;
            out_data[t].indices    = static_cast<uint32_t *>(Allocate(count * sizeof(uint32_t)));

            for (uint32_t k = 0; k < count; k++) {
                out_data[t].indices[k] =
                    static_cast<uint32_t>(PyLong_AsLong(PyList_GetItem(py_idxs, k)));
            }
        }
        // Generic used here for driven wheel
        out_data[t].driven_idx =
            static_cast<uint32_t>(GetDictionaryItem(dict, "driven_wheel", 0L));
    }
    return ParsedTracks{.count = num};
}

} // namespace CPH::Python
