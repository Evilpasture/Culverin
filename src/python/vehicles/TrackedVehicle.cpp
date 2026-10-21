// Python adapter for tracked vehicles.
//
// The wheel and track settings become plain config structs here; the tank's
// kickstart-and-neutral transmission logic lives in src/vehicles/Vehicle.cpp.

#include "vehicles/VehicleBinding.hpp"
#include "bindings/ModuleState.hpp"
#include "bindings/PythonCommon.hpp"
#include "bindings/PythonSupport.hpp"
#include "bindings/parsing/Parsers.hpp"
#include "physics/PhysicsSync.hpp"
#include <cstring>
#include <mutex>

namespace CPH::Python {

namespace {

// Every failure on the tracked path has always surfaced as this one message,
// including malformed wheel dictionaries: create_track_wheel raised its parse
// error, the caller's `python_fail` label overwrote it, and the last write wins.
PyObject *RaiseTrackedCreationFailed() {
    return PyErr_Format(PyExc_RuntimeError, "Tracked vehicle creation failed");
}

bool ParseTrackWheelConfig(PyObject *dict, CPH::TrackWheelConfig &out) {
    PosStride position{};
    if (ParseResult parsed = ParseVec3Into(PyDict_GetItemString(dict, "pos"), position);
        !parsed) {
        return false;
    }

    out.x = static_cast<float>(position.x);
    out.y = static_cast<float>(position.y);
    out.z = static_cast<float>(position.z);

    out.radius      = GetAttribute(dict, "radius", out.radius);
    out.width       = GetAttribute(dict, "width", out.width);
    out.suspension  = GetAttribute(dict, "suspension", out.suspension);
    out.friction    = GetAttribute(dict, "friction", out.friction);
    out.spring_freq = GetAttribute(dict, "spring_freq", out.spring_freq);
    out.spring_damp = GetAttribute(dict, "spring_damp", out.spring_damp);
    return true;
}

// Converts the parsed track table into the native config and frees the index
// buffers the parser allocated.
bool ParseTracks(PyObject *py_tracks, std::vector<CPH::TrackConfig> &out) {
    TrackData tracks[2];
    std::memset(tracks, 0, sizeof(tracks));

    const std::expected<ParsedTracks, ParseError> parsed_tracks =
        ParseTracksToC(py_tracks, tracks);

    if (!parsed_tracks) {
        for (TrackData &track : tracks) {
            if (track.indices != nullptr) {
                Deallocate(track.indices);
                track.indices = nullptr;
            }
        }
        return false;
    }

    out.clear();
    out.reserve(static_cast<std::size_t>(parsed_tracks->count));
    for (int t = 0; t < parsed_tracks->count; t++) {
        CPH::TrackConfig track;
        track.wheels.reserve(tracks[t].count);
        for (std::uint32_t k = 0; k < tracks[t].count; k++) {
            track.wheels.push_back(static_cast<int>(tracks[t].indices[k]));
        }
        track.driven_wheel = static_cast<int>(tracks[t].driven_idx);
        out.push_back(std::move(track));

        Deallocate(tracks[t].indices);
        tracks[t].indices = nullptr;
    }
    return true;
}

} // namespace

[[nodiscard]] PyObject *WorldCreateTrackedVehicle(PhysicsWorldObject *self,
                                                  PyObject *const *args, size_t nargsf,
                                                  PyObject *kwnames) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);

    uint64_t  chassis_raw = 0;
    PyObject *py_wheels   = nullptr;
    PyObject *py_tracks   = nullptr;
    // The parser only writes these when the keyword is supplied, so the
    // defaults have to be here; 5000 Nm / 6000 rpm are what a tracked vehicle
    // has always been built with.
    float     max_torque  = 5000.0f;
    float     max_rpm     = 6000.0f;

    const ParseResult parsed = self->parsers->CreateTracked.Parse(
        args, nargs, kwnames, chassis_raw, py_wheels, py_tracks, max_torque, max_rpm);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    if (!PyList_Check(py_wheels)) {
        return RaiseTrackedCreationFailed();
    }

    CPH::TrackedVehicleConfig config;
    config.engine.max_torque = max_torque;
    config.engine.max_rpm    = max_rpm;
    // min_rpm is not part of the tracked signature, and a tracked engine idles
    // lower than a car's: 500 rpm, not EngineConfig's 1000 rpm default.
    config.engine.min_rpm    = 500.0f;

    const Py_ssize_t wheel_count = PyList_Size(py_wheels);
    config.wheels.resize(static_cast<std::size_t>(wheel_count));
    for (Py_ssize_t i = 0; i < wheel_count; i++) {
        if (!ParseTrackWheelConfig(PyList_GetItem(py_wheels, i),
                                   config.wheels[static_cast<std::size_t>(i)])) {
            return RaiseTrackedCreationFailed();
        }
    }

    if (!ParseTracks(py_tracks, config.tracks)) {
        return RaiseTrackedCreationFailed();
    }

    // Pending body-creation commands have to run before the chassis handle can
    // resolve to a live body.
    {
        const std::scoped_lock lock(self->physics.shadow_lock);
        SyncAndFlushInternal(self);
    }

    std::expected<std::unique_ptr<CPH::Vehicle>, CPH::Error> created =
        std::unexpected(CPH::Error{CPH::VehicleError::TrackedCreationFailed});
    Py_BEGIN_ALLOW_THREADS
    created = CPH::Vehicle::CreateTracked(self->physics, BodyHandle(chassis_raw), config);
    Py_END_ALLOW_THREADS

    if (!created) {
        if (created.error().Code().Is(CPH::VehicleError::InvalidChassis)) {
            return PyErr_Format(PyExc_ValueError,
                                "Invalid chassis handle (Body is dead or pending)");
        }
        return RaiseTrackedCreationFailed();
    }

    ModuleState *state = GetModuleState(PyType_GetModule(Py_TYPE(self)));
    auto         *obj =
        PyObject_GC_New(VehicleObject, reinterpret_cast<PyTypeObject *>(state->VehicleType));
    if (obj == nullptr) {
        return nullptr;
    }
    // GC_New leaves the storage uninitialised, and the failure path below
    // decrefs the object, so every field is set before anything can fail.
    obj->world   = nullptr;
    obj->vehicle = nullptr;
    obj->parsers = nullptr;

    obj->parsers = CPH::Python::NewParserGroup<CPH::Python::VehicleParsers>();
    if (obj->parsers == nullptr) {
        Py_DECREF(reinterpret_cast<PyObject *>(obj));
        return PyErr_NoMemory();
    }

    obj->vehicle = created->release();
    obj->world   = reinterpret_cast<PhysicsWorldObject *>(Py_NewRef(self));

    PyObject_GC_Track(reinterpret_cast<PyObject *>(obj));
    return reinterpret_cast<PyObject *>(obj);
}

// --- Controls ----------------------------------------------------------------

[[nodiscard]] PyObject *VehicleSetTankInput(VehicleObject *self, PyObject *const *args,
                                            size_t nargsf, PyObject *kwnames) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);

    float left  = 0.0f;
    float right = 0.0f;
    float brake = 0.0f;

    const ParseResult parsed = self->parsers->TankInput.Parse(args, nargs, kwnames, left, right,
                                                              brake);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    if (self->vehicle != nullptr) {
        self->vehicle->SetTankInput(left, right, brake);
    }
    Py_RETURN_NONE;
}

} // namespace CPH::Python
