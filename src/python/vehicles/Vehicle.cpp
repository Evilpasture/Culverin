// Python adapter for CPH::Vehicle (wheeled).
//
// Everything here parses Python objects into native config structs, calls the
// native vehicle, and packs the result. The arcade transmission state machine,
// the wheel queries and the debug dump live in src/vehicles/Vehicle.cpp.

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

// Wheel defaults come from a default-constructed WheelConfig, so the native
// struct stays the single source of truth for them.
bool ParseWheelConfig(PyObject *dict, CPH::WheelConfig &out) {
    PosStride position{};
    if (ParseResult parsed = ParseVec3Into(PyDict_GetItemString(dict, "pos"), position);
        !parsed) {
        // Deliberate message override: the wheel dict is user data, and this is
        // the diagnostic callers have always seen for a malformed entry.
        PyErr_SetString(PyExc_ValueError, "Wheel 'pos' must be a sequence of 3 real numbers");
        return false;
    }

    out.x = static_cast<float>(position.x);
    out.y = static_cast<float>(position.y);
    out.z = static_cast<float>(position.z);

    out.radius           = GetAttribute(dict, "radius", out.radius);
    out.width            = GetAttribute(dict, "width", out.width);
    out.brakeTorque     = GetAttribute(dict, "brake_torque", out.brakeTorque);
    out.handbrakeTorque = GetAttribute(dict, "handbrake_torque", out.handbrakeTorque);
    out.suspension       = GetAttribute(dict, "suspension", out.suspension);
    out.springFreq      = GetAttribute(dict, "spring_freq", out.springFreq);
    out.springDamp      = GetAttribute(dict, "spring_damp", out.springDamp);
    return true;
}

// The drive string has always been matched case-sensitively against "FWD" and
// "AWD", with everything else -- including a non-string -- meaning RWD.
CPH::DriveMode ParseDriveMode(PyObject *py_drive) {
    if (py_drive != nullptr && PyUnicode_Check(py_drive)) {
        if (const char *name = PyUnicode_AsUTF8(py_drive); name != nullptr) {
            if (std::strcmp(name, "FWD") == 0) {
                return CPH::DriveMode::Fwd;
            }
            if (std::strcmp(name, "AWD") == 0) {
                return CPH::DriveMode::Awd;
            }
            return CPH::DriveMode::Rwd;
        }
        PyErr_Clear();
    }
    return CPH::DriveMode::Rwd;
}

void ParseEngine(PyObject *py_engine, CPH::EngineConfig &out) {
    out.maxTorque = GetAttribute(py_engine, "max_torque", out.maxTorque);
    out.maxRpm    = GetAttribute(py_engine, "max_rpm", out.maxRpm);
    out.minRpm    = GetAttribute(py_engine, "min_rpm", out.minRpm);
    out.inertia    = GetAttribute(py_engine, "inertia", out.inertia);
}

void ParseTransmission(PyObject *py_transmission, CPH::TransmissionConfig &out) {
    if (py_transmission != nullptr && py_transmission != Py_None) {
        if (PyObject *mode = PyObject_GetAttrString(py_transmission, "mode"); mode != nullptr) {
            out.mode = static_cast<int>(PyLong_AsLong(mode));
            Py_DECREF(mode);
        }
        PyErr_Clear();

        PyObject *ratios = PyObject_GetAttrString(py_transmission, "ratios");
        if (ratios != nullptr && PyList_Check(ratios)) {
            const Py_ssize_t count = PyList_Size(ratios);
            out.gearRatios.clear();
            out.gearRatios.reserve(static_cast<std::size_t>(count));
            for (Py_ssize_t i = 0; i < count; i++) {
                out.gearRatios.push_back(
                    static_cast<float>(PyFloat_AsDouble(PyList_GetItem(ratios, i))));
            }
        }
        Py_XDECREF(ratios);
        PyErr_Clear();
    }

    out.clutchStrength    = GetAttribute(py_transmission, "clutch_strength",
                                          out.clutchStrength);
    out.differentialRatio = GetAttribute(py_transmission, "differential_ratio",
                                          out.differentialRatio);
}

// (position, rotation) pair, the shape both wheel queries have always returned.
//
// TODO(smell): PyTuple_Pack already increfs each element, so the fresh
// references from PyFloat_FromDouble leak on every call. Switching to
// PyTuple_SET_ITEM (which steals) fixes it but changes observable refcounts, so
// it is left for a dedicated fix.
PyObject *PackTransform(const CPH::WheelTransform &transform) {
    PyObject *position = PyTuple_Pack(3, PyFloat_FromDouble(transform.px),
                                      PyFloat_FromDouble(transform.py),
                                      PyFloat_FromDouble(transform.pz));
    PyObject *rotation = PyTuple_Pack(4, PyFloat_FromDouble(transform.rx),
                                      PyFloat_FromDouble(transform.ry),
                                      PyFloat_FromDouble(transform.rz),
                                      PyFloat_FromDouble(transform.rw));
    PyObject *result = PyTuple_Pack(2, position, rotation);
    Py_DECREF(position);
    Py_DECREF(rotation);
    return result;
}

// Bad handles and short wheel lists have always been ValueErrors; everything
// else is a RuntimeError carrying the native message verbatim.
PyObject *RaiseVehicleError(const CPH::Error &error) {
    const std::string message(error.Message());
    if (error.Code().Is(CPH::VehicleError::InvalidChassis) ||
        error.Code().Is(CPH::VehicleError::TooFewWheels)) {
        PyErr_SetString(PyExc_ValueError, message.c_str());
    } else {
        PyErr_SetString(PyExc_RuntimeError, message.c_str());
    }
    return nullptr;
}

bool ParseWheelConfigs(PyObject *py_wheels, std::vector<CPH::WheelConfig> &out) {
    const Py_ssize_t count = PyList_Size(py_wheels);
    out.resize(static_cast<std::size_t>(count));
    for (Py_ssize_t i = 0; i < count; i++) {
        if (!ParseWheelConfig(PyList_GetItem(py_wheels, i), out[static_cast<std::size_t>(i)])) {
            return false;
        }
    }
    return true;
}

} // namespace

// --- Creation ----------------------------------------------------------------

[[nodiscard]] PyObject *WorldCreateVehicle(PhysicsWorldObject *self, PyObject *const *args,
                                           size_t nargsf, PyObject *kwnames) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);

    uint64_t  chassis_raw = 0;
    PyObject *py_wheels   = nullptr;
    PyObject *py_drive    = nullptr;
    PyObject *py_engine   = nullptr;
    PyObject *py_trans    = nullptr;

    const ParseResult parsed = self->parsers->createVehicle.Parse(
        args, nargs, kwnames, chassis_raw, py_wheels, py_drive, py_engine, py_trans);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    if (!PyList_Check(py_wheels) || PyList_Size(py_wheels) < 2) {
        return PyErr_Format(PyExc_ValueError, "Wheels must be a list of at least 2 dictionaries");
    }

    CPH::WheeledVehicleConfig config;
    if (!ParseWheelConfigs(py_wheels, config.wheels)) {
        return nullptr;
    }
    config.drive = ParseDriveMode(py_drive);
    ParseEngine(py_engine, config.engine);
    ParseTransmission(py_trans, config.transmission);

    // Pending body-creation commands have to run before the chassis handle can
    // resolve to a live body.
    {
        const std::scoped_lock lock(self->physics.shadowLock);
        SyncAndFlushInternal(self);
    }

    std::expected<std::unique_ptr<CPH::Vehicle>, CPH::Error> created =
        std::unexpected(CPH::Error{CPH::VehicleError::CreationFailed});
    Py_BEGIN_ALLOW_THREADS
    created = CPH::Vehicle::CreateWheeled(self->physics, BodyHandle(chassis_raw), config);
    Py_END_ALLOW_THREADS

    if (!created) {
        return RaiseVehicleError(created.error());
    }

    ModuleState *state = GetModuleState(PyType_GetModule(Py_TYPE(self)));
    auto         *obj  = PyObject_GC_New(VehicleObject,
                                         reinterpret_cast<PyTypeObject *>(state->vehicleType));
    if (obj == nullptr) {
        return nullptr;
    }
    // GC_New leaves the storage uninitialised, and the failure paths below
    // decref the object, so every field is set before anything can fail.
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

    // TODO(bug): obj->world already took a strong reference via Py_NewRef above,
    // and VehicleDealloc releases exactly one, so this extra INCREF leaks a world
    // reference per vehicle and keeps the world alive as long as any vehicle
    // object exists. WorldCreateTrackedVehicle does not do this. Preserved as-is;
    // dropping the line is the fix and is a lifecycle change.
    Py_INCREF(self);

    PyObject_GC_Track(reinterpret_cast<PyObject *>(obj));
    return reinterpret_cast<PyObject *>(obj);
}

// --- Controls ----------------------------------------------------------------

[[nodiscard]] PyObject *VehicleSetInput(VehicleObject *self, PyObject *const *args, size_t nargsf,
                                        PyObject *kwnames) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);

    float forward   = 0.0f;
    float right     = 0.0f;
    float brake     = 0.0f;
    float handbrake = 0.0f;

    const ParseResult parsed =
        self->parsers->vehicleInput.Parse(args, nargs, kwnames, forward, right, brake, handbrake);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    if (self->vehicle != nullptr) {
        self->vehicle->SetInput(forward, right, brake, handbrake);
    }
    Py_RETURN_NONE;
}

// --- Queries -----------------------------------------------------------------

namespace {

PyObject *WheelQuery(VehicleObject *self, PyObject *const *args, size_t nargsf, PyObject *kwnames,
                     bool local) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);

    uint32_t index = 0;
    if (self->world == nullptr) {
        Py_RETURN_NONE;
    }

    const ParseResult parsed =
        self->world->parsers->wheelIdx.Parse(args, nargs, kwnames, index);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    if (self->vehicle == nullptr) {
        Py_RETURN_NONE;
    }

    const std::optional<CPH::WheelTransform> transform =
        local ? self->vehicle->GetWheelLocalTransform(index) : self->vehicle->GetWheelTransform(index);
    if (!transform) {
        Py_RETURN_NONE;
    }
    return PackTransform(*transform);
}

} // namespace

[[nodiscard]] PyObject *VehicleGetWheelTransform(VehicleObject *self, PyObject *const *args,
                                                 size_t nargsf, PyObject *kwnames) {
    return WheelQuery(self, args, nargsf, kwnames, false);
}

[[nodiscard]] PyObject *VehicleGetWheelLocalTransform(VehicleObject *self, PyObject *const *args,
                                                      size_t nargsf, PyObject *kwnames) {
    return WheelQuery(self, args, nargsf, kwnames, true);
}

[[nodiscard]] PyObject *VehicleGetDebugState(VehicleObject *self,
                                             PyObject *ignored [[maybe_unused]]) {
    if (self->vehicle != nullptr) {
        self->vehicle->LogDebugState();
    }
    Py_RETURN_NONE;
}

// --- Lifecycle ---------------------------------------------------------------

[[nodiscard]] int VehicleTraverse(VehicleObject *self, visitproc visit, void *arg) {
    Py_VISIT(self->world);
    return 0;
}

[[nodiscard]] int VehicleClear(VehicleObject *self) {
    Py_CLEAR(self->world);
    return 0;
}

[[nodiscard]] PyObject *VehicleDestroy(VehicleObject *self,
                                       PyObject *ignored [[maybe_unused]]) {
    delete self->vehicle;
    self->vehicle = nullptr;
    Py_RETURN_NONE;
}

void VehicleDealloc(VehicleObject *self) {
    PyObject_GC_UnTrack(self);

    // The vehicle is registered with the physics system and has to leave it
    // before the world reference goes away.
    delete self->vehicle;
    self->vehicle = nullptr;

    delete self->parsers;
    self->parsers = nullptr;

    Py_XDECREF(self->world);
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

[[nodiscard]] PyObject *VehicleGetWheelCount(VehicleObject *self,
                                             [[maybe_unused]] void *closure) {
    return PyLong_FromUnsignedLong(self->vehicle != nullptr ? self->vehicle->wheel_count() : 0);
}

static PyMethodDef kVehicleMethods[] = {

    CPH::Python::FastCallMethod("set_input", VehicleSetInput),
    CPH::Python::FastCallMethod("set_tank_input", VehicleSetTankInput),
    CPH::Python::FastCallMethod("get_wheel_transform", VehicleGetWheelTransform),
    CPH::Python::FastCallMethod("get_wheel_local_transform", VehicleGetWheelLocalTransform),
    CPH::Python::NoArgsMethod("destroy", VehicleDestroy),
    CPH::Python::NoArgsMethod("get_debug_state", VehicleGetDebugState),
    {nullptr, nullptr, 0, nullptr}};

static PyGetSetDef VehicleGetSets[] = {
    CPH::Python::GetSet("wheel_count", VehicleGetWheelCount),
    {},
};

static PyType_Slot VehicleSlots[] = {
    {.slot = Py_tp_dealloc, .pfunc = reinterpret_cast<void *>(VehicleDealloc)},
    {.slot = Py_tp_traverse, .pfunc = reinterpret_cast<void *>(VehicleTraverse)},
    {.slot = Py_tp_clear, .pfunc = reinterpret_cast<void *>(VehicleClear)},
    {.slot = Py_tp_methods, .pfunc = reinterpret_cast<void *>(kVehicleMethods)},
    {.slot = Py_tp_getset, .pfunc = reinterpret_cast<void *>(VehicleGetSets)},
    {},
};

PyType_Spec VehicleSpec = {
    .name      = "culverin._culverin_c.Vehicle",
    .basicsize = sizeof(VehicleObject),
    .flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .slots     = VehicleSlots,
};

} // namespace CPH::Python
