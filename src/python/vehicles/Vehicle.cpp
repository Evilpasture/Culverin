// Python adapter for CPH::Vehicle (wheeled).
//
// Everything here parses Python objects into native config structs, calls the
// native vehicle, and packs the result. The arcade transmission state machine,
// the wheel queries and the debug dump live in src/vehicles/Vehicle.cpp.

#include "vehicles/VehicleBinding.hpp"
#include "bindings/ModuleState.hpp"
#include "bindings/ThreadState.hpp"
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

    // Optional. Absent leaves the flag unset, which keeps the historical
    // "steers if its local z is positive" inference; present says so outright,
    // and is the only way to make a wheel on a -Z-facing chassis steer.
    if (PyObject *steers = PyDict_GetItemString(dict, "steers"); steers != nullptr) {
        if (steers == Py_True) {
            out.steers = true;
        } else if (steers == Py_False) {
            out.steers = false;
        } else {
            PyErr_SetString(PyExc_TypeError, "Wheel 'steers' must be a bool");
            return false;
        }
    }

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
// Every float is built with PyFloat_FromDouble and handed straight to
// PyTuple_SET_ITEM, which *steals* the reference. The previous spelling passed
// those fresh references to PyTuple_Pack, which increfs instead of stealing, so
// seven float objects leaked on every get_wheel_transform /
// get_wheel_local_transform call. The leak was invisible to callers -- the
// returned tuples are identical either way -- so it is fixed here rather than
// preserved.
// Element type is the caller's, not float: WheelTransform keeps world
// positions in double precision (px/py/pz are double) and only the rotation in
// float, so a float buffer here would quietly round the translation.
template <typename T>
PyObject *PackFloats(const T *values, int count) {
    PyObject *tuple = PyTuple_New(count);
    if (tuple == nullptr) [[unlikely]] {
        return nullptr;
    }
    for (int i = 0; i < count; ++i) {
        PyObject *item = PyFloat_FromDouble(values[i]);
        if (item == nullptr) [[unlikely]] {
            // SET_ITEM stole every reference stored so far; the untouched
            // slots are NULL, which Py_DECREF tolerates.
            Py_DECREF(tuple);
            return nullptr;
        }
        PyTuple_SET_ITEM(tuple, i, item);
    }
    return tuple;
}

PyObject *PackTransform(const CPH::WheelTransform &transform) {
    const double position_values[3] = {transform.px, transform.py, transform.pz};
    const float  rotation_values[4] = {transform.rx, transform.ry, transform.rz, transform.rw};

    PyObject *position = PackFloats(position_values, 3);
    if (position == nullptr) [[unlikely]] {
        return nullptr;
    }
    PyObject *rotation = PackFloats(rotation_values, 4);
    if (rotation == nullptr) [[unlikely]] {
        Py_DECREF(position);
        return nullptr;
    }
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
    {
        const AllowThreads allow;
    created = CPH::Vehicle::CreateWheeled(self->physics, BodyHandle(chassis_raw), config);
    }

    if (!created) {
        return RaiseVehicleError(created.error());
    }

    ModuleState *state = GetModuleState(PyType_GetModule(Py_TYPE(self)));
    auto         *obj  = VehicleObject::New(reinterpret_cast<PyTypeObject *>(state->vehicleType));
    if (obj == nullptr) {
        return nullptr;
    }
    // The base zeroes every member, so the failure path below can decref a
    // half-built object safely.

    obj->parsers = CPH::Python::NewParserGroup<CPH::Python::VehicleParsers>();
    if (obj->parsers == nullptr) {
        Py_DECREF(reinterpret_cast<PyObject *>(obj));
        return PyErr_NoMemory();
    }

    obj->vehicle = created->release();

    // Exactly one strong reference to the world, released exactly once by the
    // GC hooks. There used to be a second
    // Py_INCREF(self) here, which nothing balanced: every wheeled vehicle
    // leaked a whole PhysicsWorld reference, so the world -- and everything it
    // owned -- stayed alive as long as any vehicle object existed. Tracked
    // vehicles never did this, so the two paths disagreed.
    obj->world = reinterpret_cast<PhysicsWorldObject *>(Py_NewRef(self));

    return obj->Publish();
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
//
// dealloc, traverse and clear all come from PyGcObject; VehicleBinding.hpp
// declares what they act on.

[[nodiscard]] PyObject *VehicleDestroy(VehicleObject *self,
                                       PyObject *ignored [[maybe_unused]]) {
    delete self->vehicle;
    self->vehicle = nullptr;
    Py_RETURN_NONE;
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
    CPH::Python::GcDeallocSlot<VehicleObject>(),
    CPH::Python::GcTraverseSlot<VehicleObject>(),
    CPH::Python::GcClearSlot<VehicleObject>(),
    {.slot = Py_tp_methods, .pfunc = reinterpret_cast<void *>(kVehicleMethods)},
    {.slot = Py_tp_getset, .pfunc = reinterpret_cast<void *>(VehicleGetSets)},
    {},
};

PyType_Spec VehicleSpec = CPH::Python::GcTypeSpec("culverin._culverin_c.Vehicle",
                                                   sizeof(VehicleObject), VehicleSlots);

} // namespace CPH::Python
