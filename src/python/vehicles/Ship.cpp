// Python adapter for CPH::Ship.
//
// The flight model, the step listener and the sled resolution all live in
// src/vehicles/Ship.cpp. Everything below parses arguments, calls the native
// ship, and maps failures onto exceptions.

#include "vehicles/ShipBinding.hpp"
#include "bindings/ModuleState.hpp"
#include "bindings/PythonCommon.hpp"
#include "bindings/PythonSupport.hpp"
#include "physics/PhysicsSync.hpp"
#include <mutex>

namespace CPH::Python {

[[nodiscard]] PyObject * WorldCreateShip(PhysicsWorldObject *self, PyObject *const *args,
                                          size_t nargsf, PyObject *kwnames) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    uint64_t sled_raw      = 0;
    float kp               = 0.0f;
    float kd               = 0.0f;
    float throttle         = 0.0f;
    float steer            = 0.0f;
    float banking          = 0.15f;
    float grip             = 500.0f;
    float drag             = 10.0f;

    const ParseResult parsed = self->parsers->CreateShip.Parse(
        args, nargs, kwnames, sled_raw, kp, kd, throttle, steer, banking, grip, drag);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    // Pending commands have to be applied before the sled handle can resolve.
    {
        const std::scoped_lock lock(self->physics.shadow_lock);
        SyncAndFlushInternal(self);
    }

    const CPH::ShipConfig config{.sled_handle      = BodyHandle(sled_raw),
                                 .kp               = kp,
                                 .kd               = kd,
                                 .throttle_force   = throttle,
                                 .steer_speed      = steer,
                                 .banking_strength = banking,
                                 .lateral_grip     = grip,
                                 .linear_drag      = drag};

    std::unique_ptr<CPH::Ship> ship;
    Py_BEGIN_ALLOW_THREADS
    ship = CPH::Ship::Create(self->physics, config);
    Py_END_ALLOW_THREADS

    if (!ship) {
        return PyErr_Format(PyExc_ValueError, "Invalid sled handle.");
    }

    ModuleState *st = GetModuleState(PyType_GetModule(Py_TYPE(self)));
    auto obj = PyObject_GC_New(ShipObject, reinterpret_cast<PyTypeObject *>(st->ShipType));
    if (!obj) {
        return nullptr;
    }

    obj->parsers = CPH::Python::NewParserGroup<CPH::Python::ShipParsers>();
    if (!obj->parsers) {
        Py_DECREF(obj);
        PyErr_NoMemory();
        return nullptr;
    }

    obj->world = reinterpret_cast<PhysicsWorldObject *>(Py_NewRef(self));
    // The ship is registered with the physics system and must be destroyed
    // before the world reference is dropped; ShipDealloc enforces that order.
    obj->ship = ship.release();

    PyObject_GC_Track(reinterpret_cast<PyObject *>(obj));
    return reinterpret_cast<PyObject *>(obj);
}

[[nodiscard]] PyObject * ShipSetInput(ShipObject *self, PyObject *const *args, size_t nargsf,
                                       PyObject *kwnames) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    float fwd              = 0.0f;
    float right            = 0.0f;

    const ParseResult parsed = self->parsers->ShipInput.Parse(args, nargs, kwnames, fwd, right);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    // No locks and no GIL assumptions: the listener running in the physics
    // step picks these up from the atomics.
    self->ship->SetInput(fwd, right);
    Py_RETURN_NONE;
}

void ShipDealloc(ShipObject *self) {
    PyObject_GC_UnTrack(self);

    // Destroying the ship removes its step listener from the physics system.
    delete self->ship;
    self->ship = nullptr;

    delete self->parsers;
    self->parsers = nullptr;

    Py_XDECREF(self->world);
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

[[nodiscard]] int ShipTraverse(ShipObject *self, visitproc visit, void *arg) {
    Py_VISIT(self->world);
    return 0;
}

[[nodiscard]] int ShipClear(ShipObject *self) {
    Py_CLEAR(self->world);
    return 0;
}

static PyMethodDef kShipMethods[] = {CPH::Python::FastCallMethod("set_input", ShipSetInput),
                                     {nullptr, nullptr, 0, nullptr}};

static PyType_Slot ShipSlots[] = {
    {.slot = Py_tp_dealloc, .pfunc = reinterpret_cast<void *>(ShipDealloc)},
    {.slot = Py_tp_traverse, .pfunc = reinterpret_cast<void *>(ShipTraverse)},
    {.slot = Py_tp_clear, .pfunc = reinterpret_cast<void *>(ShipClear)},
    {.slot = Py_tp_methods, .pfunc = reinterpret_cast<void *>(kShipMethods)},
    {},
};

PyType_Spec ShipSpec = {
    .name      = "culverin._culverin_c.Ship",
    .basicsize = sizeof(ShipObject),
    .flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .slots     = ShipSlots,
};

} // namespace CPH::Python
