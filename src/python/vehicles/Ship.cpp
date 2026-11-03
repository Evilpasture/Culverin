// Python adapter for CPH::Ship.
//
// The flight model, the step listener and the sled resolution all live in
// src/vehicles/Ship.cpp. Everything below parses arguments, calls the native
// ship, and maps failures onto exceptions.

#include "vehicles/ShipBinding.hpp"
#include "bindings/ModuleState.hpp"
#include "bindings/ThreadState.hpp"
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

    const ParseResult parsed = self->parsers->createShip.Parse(
        args, nargs, kwnames, sled_raw, kp, kd, throttle, steer, banking, grip, drag);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    // Pending commands have to be applied before the sled handle can resolve.
    {
        const std::scoped_lock lock(self->physics.shadowLock);
        SyncAndFlushInternal(self);
    }

    const CPH::ShipConfig config{.sledHandle      = BodyHandle(sled_raw),
                                 .kp               = kp,
                                 .kd               = kd,
                                 .throttleForce   = throttle,
                                 .steerSpeed      = steer,
                                 .bankingStrength = banking,
                                 .lateralGrip     = grip,
                                 .linearDrag      = drag};

    std::unique_ptr<CPH::Ship> ship;
    {
        const AllowThreads allow;
    ship = CPH::Ship::Create(self->physics, config);
    }

    if (!ship) {
        return PyErr_Format(PyExc_ValueError, "Invalid sled handle.");
    }

    ModuleState *st = GetModuleState(PyType_GetModule(Py_TYPE(self)));
    auto *obj = ShipObject::New(reinterpret_cast<PyTypeObject *>(st->shipType));
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
    // before the world reference is dropped; ShipObject::BeforeClear enforces
    // that order.
    obj->ship = ship.release();

    return obj->Publish();
}

[[nodiscard]] PyObject * ShipSetInput(ShipObject *self, PyObject *const *args, size_t nargsf,
                                       PyObject *kwnames) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    float fwd              = 0.0f;
    float right            = 0.0f;

    const ParseResult parsed = self->parsers->shipInput.Parse(args, nargs, kwnames, fwd, right);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    // No locks and no GIL assumptions: the listener running in the physics
    // step picks these up from the atomics.
    self->ship->SetInput(fwd, right);
    Py_RETURN_NONE;
}

// dealloc, traverse and clear all come from PyGcObject; ShipBinding.hpp says
// what they act on. In particular BeforeClear() destroys the ship *before* the
// world reference is dropped, which the hand-written tp_clear did not.

static PyMethodDef kShipMethods[] = {CPH::Python::FastCallMethod("set_input", ShipSetInput),
                                     {nullptr, nullptr, 0, nullptr}};

static PyType_Slot ShipSlots[] = {
    CPH::Python::GcDeallocSlot<ShipObject>(),
    CPH::Python::GcTraverseSlot<ShipObject>(),
    CPH::Python::GcClearSlot<ShipObject>(),
    {.slot = Py_tp_methods, .pfunc = reinterpret_cast<void *>(kShipMethods)},
    {},
};

PyType_Spec ShipSpec =
    CPH::Python::GcTypeSpec("culverin._culverin_c.Ship", sizeof(ShipObject), ShipSlots);

} // namespace CPH::Python
