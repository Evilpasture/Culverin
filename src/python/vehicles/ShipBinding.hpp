#pragma once

// Thin adapter: the flight model is CPH::Ship (src/vehicles/); this object
// holds it plus the CPython header and the argument parsers.

#include "bindings/PyGcObject.hpp"
#include "core/Compiler.hpp"
#include "physics/PhysicsWorldBinding.hpp"
#include "vehicles/Ship.hpp"
#include "vehicles/ShipParsers.hpp"
#include <Python.h>

namespace CPH::Python {

// Lifetime, traverse/clear and dealloc come from PyGcObject. The PyObject_HEAD
// is supplied by the base.
struct ShipObject : PyGcObject<ShipObject> {
    PhysicsWorldObject       *world   = nullptr;
    // Owned raw pointer for the same reason as VehicleObject::vehicle: this
    // storage comes from PyObject_GC_New.
    CPH::Ship               *ship    = nullptr;
    CPH::Python::ShipParsers *parsers = nullptr;

    // ~Ship removes its step listener through the world's trampoline lock, so
    // the ship must be destroyed while the world reference is still held. That
    // ordering is the whole reason this hook exists: the previous tp_clear
    // dropped the world first and left ~Ship with a dangling world_.
    void BeforeClear() noexcept {
        delete ship;
        ship = nullptr;
    }

    template <typename Visitor>
    void VisitRefs(Visitor &&visit) noexcept {
        visit(world);
    }

    void BeforeFree() noexcept {
        delete parsers;
        parsers = nullptr;
    }
};

} // namespace CPH::Python
