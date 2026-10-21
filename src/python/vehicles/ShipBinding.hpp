#pragma once

// Thin adapter: the flight model is CPH::Ship (src/vehicles/); this object
// holds it plus the CPython header and the argument parsers.

#include "core/Compiler.hpp"
#include "physics/PhysicsWorldBinding.hpp"
#include "vehicles/Ship.hpp"
#include "vehicles/ShipParsers.hpp"
#include <Python.h>

namespace CPH::Python {

struct ShipObject {
    PyObject_HEAD
    PhysicsWorldObject *world;
    // Owned, and deliberately a raw pointer: ships are allocated with
    // PyObject_GC_New, which does NOT zero the storage the way tp_alloc does,
    // so a non-trivial member would be assigned over indeterminate bytes.
    CPH::Ship *ship;
    CPH::Python::ShipParsers *parsers;
};

} // namespace CPH::Python
