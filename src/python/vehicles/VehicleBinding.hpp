#pragma once

// Thin adapter: the vehicle is CPH::Vehicle (src/vehicles/); this object holds
// it plus the CPython header and the argument parsers. Turning the Python wheel
// and drivetrain dictionaries into native config structs happens in Vehicle.cpp
// -- that is parsing, which is what this layer is for.

#include "core/Compiler.hpp"
#include "physics/PhysicsWorldBinding.hpp"
#include "vehicles/Vehicle.hpp"
#include "vehicles/VehicleParsers.hpp"
#include <Python.h>

namespace CPH::Python {

struct VehicleObject {
    PyObject_HEAD
    PhysicsWorldObject *world;
    // Owned, and deliberately a raw pointer: vehicles are allocated with
    // PyObject_GC_New, which does NOT zero the storage the way tp_alloc does,
    // so a non-trivial member would be assigned over indeterminate bytes.
    CPH::Vehicle *vehicle;
    CPH::Python::VehicleParsers *parsers;
};

extern PyType_Spec VehicleSpec;

[[nodiscard]] PyObject *WorldCreateVehicle(PhysicsWorldObject *self, PyObject *const *args,
                                           size_t nargsf, PyObject *kwnames);
[[nodiscard]] PyObject *WorldCreateTrackedVehicle(PhysicsWorldObject *self, PyObject *const *args,
                                                  size_t nargsf, PyObject *kwnames);
[[nodiscard]] PyObject *VehicleSetInput(VehicleObject *self, PyObject *const *args, size_t nargsf,
                                        PyObject *kwnames);
[[nodiscard]] PyObject *VehicleSetTankInput(VehicleObject *self, PyObject *const *args,
                                            size_t nargsf, PyObject *kwnames);

} // namespace CPH::Python
