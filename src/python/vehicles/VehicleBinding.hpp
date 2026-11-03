#pragma once

// Thin adapter: the vehicle is CPH::Vehicle (src/vehicles/); this object holds
// it plus the CPython header and the argument parsers. Turning the Python wheel
// and drivetrain dictionaries into native config structs happens in Vehicle.cpp
// -- that is parsing, which is what this layer is for.

#include "bindings/PyGcObject.hpp"
#include "core/Compiler.hpp"
#include "physics/PhysicsWorldBinding.hpp"
#include "vehicles/Vehicle.hpp"
#include "vehicles/VehicleParsers.hpp"
#include <Python.h>
#include "bindings/Result.hpp"

namespace CPH::Python {

// Lifetime, traverse/clear and dealloc come from PyGcObject. The PyObject_HEAD
// is supplied by the base.
struct VehicleObject : PyGcObject<VehicleObject> {
    PhysicsWorldObject           *world   = nullptr;
    // Owned, and deliberately a raw pointer: the storage comes from
    // PyObject_GC_New, which does NOT zero it the way tp_alloc does, so a
    // non-trivial member would be assigned over indeterminate bytes. (The base
    // zeroes every member for exactly this reason -- see New().)
    CPH::Vehicle                 *vehicle = nullptr;
    CPH::Python::VehicleParsers  *parsers = nullptr;

    // The vehicle is registered with the physics system and has to leave it
    // before the world reference goes away.
    void BeforeClear() noexcept {
        delete vehicle;
        vehicle = nullptr;
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

extern PyType_Spec VehicleSpec;

[[nodiscard]] BoundResult WorldCreateVehicle(PhysicsWorldObject *self, PyObject *const *args,
                                           size_t nargsf, PyObject *kwnames);
[[nodiscard]] BoundResult WorldCreateTrackedVehicle(PhysicsWorldObject *self, PyObject *const *args,
                                                  size_t nargsf, PyObject *kwnames);
[[nodiscard]] BoundResult VehicleSetInput(VehicleObject *self, PyObject *const *args, size_t nargsf,
                                        PyObject *kwnames);
[[nodiscard]] BoundResult VehicleSetTankInput(VehicleObject *self, PyObject *const *args,
                                            size_t nargsf, PyObject *kwnames);

} // namespace CPH::Python
