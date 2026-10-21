#pragma once
#include "jolt/Jolt.hpp"
#include "vehicles/VehicleParsers.hpp"
#include <Jolt/Physics/Vehicle/VehicleCollisionTester.h>
#include <Jolt/Physics/Vehicle/VehicleConstraint.h>
#include <Python.h>

namespace CPH::Python {

constexpr uint32_t kLayerStatic  = 1;
constexpr uint32_t kLayerDynamic = 2;
constexpr uint32_t kLayerSpecial = 3;

// Wheel, engine and transmission settings are deliberately NOT stored here.
// WheelSettings are held by the constraint itself (Wheel keeps a
// RefConst<WheelSettings>), and the engine/transmission settings are copied by
// value into the controller at construction, so nothing needs to be kept alive
// past WorldCreateVehicle. The five ownership pointers this struct used to
// carry existed only to free those objects during teardown.
struct VehicleObject {
    PyObject_HEAD JPH::VehicleConstraint *vehicle;

    // Creation reference to the collision tester. VehicleConstraint keeps its
    // own RefConst, so this is dropped with Release() during cleanup.
    JPH::VehicleCollisionTester *tester;
    struct PhysicsWorldObject *world;
    CPH::Python::VehicleParsers *parsers;

    uint32_t num_wheels;
    int current_gear;
};

} // namespace CPH::Python
