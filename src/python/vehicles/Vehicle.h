#pragma once
#include "joltc.h"
#include "vehicles/VehicleParsers.hpp"
#include <Python.h>

constexpr uint32_t kLayerStatic  = 1;
constexpr uint32_t kLayerDynamic = 2;
constexpr uint32_t kLayerSpecial = 3;

typedef struct {
    JPH_LinearCurve *f_curve;
    JPH_LinearCurve *t_curve;
    JPH_WheelSettings **w_settings;
    JPH_WheeledVehicleControllerSettings *v_ctrl;
    JPH_VehicleTransmissionSettings *v_trans_set;
    JPH_VehicleCollisionTesterRay *tester;
    JPH_VehicleConstraint *j_veh;
    bool is_added_to_world;
} CPH_VehicleResources;

typedef struct CPH_VehicleObject {
    PyObject_HEAD JPH_VehicleConstraint *vehicle;
    JPH_VehicleCollisionTester *tester;
    struct CPH_PhysicsWorldObject *world;

    // Ownership tracking for cleanup
    JPH_WheelSettings **wheel_settings;
    JPH_VehicleControllerSettings *controller_settings;
    JPH_VehicleTransmissionSettings *transmission_settings; // NEW: Keep alive
    JPH_LinearCurve *friction_curve;
    JPH_LinearCurve *torque_curve;
    CPH::Python::VehicleParsers *parsers;

    uint32_t num_wheels;
    int current_gear;
} CPH_VehicleObject;