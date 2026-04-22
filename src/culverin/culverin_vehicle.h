#pragma once
#include "joltc.h"
#include <Python.h>

constexpr uint32_t LAYER_STATIC  = 1;
constexpr uint32_t LAYER_DYNAMIC = 2;
constexpr uint32_t LAYER_SPECIAL = 3;

typedef struct {
    JPH_LinearCurve *f_curve;
    JPH_LinearCurve *t_curve;
    JPH_WheelSettings **w_settings;
    JPH_WheeledVehicleControllerSettings *v_ctrl;
    JPH_VehicleTransmissionSettings *v_trans_set;
    JPH_VehicleCollisionTesterRay *tester;
    JPH_VehicleConstraint *j_veh;
    bool is_added_to_world;
} VehicleResources;

typedef struct VehicleObject {
    PyObject_HEAD JPH_VehicleConstraint *vehicle;
    JPH_VehicleCollisionTester *tester;
    struct PhysicsWorldObject *world;

    // Ownership tracking for cleanup
    JPH_WheelSettings **wheel_settings;
    JPH_VehicleControllerSettings *controller_settings;
    JPH_VehicleTransmissionSettings *transmission_settings; // NEW: Keep alive
    JPH_LinearCurve *friction_curve;
    JPH_LinearCurve *torque_curve;

    uint32_t num_wheels;
    int current_gear;
} VehicleObject;