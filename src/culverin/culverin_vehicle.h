#pragma once
#include "culverin.h"

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

PyCFunction_DeclareMethodFromModule PhysicsWorld_create_vehicle(PhysicsWorldObject *self,
                                                                PyObject *const *args,
                                                                Py_ssize_t nargs,
                                                                PyObject *kwnames);
void cleanup_vehicle_resources(VehicleResources *r, uint32_t num_wheels,
                               struct PhysicsWorldObject *self);

PyCFunction_DeclareMethodFromModule Vehicle_set_input(VehicleObject *self, PyObject *const *args,
                                                      Py_ssize_t nargs, PyObject *kwnames);

PyCFunction_DeclareMethodFromModule Vehicle_get_wheel_transform(VehicleObject *self,
                                                                PyObject *const *args,
                                                                Py_ssize_t nargs,
                                                                PyObject *kwnames);

PyCFunction_DeclareMethodFromModule Vehicle_get_wheel_local_transform(VehicleObject *self,
                                                                      PyObject *const *args,
                                                                      Py_ssize_t nargs,
                                                                      PyObject *kwnames);

PyCFunction_DeclareMethodFromModule Vehicle_get_debug_state(VehicleObject *self,
                                                            PyObject *Py_UNUSED(ignored));

PyType_DeclareSlot_StatusFromModule Vehicle_traverse(VehicleObject *self, visitproc visit,
                                                     void *arg);

PyType_DeclareSlot_StatusFromModule Vehicle_clear(VehicleObject *self);

PyCFunction_DeclareMethodFromModule Vehicle_destroy(VehicleObject *self,
                                                    PyObject *Py_UNUSED(ignored));

PyType_DeclareSlot_VoidFromModule Vehicle_dealloc(VehicleObject *self);
