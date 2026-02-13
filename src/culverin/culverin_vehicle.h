#pragma once
#include "culverin.h"

constexpr uint32_t LAYER_STATIC = 1;
constexpr uint32_t LAYER_DYNAMIC = 2;
constexpr uint32_t LAYER_SPECIAL  = 3;

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

PyObject *PhysicsWorld_create_vehicle(struct PhysicsWorldObject *self,
                                      PyObject *args, PyObject *kwds);
void cleanup_vehicle_resources(VehicleResources *r, uint32_t num_wheels,
                               struct PhysicsWorldObject *self);

PyObject *Vehicle_set_input(VehicleObject *self, PyObject *args,
                            PyObject *kwds);

PyObject *Vehicle_get_wheel_transform(VehicleObject *self, PyObject *args);

PyObject *Vehicle_get_wheel_local_transform(VehicleObject *self,
                                            PyObject *args);

PyObject *Vehicle_get_debug_state(VehicleObject *self,
                                  PyObject *Py_UNUSED(ignored));

int Vehicle_traverse(VehicleObject *self, visitproc visit, void *arg);

int Vehicle_clear(VehicleObject *self);

PyObject *Vehicle_destroy(VehicleObject *self, PyObject *Py_UNUSED(ignored));

void Vehicle_dealloc(VehicleObject *self);