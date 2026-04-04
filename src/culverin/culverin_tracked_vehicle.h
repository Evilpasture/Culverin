#pragma once
#include "culverin.h"
#include "culverin_vehicle.h"

CULV_MAYBE_UNUSED static constexpr uint32_t TRACKED_LAYER_MOVING   = 0;
CULV_MAYBE_UNUSED static constexpr uint32_t TRACKED_LAYER_STATIC   = 1;
CULV_MAYBE_UNUSED static constexpr uint32_t TRACKED_LAYER_DRIVABLE = 2;

struct PhysicsWorldObject;
typedef struct PhysicsWorldObject PhysicsWorldObject;

struct VehicleObject;
typedef struct VehicleObject VehicleObject;

typedef struct {
    float torque;
    float max_rpm;
    float min_rpm;
} TrackedEngineConfig;

PyCFunction_DeclareMethodFromModule PhysicsWorld_create_tracked_vehicle(PhysicsWorldObject *self,
                                                                        PyObject *const *args,
                                                                        Py_ssize_t nargs,
                                                                        PyObject *kwnames);

PyCFunction_DeclareMethodFromModule Vehicle_set_tank_input(VehicleObject *self,
                                                           PyObject *const *args, Py_ssize_t nargs,
                                                           PyObject *kwnames);
