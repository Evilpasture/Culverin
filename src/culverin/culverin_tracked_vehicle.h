#pragma once
#include "culverin.h"
#include "culverin_vehicle.h"

struct PhysicsWorldObject;
typedef struct PhysicsWorldObject PhysicsWorldObject;

struct VehicleObject;
typedef struct VehicleObject VehicleObject;

typedef struct {
  float torque;
  float max_rpm;
  float min_rpm;
} TrackedEngineConfig;

// Helper: Only parse Python data into C structs (Must hold GIL)
typedef struct {
    uint32_t* indices;
    uint32_t count;
    uint32_t driven_idx;
} TrackData;

PyObject *PhysicsWorld_create_tracked_vehicle(struct PhysicsWorldObject *self,
                                              PyObject *args, PyObject *kwds);

PyObject *Vehicle_set_tank_input(struct VehicleObject *self, PyObject *args,
                                 PyObject *kwds);