#pragma once
#include "culverin.h"

typedef struct {
    float torque;
    float max_rpm;
    float min_rpm;
} TrackedEngineConfig;


PyObject *PhysicsWorld_create_tracked_vehicle(struct PhysicsWorldObject *self,
                                                     PyObject *args,
                                                     PyObject *kwds);