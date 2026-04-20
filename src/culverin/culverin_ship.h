#pragma once
#include "culverin.h"
#include <Python.h>

typedef struct ShipObject {
    PyObject_HEAD
    struct PhysicsWorldObject *world;
    JPH_PhysicsStepListener *listener; // Native listener pointer
    JPH_BodyID sled_bid;               // Resolved ID for the heavy part
    uint64_t sled_h_raw;               // Added to store the Python-side handle
    
    // Inputs (Atomics so Python can write while C reads)
    CULV_ATOMIC(float) input_fwd;
    CULV_ATOMIC(float) input_right;

    // Config
    float kp;
    float kd;
    float throttle_force;
    float steer_speed;
    float banking_strength;
    float lateral_grip;
    float linear_drag;
} ShipObject;

extern PyType_Spec Ship_spec;

PyCFunction_DeclareMethodFromModule PhysicsWorld_create_ship(PhysicsWorldObject *self,
                                                             PyObject *const *args,
                                                             Py_ssize_t nargs,
                                                             PyObject *kwnames);

PyCFunction_DeclareMethodFromModule Ship_set_input(ShipObject *self, PyObject *const *args,
                                                   Py_ssize_t nargs, PyObject *kwnames);