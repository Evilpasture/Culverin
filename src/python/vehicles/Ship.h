#pragma once
#include "bindings/physics/PhysicsWorld.h"
#include "core/CompilerSpecifics.h"
#include "joltc.h"
#include "vehicles/ShipParsers.hpp"
#include <Python.h>

typedef struct CPH_ShipObject {
    PyObject_HEAD CPH_PhysicsWorldObject *world;
    JPH_PhysicsStepListener *listener; // Native listener pointer
    CPH::Python::ShipParsers *parsers;
    JPH_BodyID sled_bid; // Resolved ID for the heavy part
    uint64_t sled_h_raw; // Added to store the Python-side handle

    // Inputs (Atomics so Python can write while C reads)
    CPH_ATOMIC(float) input_fwd;
    CPH_ATOMIC(float) input_right;

    // Config
    float kp;
    float kd;
    float throttle_force;
    float steer_speed;
    float banking_strength;
    float lateral_grip;
    float linear_drag;
} CPH_ShipObject;
