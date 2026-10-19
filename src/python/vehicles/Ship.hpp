#pragma once
#include "core/CompilerSpecifics.h"
#include "joltc.h"
#include "physics/PhysicsWorldBinding.hpp"
#include "vehicles/ShipParsers.hpp"
#include <Python.h>

namespace CPH::Python {

struct ShipObject {
    PyObject_HEAD PhysicsWorldObject *world;
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
};

} // namespace CPH::Python
