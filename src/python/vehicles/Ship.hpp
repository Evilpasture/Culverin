#pragma once
#include "core/Compiler.hpp"
#include "joltc.h"
#include "physics/PhysicsWorldBinding.hpp"
#include "vehicles/ShipParsers.hpp"
#include <Jolt/Physics/PhysicsStepListener.h>
#include <Python.h>

namespace CPH::Python {

struct ShipObject {
    PyObject_HEAD PhysicsWorldObject *world;
    // Owned here; PhysicsSystem::AddStepListener stores a non-owning pointer,
    // so this is deleted after RemoveStepListener.
    JPH::PhysicsStepListener *listener;
    CPH::Python::ShipParsers *parsers;
    JPH_BodyID sled_bid; // Resolved ID for the heavy part
    uint64_t sled_h_raw; // Added to store the Python-side handle

    // Inputs (Atomics so Python can write while C reads)
    std::atomic<float> input_fwd;
    std::atomic<float> input_right;

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
