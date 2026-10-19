#pragma once
#include "ragdolls/RagdollParsers.hpp"
#include <Python.h>
#include <joltc.h>

namespace CPH::Python {

// --- Ragdoll Structures ---

struct SkeletonObject {
    PyObject_HEAD JPH_Skeleton *skeleton;
    CPH::Python::SkeletonParsers *parsers;
};

struct RagdollSettingsObject {
    PyObject_HEAD JPH_RagdollSettings *settings;
    struct PhysicsWorldObject *world; // Kept to access Shape Cache
    CPH::Python::RagdollSettingsParsers *parsers;
};

struct RagdollObject {
    PyObject_HEAD JPH_Ragdoll *ragdoll;
    struct PhysicsWorldObject *world;

    // We must track the handles of the parts so we can
    // invalid the slots when the ragdoll is destroyed.
    size_t body_count;
    uint32_t *body_slots;
    CPH::Python::RagdollParsers *parsers;
};

} // namespace CPH::Python
