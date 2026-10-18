#pragma once
#include "ragdolls/RagdollParsers.hpp"
#include <Python.h>
#include <joltc.h>

// --- Ragdoll Structures ---

typedef struct CPH_SkeletonObject {
    PyObject_HEAD JPH_Skeleton *skeleton;
    CPH::Python::SkeletonParsers *parsers;
} CPH_SkeletonObject;

typedef struct {
    PyObject_HEAD JPH_RagdollSettings *settings;
    struct CPH_PhysicsWorldObject *world; // Kept to access Shape Cache
    CPH::Python::RagdollSettingsParsers *parsers;
} CPH_RagdollSettingsObject;

typedef struct {
    PyObject_HEAD JPH_Ragdoll *ragdoll;
    struct CPH_PhysicsWorldObject *world;

    // We must track the handles of the parts so we can
    // invalid the slots when the ragdoll is destroyed.
    size_t body_count;
    uint32_t *body_slots;
    CPH::Python::RagdollParsers *parsers;
} CPH_RagdollObject;