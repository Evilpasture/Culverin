#pragma once
#include "bindings/parsing/ParserRegistry.hpp"
#include <Python.h>
#include <joltc.h>

// --- Ragdoll Structures ---

typedef struct CPH_SkeletonObject {
    PyObject_HEAD JPH_Skeleton *skeleton;
    CPH_SkeletonParsers *parsers;
} CPH_SkeletonObject;

typedef struct {
    PyObject_HEAD JPH_RagdollSettings *settings;
    struct CPH_PhysicsWorldObject *world; // Kept to access Shape Cache
    CPH_RagdollSettingsParsers *parsers;
} CPH_RagdollSettingsObject;

typedef struct {
    PyObject_HEAD JPH_Ragdoll *ragdoll;
    struct CPH_PhysicsWorldObject *world;

    // We must track the handles of the parts so we can
    // invalid the slots when the ragdoll is destroyed.
    size_t body_count;
    uint32_t *body_slots;
    CPH_RagdollParsers *parsers;
} CPH_RagdollObject;