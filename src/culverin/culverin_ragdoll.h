#pragma once
#include <Python.h>
#include <joltc.h>

// --- Ragdoll Structures ---

typedef struct SkeletonObject {
    PyObject_HEAD JPH_Skeleton *skeleton;
} SkeletonObject;

typedef struct {
    PyObject_HEAD JPH_RagdollSettings *settings;
    struct PhysicsWorldObject *world; // Kept to access Shape Cache
} RagdollSettingsObject;

typedef struct {
    PyObject_HEAD JPH_Ragdoll *ragdoll;
    struct PhysicsWorldObject *world;

    // We must track the handles of the parts so we can
    // invalid the slots when the ragdoll is destroyed.
    size_t body_count;
    uint32_t *body_slots;
} RagdollObject;