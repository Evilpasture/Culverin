#pragma once
#include "bindings/BindingCommon.h"
#include "bindings/parsing/ArgIndices.h"
#include <Python.h>

// --- Character Object ---
typedef struct CPH_CharacterObject {
    PyObject_HEAD JPH_CharacterVirtual *character;
    struct CPH_PhysicsWorldObject *world;
    CPH_ATOMIC(BodyHandle) handle;

    // Filters and listeners
    JPH_BodyFilter *body_filter;
    JPH_ShapeFilter *shape_filter;
    JPH_BroadPhaseLayerFilter *bp_filter;
    JPH_ObjectLayerFilter *obj_filter;
    JPH_CharacterContactListener *listener;

    // Parsers
    CPH_CharacterParsers *parsers;

    // ATOMIC INPUTS: Read by Jolt worker threads in callbacks
    CPH_ATOMIC(float) push_strength;
    CPH_ATOMIC(float) last_vx;
    CPH_ATOMIC(float) last_vy;
    CPH_ATOMIC(float) last_vz;

    // Non-atomic: Used by main thread only for rendering
    // AVOID FALSE SHARING.
    JPH_Real prev_px, prev_py, prev_pz;
    float prev_rx, prev_ry, prev_rz, prev_rw;
} CPH_CharacterObject;

typedef struct {
    float height;
    float radius;
    float max_slope;
} CPH_CharacterParams;

/* We expose the Procs table so the main module can assign it
   when creating the Character Virtual instance.
*/

extern const JPH_CharacterContactListener_Procs kCharListenerProcs;

extern PyType_Spec Character_spec;

struct CPH_PhysicsWorldObject;

PyCFunction_DeclareMethodFromModule CPH_PyPhysicsWorldCreateCharacter(
    struct CPH_PhysicsWorldObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
