#pragma once
#include "culverin.h"
#include <Python.h>

// --- Character Object ---
typedef struct CharacterObject {
    PyObject_HEAD JPH_CharacterVirtual *character;
    struct PhysicsWorldObject *world;
    CULV_ATOMIC(BodyHandle) handle;

    // Filters and listeners
    JPH_BodyFilter *body_filter;
    JPH_ShapeFilter *shape_filter;
    JPH_BroadPhaseLayerFilter *bp_filter;
    JPH_ObjectLayerFilter *obj_filter;
    JPH_CharacterContactListener *listener;

    // ATOMIC INPUTS: Read by Jolt worker threads in callbacks
    CULV_ATOMIC(float) push_strength;
    CULV_ATOMIC(float) last_vx;
    CULV_ATOMIC(float) last_vy;
    CULV_ATOMIC(float) last_vz;

    // Non-atomic: Used by main thread only for rendering
    // AVOID FALSE SHARING.
    JPH_Real prev_px, prev_py, prev_pz;
    float prev_rx, prev_ry, prev_rz, prev_rw;
} CharacterObject;

typedef struct {
    float height;
    float radius;
    float max_slope;
} CharacterParams;

/* We expose the Procs table so the main module can assign it
   when creating the Character Virtual instance.
*/

extern const JPH_CharacterContactListener_Procs char_listener_procs;

extern PyGetSetDef Character_getset[];
extern PyMethodDef Character_methods[];

extern const PyType_Spec Character_spec;

PyCFunction_DeclareMethodFromModule PhysicsWorld_create_character(PhysicsWorldObject *self,
                                                                  PyObject *const *args,
                                                                  Py_ssize_t nargs,
                                                                  PyObject *kwnames);