#pragma once
#include "culverin.h"
#include <Python.h>


// --- Character Object ---
typedef struct CharacterObject {
  PyObject_HEAD JPH_CharacterVirtual *character;
  struct PhysicsWorldObject *world;
  BodyHandle handle;

  // Filters and listeners
  JPH_BodyFilter *body_filter;
  JPH_ShapeFilter *shape_filter;
  JPH_BroadPhaseLayerFilter *bp_filter;
  JPH_ObjectLayerFilter *obj_filter;
  JPH_CharacterContactListener *listener;

  // ATOMIC INPUTS: Read by Jolt worker threads in callbacks
  _Atomic float push_strength;
  _Atomic float last_vx;
  _Atomic float last_vy;
  _Atomic float last_vz;

  // Non-atomic: Used by main thread only for rendering
  // AVOID FALSE SHARING.
#if defined(_MSC_VER)
  __declspec(align(64))
#else
  _Alignas(64)
#endif
  float prev_px,
      prev_py, prev_pz;
  float prev_rx, prev_ry, prev_rz, prev_rw;
} CharacterObject;

/* We expose the Procs table so the main module can assign it
   when creating the Character Virtual instance.
*/
extern const JPH_CharacterContactListener_Procs char_listener_procs;

PyObject *Character_move(CharacterObject *self, PyObject *args, PyObject *kwds);

PyObject *Character_get_position(CharacterObject *self,
                                 PyObject *Py_UNUSED(ignored));

PyObject *Character_set_position(CharacterObject *self, PyObject *args,
                                 PyObject *kwds);

PyObject *Character_set_rotation(CharacterObject *self, PyObject *args,
                                 PyObject *kwds);

PyObject *Character_is_grounded(CharacterObject *self, PyObject *args);

PyObject *Character_set_strength(CharacterObject *self, PyObject *args);

PyObject *Character_get_render_transform(CharacterObject *self, PyObject *arg);

int Character_traverse(CharacterObject *self, visitproc visit, void *arg);

int Character_clear(CharacterObject *self);

void Character_dealloc(CharacterObject *self);

PyObject *PhysicsWorld_create_character(PhysicsWorldObject *self,
                                        PyObject *args, PyObject *kwds);