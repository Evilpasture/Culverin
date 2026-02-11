#pragma once
#include "culverin.h"

// Helper for Overlap Callbacks
typedef struct {
  PhysicsWorldObject *world;
  uint64_t *hits; // C array to store baked handles
  size_t count;
  size_t capacity;
} OverlapContext;

PyObject *PhysicsWorld_overlap_sphere(PhysicsWorldObject *self, PyObject *args,
                                      PyObject *kwds);

PyObject *PhysicsWorld_overlap_aabb(PhysicsWorldObject *self, PyObject *args,
                                    PyObject *kwds);

PyObject *PhysicsWorld_raycast(PhysicsWorldObject *self, PyObject *args,
                               PyObject *kwds);

static PyObject *PhysicsWorld_raycast_batch(PhysicsWorldObject *self,
                                            PyObject *const *args, 
                                            size_t nargsf, 
                                            PyObject *kwnames);

PyObject *PhysicsWorld_shapecast(PhysicsWorldObject *self, PyObject *args,
                                 PyObject *kwds);