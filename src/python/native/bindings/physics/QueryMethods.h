#pragma once

#include "bindings/physics/PhysicsWorld.h"
#include "physics/Query.h"
#include <Python.h>

PyCFunction_DeclareMethodFromModule PhysicsWorld_overlap_sphere(PhysicsWorldObject *self,
                                                                PyObject *const *args,
                                                                size_t nargsf, PyObject *kwnames);

PyCFunction_DeclareMethodFromModule PhysicsWorld_overlap_aabb(PhysicsWorldObject *self,
                                                              PyObject *const *args, size_t nargsf,
                                                              PyObject *kwnames);

PyCFunction_DeclareMethodFromModule PhysicsWorld_raycast(PhysicsWorldObject *self,
                                                         PyObject *const *args, size_t nargsf,
                                                         PyObject *kwnames);

PyCFunction_DeclareMethodFromModule PhysicsWorld_raycast_batch(PhysicsWorldObject *self,
                                                               PyObject *const *args, size_t nargsf,
                                                               PyObject *kwnames);

PyCFunction_DeclareMethodFromModule PhysicsWorld_shapecast(PhysicsWorldObject *self,
                                                           PyObject *const *args, size_t nargsf,
                                                           PyObject *kwnames);
