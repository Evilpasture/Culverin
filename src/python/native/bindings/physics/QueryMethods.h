#pragma once

#include "bindings/physics/PhysicsWorld.h"
#include "physics/Query.h"
#include <Python.h>

PyCFunction_DeclareMethodFromModule CPH_PyPhysicsWorldOverlapSphere(CPH_PhysicsWorldObject *self,
                                                                PyObject *const *args,
                                                                size_t nargsf, PyObject *kwnames);

PyCFunction_DeclareMethodFromModule CPH_PyPhysicsWorldOverlapAabb(CPH_PhysicsWorldObject *self,
                                                              PyObject *const *args, size_t nargsf,
                                                              PyObject *kwnames);

PyCFunction_DeclareMethodFromModule CPH_PyPhysicsWorldRaycast(CPH_PhysicsWorldObject *self,
                                                         PyObject *const *args, size_t nargsf,
                                                         PyObject *kwnames);

PyCFunction_DeclareMethodFromModule CPH_PyPhysicsWorldRaycastBatch(CPH_PhysicsWorldObject *self,
                                                               PyObject *const *args, size_t nargsf,
                                                               PyObject *kwnames);

PyCFunction_DeclareMethodFromModule CPH_PyPhysicsWorldShapecast(CPH_PhysicsWorldObject *self,
                                                           PyObject *const *args, size_t nargsf,
                                                           PyObject *kwnames);
