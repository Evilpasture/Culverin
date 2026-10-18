#pragma once

#include "bindings/physics/PhysicsWorld.h"
#include "physics/Query.h"
#include <Python.h>

PyCFunction_DeclareMethodFromModule CPH_PyWorld_OverlapSphere(CPH_PhysicsWorldObject *self,
                                                                PyObject *const *args,
                                                                size_t nargsf, PyObject *kwnames);

PyCFunction_DeclareMethodFromModule CPH_PyWorld_OverlapAabb(CPH_PhysicsWorldObject *self,
                                                              PyObject *const *args, size_t nargsf,
                                                              PyObject *kwnames);

PyCFunction_DeclareMethodFromModule CPH_PyWorld_Raycast(CPH_PhysicsWorldObject *self,
                                                         PyObject *const *args, size_t nargsf,
                                                         PyObject *kwnames);

PyCFunction_DeclareMethodFromModule CPH_PyWorld_RaycastBatch(CPH_PhysicsWorldObject *self,
                                                               PyObject *const *args, size_t nargsf,
                                                               PyObject *kwnames);

PyCFunction_DeclareMethodFromModule CPH_PyWorld_Shapecast(CPH_PhysicsWorldObject *self,
                                                           PyObject *const *args, size_t nargsf,
                                                           PyObject *kwnames);
