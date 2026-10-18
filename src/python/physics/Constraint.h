#pragma once
#include "physics/PhysicsWorldBinding.h"
#include <Python.h>

PyCFunction_DeclareMethodFromModule CPH_PyWorld_CreateConstraint(CPH_PhysicsWorldObject *self,
                                                                 PyObject *const *args,
                                                                 size_t nargsf, PyObject *kwnames);

PyCFunction_DeclareMethodFromModule CPH_PyWorld_DestroyConstraint(CPH_PhysicsWorldObject *self,
                                                                  PyObject *const *args,
                                                                  size_t nargsf, PyObject *kwnames);

PyCFunction_DeclareMethodFromModule CPH_PyWorld_SetConstraintTarget(CPH_PhysicsWorldObject *self,
                                                                    PyObject *const *args,
                                                                    size_t nargsf,
                                                                    PyObject *kwnames);

PyCFunction_DeclareMethodFromModule CPH_PyWorld_GetConstraintType(CPH_PhysicsWorldObject *self,
                                                                  PyObject *const *args,
                                                                  size_t nargsf, PyObject *kwnames);