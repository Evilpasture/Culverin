#pragma once
#include "bindings/physics/PhysicsWorld.h"
#include <Python.h>

PyCFunction_DeclareMethodFromModule CPH_PyPhysicsWorldCreateConstraint(CPH_PhysicsWorldObject *self,
                                                                   PyObject *const *args,
                                                                   size_t nargsf,
                                                                   PyObject *kwnames);

PyCFunction_DeclareMethodFromModule CPH_PyPhysicsWorldDestroyConstraint(CPH_PhysicsWorldObject *self,
                                                                    PyObject *const *args,
                                                                    size_t nargsf,
                                                                    PyObject *kwnames);

PyCFunction_DeclareMethodFromModule CPH_PyPhysicsWorldSetConstraintTarget(CPH_PhysicsWorldObject *self,
                                                                       PyObject *const *args,
                                                                       size_t nargsf,
                                                                       PyObject *kwnames);

PyCFunction_DeclareMethodFromModule CPH_PyPhysicsWorldGetConstraintType(CPH_PhysicsWorldObject *self,
                                                                     PyObject *const *args,
                                                                     size_t nargsf,
                                                                     PyObject *kwnames);