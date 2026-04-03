#pragma once
#include "culverin.h"

PyCFunction_DeclareMethodFromModule PhysicsWorld_create_constraint(PhysicsWorldObject *self,
                                                                   PyObject *const *args,
                                                                   size_t nargsf,
                                                                   PyObject *kwnames);

PyCFunction_DeclareMethodFromModule PhysicsWorld_destroy_constraint(PhysicsWorldObject *self,
                                                                    PyObject *const *args,
                                                                    size_t nargsf,
                                                                    PyObject *kwnames);

PyCFunction_DeclareMethodFromModule PhysicsWorld_set_constraint_target(PhysicsWorldObject *self,
                                                                       PyObject *const *args,
                                                                       size_t nargsf,
                                                                       PyObject *kwnames);
                                                                       
PyCFunction_DeclareMethodFromModule PhysicsWorld_get_constraint_type(PhysicsWorldObject *self,
                                                                     PyObject *const *args,
                                                                     size_t nargsf,
                                                                     PyObject *kwnames);