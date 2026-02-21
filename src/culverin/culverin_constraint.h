#pragma once
#include "culverin.h"

PyObject *PhysicsWorld_create_constraint(PhysicsWorldObject *self,
                                         PyObject *const *args, size_t nargsf,
                                         PyObject *kwnames);

PyObject *PhysicsWorld_destroy_constraint(PhysicsWorldObject *self,
                                          PyObject *const *args, size_t nargsf,
                                          PyObject *kwnames);

PyObject *PhysicsWorld_set_constraint_target(PhysicsWorldObject *self,
                                             PyObject *const *args, size_t nargsf,
                                             PyObject *kwnames);