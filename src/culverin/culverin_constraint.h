#pragma once
#include "culverin.h"

PyObject *PhysicsWorld_create_constraint(PhysicsWorldObject *self,
                                                PyObject *args,
                                                PyObject *kwds);

PyObject *PhysicsWorld_destroy_constraint(PhysicsWorldObject *self,
                                                 PyObject *args,
                                                 PyObject *kwds);

PyObject *PhysicsWorld_set_constraint_target(PhysicsWorldObject *self,
                                                    PyObject *args, PyObject *kwds);