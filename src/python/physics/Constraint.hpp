#pragma once
#include "physics/PhysicsWorldBinding.hpp"
#include <Python.h>

namespace CPH::Python {

[[nodiscard]] PyObject * WorldCreateConstraint(PhysicsWorldObject *self,
                                                                 PyObject *const *args,
                                                                 size_t nargsf, PyObject *kwnames);

[[nodiscard]] PyObject * WorldDestroyConstraint(PhysicsWorldObject *self,
                                                                  PyObject *const *args,
                                                                  size_t nargsf, PyObject *kwnames);

[[nodiscard]] PyObject * WorldSetConstraintTarget(PhysicsWorldObject *self,
                                                                    PyObject *const *args,
                                                                    size_t nargsf,
                                                                    PyObject *kwnames);

[[nodiscard]] PyObject * WorldGetConstraintType(PhysicsWorldObject *self,
                                                                  PyObject *const *args,
                                                                  size_t nargsf, PyObject *kwnames);

} // namespace CPH::Python
