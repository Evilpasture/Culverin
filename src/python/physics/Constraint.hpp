#pragma once
#include "physics/PhysicsWorldBinding.hpp"
#include <Python.h>
#include "bindings/Result.hpp"

namespace CPH::Python {

[[nodiscard]] BoundResult WorldCreateConstraint(PhysicsWorldObject *self,
                                                                 PyObject *const *args,
                                                                 size_t nargsf, PyObject *kwnames);

[[nodiscard]] BoundResult WorldDestroyConstraint(PhysicsWorldObject *self,
                                                                  PyObject *const *args,
                                                                  size_t nargsf, PyObject *kwnames);

[[nodiscard]] BoundResult WorldSetConstraintTarget(PhysicsWorldObject *self,
                                                                    PyObject *const *args,
                                                                    size_t nargsf,
                                                                    PyObject *kwnames);

[[nodiscard]] BoundResult WorldGetConstraintType(PhysicsWorldObject *self,
                                                                  PyObject *const *args,
                                                                  size_t nargsf, PyObject *kwnames);

} // namespace CPH::Python
