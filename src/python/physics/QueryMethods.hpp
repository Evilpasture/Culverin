#pragma once

#include "physics/PhysicsWorldBinding.hpp"
#include "physics/Query.h"
#include <Python.h>

namespace CPH::Python {

[[nodiscard]] PyObject * WorldOverlapSphere(PhysicsWorldObject *self,
                                                              PyObject *const *args, size_t nargsf,
                                                              PyObject *kwnames);

[[nodiscard]] PyObject * WorldOverlapAabb(PhysicsWorldObject *self,
                                                            PyObject *const *args, size_t nargsf,
                                                            PyObject *kwnames);

[[nodiscard]] PyObject * WorldRaycast(PhysicsWorldObject *self,
                                                        PyObject *const *args, size_t nargsf,
                                                        PyObject *kwnames);

[[nodiscard]] PyObject * WorldRaycastBatch(PhysicsWorldObject *self,
                                                             PyObject *const *args, size_t nargsf,
                                                             PyObject *kwnames);

[[nodiscard]] PyObject * WorldShapecast(PhysicsWorldObject *self,
                                                          PyObject *const *args, size_t nargsf,
                                                          PyObject *kwnames);

} // namespace CPH::Python
