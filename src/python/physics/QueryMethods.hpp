#pragma once

#include "physics/PhysicsWorldBinding.hpp"
#include "physics/Query.hpp"
#include <Python.h>
#include "bindings/Result.hpp"

namespace CPH::Python {

[[nodiscard]] BoundResult WorldOverlapSphere(PhysicsWorldObject *self,
                                                              PyObject *const *args, size_t nargsf,
                                                              PyObject *kwnames);

[[nodiscard]] BoundResult WorldOverlapAabb(PhysicsWorldObject *self,
                                                            PyObject *const *args, size_t nargsf,
                                                            PyObject *kwnames);

[[nodiscard]] BoundResult WorldRaycast(PhysicsWorldObject *self,
                                                        PyObject *const *args, size_t nargsf,
                                                        PyObject *kwnames);

[[nodiscard]] BoundResult WorldRaycastBatch(PhysicsWorldObject *self,
                                                             PyObject *const *args, size_t nargsf,
                                                             PyObject *kwnames);

[[nodiscard]] BoundResult WorldShapecast(PhysicsWorldObject *self,
                                                          PyObject *const *args, size_t nargsf,
                                                          PyObject *kwnames);

} // namespace CPH::Python
