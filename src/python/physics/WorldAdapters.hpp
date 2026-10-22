#pragma once

#include "bindings/PythonCommon.hpp"
#include "physics/PhysicsWorldState.hpp"
#include <Python.h>

namespace CPH::Python {

struct PhysicsWorldObject;

// Python entry points translate Python-side concerns and exceptions around the
// native core operations declared in PhysicsWorld.h.
int WorldResize(struct PhysicsWorldObject *self, size_t new_capacity);
void WorldFreeMembers(struct PhysicsWorldObject *self);

int InitSettings(struct PhysicsWorldObject *self, PyObject *settings_dict,
                     JPH::Float3 *gravity, WorldSettings *settings);
int GetBakedSceneCount(PyObject *baked, size_t *out_count);
int LoadBakedScene(struct PhysicsWorldObject *self, PyObject *baked);

[[nodiscard]] int WorldGetbuffer(struct PhysicsWorldObject *self,
                                                          Py_buffer *view,
                                                          [[maybe_unused]] int flags);
void WorldReleasebuffer(struct PhysicsWorldObject *self,
                                                            Py_buffer *view);

} // namespace CPH::Python
