#pragma once

#include "bindings/BindingCommon.h"
#include "physics/PhysicsWorldState.h"
#include <Python.h>

struct CPH_PhysicsWorldObject;

// Python entry points translate Python-side concerns and exceptions around the
// native core operations declared in PhysicsWorld.h.
int CPH_PyPhysicsWorldResize(struct CPH_PhysicsWorldObject *self, size_t new_capacity);
void CPH_PyPhysicsWorldFreeMembers(struct CPH_PhysicsWorldObject *self);

int CPH_InitSettings(struct CPH_PhysicsWorldObject *self, PyObject *settings_dict, CPH_GravityVector *gravity,
                  CPH_WorldSettings *settings);
int CPH_GetBakedSceneCount(PyObject *baked, size_t *out_count);
int CPH_LoadBakedScene(struct CPH_PhysicsWorldObject *self, PyObject *baked);

PyType_DeclareSlot_StatusFromModule CPH_PyPhysicsWorldGetbuffer(struct CPH_PhysicsWorldObject *self,
                                                           Py_buffer *view,
                                                           [[maybe_unused]] int flags);
PyType_DeclareSlot_VoidFromModule CPH_PyPhysicsWorldReleasebuffer(struct CPH_PhysicsWorldObject *self,
                                                             Py_buffer *view);
