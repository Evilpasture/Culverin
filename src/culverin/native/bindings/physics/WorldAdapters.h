#pragma once

#include "bindings/BindingCommon.h"
#include "physics/PhysicsWorldState.h"
#include <Python.h>

struct PhysicsWorldObject;

// Python entry points translate Python-side concerns and exceptions around the
// native core operations declared in PhysicsWorldCore.h.
int PhysicsWorld_resize(struct PhysicsWorldObject *self, size_t new_capacity);
void PhysicsWorld_free_members(struct PhysicsWorldObject *self);

int init_settings(struct PhysicsWorldObject *self, PyObject *settings_dict, GravityVector *gravity,
                  WorldSettings *settings);
int load_baked_scene(struct PhysicsWorldObject *self, PyObject *baked);

PyType_DeclareSlot_StatusFromModule PhysicsWorld_getbuffer(struct PhysicsWorldObject *self,
                                                           Py_buffer *view,
                                                           CULV_MAYBE_UNUSED int flags);
PyType_DeclareSlot_VoidFromModule PhysicsWorld_releasebuffer(struct PhysicsWorldObject *self,
                                                             Py_buffer *view);
