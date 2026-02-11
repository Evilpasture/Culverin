#pragma once
#include "culverin.h"

// --- Ragdoll Structures ---

typedef struct SkeletonObject {
  PyObject_HEAD JPH_Skeleton *skeleton;
} SkeletonObject;

typedef struct {
  PyObject_HEAD JPH_RagdollSettings *settings;
  struct PhysicsWorldObject *world; // Kept to access Shape Cache
} RagdollSettingsObject;

typedef struct {
  PyObject_HEAD JPH_Ragdoll *ragdoll;
  struct PhysicsWorldObject *world;

  // We must track the handles of the parts so we can
  // invalid the slots when the ragdoll is destroyed.
  size_t body_count;
  uint32_t *body_slots;
} RagdollObject;

PyObject *Skeleton_add_joint(SkeletonObject *self, PyObject *args);

PyObject *Skeleton_get_joint_index(SkeletonObject *self, PyObject *args);

PyObject *Skeleton_finalize(SkeletonObject *self, PyObject *args);

PyObject *PhysicsWorld_create_ragdoll_settings(struct PhysicsWorldObject *self,
                                                      PyObject *args);

PyObject *RagdollSettings_add_part(RagdollSettingsObject *self,
                                          PyObject *args, PyObject *kwds);

PyObject *RagdollSettings_stabilize(RagdollSettingsObject *self,
                                           PyObject *args);

PyObject *PhysicsWorld_create_ragdoll(struct PhysicsWorldObject *self,
                                             PyObject *args, PyObject *kwds);

PyObject *Ragdoll_drive_to_pose(RagdollObject *self, PyObject *args,
                                       PyObject *kwds);
                                       
PyObject *Ragdoll_get_body_ids(RagdollObject *self, PyObject *args);

PyObject *Ragdoll_get_debug_info(RagdollObject *self,
                                        PyObject *Py_UNUSED(ignored));

void Skeleton_dealloc(SkeletonObject *self);

PyObject *Skeleton_new(PyTypeObject *type, PyObject *args,
                              PyObject *kwds);

void RagdollSettings_dealloc(RagdollSettingsObject *self);

void Ragdoll_dealloc(RagdollObject *self);