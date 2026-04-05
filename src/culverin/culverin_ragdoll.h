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

PyCFunction_DeclareMethodFromModule Skeleton_add_joint(SkeletonObject *self, PyObject *const *args,
                                                       Py_ssize_t nargs, PyObject *kwnames);

PyCFunction_DeclareMethodFromModule Skeleton_get_joint_index(SkeletonObject *self,
                                                             PyObject *const *args,
                                                             Py_ssize_t nargs, PyObject *kwnames);

PyCFunction_DeclareMethodFromModule Skeleton_finalize(SkeletonObject *self,
                                                      CULV_MAYBE_UNUSED PyObject *args);

PyCFunction_DeclareMethodFromModule PhysicsWorld_create_ragdoll_settings(PhysicsWorldObject *self,
                                                                         PyObject *const *args,
                                                                         Py_ssize_t nargs,
                                                                         PyObject *kwnames);

PyCFunction_DeclareMethodFromModule RagdollSettings_add_part(RagdollSettingsObject *self,
                                                             PyObject *const *args,
                                                             Py_ssize_t nargs, PyObject *kwnames);

PyCFunction_DeclareMethodFromModule RagdollSettings_stabilize(RagdollSettingsObject *self,
                                                              PyObject *args);

PyCFunction_DeclareMethodFromModule PhysicsWorld_create_ragdoll(PhysicsWorldObject *self,
                                                                PyObject *const *args,
                                                                Py_ssize_t nargs,
                                                                PyObject *kwnames);

PyCFunction_DeclareMethodFromModule Ragdoll_drive_to_pose(RagdollObject *self,
                                                          PyObject *const *args, Py_ssize_t nargs,
                                                          PyObject *kwnames);

PyCFunction_DeclareMethodFromModule Ragdoll_get_body_handles(RagdollObject *self, PyObject *args);

PyCFunction_DeclareMethodFromModule Ragdoll_get_debug_info(RagdollObject *self,
                                                           PyObject *Py_UNUSED(ignored));

PyType_DeclareSlot_VoidFromModule Skeleton_dealloc(SkeletonObject *self);

PyType_DeclareSlot_ObjectFromModule Skeleton_new(PyTypeObject *type, PyObject *args,
                                                 PyObject *kwds);

PyType_DeclareSlot_VoidFromModule RagdollSettings_dealloc(RagdollSettingsObject *self);

PyType_DeclareSlot_VoidFromModule Ragdoll_dealloc(RagdollObject *self);
